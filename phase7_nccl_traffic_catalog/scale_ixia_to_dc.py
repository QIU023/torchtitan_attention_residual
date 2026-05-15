#!/usr/bin/env python3
"""Scale 8-GPU NCCL trace IXIA configs to datacenter-scale presets.

Takes a source ``ixia_config.json`` (emitted by ``phase7_nccl_traffic_catalog/flows_to_ixia.py``
from a real 8-GPU torchtitan trace) and emits a scaled-up config that
projects the source mesh's collective sequence onto a published large-LLM
parallelism plan (Llama 3 405B / DeepSeek-V3 671B / Kimi-K2-style 10K-GPU).

What the script does
--------------------
For each source ``trafficItem`` (already tagged with ``axis_guess`` =
fsdp/tp/pp/ep/cp by ``flows_to_ixia.py``), the script:

1. Looks up the target topology's rank count for that axis (e.g. target
   FSDP=128, TP=8, PP=16, EP=8 — paper-published numbers).
2. Re-emits ONE aggregated traffic item per (axis, source-rank-pair)
   tagged with ``target_axis_ranks`` so IxNetwork's "endpoint multiplier"
   feature can fan it out to the full datacenter scale without bloating
   the JSON to millions of items.
3. Scales ``frameCount`` (i.e. cumulative bytes/call) by axis-specific
   formulas:
     - DP / FSDP all-gather + reduce-scatter: bytes/call ∝
       (target_params / source_params) × (source_fsdp / target_fsdp)
       (each rank holds 1/fsdp of the params, so per-call AG bytes
       scale with model size and inverse with shard ranks).
     - TP all-reduce / RS+AG: bytes/call ∝ hidden(target) / hidden(source).
     - PP send/recv: bytes/call ∝ (hidden × seq_len) ratio.
     - EP all-to-all: bytes/call ∝ (hidden × seq × experts_per_token).
     - CP all-gather of K/V: bytes/call ∝ (hidden × seq) ratio.
4. Scales the per-axis call count by depth ratio
   (target_layers / source_layers).
5. Sets ``interFrameGap_ns`` from the requested target link bandwidth
   (default 400 Gbps NDR-class).

Output JSON includes a ``scaled`` summary block listing per-axis aggregate
bytes/step at target scale, so a network engineer can sanity-check the
fabric demand before pushing into IxNetwork.

Target topology references
--------------------------

* **llama3-405b-16k** — Llama 3 paper (Meta AI, 2024), TP=8 PP=16 FSDP=128
  on 16,384 H100. Dense, no MoE/CP in the public 16K-GPU configuration.
* **deepseek-v3-671b-2k** — DeepSeek-V3 technical report (2024), PP=16
  FSDP=16 EP=8 (for 256-expert MoE), TP=1, on 2,048 H800.
* **kimi-k2-style-10k** — synthesized 10,240-GPU plan inspired by
  Kimi K2 / DeepSeek-V3 public hints: TP=4 PP=16 FSDP=20 EP=8 with
  trillion-param MoE; not a paper-faithful number, marked as such in
  the output metadata.

Usage
-----

::

    python phase7_nccl_traffic_catalog/scale_ixia_to_dc.py \\
        --source phase5_vlm_multimodal_sft/runs/v11_4d_fsdp2_pp2_tp2_ep2_continue_8gpu_from_p4_step8000/tier_b_trace/ixia_config.json \\
        --preset llama3-405b-16k \\
        --link-bandwidth-gbps 400 \\
        --out phase7_nccl_traffic_catalog/scaled_configs/v11_pretrain_to_llama3-405b-16k.json
"""
from __future__ import annotations

import argparse
import json
from copy import deepcopy
from pathlib import Path
from typing import Any


# ---- Source mesh: the actual 8-GPU box used in this project ----
SOURCE_MESH: dict[str, Any] = {
    "world_size": 8,
    "fsdp": 2,
    "pp": 2,
    "tp": 2,
    "ep": 2,
    "cp": 1,
    "model_params_b": 0.436,
    "active_params_b": 0.436,
    "hidden_size": 1168,
    "num_layers": 16,
    "num_experts": 8,
    "experts_per_token": 2,
    "seq_len": 256,
    "ref": "Phase 5 v11_4d AttnRes-Kimi-Linear pretrain on 8x RTX 5090 PCIe.",
}

# ---- Target mesh presets (paper-published or paper-shape-aligned) ----
TARGET_PRESETS: dict[str, dict[str, Any]] = {
    "llama3-405b-16k": {
        "world_size": 16384,
        "fsdp": 128,
        "pp": 16,
        "tp": 8,
        "ep": 1,
        "cp": 1,
        "model_params_b": 405,
        "active_params_b": 405,
        "hidden_size": 16384,
        "num_layers": 126,
        "num_experts": 1,
        "experts_per_token": 1,
        "seq_len": 8192,
        "ref": "Llama 3 405B, Meta AI 2024 (arxiv:2407.21783). Dense, "
               "TP=8 PP=16 FSDP=128 on 16,384 H100.",
    },
    "deepseek-v3-671b-2k": {
        "world_size": 2048,
        "fsdp": 16,
        "pp": 16,
        "tp": 1,
        "ep": 8,
        "cp": 1,
        "model_params_b": 671,
        "active_params_b": 37,
        "hidden_size": 7168,
        "num_layers": 61,
        "num_experts": 256,
        "experts_per_token": 8,
        "seq_len": 4096,
        "ref": "DeepSeek-V3 671B MoE, DeepSeek 2024 (arxiv:2412.19437). "
               "PP=16 FSDP=16 EP=8 TP=1 on 2,048 H800.",
    },
    "kimi-k2-style-10k": {
        "world_size": 10240,
        "fsdp": 20,
        "pp": 16,
        "tp": 4,
        "ep": 8,
        "cp": 1,
        "model_params_b": 1000,
        "active_params_b": 50,
        "hidden_size": 8192,
        "num_layers": 80,
        "num_experts": 512,
        "experts_per_token": 8,
        "seq_len": 8192,
        "ref": "Synthesized: trillion-param MoE on 10,240 H100/H200, "
               "shape inspired by DeepSeek-V3 and Kimi K2 public hints. "
               "Not a paper-faithful plan, included as a 10K-GPU MoE "
               "reference point.",
    },
}


def make_endpoints(world_size: int) -> list[dict]:
    """Synthesize a flat endpoint list with synthetic IPv4/MAC.

    Pack 2 ports per card so the IxNetwork chassis layout stays linear
    (a 16,384-rank target needs 8,192 cards x 2 ports). Real datacenter
    layouts use 8 NICs/node x 1 port each; pick whichever the IXIA
    chassis supports, this is just a starting representation.
    """
    eps = []
    for r in range(world_size):
        card = (r // 2) + 1
        port = (r % 2) + 1
        oct3 = (r >> 16) & 0xff
        oct2 = (r >> 8) & 0xff
        oct1 = (r & 0xff) + 1
        eps.append({
            "name": f"rank_{r}",
            "port": f"card_{card}/port_{port}",
            "ipv4": f"10.{oct3}.{oct2}.{oct1}",
            "mac": f"02:{oct3:02x}:{oct2:02x}:{oct1:02x}:00:00",
        })
    return eps


def axis_rank_count(target: dict, axis: str) -> int:
    """Number of ranks participating in a collective on this axis."""
    a = axis.lower()
    return {
        "dp": target["fsdp"],
        "fsdp": target["fsdp"],
        "tp": target["tp"],
        "pp": 2,
        "ep": target["ep"],
        "cp": target["cp"],
    }.get(a, 2)


def per_call_byte_scale(source: dict, target: dict, axis: str) -> float:
    """Multiplier for ``frameCount * frameSize`` of one collective call.

    Returns the scale factor accounting for both model-size growth and
    per-axis rank count change. See module docstring for formulas.
    """
    p_t, p_s = target["model_params_b"], source["model_params_b"]
    h_t, h_s = target["hidden_size"], source["hidden_size"]
    seq_t, seq_s = target["seq_len"], source["seq_len"]
    a = axis.lower()

    if a in ("dp", "fsdp"):
        return (p_t / p_s) * (source["fsdp"] / target["fsdp"])
    if a == "tp":
        return h_t / h_s
    if a == "pp":
        return (h_t * seq_t) / (h_s * seq_s)
    if a == "ep":
        return (h_t * seq_t * target["experts_per_token"]) / \
               (h_s * seq_s * source["experts_per_token"])
    if a == "cp":
        return (h_t * seq_t) / (h_s * seq_s)
    return 1.0


def per_call_count_scale(source: dict, target: dict) -> float:
    """Number-of-calls-per-step depth scaling: more layers → more calls."""
    return target["num_layers"] / source["num_layers"]


def compute_ifg_for_link(frame_size: int, link_bw_gbps: float,
                         line_rate_pct: float = 0.8) -> int:
    """Compute inter-frame gap in ns for the requested link saturation.

    bytes/sec_used = line_rate_pct * link_bw_gbps * 1e9 / 8
    sec_per_frame_used = frame_size / bytes/sec_used
    sec_per_frame_total = sec_per_frame_used / line_rate_pct
    ifg = sec_per_frame_total - serialize_time(=sec_per_frame_used at line rate)
        = sec_per_frame_used * (1 / line_rate_pct - 1)
    """
    bps_used = link_bw_gbps * line_rate_pct * 1e9
    serialize_ns = frame_size * 8 / bps_used * 1e9
    ifg_ns = serialize_ns * (1.0 / line_rate_pct - 1.0)
    return max(4, int(ifg_ns))


def scale_traffic_item(item: dict, source: dict, target: dict,
                       jumbo_size: int, link_bw_gbps: float) -> dict:
    """Scale one source trafficItem to a single target trafficItem.

    The ``target_axis_ranks`` field tells IxNetwork (or the operator) how
    many parallel rings of this collective exist in the target topology
    — use the GUI's endpoint multiplier to fan out, do not expand here.
    """
    axis = item.get("metadata", {}).get("axis_guess", "dp")
    bytes_factor = per_call_byte_scale(source, target, axis)
    count_factor = per_call_count_scale(source, target)
    target_axis_ranks = axis_rank_count(target, axis)

    src_bytes = item["frameCount"] * item["frameSize"]["value"]
    new_bytes_per_call = max(1, int(src_bytes * bytes_factor))
    new_frame_count = max(1, new_bytes_per_call // jumbo_size)

    # Apply depth (number-of-calls) scaling on top
    new_frame_count = max(1, int(new_frame_count * count_factor))
    new_total_bytes = new_frame_count * jumbo_size

    new_ifg = compute_ifg_for_link(jumbo_size, link_bw_gbps)
    duration_ns = new_frame_count * (new_ifg + jumbo_size * 8 * 1000 / (link_bw_gbps * 1e3))

    out = deepcopy(item)
    out["name"] = f"{item['name']}_scaled"
    out["frameSize"] = {"type": "fixed", "value": jumbo_size}
    out["frameCount"] = new_frame_count
    out["interFrameGap_ns"] = new_ifg
    out["duration_us"] = max(1, int(duration_ns / 1000))
    out["metadata"]["scaled_from_source"] = True
    out["metadata"]["bytes_total"] = new_total_bytes
    out["metadata"]["target_axis_ranks"] = target_axis_ranks
    out["metadata"]["bytes_factor"] = round(bytes_factor, 3)
    out["metadata"]["count_factor"] = round(count_factor, 3)
    return out


def scale_config(source_path: Path, target_preset: str,
                 link_bw_gbps: float, jumbo_size: int,
                 source_steps: int, target_steps: int,
                 target_step_time_sec: float,
                 out_path: Path) -> dict:
    target = TARGET_PRESETS[target_preset]
    src = json.load(open(source_path))

    # Normalize source bytes to per-step, then scale to target_steps so
    # readers can compare apples-to-apples regardless of how many steps
    # the source trace happened to record (50 / 100 / 5000).
    step_normalize = target_steps / max(1, source_steps)

    target_endpoints = make_endpoints(target["world_size"])
    target_items = []
    for it in src["trafficItems"]:
        scaled = scale_traffic_item(it, SOURCE_MESH, target, jumbo_size, link_bw_gbps)
        scaled["frameCount"] = max(1, int(scaled["frameCount"] * step_normalize))
        scaled["metadata"]["bytes_total"] = scaled["frameCount"] * jumbo_size
        scaled["metadata"]["step_normalize"] = round(step_normalize, 4)
        target_items.append(scaled)

    # Per-axis aggregate
    axis_bytes: dict[str, int] = {}
    axis_count: dict[str, int] = {}
    for it in target_items:
        a = it.get("metadata", {}).get("axis_guess", "unknown")
        axis_bytes[a] = axis_bytes.get(a, 0) + it["metadata"]["bytes_total"]
        axis_count[a] = axis_count.get(a, 0) + 1

    total_bytes = sum(axis_bytes.values())
    aggregate_gbps = (total_bytes * 8 / 1e9) / (target_steps * target_step_time_sec) \
                     if (target_steps * target_step_time_sec) else 0.0

    out = {
        "schema": "ixnetwork-traffic-config-v1",
        "scaled": {
            "source": {
                "path": str(source_path),
                "world_size": SOURCE_MESH["world_size"],
                "model_params_b": SOURCE_MESH["model_params_b"],
                "axes": {k: SOURCE_MESH[k] for k in ("fsdp", "pp", "tp", "ep", "cp")},
            },
            "target_preset": target_preset,
            "target": {
                "world_size": target["world_size"],
                "model_params_b": target["model_params_b"],
                "active_params_b": target.get("active_params_b"),
                "axes": {k: target[k] for k in ("fsdp", "pp", "tp", "ep", "cp")},
                "hidden_size": target["hidden_size"],
                "num_layers": target["num_layers"],
                "ref": target["ref"],
            },
            "link_bandwidth_gbps": link_bw_gbps,
            "jumbo_frame_size": jumbo_size,
            "source_steps_in_trace": source_steps,
            "target_steps_in_window": target_steps,
            "step_normalize_factor": round(step_normalize, 4),
            "target_step_time_sec_assumed": target_step_time_sec,
            "axis_bytes_total_GB": {k: round(v / 1e9, 2) for k, v in axis_bytes.items()},
            "axis_item_count": axis_count,
            "total_bytes_GB": round(total_bytes / 1e9, 2),
            "total_bytes_per_step_GB": round(total_bytes / 1e9 / target_steps, 2),
            "aggregate_demand_Gbps": round(aggregate_gbps, 2),
            "per_port_demand_Gbps": (
                round(aggregate_gbps / target["world_size"], 2)
                if target["world_size"] else 0.0
            ),
            "n_traffic_items": len(target_items),
        },
        "topology": {"endpoints": target_endpoints},
        "trafficItems": target_items,
    }
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w") as f:
        json.dump(out, f, indent=2)
    return out["scaled"]


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--source", type=Path, required=True,
                   help="Path to source ixia_config.json (8-GPU trace).")
    p.add_argument("--preset", choices=list(TARGET_PRESETS.keys()), required=True)
    p.add_argument("--link-bandwidth-gbps", type=float, default=400,
                   help="Target link bandwidth (default 400 Gbps NDR-class).")
    p.add_argument("--jumbo-frame-size", type=int, default=9000)
    p.add_argument("--source-steps", type=int, default=5000,
                   help="Steps recorded in the source trace (default 5000 — "
                        "matches v11 5000-step pretrain trace; use 50 for "
                        "tier_b smoke traces).")
    p.add_argument("--target-steps", type=int, default=50,
                   help="Steps to project into the target IXIA window "
                        "(default 50, a typical IxNetwork run length).")
    p.add_argument("--target-step-time-sec", type=float, default=5.0,
                   help="Wallclock per step at target scale, used to "
                        "convert aggregate bytes into Gbps. Default 5s "
                        "(typical 4D pretrain). Use ~0.2 for inference.")
    p.add_argument("--out", type=Path, required=True)
    args = p.parse_args()

    summary = scale_config(args.source, args.preset, args.link_bandwidth_gbps,
                           args.jumbo_frame_size,
                           args.source_steps, args.target_steps,
                           args.target_step_time_sec,
                           args.out)
    print(f"\nScaled config written: {args.out}")
    print(f"  source:  {summary['source']['world_size']} GPU @ "
          f"{summary['source']['model_params_b']}B params, "
          f"axes={summary['source']['axes']}")
    print(f"  target:  {summary['target']['world_size']} GPU @ "
          f"{summary['target']['model_params_b']}B params, "
          f"axes={summary['target']['axes']}")
    print(f"  link:    {summary['link_bandwidth_gbps']} Gbps per port")
    print(f"  steps:   source={summary['source_steps_in_trace']} -> "
          f"target window={summary['target_steps_in_window']}")
    print(f"  total:   {summary['total_bytes_GB']:.1f} GB across "
          f"{summary['target_steps_in_window']} steps")
    print(f"  per-step: {summary['total_bytes_per_step_GB']:.2f} GB / step")
    print(f"  by axis: " + ", ".join(f"{k}={v:.1f}GB"
                                      for k, v in summary['axis_bytes_total_GB'].items()))
    print(f"  aggregate fabric demand: {summary['aggregate_demand_Gbps']:.1f} Gbps "
          f"(at {summary['target_step_time_sec_assumed']}s/step)")
    print(f"  per-port average:        {summary['per_port_demand_Gbps']:.2f} Gbps")
    print(f"  traffic items:           {summary['n_traffic_items']}")


if __name__ == "__main__":
    main()
