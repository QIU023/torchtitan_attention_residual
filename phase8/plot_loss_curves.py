"""Phase 8 — extract loss + grad_norm from train.log files (no TB dep)."""
from pathlib import Path
import re
import json

WORKSPACE = Path("/root/torchtitan_attention_residual/phase5/runs")
RUNS = {
    "v11_4d_pretrain": WORKSPACE / "v11_4d_fsdp2_pp2_tp2_ep2_continue_8gpu_from_p4_step8000",
    "v12_4d_no_tp_pretrain": WORKSPACE / "v12_4d_fsdp2_dp2_pp2_ep2_continue_8gpu_from_p4_step8000",
    "sft_v11_llava_instruct": WORKSPACE / "sft_v11_llava_instruct_150k_4d",
}
ANSI = re.compile(r"\x1b\[[0-9;]*m")
PATTERN = re.compile(r"step:\s*(\d+)\s+loss:\s*([\d.]+)\s+grad_norm:\s*([\d.]+).*?memory:\s*([\d.]+)GiB.*?tps:\s*([\d,]+).*?mfu:\s*([\d.]+)")

result = {}
for name, run_dir in RUNS.items():
    log = run_dir / "train.log"
    if not log.exists():
        continue
    rows = []
    with log.open() as f:
        for line in f:
            m = PATTERN.search(ANSI.sub("", line))
            if m:
                rows.append({
                    "step": int(m.group(1)),
                    "loss": float(m.group(2)),
                    "grad_norm": float(m.group(3)),
                    "memory_gib": float(m.group(4)),
                    "tps": int(m.group(5).replace(",", "")),
                    "mfu_pct": float(m.group(6)),
                })
    print(f"{name}: {len(rows)} steps")
    if rows:
        print(f"  step range: {rows[0]['step']}–{rows[-1]['step']}")
        print(f"  loss: {rows[0]['loss']:.4f} → {rows[-1]['loss']:.4f}")
        print(f"  TPS avg: {sum(r['tps'] for r in rows) / len(rows):.0f}")
        print(f"  MFU avg: {sum(r['mfu_pct'] for r in rows) / len(rows):.2f}%")
    result[name] = rows

out = Path("/root/torchtitan_attention_residual/phase8/eval_results/loss_curves.json")
out.write_text(json.dumps({k: v[::5] for k, v in result.items()}, indent=2))
print(f"wrote {out} (5x downsampled)")
