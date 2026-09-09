#!/usr/bin/env python3
"""Full TP/SP matrix on the 8-GPU box: dp/ep x tp{1,2,4} x SP{on,off} x backend, 10 steps.

Greedy GPU packing over mx3.sh cells; per-cell triton cache; summary tables in
PR 4500's format (loss and grad norm per configuration, steps 1/3/10, percent
against the stream's parent-tree reference). Usage: run_full.py [--summarize-only TAG]
"""
import os, subprocess, sys, time, glob, re, json

MX = "/workspace/torchtitan_attention_residual/phase13_k3like_48b_posttrain/matrix_scripts/mx3.sh"
TREE = os.environ.get("TREE", "/tmp/wt_tp4527")
PARENT = os.environ.get("PARENT_TREE", "/tmp/wt_pr4527")
TAG = os.environ.get("TAG", "tpfull")
B1 = "--training.num-tokens-per-train-step 256 --training.num-tokens-per-microbatch-per-dp-rank 256"
B2 = "--training.num-tokens-per-train-step 512 --training.num-tokens-per-microbatch-per-dp-rank 256"
D = "--parallelism.data_parallel_shard_degree"; T = "--parallelism.tensor_parallel_degree"; E = "--parallelism.expert_parallel_degree"
ST = "--parallelism.spmd_backend spmd_types"; PD = "--parallelism.spmd_backend partial_dtensor"; NOSP = "--parallelism.no-enable-sequence-parallel"

def cells():
    out = []
    for stream, dp, ep, batch in (("dp1", 1, 1, B1), ("dp2", 2, 1, B2), ("dp2ep2", 2, 2, B2)):
        base = f"{D} {dp}" + (f" {E} {ep}" if ep > 1 else "")
        for be, bef in (("pd", PD), ("st", ST)):
            out.append((f"{stream}_parent_tp1_{be}", dp, PARENT, batch, f"{base} {bef}"))
            out.append((f"{stream}_tp1_{be}", dp, TREE, batch, f"{base} {bef}"))
            for tp in (2, 4):
                if dp * tp > 8: continue
                for sp, spf in (("sp", ""), ("nosp", NOSP)):
                    out.append((f"{stream}_tp{tp}_{sp}_{be}", dp * tp, TREE, batch, f"{base} {T} {tp} {bef} {spf}".strip()))
    return out

def launch(cell, gpus):
    name, n, tree, batch, flags = cell
    env = dict(os.environ, VENV="/workspace/venv_bfx9", PYPRE="/tmp/attn_gym_up", MEASURE_STEPS="10", WARM_STEPS="1",
               SEED_ROOT="/workspace/.mx3_seeds_tp4527", SEED_CFG="kimi_k3_debugmodel",
               CUDA_VISIBLE_DEVICES=",".join(map(str, gpus)), TRITON_CACHE_DIR=f"/workspace/.triton_{TAG}/{name}",
               TITAN=tree, CFG="kimi_k3_debugmodel", BATCH=batch, CELLS=f"{name}|{n}|{flags}")
    os.makedirs(env["TRITON_CACHE_DIR"], exist_ok=True)
    log = open(f"/workspace/{TAG}_{name}.launch.log", "w")
    return subprocess.Popen(["bash", MX, f"{TAG}_{name}"], env=env, stdout=log, stderr=subprocess.STDOUT)

def run():
    queue = [c for c in cells() if not (parse(c[0]) or {}).get(10)]  # skip cells whose measure log reached step 10
    queue.sort(key=lambda c: -c[1]); free = [int(g) for g in os.environ.get("GPUS", "0,1,2,3,4,5,6,7").split(",")]; running = []
    print(f"{len(queue)} cells", flush=True)
    while queue or running:
        for p, cell, gpus in list(running):
            if p.poll() is not None:
                running.remove((p, cell, gpus)); free += gpus; print(time.strftime("%H:%M"), "done", cell[0], "rc", p.returncode, flush=True)
        launched = True
        while launched and queue:
            launched = False
            for cell in queue:
                if cell[1] <= len(free):
                    gpus = sorted(free)[:cell[1]]; free = [g for g in free if g not in gpus]
                    running.append((launch(cell, gpus), cell, gpus)); queue.remove(cell); launched = True
                    print(time.strftime("%H:%M"), "start", cell[0], "on", gpus, flush=True); break
        time.sleep(15)

def parse(name):
    dirs = sorted(glob.glob(f"/workspace/mx3_{TAG}_{name}_*"))
    if not dirs: return None
    logs = glob.glob(f"{dirs[-1]}/*measure.log")
    if not logs: return {"abort": "no measure log"}
    txt = open(logs[0], errors="replace").read()
    res = {}
    for m in re.finditer(r"step:\s*(\d+)\s.*?loss:\s*([0-9.]+).*?grad_norm:\s*([0-9.]+)", txt):
        res[int(m.group(1))] = (float(m.group(2)), float(m.group(3)))
    ab = re.search(r"ABORT.*", open(f"{dirs[-1]}/results.txt").read())
    if ab: res["abort"] = ab.group(0)[:60]
    return res

def fmt(v, ref):
    if v is None or ref is None: return "n/a"
    if v == ref: return f"`{v:.6f}` (bitwise)" if isinstance(v, float) and v < 100 else f"`{v}` (bitwise)"
    return f"`{v:.6f}` (`{100*abs(v-ref)/ref:.3g}%`)" if v < 100 else f"`{v}` (`{100*abs(v-ref)/ref:.3g}%`)"

def summarize():
    names = [c[0] for c in cells()]; R = {n: parse(n) for n in names}
    for stream in ("dp1", "dp2", "dp2ep2"):
        ref = R.get(f"{stream}_parent_tp1_pd") or {}
        cols = [n for n in names if n.startswith(stream + "_")]
        print(f"\n### stream {stream} (reference {stream}_parent_tp1_pd)\n")
        print("| cell | " + " | ".join(f"loss s{s} | grad norm s{s}" for s in (1, 3, 10)) + " |")
        print("| --- |" + " ---: |" * 6)
        for n in cols:
            r = R.get(n) or {}
            row = []
            for s in (1, 3, 10):
                v = r.get(s); rv = ref.get(s)
                if v is None: row += [r.get("abort", "missing") if s == 1 else "", ""]; continue
                row += [fmt(v[0], rv[0] if rv else None), fmt(v[1], rv[1] if rv else None)]
            print(f"| {n} | " + " | ".join(row) + " |")

if __name__ == "__main__":
    if "--summarize-only" in sys.argv: summarize()
    else: run(); summarize(); print("FULL-DONE")
