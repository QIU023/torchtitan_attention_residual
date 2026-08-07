import glob, json, os, statistics, sys
o = sys.argv[1]
def merge(n):
    d = {}
    for f in sorted(glob.glob(os.path.join(o, n + ".json.r*"))):
        d.update(json.load(open(f)))
    return d or None
DP1 = ["pp2", "cp2", "tp2", "pp2_cp2"]
DP2 = ["fsdp2", "fsdp2_pp2", "fsdp2_cp2", "fsdp2_pp2_cp2",
       "ep2_fsdp2", "ep2_fsdp2_pp2", "ep2_fsdp2_cp2"]
for legs, refname in ((DP1, "ref"), (DP2, "ref_dp2")):
    base = merge(refname)
    if not base:
        print(f"\n!! reference {refname} missing"); continue
    print(f"\n== against {refname} ({len(base)} params) ==")
    for n in legs:
        d = merge(n)
        if not d:
            print(f"  {n:<18} NO DUMP"); continue
        rows = [(k, base[k]/d[k]) for k in base
                if base[k] > 1e-9 and d.get(k, 0) > 1e-9]
        if not rows:
            print(f"  {n:<18} no comparable params"); continue
        rows.sort(key=lambda r: -abs(r[1]-1))
        lo = [r for r in rows if "lora_" in r[0]]
        lmax = max(abs(r[1]-1) for r in lo) if lo else float("nan")
        big = [r for r in rows if "lora_" in r[0] or "res_" in r[0]]
        print(f"  {n:<18} n={len(rows):<4} max={max(abs(r[1]-1) for r in rows):.5f} "
              f"med={statistics.median(abs(r[1]-1) for r in rows):.5f} "
              f"LoRAmax={lmax:.5f}  worst={'.'.join(rows[0][0].split('.')[-2:])}")
