import glob, os, sys, torch
O = sys.argv[1]
def load(cell):
    d = {}
    for f in glob.glob(f"{O}/qbdump_{cell}/rank*_call*_layer*.pt"):
        r = torch.load(f); base = os.path.basename(f)
        rank = int(base.split("_")[0][4:]); call = int(base.split("_")[1][4:])
        d.setdefault((call, r["fqn"]), []).append((rank, r))
    return d
A, B = load("dp2"), load("pp2")
print(f"dp2 entries {len(A)}  pp2 entries {len(B)}")
for key in sorted(set(A) | set(B)):
    a = A.get(key, []); b = B.get(key, [])
    if not a or not b: print(key, "missing on one side", len(a), len(b)); continue
    ra, rb = a[0][1], b[0][1]
    hist_eq = torch.equal(ra["hist"], rb["hist"]); nb_eq = torch.equal(ra["bias_next"], rb["bias_next"]); bb_eq = torch.equal(ra["bias_before"], rb["bias_before"])
    dh = (ra["hist"].double() - rb["hist"].double()).abs().max().item(); dn = (ra["bias_next"] - rb["bias_next"]).abs().max().item()
    # consistency across dp ranks within a cell (they should be identical after the reduce)
    intra = all(torch.equal(a[0][1]["bias_next"], x[1]["bias_next"]) for x in a) and all(torch.equal(b[0][1]["bias_next"], x[1]["bias_next"]) for x in b)
    print(f"call {key[0]} {key[1]:22s} hist bitwise={hist_eq!s:5s} (max|diff| {dh:g})  bias_before bitwise={bb_eq!s:5s}  bias_next bitwise={nb_eq!s:5s} (max|diff| {dn:.3e})  intra-cell consistent={intra}  dp2 ranks={sorted(x[0] for x in a)} pp2 ranks={sorted(x[0] for x in b)}")
