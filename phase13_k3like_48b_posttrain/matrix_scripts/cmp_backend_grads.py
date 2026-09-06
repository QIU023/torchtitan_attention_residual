"""Compare two step-1 gradient dumps (rank 0, every parameter, own dtype) of the same cell under two
backends: bitwise count, relative norm difference, largest elementwise relative difference, and the
sign-flip fraction (elements whose gradient sign differs). Usage: cmp_backend_grads.py A.pt B.pt"""
import sys, torch, re
a = torch.load(sys.argv[1]); b = torch.load(sys.argv[2])
keys = [k for k in a if k in b]
missing = sorted(set(a) ^ set(b))
tot = 0; flips = 0; bitwise = 0; worst = []
for k in keys:
    x = a[k].float(); y = b[k].float()
    if x.shape != y.shape:
        print("SHAPE", k, tuple(x.shape), tuple(y.shape)); continue
    n = x.numel(); tot += n
    eq = torch.equal(x, y); bitwise += eq
    f = ((torch.sign(x) != torch.sign(y)) & ((x != 0) | (y != 0))).sum().item(); flips += f
    rel = ((x - y).norm() / max(x.norm().item(), 1e-30)).item()
    worst.append((rel, k, f / max(n, 1), n))
worst.sort(reverse=True)
print(f"params compared {len(keys)}  bitwise {bitwise}/{len(keys)}  elements {tot}  sign flips {flips} ({100*flips/max(tot,1):.4f}%)")
if missing: print("only in one dump:", missing[:10], "..." if len(missing) > 10 else "")
print("largest relative norm differences:")
for rel, k, fr, n in worst[:12]:
    print(f"  {rel:.3e}  flips {100*fr:.3f}%  n={n:<10d} {re.sub(r'_checkpoint_wrapped_module\.', '', k)}")
print(f"median relative norm difference {sorted(w[0] for w in worst)[len(worst)//2]:.3e}")
