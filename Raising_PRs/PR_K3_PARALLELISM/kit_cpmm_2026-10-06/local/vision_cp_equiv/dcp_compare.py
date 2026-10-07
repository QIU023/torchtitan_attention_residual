import sys
import torch
out_dir = sys.argv[1]
bad = total = 0
for rank in range(4):
    a = torch.load(f"{out_dir}/old.{rank}.pt")
    b = torch.load(f"{out_dir}/new.{rank}.pt")
    if rank == 0: print("old from", a.pop("_tree")); print("new from", b.pop("_tree"))
    else: a.pop("_tree"); b.pop("_tree")
    assert a.keys() == b.keys()
    for name in a:
        (oa, ga), (ob, gb) = a[name], b[name]
        total += 1 + len(ga)
        if not torch.equal(oa, ob):
            bad += 1; print("output differs", rank, name)
        assert ga.keys() == gb.keys()
        for p in ga:
            if not torch.equal(ga[p], gb[p]):
                bad += 1; print("grad differs", rank, name, p)
print(f"{total - bad} of {total} tensors bitwise equal ({len(a)} cases x 4 ranks)")
