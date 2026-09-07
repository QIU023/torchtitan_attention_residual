"""Compare the vLLM rollout model after verl's first sync with the same model loaded from disk (vLLM-internal names)."""
import glob, torch, collections
disk = torch.load("/workspace/vllm_disk_params.pt")
f = sorted(glob.glob("/workspace/vllm_sync_dump/vllm_after_sync_*.pt"))[0]; sync = torch.load(f); print("synced dump:", f, len(sync), "tensors; disk:", len(disk))
same, diff, missing = 0, [], []
for k, v in disk.items():
    if k not in sync: missing.append(k); continue
    a, b = v.float(), sync[k].float()
    if a.shape != b.shape: diff.append((float("inf"), k, f"shape {tuple(a.shape)} vs {tuple(b.shape)}")); continue
    m = (a - b).abs().max().item()
    if m == 0: same += 1
    else: diff.append((m, k, f"max|d|={m:.3g} |disk|={a.abs().mean():.3g} |sync|={b.abs().mean():.3g}"))
extra = [k for k in sync if k not in disk]
print(f"identical: {same}, differing: {len(diff)}, only in disk dump: {len(missing)}, only in synced dump: {len(extra)}")
by_kind = collections.Counter()
for m, k, _ in diff:
    kind = ".".join(p for p in k.split(".") if not p.isdigit()); by_kind[kind] += 1
print("differing tensors by kind:"); [print(f"  {c:4d}  {kind}") for kind, c in by_kind.most_common()]
print("worst 12:"); [print(f"  {k}: {s}") for m, k, s in sorted(diff, reverse=True)[:12]]
if missing: print("only in disk:", missing[:10])
if extra: print("only in synced:", extra[:10])
