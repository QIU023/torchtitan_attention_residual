"""LOCAL PROBE HACK (not committed): GN_FP32=1 makes clip_grad_norm_ take the total norm of float32 copies of the
gradients, so the norm (and the clip factor, max_norm / norm, which fires every step here) no longer depends on how
PP partitions the parameters. Same effect as torchtitan 5e88ff897 + pytorch PR 194033 for this purpose. argv[1] = tree."""
import pathlib, sys
p = pathlib.Path(sys.argv[1]) / "torchtitan/distributed/utils.py"; s = p.read_text()
anchor = "    grads = [p.grad for p in parameters if p.grad is not None]\n    total_norm = torch.nn.utils.get_total_norm(\n"
assert s.count(anchor) == 1 and "GN_FP32" not in s, s.count(anchor)
hack = ("    grads = [p.grad for p in parameters if p.grad is not None]\n"
        "    if __import__(\"os\").environ.get(\"GN_FP32\") == \"1\":  # LOCAL PROBE HACK (not committed): norm in float32\n"
        "        grads = [g.float() for g in grads]\n"
        "    total_norm = torch.nn.utils.get_total_norm(\n")
p.write_text(s.replace(anchor, hack, 1)); print("patched", p)
