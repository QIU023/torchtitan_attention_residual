"""LOCAL PROBE HACK (not committed): GRAD_DUMP=<prefix> saves every parameter gradient at step 1, per rank,
before clipping. argv[1] = tree."""
import pathlib, sys
p = pathlib.Path(sys.argv[1]) / "torchtitan/trainer.py"; s = p.read_text()
if "GRAD_DUMP" in s:
    print("already patched"); sys.exit(0)
anchor = "            grad_norm = dist_utils.clip_grad_norm_(\n"
assert s.count(anchor) == 1, s.count(anchor)
hack = '''            if os.environ.get("GRAD_DUMP") and self.step == 1:  # LOCAL PROBE HACK (not committed)
                import torch.distributed as _dist

                _d = {}
                for _m in self.model_parts:
                    for _n, _p in _m.named_parameters():
                        if _p.grad is None:
                            continue
                        _g = _p.grad
                        _g = _g.full_tensor() if hasattr(_g, "full_tensor") else _g
                        _d[_n] = _g.detach().cpu()
                torch.save(_d, f"{os.environ['GRAD_DUMP']}.rank{_dist.get_rank()}.pt")
'''
p.write_text(s.replace(anchor, hack + anchor, 1)); print("patched", p)
