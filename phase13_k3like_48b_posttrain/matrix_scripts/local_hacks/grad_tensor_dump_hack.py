"""LOCAL PROBE HACK (not committed): patch trainer.py to save every parameter's step-1
gradient tensor (bf16, as trained) per rank before clip_grad_norm_, when GRAD_TENSOR_DUMP
names the file prefix. The sign census (pp_step10_census.py) reads the files."""
import pathlib, sys
p = pathlib.Path(sys.argv[1]) / 'torchtitan/trainer.py'; s = p.read_text()
assert 'import os' in s, 'trainer.py has no os import'
old = '            grad_norm = dist_utils.clip_grad_norm_(\n'
assert s.count(old) == 1
new = '''            if os.environ.get("GRAD_TENSOR_DUMP") and self.step == 1:  # LOCAL PROBE HACK (not committed)
                import torch.distributed as _dist

                _d = {}
                for _m in self.model_parts:
                    for _n, _p in _m.named_parameters():
                        if _p.grad is None:
                            continue
                        _g = _p.grad
                        _g = _g.full_tensor() if hasattr(_g, "full_tensor") else _g
                        _d[_n] = _g.detach().to(torch.bfloat16).cpu()
                torch.save(_d, f"{os.environ['GRAD_TENSOR_DUMP']}.rank{_dist.get_rank()}.pt")
''' + old
p.write_text(s.replace(old, new)); print('grad tensor dump hack applied to', sys.argv[1])
