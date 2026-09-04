import pathlib, sys
p = pathlib.Path(sys.argv[1]) / 'torchtitan/trainer.py'; s = p.read_text()
assert 'import os' in s, 'trainer.py has no os import'
old = '            grad_norm = dist_utils.clip_grad_norm_(\n'
assert s.count(old) == 1
new = '''            if os.environ.get("GRAD_DUMP"):  # LOCAL PROBE HACK (not committed)
                import torch.distributed as _dist

                with open(
                    f"{os.environ['GRAD_DUMP']}.rank{_dist.get_rank()}.step{self.step}.txt", "w"
                ) as _f:
                    for _m in self.model_parts:
                        for _n, _p in _m.named_parameters():
                            if _p.grad is None:
                                continue
                            _g = _p.grad
                            _g = _g.to_local() if hasattr(_g, "to_local") else _g
                            _f.write(f"{_n} {_g.float().norm().item():.10e} {tuple(_g.shape)}\\n")
''' + old
p.write_text(s.replace(old, new)); print('grad dump hack applied to', sys.argv[1])
