"""LOCAL PROBE HACK (not committed): save every parameter's step-1 gradient in float32 per
rank before clip_grad_norm_ (GRAD_TENSOR_DUMP names the prefix), then leave the process
(no optimizer step, so a float32 model fits a 16 GB card at dp1). For the exact-gradient
comparison of the pipeline against one GPU with the model in float32."""
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
                        _d[_n] = _g.detach().float().cpu()
                torch.save(_d, f"{os.environ['GRAD_TENSOR_DUMP']}.rank{_dist.get_rank()}.pt")
                if os.environ.get("GRAD_TENSOR_DUMP_EXIT"):
                    _dist.barrier()
                    os._exit(0)
''' + old
p.write_text(s.replace(old, new)); print('fp32 dump-and-exit hack applied to', sys.argv[1])
