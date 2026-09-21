"""LOCAL PROBE HACKS (not committed) to locate the fp32 whole-stack departure. argv[1] = tree (with the H200 hacks on).
GRAD_DUMP=<prefix>: every parameter gradient at step 1, per rank, before clipping (grads_<prefix>.rank<r>.pt);
PARAM_DUMP=<prefix>: every parameter after the step-1 optimizer update (params_<prefix>.rank<r>.pt);
GN_REPR=1: the total grad norm every step at full precision and as hex, every rank ("GNREPR r<rank> <step> <value> <hex>");
LOSS_REPR=1: every micro batch loss at step 1 at full precision (the reference's per group loss, the pipeline's per chunk
loss on the last stage) ("LOSSREPR r<rank> <index> <value> <hex>"). Undo: git checkout -- torchtitan/training_engine.py.
"""
import pathlib
import sys

tree = pathlib.Path(sys.argv[1])
p = tree / "torchtitan/training_engine.py"
s = p.read_text()
if "GRAD_DUMP" in s:
    print("already patched", p)
    sys.exit(0)

def rep(old, new):
    global s
    assert s.count(old) == 1, (s.count(old), old)
    s = s.replace(old, new, 1)

rep("        current_step = self.num_completed_steps + 1\n        grad_norm = dist_utils.clip_grad_norm_(\n",
    '''        current_step = self.num_completed_steps + 1
        if __import__("os").environ.get("GRAD_DUMP") and current_step == 1:  # LOCAL PROBE HACK (not committed)
            _d = {}
            for _m in self.model_parts:
                for _n, _p in _m.named_parameters():
                    if _p.grad is None:
                        continue
                    _g = _p.grad
                    _g = _g.full_tensor() if hasattr(_g, "full_tensor") else _g
                    _d[_n] = _g.detach().cpu()
            _dd, _bb = __import__("os").path.split(__import__("os").environ["GRAD_DUMP"])
            torch.save(_d, __import__("os").path.join(_dd, f"grads_{_bb}.rank{torch.distributed.get_rank()}.pt"))
        grad_norm = dist_utils.clip_grad_norm_(
''')
rep("        if not self.parallel_dims.pp_enabled or self.pp_has_last_stage:\n            loss_mesh = self.parallel_dims.get_optional_mesh(\"loss\")\n",
    '''        if __import__("os").environ.get("GN_REPR") == "1":  # LOCAL PROBE HACK (not committed)
            _v = float(grad_norm.full_tensor() if hasattr(grad_norm, "full_tensor") else grad_norm)
            print(f"GNREPR r{torch.distributed.get_rank()} {current_step} {_v:.10e} {_v.hex()}", flush=True)
        if not self.parallel_dims.pp_enabled or self.pp_has_last_stage:
            loss_mesh = self.parallel_dims.get_optional_mesh("loss")
''')
rep("        self.optimizers.step()\n        self.lr_schedulers.step()\n",
    '''        self.optimizers.step()
        if __import__("os").environ.get("PARAM_DUMP") and current_step == 1:  # LOCAL PROBE HACK (not committed)
            _d = {}
            for _m in self.model_parts:
                for _n, _p in _m.named_parameters():
                    _t = _p.data
                    _t = _t.full_tensor() if hasattr(_t, "full_tensor") else _t
                    _d[_n] = _t.detach().cpu()
            _dd, _bb = __import__("os").path.split(__import__("os").environ["PARAM_DUMP"])
            torch.save(_d, __import__("os").path.join(_dd, f"params_{_bb}.rank{torch.distributed.get_rank()}.pt"))
        self.lr_schedulers.step()
''')
rep("            detached_losses = [loss.detach() for loss in losses]\n",
    '''            detached_losses = [loss.detach() for loss in losses]
            if __import__("os").environ.get("LOSS_REPR") == "1" and self.num_completed_steps == 0:  # LOCAL PROBE HACK (not committed)
                for _i, _l in enumerate(detached_losses):
                    _v = float(_l)
                    print(f"LOSSREPR r{torch.distributed.get_rank()} {_i} {_v:.10e} {_v.hex()}", flush=True)
''')
p.write_text(s)

# the reference's per micro batch group loss (trainer loop, NOSYNC_GA branch of the equalization hack)
p = tree / "torchtitan/trainer.py"
s = p.read_text()
if "LOSS_REPR" not in s:
    rep("                _probe_losses.append(detached_loss.clone())\n",
        '''                _probe_losses.append(detached_loss.clone())
                if os.environ.get("LOSS_REPR") == "1" and engine.num_completed_steps == 0:  # LOCAL PROBE HACK (not committed)
                    _v = float(detached_loss)
                    print(f"LOSSREPR r{torch.distributed.get_rank()} {fwd_bwd_index} {_v:.10e} {_v.hex()}", flush=True)
''')
    p.write_text(s)
print("fp32 locate hacks applied to", tree)
