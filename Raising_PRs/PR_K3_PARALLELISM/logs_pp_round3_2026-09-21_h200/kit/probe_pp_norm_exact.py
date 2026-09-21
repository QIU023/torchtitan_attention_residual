"""LOCAL PROBE HACK (not committed): PP_NORM_EXACT=<fqn order file> makes a pipeline run take its total grad norm the way
the single device reference does: every rank's per parameter norms are gathered over the pp group, put in the reference's
parameter order (the file: one FQN per line, the grads dump's key order) and reduced by one torch.linalg.vector_norm over
that stack, instead of core's per rank norm, square, all-reduce, sqrt. argv[1] = tree (with probe_fp32_locate.py on)."""
import pathlib
import sys

tree = pathlib.Path(sys.argv[1])
p = tree / "torchtitan/training_engine.py"
s = p.read_text()
if "PP_NORM_EXACT" in s:
    print("already patched", p)
    sys.exit(0)
old = '''        grad_norm = dist_utils.clip_grad_norm_(
            [p for model in self.model_parts for p in model.parameters()],
            self.config.training.max_norm,
            foreach=True,
            pp_mesh=self.parallel_dims.get_optional_mesh("pp"),
            ep_enabled=self.parallel_dims.ep_enabled,
        )
'''
new = '''        if __import__("os").environ.get("PP_NORM_EXACT") and self.parallel_dims.pp_enabled:  # LOCAL PROBE HACK (not committed): the total norm in the reference's reduction order
            _order = [_l.strip() for _l in open(__import__("os").environ["PP_NORM_EXACT"]) if _l.strip()]
            _named = [(_n, _p.grad) for _m in self.model_parts for _n, _p in _m.named_parameters() if _p.grad is not None]
            _norms = torch._foreach_norm([_g for _, _g in _named], 2.0)
            _local = {_n: float(_v.full_tensor() if hasattr(_v, "full_tensor") else _v) for (_n, _), _v in zip(_named, _norms)}
            _pp = self.parallel_dims.get_optional_mesh("pp")
            _gathered = [None] * _pp.size()
            torch.distributed.all_gather_object(_gathered, _local, group=_pp.get_group())
            _all = {}
            for _g in _gathered:
                _all.update(_g)
            assert set(_all) == set(_order), (len(_all), len(_order), sorted(set(_all) ^ set(_order))[:5])
            grad_norm = torch.linalg.vector_norm(torch.tensor([_all[_n] for _n in _order], dtype=torch.float32, device=self.device))
            torch.nn.utils.clip_grads_with_norm_(
                [p for model in self.model_parts for p in model.parameters()], self.config.training.max_norm, grad_norm, foreach=True
            )
        else:
            grad_norm = dist_utils.clip_grad_norm_(
                [p for model in self.model_parts for p in model.parameters()],
                self.config.training.max_norm,
                foreach=True,
                pp_mesh=self.parallel_dims.get_optional_mesh("pp"),
                ep_enabled=self.parallel_dims.ep_enabled,
            )
'''
assert s.count(old) == 1, s.count(old)
p.write_text(s.replace(old, new, 1))
print("PP_NORM_EXACT hack applied to", tree)
