"""LOCAL PROBE HACK (not committed): every router call records its top-k expert ids (ROUTER_DUMP
set); the trainer's step-1 dump writes them per rank as {fqn: [ids per call]} next to the
gradients, so dp1 and a pipeline run can be compared micro-batch by micro-batch."""
import pathlib, sys
root = pathlib.Path(sys.argv[1])
m = root / 'torchtitan/models/common/moe.py'; s = m.read_text()
old = "        topk_scores_TK = topk_scores_TK * self.route_scale\n"
assert s.count(old) == 1
new = old + '''        if os.environ.get("ROUTER_DUMP"):  # LOCAL PROBE HACK (not committed)
            self._route_log = getattr(self, "_route_log", []) + [topk_expert_ids_TK.detach().to(torch.int16).cpu()]
'''
s = s.replace(old, new)
if "\nimport os\n" not in s:
    s = s.replace("import torch\n", "import os\n\nimport torch\n", 1)
m.write_text(s)
t = root / 'torchtitan/trainer.py'; u = t.read_text()
old = '''                torch.save(_d, f"{os.environ['GRAD_TENSOR_DUMP']}.rank{_dist.get_rank()}.pt")
'''
assert u.count(old) == 1
new = old + '''                if os.environ.get("ROUTER_DUMP"):
                    _r = {}
                    for _m in self.model_parts:
                        for _n, _mod in _m.named_modules():
                            if hasattr(_mod, "_route_log"):
                                _r[_n] = _mod._route_log
                    torch.save(_r, f"{os.environ['GRAD_TENSOR_DUMP']}.rank{_dist.get_rank()}.router.pt")
'''
t.write_text(u.replace(old, new)); print("router dump hack applied to", root)
