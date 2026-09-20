"""LOCAL PROBE HACK (not committed): QB_DUMP=<dir> makes the quantile-balancing hook save, per call and per
MoE layer (by FQN), the reduced histogram, the expert bias before, and the next bias, in full precision."""
import pathlib, sys
p = pathlib.Path(sys.argv[1]) / "torchtitan/components/optimizer/optimizer.py"
s = p.read_text()
old = '''        for histogram_EB, (moe, router) in zip(
            histograms,
            moe_layers,
            strict=True,
        ):
            expert_bias_E = moe.expert_bias_E
            assert expert_bias_E is not None
            quantile_balancer = router.quantile_balancer
            next_expert_bias_E = quantile_balancer.estimate_expert_bias(
                histogram_EB,
                expert_bias_E,
            )
            expert_bias_E.copy_(next_expert_bias_E)'''
new = '''        _dump = __import__("os").environ.get("QB_DUMP")  # LOCAL PROBE HACK (not committed)
        if _dump:
            _names = {}
            for _part in model_parts:
                for _n, _m in _part.named_modules():
                    _names[id(_m)] = _n
            _call = getattr(_update_expert_bias, "_calls", 0); _update_expert_bias._calls = _call + 1
        for _i, (histogram_EB, (moe, router)) in enumerate(zip(
            histograms,
            moe_layers,
            strict=True,
        )):
            expert_bias_E = moe.expert_bias_E
            assert expert_bias_E is not None
            quantile_balancer = router.quantile_balancer
            next_expert_bias_E = quantile_balancer.estimate_expert_bias(
                histogram_EB,
                expert_bias_E,
            )
            if _dump and _call < 3:
                import os as _os
                _os.makedirs(_dump, exist_ok=True)
                torch.save({"fqn": _names.get(id(moe), f"?{_i}"), "hist": histogram_EB.detach().cpu(), "bias_before": expert_bias_E.detach().cpu().clone(), "bias_next": next_expert_bias_E.detach().cpu().clone(), "tokens": router.tokens_per_expert_E.detach().cpu().clone()},
                           f"{_dump}/rank{torch.distributed.get_rank()}_call{_call}_layer{_i}.pt")
            expert_bias_E.copy_(next_expert_bias_E)'''
assert old in s and "QB_DUMP" not in s, "anchor"
p.write_text(s.replace(old, new, 1)); print("QB dump hack applied")
