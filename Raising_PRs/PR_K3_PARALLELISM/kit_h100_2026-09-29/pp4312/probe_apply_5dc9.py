"""LOCAL PROBE HACKS for PR 4312's H100 matrix on main 5dc97a3e7 and trees on it (never committed).
argv[1] = tree. Undo: git -C <tree> checkout -- torchtitan/models/kimi_k3/model.py torchtitan/trainer.py
torchtitan/training_engine.py torchtitan/distributed/utils.py
ATTN_RES_NAIVE (naive transport), MB_REVERSE, NOSYNC_GA (the reference accumulates like the pipeline),
GN_FP32 (total grad norm in float32), and the reference logging its step loss the way the pipeline does."""
import pathlib
import sys

tree = pathlib.Path(sys.argv[1])


def patch(rel, old, new, marker):
    p = tree / rel
    s = p.read_text()
    if marker in s:
        return
    assert s.count(old) == 1, (rel, s.count(old))
    p.write_text(s.replace(old, new, 1))


patch(
    "torchtitan/models/kimi_k3/model.py",
    "        return pipeline_kimi_k3(self, **kwargs)\n",
    '        return pipeline_kimi_k3(self, attn_res_cache=__import__("os").environ.get("ATTN_RES_NAIVE") != "1", **kwargs)  # LOCAL PROBE HACK\n',
    "ATTN_RES_NAIVE",
)
patch(
    "torchtitan/trainer.py",
    "            microbatch_groups.append(microbatch_group)\n        sl.log_trace_scalar({\"local_valid_tokens\": local_valid_tokens})\n",
    "            microbatch_groups.append(microbatch_group)\n"
    "        if os.environ.get(\"MB_REVERSE\") == \"1\":  # LOCAL PROBE HACK: accumulation order only\n"
    "            microbatch_groups.reverse()\n"
    "        sl.log_trace_scalar({\"local_valid_tokens\": local_valid_tokens})\n",
    "MB_REVERSE",
)
patch(
    "torchtitan/training_engine.py",
    "                part.set_requires_all_reduce(is_last)  # pyrefly: ignore[not-callable]\n",
    "                part.set_requires_all_reduce(is_last)  # pyrefly: ignore[not-callable]\n"
    "\n        if __import__(\"os\").environ.get(\"NOSYNC_GA\") == \"1\" and not self.parallelism_context.pp_enabled:  # LOCAL PROBE HACK\n"
    "            _last = accumulation_index == self.num_accumulation_steps - 1\n"
    "            for _part in self.model_parts:\n"
    "                _part.set_is_last_backward(_last)\n"
    "                _part.set_reshard_after_backward(_last)\n"
    "                _part.set_requires_gradient_sync(_last)\n",
    "NOSYNC_GA",
)
patch(
    "torchtitan/distributed/utils.py",
    "    grads = [p.grad for p in parameters if p.grad is not None]\n    total_norm = torch.nn.utils.get_total_norm(\n",
    "    grads = [p.grad for p in parameters if p.grad is not None]\n"
    "    if __import__(\"os\").environ.get(\"GN_FP32\") == \"1\":  # LOCAL PROBE HACK: norm in float32\n"
    "        grads = [g.float() for g in grads]\n"
    "    total_norm = torch.nn.utils.get_total_norm(\n",
    "GN_FP32",
)
patch(
    "torchtitan/trainer.py",
    """            if should_log:
                if accumulated_loss is None:
                    # Take ownership before the next replay overwrites the
                    # graph-owned output. Later losses accumulate in place.
                    accumulated_loss = detached_loss.clone()
                else:
                    accumulated_loss.add_(detached_loss)
""",
    """            if should_log and os.environ.get("NOSYNC_GA") == "1":  # LOCAL PROBE HACK: the step loss as the pipeline's last stage sums it
                _probe_losses = globals().setdefault("_PROBE_LOSSES", [])
                _probe_losses.append(detached_loss.clone())
                if fwd_bwd_index == len(microbatch_groups) - 1:
                    accumulated_loss = torch.sum(torch.stack(_probe_losses)).to(detached_loss.device)
                    _probe_losses.clear()
            elif should_log:
                if accumulated_loss is None:
                    # Take ownership before the next replay overwrites the
                    # graph-owned output. Later losses accumulate in place.
                    accumulated_loss = detached_loss.clone()
                else:
                    accumulated_loss.add_(detached_loss)
""",
    "_PROBE_LOSSES",
)
print("4312 matrix probe hacks applied to", tree)
