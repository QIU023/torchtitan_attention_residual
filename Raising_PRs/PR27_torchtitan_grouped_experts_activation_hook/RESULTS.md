# PR27: does the hook change a stock model's arithmetic?

No. Every numeric scalar is bit-identical over 10 steps.

`deepseek_v3_debugmodel`, 2 ranks, seq 512, `--debug.seed 42 --debug.deterministic`,
run twice on the same tree: once as committed, once with `gate_up_combine.patch` reversed
so `GroupedExperts.forward` computes `F.silu(...) * ...` inline again. Reversing only that
hunk leaves PR28's `router_input_BLD` in place, so the two legs differ by this change and
nothing else.

deepseek_v3 rather than our own model on purpose: it is upstream's, it uses
`GroupedExperts` unmodified, and an identical run on the K3 model would only show that our
subclass agrees with itself.

Read from the TensorBoard event files at full precision, not from stdout -- the repo's own
guidance is that five significant digits is not enough to call two runs identical.

    scalar                          points  differing
    grad_norm                           10          0
    loss_metrics/global_avg_loss        10          0
    loss_metrics/global_max_loss        10          0
    lr/AdamW                            10          0
    n_tokens_seen                       10          0
    memory/num_alloc_retries            10          0
    memory/num_ooms                     10          0
    memory/max_active(GiB)              10         10
    memory/max_reserved(GiB)            10         10
    mfu(%) / tflops / throughput(tps)   30         30
    time_metrics/*                      30         30

Everything that differs is telemetry that varies run to run.

One entry is worth reading rather than skipping: `memory/max_active(%)` is LOWER with the
hook, 6.9072 against 6.9198. Giving the two grouped-mm results names (`gate_RF`, `up_RF`)
where the original nested one call inside `F.silu` could have extended their lifetimes; it
did not. That is allocator behaviour and not a number to claim in the PR -- what it rules
out is the opposite reading.

Reproduce with `run_stock_identity.sh` (TAG=hook, then reverse the patch and TAG=base).
