"""Minimal GRPO loop against veRL's torchtitan engine and our K3 model.

Not a training run. It exercises the loop's plumbing end to end -- group
sampling, a programmable reward, group-relative advantages, a policy-gradient
step -- on dummy GSM8K data, so a break in the wiring surfaces on hardware that
cannot host real RL.

The reward is exact-match on the '#### <number>' marker, which is why GSM8K was
chosen (DATASET_SELECTION): it needs no reward model and no judge, so the loop
under test is the RL loop rather than a second training problem.

Usage: torchrun --nproc_per_node=N grpo_min_smoke.py [--data DIR] [--steps N]
"""

from __future__ import annotations

import argparse
import json
import os
import re
import sys

import torch


def extract_answer(text: str) -> str | None:
    """GSM8K ground truth is the number after the last '####'."""
    m = re.findall(r"####\s*(-?[\d.,]+)", text)
    return m[-1].replace(",", "").strip() if m else None


def reward_exact_match(sample: str, gold: str) -> float:
    """1.0 when the sampled answer matches the gold number, else 0.0."""
    a, b = extract_answer(sample), extract_answer(gold)
    return 1.0 if (a is not None and a == b) else 0.0


def group_relative_advantages(rewards: torch.Tensor) -> torch.Tensor:
    """GRPO's advantage: reward standardized WITHIN each prompt's group.

    Group-relative is the whole point -- it removes the need for a value model,
    which is what makes GRPO runnable here at all. Degenerate groups (every
    sample identical) give zero advantage rather than a divide-by-zero.
    """
    mean = rewards.mean(dim=-1, keepdim=True)
    std = rewards.std(dim=-1, keepdim=True)
    return (rewards - mean) / torch.clamp(std, min=1e-6)


def _policy_gradient_step(engine, rows, adv) -> float:
    """Backprop the GRPO objective through the actor and report the grad norm.

    -(advantage * mean logprob of the sequence), summed over the group. The
    sampler is stubbed -- sequences are the dummy answers rather than model
    samples -- because what is under test is that the objective reaches the
    actor's parameters through veRL's engine, not the sample quality. A real
    loop replaces the token source and nothing else here.
    """
    import torch.nn.functional as F

    if len(engine.module) > 1:
        # PP splits the model into stages that must be driven by the pipeline
        # schedule; calling a stage directly hands stage 1 the raw token ids
        # instead of stage 0's hidden states ("normalized_shape=[512] ... got
        # input of size [1, 8, 128]").
        #
        # engine.forward_backward_batch is the right entry -- it owns the
        # schedule, micro-batching and the SPMD mesh context -- but its
        # TensorDict contract did not come together in four attempts here
        # (max_token_len_per_gpu lives on the batch rather than the config, and
        # the batch then fails with "values expected sparse tensor layout").
        # Rather than keep guessing at it, the direct path stays as the
        # verified one and PP is recorded as unsupported by this smoke.
        raise NotImplementedError(
            "GRPO smoke does not support PP: needs engine.forward_backward_batch, "
            "whose batch contract is not yet worked out. See "
            "GRPO_PARALLEL_STATUS."
        )

    module = engine.module[0]
    device = next(module.parameters()).device
    vocab = module.vocab_size if hasattr(module, "vocab_size") else 2016

    # Deterministic token ids, so the step is reproducible without a tokenizer.
    # Length 128: KDA's training path asserts T > 64 (chunk mode).
    # One sequence per GROUP MEMBER, not per prompt. Averaging advantages over
    # the group would give exactly zero -- group-relative standardization makes
    # each group's mean zero by construction -- so a per-prompt sequence yields
    # a zero objective and a zero gradient, and the smoke would pass having
    # tested nothing.
    n_prompt, n_group = adv.shape
    g = torch.Generator(device="cpu").manual_seed(0)
    ids = torch.randint(
        0, vocab, (n_prompt * n_group, 128), generator=g
    ).to(device)

    # bf16 autocast is load-bearing here, not an optimization: without FSDP's
    # mixed-precision cast the KDA params stay fp32 and fla's kernel asks for
    # 108,160 B of dynamic shared memory against this GPU's 101,376 B limit.
    with torch.autocast("cuda", dtype=torch.bfloat16):
        logits = module(ids)
    if isinstance(logits, (tuple, list)):
        logits = logits[0]
    logprobs = F.log_softmax(logits.float(), dim=-1)
    tok_lp = logprobs[:, :-1, :].gather(-1, ids[:, 1:].unsqueeze(-1)).squeeze(-1)
    seq_lp = tok_lp.mean(dim=-1)                      # [prompts * group]

    a = adv.to(device).reshape(-1)                    # matching flat order
    loss = -(a * seq_lp).sum()
    loss.backward()

    total = 0.0
    for m in engine.module:
        for p in m.parameters():
            if p.grad is not None:
                gg = p.grad
                gg = gg.full_tensor() if hasattr(gg, "full_tensor") else gg
                total += float(gg.float().pow(2).sum())
    norm = total ** 0.5
    print(f"[GRPO]   loss={float(loss):.6f} grad_norm={norm:.6f}", flush=True)
    # A zero gradient means the objective never reached the parameters, which
    # is a passing run that tested nothing. It happened once here: averaging
    # advantages over the group gives exactly zero, because group-relative
    # standardization makes each group's mean zero by construction.
    assert norm > 0.0, "GRPO objective produced no gradient"
    return float(loss)


def load_dummy(path: str, n: int) -> list[dict]:
    rows = []
    with open(os.path.join(path, "gsm8k_dummy", "data.jsonl")) as f:
        for line in f:
            rows.append(json.loads(line))
            if len(rows) >= n:
                break
    return rows


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="/workspace/dummy_data")
    ap.add_argument("--steps", type=int, default=3)
    ap.add_argument("--group", type=int, default=4)
    ap.add_argument("--prompts", type=int, default=2)
    ap.add_argument(
        "--engine", action="store_true",
        help="drive veRL's torchtitan engine; without it only the reward and "
             "advantage path runs, which needs no GPU",
    )
    args = ap.parse_args()

    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
    engine = None
    if args.engine:
        import torch.distributed as dist

        from verl_actor_smoke import build_engine

        engine = build_engine()
        if dist.get_rank() == 0:
            print("[GRPO] engine ready", flush=True)

    rows = load_dummy(args.data, args.prompts)
    print(f"[GRPO] {len(rows)} prompts, group={args.group}", flush=True)
    assert rows, "no dummy rows loaded"

    for step in range(1, args.steps + 1):
        # Reward path first, on the real strings, so an error in extraction or
        # scoring surfaces even if generation is stubbed.
        rewards = torch.tensor([
            [reward_exact_match(r["answer"] if g == 0 else "#### 0", r["answer"])
             for g in range(args.group)]
            for r in rows
        ], dtype=torch.float32)
        adv = group_relative_advantages(rewards)
        print(f"[GRPO] step {step} rewards={rewards.tolist()} "
              f"adv[0]={[round(x, 4) for x in adv[0].tolist()]}", flush=True)
        assert torch.isfinite(adv).all(), "advantages must be finite"

        if engine is not None:
            # One policy-gradient step. The loss is -(advantage * logprob)
            # summed over the group; with a stubbed sampler the logprobs come
            # from the model's own forward on the prompt, which is enough to
            # prove gradients flow from the GRPO objective into the actor.
            engine.optimizer_zero_grad()
            loss = _policy_gradient_step(engine, rows, adv)
            engine.optimizer_step()
            print(f"[GRPO] step {step} objective={loss:.6f}", flush=True)

    if engine is not None:
        import torch.distributed as dist

        dist.destroy_process_group()
    print("[GRPO] PASS", flush=True)


if __name__ == "__main__":
    main()
