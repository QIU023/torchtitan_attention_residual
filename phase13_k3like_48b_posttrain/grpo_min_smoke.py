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
    """Backprop the GRPO objective through veRL's engine and report the norm.

    Goes through ``forward_backward_batch``, not a direct module call. That is
    what makes TP and PP reachable: under TP the wrapped model expects a DTensor
    input and a plain one raises "mixed torch.Tensor and DTensor", and under PP
    the stages have to be driven by the pipeline schedule. The engine owns both,
    plus micro-batching and the SPMD mesh context.

    ``use_dynamic_bsz`` is turned OFF. The default path routes through
    ``rearrange_micro_batches``, whose token-budget bookkeeping did not hold for
    this batch; the fixed-size path only needs ``micro_batch_size_per_gpu`` and
    keeps the whole batch in one micro batch, which is what the smoke wants
    anyway.

    The sampler is stubbed -- sequences are deterministic ids, not model samples
    -- so this establishes that the objective reaches the actor's parameters,
    not sample quality.
    """
    import torch.nn.functional as F
    from tensordict import TensorDict

    from verl.utils import tensordict_utils as tu

    n_prompt, n_group = adv.shape
    n_seq, seq_len = n_prompt * n_group, 128  # KDA asserts T > 64 in training
    device = torch.device(f"cuda:{torch.cuda.current_device()}")

    g = torch.Generator(device="cpu").manual_seed(0)
    flat = torch.randint(0, 2016, (n_seq, seq_len), generator=g).to(device)
    a = adv.reshape(-1).to(device)

    # veRL's engine runs in NO_PADDING mode: prepare_model_inputs calls
    # .values() and .offsets() on input_ids, so the batch has to carry NESTED
    # tensors (variable-length sequences packed jagged), not padded strided
    # ones. Turning the un-padding off is not an alternative -- the rest of the
    # path assumes the same layout.
    def _nest(t):
        return torch.nested.nested_tensor(
            list(t.unbind(0)), layout=torch.jagged, device=device
        )

    ids = _nest(flat)

    data = TensorDict(
        {
            "input_ids": ids,
            "position_ids": _nest(
                torch.arange(seq_len, device=device)
                .unsqueeze(0)
                .expand(n_seq, -1)
                .contiguous()
            ),
            "loss_mask": torch.ones_like(flat),
            "responses": flat,
            "advantages": a.unsqueeze(-1).expand(-1, seq_len),
        },
        batch_size=[n_seq],
    )
    # The engine reads sampling metadata off the batch too.
    tu.assign_non_tensor_data(data, "temperature", 1.0)
    tu.assign_non_tensor_data(data, "use_dynamic_bsz", False)
    tu.assign_non_tensor_data(data, "micro_batch_size_per_gpu", n_seq)


    def grpo_loss(model_output, data, dp_group=None, **kw):
        """-(advantage * logprob), the policy-gradient objective.

        The engine computes token log-probs itself and hands them over in
        model_output, so this only has to weight them. Group-relative
        advantages are already standardized, so no baseline is subtracted.
        """
        lp = model_output["log_probs"]
        a = data["advantages"]
        if hasattr(lp, "values"):        # nested, as NO_PADDING mode packs it
            lp = lp.values()
            a = a.reshape(-1)[: lp.numel()]
        else:
            a = a[..., : lp.shape[-1]]
        return -(a * lp).mean(), {}

    engine.forward_backward_batch(data, loss_function=grpo_loss)

    total = 0.0
    for m in engine.module:
        for p in m.parameters():
            if p.grad is not None:
                gg = p.grad
                gg = gg.full_tensor() if hasattr(gg, "full_tensor") else gg
                total += float(gg.float().pow(2).sum())
    norm = total ** 0.5
    print(f"[GRPO]   grad_norm={norm:.6f}", flush=True)
    # A zero gradient means the objective never reached the parameters, which is
    # a passing run that tested nothing. It happened once: averaging advantages
    # over the group gives exactly zero, because group-relative standardization
    # makes each group's mean zero by construction.
    assert norm > 0.0, "GRPO objective produced no gradient"
    return norm


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
            print(f"[GRPO] step {step} grad_norm={loss:.6f}", flush=True)

    if engine is not None:
        import torch.distributed as dist

        dist.destroy_process_group()
    print("[GRPO] PASS", flush=True)


if __name__ == "__main__":
    main()
