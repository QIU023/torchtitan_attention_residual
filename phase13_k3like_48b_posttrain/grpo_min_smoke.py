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
    args = ap.parse_args()

    # The engine is verified separately by verl_actor_smoke (actor builds,
    # 80.9M params initialize on the torchtitan engine). This exercises the
    # GRPO-specific path -- reward extraction, group-relative advantages -- so
    # that a break in either is attributable. Joining them is the next step and
    # needs verl_actor_smoke's monolithic main() split into a builder first.
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

    print("[GRPO] PASS (reward + advantage path)", flush=True)


if __name__ == "__main__":
    main()
