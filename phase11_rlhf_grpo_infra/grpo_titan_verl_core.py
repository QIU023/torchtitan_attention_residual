"""GRPO closed loop on the titan Kimi-Linear/AttnRes model using veRL's
REAL core algorithm -- not a hand-rolled advantage/loss.

Upgrades grpo_titan_standalone.py by swapping the hand-written pieces for
veRL core:
  * advantages  -> verl.trainer.ppo.core_algos.compute_grpo_outcome_advantage
  * policy loss -> compute_policy_loss_vanilla (the registered PPO-clip loss)
proving veRL's GRPO math composes with our AttnRes titan model. The
rollout is still an in-process full-recompute decode (KDA chunk-mode stays
valid since prompt > 64) -- that is the pre-7.27 stand-in for a K3
inference server; on 7.27 it swaps for rollout.name=vllm (official K3),
while THIS advantage+loss+actor path is unchanged. This is the test
harness for the permanent (Bucket A) GRPO wiring, runnable on one 5090.

Random-init fixture => reward ~0 (mechanism demo, not a learning demo);
real GSM8K exact-match plugs into reward_fn. Text-only.
"""
import numpy as np
import torch
import torch.nn.functional as F
from omegaconf import OmegaConf


def rollout_group(model, prompt, n, gen_len, temp=1.0):
    """In-process recompute rollout: n sampled completions + per-token
    old-policy logprobs (bs=n, gen_len)."""
    seqs = prompt.repeat(n, 1)
    old_logp = torch.zeros(n, gen_len, device=prompt.device)
    for t in range(gen_len):
        with torch.no_grad():
            logits = model(seqs)[:, -1, :].float() / temp
        logprobs = F.log_softmax(logits, dim=-1)
        nxt = torch.multinomial(logprobs.exp(), 1)
        old_logp[:, t] = logprobs.gather(1, nxt).squeeze(1)
        seqs = torch.cat([seqs, nxt], dim=1)
    return seqs, old_logp


def reward_fn(seqs, prompt_len):
    """Toy verifiable reward (fraction of even token-ids) so GRPO has a
    non-constant outcome signal to normalize. Real GSM8K exact-match
    (answer parse == gold) plugs in here -- a scalar per response."""
    comp = seqs[:, prompt_len:]
    return (comp % 2 == 0).float().mean(dim=1)


def actor_logprobs(model, seqs, prompt_len):
    """Recompute per-token response logprobs WITH grad (the actor pass)."""
    comp = seqs[:, prompt_len:]
    logits = model(seqs)[:, prompt_len - 1 : -1, :].float()
    return F.log_softmax(logits, dim=-1).gather(
        2, comp.unsqueeze(-1)
    ).squeeze(-1)


def main():
    from verl.trainer.ppo.core_algos import (
        compute_grpo_outcome_advantage,
        compute_policy_loss_vanilla,
    )

    from torchtitan.experiments.kimi_k3 import model_registry

    torch.cuda.set_device(0)
    torch.manual_seed(0)
    # 194m graft flavor: AttnRes is ON -- GRPO must compose with the graft.
    spec = model_registry("kimi_linear_194m_block_attn_res")
    with torch.device("cuda"):
        model = spec.model.build()
        model.init_weights()
    model = model.to(torch.bfloat16)
    opt = torch.optim.AdamW(
        [p for p in model.parameters() if p.requires_grad], lr=1e-5
    )
    # veRL PPO-clip loss reads clip_ratio* off the actor config.
    # global_batch_info={} -> agg_loss defaults (dp_size=1, batch_num_tokens
    # from the mask); the full trainer fills it for multi-DP normalization.
    loss_cfg = OmegaConf.create(
        {
            "clip_ratio": 0.2,
            "clip_ratio_low": 0.2,
            "clip_ratio_high": 0.2,
            "global_batch_info": {},
        }
    )

    G, GEN, PLEN = 6, 12, 72  # group size, gen len, prompt len (>64 for KDA)
    prompt = torch.randint(0, 163840, (1, PLEN), device="cuda")
    # GRPO group index: all G completions belong to one prompt-group.
    index = np.zeros(G, dtype=object)

    ok = 0
    for step in range(6):
        model.eval()
        seqs, old_logp = rollout_group(model, prompt, G, GEN)
        # outcome reward -> token_level_rewards at the last response token
        r = reward_fn(seqs, PLEN)
        token_rewards = torch.zeros(G, GEN, device="cuda")
        token_rewards[:, -1] = r
        resp_mask = torch.ones(G, GEN, device="cuda")
        # veRL core: group-relative outcome advantage
        advantages, _ = compute_grpo_outcome_advantage(
            token_level_rewards=token_rewards,
            response_mask=resp_mask,
            index=index,
        )
        # actor pass (with grad) + veRL core PPO-clip policy loss
        model.train()
        logp = actor_logprobs(model, seqs, PLEN)
        pg_loss, metrics = compute_policy_loss_vanilla(
            old_log_prob=old_logp,
            log_prob=logp,
            advantages=advantages,
            response_mask=resp_mask,
            loss_agg_mode="token-mean",
            config=loss_cfg,
        )
        if not torch.isfinite(pg_loss):
            print(f"[GRPO-veRL] step {step}: non-finite loss (KDA), skip")
            continue
        opt.zero_grad(set_to_none=True)
        pg_loss.backward()
        gn = torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()
        ok += 1
        pgclip = float(metrics.get("pg_clipfrac", 0.0))
        print(
            f"[GRPO-veRL] step {step}: reward {r.mean():.3f} "
            f"adv[{advantages.min():.2f},{advantages.max():.2f}] "
            f"pg_loss {pg_loss.item():.4f} pg_clipfrac {pgclip:.3f} "
            f"grad_norm {gn:.3f}",
            flush=True,
        )
    assert ok >= 1, "no finite GRPO step ran"
    print(
        "[GRPO-veRL] PASS -- veRL compute_grpo_outcome_advantage + "
        "compute_policy_loss_vanilla drive the titan AttnRes model",
        flush=True,
    )


if __name__ == "__main__":
    main()
