"""Standalone GRPO loop on the titan K3 model (194m).

Demonstrates the GRPO RL algorithm on the Kimi-Linear/AttnRes model
without an inference server: full-recompute greedy/sampling rollout
(KDA chunk-mode stays valid since prompt>64), group-relative advantages,
and a REINFORCE/PPO-clip policy update. Titan-native (no transformers,
no sglang). Random-init fixture => reward signal is ~0 (mechanism demo,
not a learning demo); the point is the loop + advantages + update run.

veRL-native GRPO needs the sglang AttnRes overlay OR a sync in-process
rollout registered in veRL's (async-server) RL rollout registry -- a
QIU023/verl fork follow-up.
"""
import torch
import torch.nn.functional as F


def sample_group(model, prompt, n, gen_len, vocab, temp=1.0):
    """Full-recompute sampling rollout: n completions of the prompt."""
    seqs = prompt.repeat(n, 1)
    logps = torch.zeros(n, device=prompt.device)
    for _ in range(gen_len):
        with torch.no_grad():
            logits = model(seqs)[:, -1, :].float() / temp
        probs = F.softmax(logits, dim=-1)
        nxt = torch.multinomial(probs, 1)
        logps = logps + torch.log(probs.gather(1, nxt).squeeze(1) + 1e-9)
        seqs = torch.cat([seqs, nxt], dim=1)
    return seqs, logps


def reward_fn(seqs, prompt_len):
    """Toy verifiable reward: reward = fraction of even token-ids in the
    completion (a deterministic, gradient-free signal so the GRPO
    mechanics have something non-constant to normalize; real GSM8K
    exact-match plugs in here)."""
    comp = seqs[:, prompt_len:]
    return (comp % 2 == 0).float().mean(dim=1)


def main():
    from torchtitan.experiments.kimi_k3 import model_registry

    torch.cuda.set_device(0)
    torch.manual_seed(0)
    spec = model_registry("kimi_linear_194m_block_attn_res")
    with torch.device("cuda"):
        model = spec.model.build()
        model.init_weights()
    model = model.to(torch.bfloat16)
    opt = torch.optim.AdamW(
        [p for p in model.parameters() if p.requires_grad], lr=1e-5
    )

    G, GEN = 6, 12  # group size, gen length
    prompt = torch.randint(0, 163840, (1, 72), device="cuda")
    for step in range(6):
        # 1) rollout a group
        model.eval()
        seqs, _ = sample_group(model, prompt, G, GEN, 163840)
        # 2) rewards + group-relative (GRPO) advantages
        r = reward_fn(seqs, prompt.shape[1])
        adv = (r - r.mean()) / (r.std() + 1e-6)
        # 3) policy-gradient update: recompute logprobs WITH grad, weight
        #    each completion's token logprobs by its advantage.
        model.train()
        comp = seqs[:, prompt.shape[1]:]
        logits = model(seqs)[:, prompt.shape[1] - 1 : -1, :].float()
        logp = F.log_softmax(logits, dim=-1).gather(
            2, comp.unsqueeze(-1)
        ).squeeze(-1).sum(dim=1)
        loss = -(adv.detach() * logp).mean()
        opt.zero_grad(set_to_none=True)
        loss.backward()
        gn = torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()
        print(
            f"[GRPO] step {step}: reward mean {r.mean():.3f} std {r.std():.3f} "
            f"adv-range [{adv.min():.2f},{adv.max():.2f}] "
            f"pg_loss {loss.item():.4f} grad_norm {gn:.3f}",
            flush=True,
        )
    print("[GRPO] PASS -- rollout + group-advantage + policy update loop runs", flush=True)


if __name__ == "__main__":
    main()
