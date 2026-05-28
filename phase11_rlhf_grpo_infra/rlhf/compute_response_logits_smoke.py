"""CPU smoke test for compute_response_logits (Stage C.2).

Validates two properties against the sibling compute_token_log_probs:

(1) Shape is [T_resp, V] float32 at the same response positions.
(2) Numerical parity: gathering log_softmax of compute_response_logits
    at gen_ids reproduces compute_token_log_probs to within atol=1e-5
    (both go float32 before any reduction, so equality should hold up
    to floating-point reassociation noise).

A toy 2-layer LM is used so this runs CPU-only in seconds.
"""
from __future__ import annotations

import os
import sys

import torch
import torch.nn as nn
import torch.nn.functional as F

# Import the helpers from the torchtitan submodule directly without
# pulling in the rest of the actors package (which depends on
# torchstore / monarch / SGLang and isn't needed for this smoke).
HERE = os.path.dirname(os.path.abspath(__file__))
TT_ROOT = os.path.abspath(os.path.join(HERE, "..", "..", "torchtitan"))
sys.path.insert(0, TT_ROOT)


class ToyLM(nn.Module):
    """Minimal LM stand-in.

    Accepts the same kwargs that PolicyTrainer's compute_token_log_probs
    passes — input_ids, attention_masks, positions — and returns logits
    of shape [B, T, V]. Vision kwargs are NOT exercised by this smoke
    (text-only path is enough to assert shape + parity).
    """

    def __init__(self, vocab: int = 256, dim: int = 32):
        super().__init__()
        self.emb = nn.Embedding(vocab, dim)
        self.norm = nn.LayerNorm(dim)
        self.lm_head = nn.Linear(dim, vocab, bias=False)

    def forward(self, input_ids, attention_masks=None, positions=None, **_):
        h = self.emb(input_ids)
        h = self.norm(h)
        return self.lm_head(h)


def main():
    # The two helpers live in actors.utils which we deliberately do NOT
    # import as a package (it would import sibling modules that need
    # monarch + torchstore). Source-load the file instead.
    import importlib.util
    spec = importlib.util.spec_from_file_location(
        "_actors_utils",
        os.path.join(TT_ROOT, "torchtitan", "experiments", "rl", "actors", "utils.py"),
    )
    utils = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(utils)

    torch.manual_seed(0)
    V = 256
    model = ToyLM(vocab=V).eval()
    device = torch.device("cpu")

    prompt_ids = [3, 7, 41, 8, 19]      # T_prompt = 5
    gen_ids = [21, 5, 99, 14, 200, 11]  # T_resp = 6

    # Forward both helpers in no_grad — compute_response_logits doesn't
    # set its own no_grad context, so we wrap it here for the parity
    # comparison (in OPDTrainer the student grad flows).
    with torch.no_grad():
        token_lps = utils.compute_token_log_probs(
            model, prompt_ids, gen_ids, device,
        )
        resp_logits = utils.compute_response_logits(
            model, prompt_ids, gen_ids, device,
        )

    T_resp = len(gen_ids)
    # (1) Shape check.
    assert resp_logits.shape == (T_resp, V), \
        f"shape mismatch: got {tuple(resp_logits.shape)}, want ({T_resp}, {V})"
    assert resp_logits.dtype == torch.float32, \
        f"dtype mismatch: got {resp_logits.dtype}, want float32"
    print(f"[crl] shape OK  {tuple(resp_logits.shape)}  dtype={resp_logits.dtype}")

    # (2) Parity check: gather log_softmax(resp_logits) at gen_ids and
    # compare to compute_token_log_probs' token_lps.
    gen_ids_t = torch.tensor(gen_ids, dtype=torch.long, device=device)
    lp_gathered = F.log_softmax(resp_logits, dim=-1).gather(
        1, gen_ids_t.unsqueeze(-1)
    ).squeeze(-1)
    assert token_lps.shape == lp_gathered.shape, \
        f"parity shape mismatch: {tuple(token_lps.shape)} vs {tuple(lp_gathered.shape)}"
    diff = (token_lps - lp_gathered).abs().max().item()
    print(f"[crl] numerical parity max|diff| = {diff:.3e}  (tolerance 1e-5)")
    assert diff < 1e-5, f"compute_response_logits parity failed (diff={diff:.3e})"

    # (3) Grad path: confirm we CAN backprop through resp_logits when
    # called without an enclosing no_grad — that's the whole point of
    # not decorating with @torch.no_grad.
    model.train()
    resp_logits_grad = utils.compute_response_logits(
        model, prompt_ids, gen_ids, device,
    )
    loss = resp_logits_grad.pow(2).mean()
    loss.backward()
    g = model.lm_head.weight.grad
    assert g is not None and g.abs().sum().item() > 0, "no grad flowed to lm_head"
    print(f"[crl] grad path OK — lm_head.grad |sum|={g.abs().sum().item():.4f}")

    print("[crl] SMOKE PASSED — compute_response_logits matches compute_token_log_probs")


if __name__ == "__main__":
    main()
