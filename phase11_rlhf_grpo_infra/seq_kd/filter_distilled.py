"""Post-merge cleanup for the seq-KD distilled mix665k.

Two hygiene passes on the teacher-generated conversations before they become
SFT labels:

1. TRUNCATION: ~5% of gpt turns hit the max_new_tokens=512 cap and end
   mid-sentence. A truncated label teaches the student to never emit EOS.
   We DROP the offending gpt turn (and, since later turns conditioned on it via
   the ORIGINAL context not this text, dropping is safe) — but if dropping a
   turn would leave a human turn with no answer, we drop that human turn too so
   the conversation stays well-formed (human/gpt alternation).

2. EMPTY / DEGENERATE: drop gpt turns that are empty or pure whitespace.

A conversation with zero surviving gpt turns is dropped entirely.

Usage:
  python filter_distilled.py --in distilled_mix665k_full.json \
                             --out distilled_mix665k_full.clean.json \
                             --trunc-token-threshold 508
"""
from __future__ import annotations

import argparse
import json

from transformers import AutoTokenizer


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--in", dest="inp", required=True)
    ap.add_argument("--out", dest="out", required=True)
    ap.add_argument("--tokenizer", default="NousResearch/Meta-Llama-3.1-8B")
    ap.add_argument("--trunc-token-threshold", type=int, default=508,
                    help="gpt turns with >= this many tokens are treated as "
                         "truncated (max_new_tokens was 512)")
    args = ap.parse_args()

    tok = AutoTokenizer.from_pretrained(args.tokenizer)
    data = json.load(open(args.inp))
    print(f"loaded {len(data)} conversations")

    out = []
    n_drop_turn = n_drop_conv = n_empty = 0
    for s in data:
        convs = s["conversations"]
        kept = []
        i = 0
        # walk in (human, gpt) order; conversations are human/gpt alternating
        while i < len(convs):
            m = convs[i]
            if m["from"] == "human":
                # peek the following gpt turn
                gpt = convs[i + 1] if i + 1 < len(convs) and convs[i + 1]["from"] == "gpt" else None
                if gpt is None:
                    kept.append(m); i += 1; continue
                txt = gpt["value"].strip()
                ntok = len(tok(txt, add_special_tokens=False)["input_ids"])
                if not txt:
                    n_empty += 1; n_drop_turn += 1; i += 2; continue
                if ntok >= args.trunc_token_threshold:
                    n_drop_turn += 1; i += 2; continue  # drop both human+gpt
                kept.append(m); kept.append(gpt); i += 2
            else:
                # stray gpt (shouldn't happen) — keep as-is
                kept.append(m); i += 1

        if sum(1 for m in kept if m["from"] == "gpt") == 0:
            n_drop_conv += 1
            continue
        rec = {"id": s.get("id"), "conversations": kept}
        if "image" in s:
            rec["image"] = s["image"]
        out.append(rec)

    json.dump(out, open(args.out, "w"), ensure_ascii=False)
    print(f"kept {len(out)} conversations -> {args.out}")
    print(f"dropped: {n_drop_turn} truncated/empty gpt turns "
          f"({n_empty} empty), {n_drop_conv} fully-empty conversations")


if __name__ == "__main__":
    main()
