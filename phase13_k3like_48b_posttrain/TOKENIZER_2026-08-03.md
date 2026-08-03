# The K3 tokenizer we were missing

## What was wrong

Every debug flavor pinned `hf_assets_path="./tests/assets/tokenizer"` -- the
2016-token tokenizer torchtitan ships for its own tests, not K3's. Worse,
`/workspace/k3mini_hf/tokenizer.json`, which looked like a K3 artifact, is that
same 2016-entry file. So nothing in this repo had ever seen K3's real vocabulary.

`BUNDLED_TOKENIZER_SIZES` encodes this honestly (k3mini / debugmodel /
debugmodel8h -> 2016, everything else -> 163840), so the configs were not lying.
But it meant tokenizer parity had never been testable.

## What is there now

The official tokenizer is small enough to hold locally -- the 1.561 TB that
blocks the weights does not apply here:

    tiktoken.model         2,795,286 bytes
    tokenization_kimi.py      16,145 bytes
    tokenizer_config.json      3,478 bytes

Downloaded to `/workspace/k3_tokenizer`. Verified:

    tokenizer_class   TikTokenTokenizer
    auto_map          tokenization_kimi.TikTokenTokenizer
    BPE ranks         163,584
    added tokens      16 (ids 163584..163599)

163,584 + 16 = 163,600 against the config's `vocab_size` 163,840, so the
released vocab leaves headroom above the special tokens rather than packing them
flush -- which is why `K3_VOCAB = 163840` in model_configs is right and should
not be "corrected" to the token count.

## What this unblocks

Tokenizer parity against the release, on any flavor that opts into it. The debug
flavors keep the bundled 2016 tokenizer by default: a 163,840-row embedding and
lm_head on an 80.9M model would be 168M parameters of vocabulary alone, and the
point of those flavors is to exercise structure and parallelism cheaply. The
choice is now explicit rather than forced.
