"""CPU check of the DEP ratio data: for each (DEPR_RES, DEPR_SEQ) variant, build dep_ratio_local's DEP-off config
and pull N micro-batches. Prints the occupied positions (seq minus padding), the loss tokens, the images per
micro-batch and the patches per image. Usage: python check_batches.py [N]"""
import itertools
import os
import sys

N = int(sys.argv[1]) if len(sys.argv) > 1 else 12
for res, seq in itertools.product((224, 448, 768, 1024), (1024, 2048)):
    os.environ["DEPR_RES"], os.environ["DEPR_SEQ"] = str(res), str(seq)
    import importlib

    import dep_ratio_local

    importlib.reload(dep_ratio_local)
    cfg = dep_ratio_local.dep_off()
    tok = cfg.tokenizer.build(tokenizer_path=cfg.hf_assets_path)
    dl = cfg.dataloader.build(
        dp_world_size=1, dp_rank=0, tokenizer=tok, max_context_length=seq, num_tokens_per_microbatch=seq
    )
    occ, loss_tok, nimg, patches = [], [], [], set()
    for b in itertools.islice(iter(dl), N):
        occ.append(seq - int(b.padding_mask.sum()))
        loss_tok.append(int(b.num_valid_tokens))
        g = b.model_kwargs.get("grid_thw")
        nimg.append(0 if g is None else int(g.shape[0]))
        if g is not None:
            patches.update(int(x) for x in g.prod(-1).tolist())
    print(
        f"res {res} seq {seq}: cap {dep_ratio_local._images_cap()}, {dep_ratio_local.image_tokens(dep_ratio_local._side())} tokens per image; "
        f"occupied min/mean/max {min(occ)}/{sum(occ) / len(occ):.0f}/{max(occ)}; loss tokens mean {sum(loss_tok) / len(loss_tok):.0f}; "
        f"images per micro-batch {nimg} (mean {sum(nimg) / len(nimg):.2f}); patches per image {sorted(patches)}",
        flush=True,
    )
