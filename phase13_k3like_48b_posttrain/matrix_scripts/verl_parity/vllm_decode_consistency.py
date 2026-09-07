"""Offline vLLM with the rollout's engine settings: generate n=5 x 256 tokens from verl's prompts, keep the sampled tokens'
decode-time log-probs, then rescore the same sequences in one prefill (prompt_logprobs). Decode vs prefill self-consistency."""
import os, torch
EXPORT = "/root/models/kimi-k3-debug-nt"; D = "/workspace/logprob_dump2"; MODE = os.environ.get("LPMODE", "raw_logprobs")
def main():
    b = torch.load(f"{D}/batch.pt"); prompts, plen = b["prompts"], b["prompt_lengths"]
    rows = list(range(0, prompts.shape[0], 5))[:8]
    from vllm import LLM, SamplingParams
    kw = dict(model=EXPORT, trust_remote_code=True, dtype="bfloat16", enforce_eager=True, gpu_memory_utilization=0.35, max_model_len=1024,
              max_num_seqs=8, enable_chunked_prefill=True, max_num_batched_tokens=512, enable_prefix_caching=True, logprobs_mode=MODE,
              enable_sleep_mode=os.environ.get("SLEEP", "0") == "1", seed=int(os.environ.get("SEED", "0")))
    llm = LLM(**kw)
    reqs = [{"prompt_token_ids": prompts[i, :int(plen[i])].tolist()} for i in rows]
    outs = llm.generate(reqs, SamplingParams(n=5, max_tokens=256, temperature=1.0, top_p=1.0, top_k=-1, logprobs=0))
    seqs, dec = [], []
    for o, i in zip(outs, rows):
        for c in o.outputs:
            gen = list(c.token_ids); seqs.append(list(o.prompt_token_ids) + gen)
            dec.append(torch.tensor([lp[t].logprob for lp, t in zip(c.logprobs, gen)]))
    outs2 = llm.generate([{"prompt_token_ids": s} for s in seqs], SamplingParams(max_tokens=1, prompt_logprobs=0, temperature=1.0, logprobs=0))
    ds = []
    for s, o2, d in zip(seqs, outs2, dec):
        Lp = len(s) - len(d); plp = o2.prompt_logprobs
        pre = torch.tensor([plp[Lp + j][s[Lp + j]].logprob for j in range(len(d))]); ds.append((d - pre).abs())
    dd = torch.cat(ds); print(f"[{MODE} sleep={kw['enable_sleep_mode']}] decode vs prefill on {len(seqs)} sequences, {len(dd)} tokens: mean|d|={dd.mean():.4f} frac>0.5={(dd > 0.5).float().mean():.3f} max={dd.max():.3f} mean(decode)={torch.cat(dec).mean():.3f}", flush=True)
    torch.save({"seqs": seqs, "dec": dec}, f"{D}/vllm_offline_gen_{MODE}.pt")
if __name__ == "__main__": main()
