"""Rescore the same sequences with offline vLLM prompt_logprobs (same engine settings as the rollout)."""
import os, torch
EXPORT = os.environ.get("EXPORT", "/root/models/kimi-k3-debug-nt"); D = os.environ.get("DUMP", "/workspace/logprob_dump2")
def main():
    tt = torch.load(f"{D}/rescore_tt.pt")
    from vllm import LLM, SamplingParams
    llm = LLM(model=EXPORT, trust_remote_code=True, dtype="bfloat16", enforce_eager=os.environ.get("EAGER", "0") == "1",
              gpu_memory_utilization=0.35, max_model_len=1024, max_num_seqs=8, enable_chunked_prefill=True,
              max_num_batched_tokens=512, enable_prefix_caching=os.environ.get("PC", "1") == "1")
    keys = sorted(tt); reqs = [{"prompt_token_ids": tt[i]["seq"].tolist()} for i in keys]
    outs = llm.generate(reqs, SamplingParams(max_tokens=1, prompt_logprobs=0, temperature=1.0, logprobs=0))
    res = {}
    for i, o in zip(keys, outs):
        Lp, Lr, seq = tt[i]["Lp"], tt[i]["Lr"], tt[i]["seq"].tolist(); plp = o.prompt_logprobs
        res[i] = torch.tensor([plp[Lp + j][seq[Lp + j]].logprob for j in range(Lr)])
    torch.save(res, f"{D}/rescore_vllm{os.environ.get('TAG','')}.pt"); print("saved", len(res), flush=True)
if __name__ == "__main__": main()
