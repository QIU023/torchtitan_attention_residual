"""Dump the vLLM K3 model's parameters and buffers (vLLM-internal names) after a normal disk load, via collective_rpc."""
import os, torch
EXPORT = "/root/models/kimi-k3-debug-nt"; OUT = os.environ.get("OUT", "/workspace/vllm_disk_params.pt")
def _dump(self):
    m = self.model_runner.model
    d = {n: p.detach().to("cpu") for n, p in m.named_parameters()}
    d.update({("buffer:" + n): b.detach().to("cpu") for n, b in m.named_buffers()})
    torch.save(d, OUT); return len(d)
def main():
    from vllm import LLM
    llm = LLM(model=EXPORT, trust_remote_code=True, dtype="bfloat16", enforce_eager=True, gpu_memory_utilization=0.35, max_model_len=1024, max_num_seqs=8)
    print("dumped", llm.collective_rpc(_dump), flush=True)
if __name__ == "__main__": main()
