"""LOCAL PROBE HACK (not committed): with EXPERTS_FP32 set, the routed experts run as a float32
per-expert matmul loop instead of the bf16 grouped GEMM (the one place a float32 model still
rounds to bf16), so the pipeline's gradients can be compared exactly against one GPU."""
import pathlib, sys
root = pathlib.Path(sys.argv[1])
c = root / 'torchtitan/models/common/moe.py'; s = c.read_text()
old = "        return torch._grouped_mm(A, weight_EOI.bfloat16().transpose(-2, -1), offs=offs)\n"
assert s.count(old) == 1
new = '''        if os.environ.get("EXPERTS_FP32"):  # LOCAL PROBE HACK (not committed)
            out = A.new_zeros(A.shape[0], weight_EOI.shape[1], dtype=torch.float32)
            ends = offs.tolist(); start = 0
            for e, end in enumerate(ends):
                if end > start:
                    out[start:end] = A[start:end].float() @ weight_EOI[e].float().t()
                start = end
            return out
        return torch._grouped_mm(A, weight_EOI.bfloat16().transpose(-2, -1), offs=offs)
'''
s = s.replace(old, new)
n = s.count("A=x_RD.bfloat16()"); s = s.replace("A=x_RD.bfloat16()", "A=(x_RD if os.environ.get(\"EXPERTS_FP32\") else x_RD.bfloat16())")
if "\nimport os\n" not in s: s = s.replace("import torch\n", "import os\n\nimport torch\n", 1)
c.write_text(s); print("common/moe.py: grouped_mm fp32 path;", n, "call sites")
k = root / 'torchtitan/models/kimi_k3/moe.py'; t = k.read_text()
n2 = t.count("A=x_RD.bfloat16(),"); t = t.replace("A=x_RD.bfloat16(),", "A=(x_RD if os.environ.get(\"EXPERTS_FP32\") else x_RD.bfloat16()),")
if "\nimport os\n" not in t: t = t.replace("import torch\n", "import os\n\nimport torch\n", 1)
k.write_text(t); print("kimi_k3/moe.py:", n2, "call sites")
