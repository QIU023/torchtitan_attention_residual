import importlib.util, sys, random
def load(path, name):
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec); sys.modules[name] = mod; spec.loader.exec_module(mod); return mod
old = load('../fig11/old/plan_6b580e438.py', 'old_plan')
new = load('torchtitan/models/kimi_k3/pipeline_parallel/vision_dep/plan.py', 'new_plan')
rng = random.Random(0)
cases = 0
for trial in range(300):
    m = rng.choice([2, 4, 6, 8, 16])
    pp = rng.choice([2, 3, 4, 8])
    loads = {mb: rng.choice([16, 64, 256, 1024]) for mb in range(m) if rng.random() < 0.8}
    if not loads: continue
    for trainable in (True, False):
        kw = dict(num_microbatches=m, num_ranks=pp, stage0_rank=0, trainable=trainable)
        a, b = old.VisionDepPlan(loads, **kw), new.VisionDepPlan(loads, **kw)
        for name in ('encode_rank', 'backward_rank', 'prologue', 'epilogue', 'anchored', 'posts', 'placed'):
            assert getattr(a, name) == getattr(b, name), (name, loads, pp, trainable)
        cases += 1
print('K2.5 form (no pipeline order): old and new plans equal in', cases, 'cases')
