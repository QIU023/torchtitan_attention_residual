import torch, torch.distributed as dist
dist.init_process_group("nccl")
r = dist.get_rank()
torch.cuda.set_device(r)
t = torch.ones(1024, device="cuda") * r
dist.all_reduce(t)
assert int(t[0]) == sum(range(dist.get_world_size())), t[0]
if r == 0:
    print("ALLREDUCE_OK", int(t[0]))
dist.destroy_process_group()
