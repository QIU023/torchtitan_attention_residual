# Two-node (2x 8x5060Ti) setup -- vast.ai, same-LAN pair (2026-07-26)

Topology discovered from the instances API:

| | instance | machine | host LAN IP | SSH (public) | direct port range |
|---|---|---|---|---|---|
| node 0 (this box) | 45781657 | 53171 | **192.168.50.17** | :10136 | 10024-10193 |
| node 1 | 45949215 | 52572 | **192.168.50.19** | :11034 | 11034-11161 |

Same operator (host 309585), same public IP 157.211.174.147, same /24 LAN.
Verified from inside node 0's container: TCP to `192.168.50.19:11034` connects
-- so cross-node traffic rides the LAN, not the 92.7 Mbps WAN uplink.

## The two transport layers (do not conflate)

1. **torchrun rendezvous (c10d TCP store)**: one fixed port. Works TODAY over
   the LAN + the direct-forwarded port (node 0 binds container port 10193,
   which the engine forwards 1:1; node 1 dials `192.168.50.17:10193`).
2. **NCCL data plane**: every rank listens on EPHEMERAL ports on its
   container's docker-bridge iface (`172.17.0.x` -- both nodes use the same
   private subnet, mutually unroutable). This CANNOT cross the NAT no matter
   which ports are forwarded, and `/dev/net/tun` is absent in these containers
   so a DIY WireGuard/tailscale overlay is not possible either.
   **The vast overlay network is the only supported path** (it gives both
   containers routable addresses on one virtual LAN):
   https://docs.vast.ai/multi-node-training-using-torch-nccl

## Step 1 -- overlay (run on YOUR machine with your account API key)

The in-container key lacks permissions for these. `join overlay` exists in the
current CLI, so the EXISTING instances can be joined -- no re-renting, unless
step (a) shows no cluster.

```bash
pip install -U vastai        # the overlay commands need the current CLI
# (a) are these machines registered as a physical cluster by the operator?
vastai show instances --raw | grep -iE '"id"|machine_id|cluster_id'
# (b) if a cluster_id shows up:
vastai create overlay <CLUSTER_ID> k3net
vastai join overlay k3net 45781657
vastai join overlay k3net 45949215
# (c) verify: a new interface appears inside each container
vastai show overlays
```

If (a) shows **no cluster_id**, the operator has not registered these machines
as a cluster and the overlay cannot be created by a renter. Options then:
message the host via the vast console to register the pair, or rent a pair
from a listing that advertises `cluster_id != None`
(`vastai search offers 'cluster_id!=None num_gpus=8 gpu_name=RTX_5060_Ti'`).
The bandwidth number from step 2 tells you whether this pair is even worth
that effort.

## Step 2 -- bandwidth (works NOW, no overlay needed)

node 0 already runs `iperf3 -s -p 10193` (syncthing stopped to free the port).
On node 1:

```bash
apt-get install -y iperf3
iperf3 -c 192.168.50.17 -p 10193        # node1 -> node0
iperf3 -c 192.168.50.17 -p 10193 -R     # node0 -> node1 (reverse, same server)
iperf3 -c 192.168.50.17 -p 10193 -P 4   # 4 parallel streams (NCCL-like)
```

**MEASURED 2026-07-26: 1 GbE, saturated.** Single stream 936 Mbit/s, reverse
938 Mbit/s, 4 parallel streams 930 Mbit/s (so the cap is the link, not per-flow)
-- symmetric, and 10x the 92.7 Mbit/s WAN uplink the machine advertises. High
retransmits at line rate (~6k per 10 s single-stream, ~9.7k with 4 streams)
imply shallow buffers, i.e. latency jitter under load; irrelevant at debug
scale.

What 117 MB/s means per optimizer step, cross-node:

| what crosses | volume/step | time | verdict |
|---|---|---|---|
| HSDP `dp_replicate` of 48B LoRA grads (7.6 M trainable -> 15.2 MB, all-reduce ~2x) | 30 MB | **0.26 s** | fine |
| PP stage boundary, 48B seq8192 B=1, 8 microbatches fwd+bwd | ~600 MB | 5.1 s | usable for validation, not for throughput |
| PP stage boundary, debugmodel8h seq512 (what the 5D cells use) | ~1 MB/microbatch | negligible | fine |
| FSDP `dp_shard` shipping parameters (bf16 98 GB / MXFP4 25 GB) | 25-150 GB | 3.5-21 min | dead |
| TP (27 layers x 2 collectives, plus ~108 sequential round trips) | ~4 GB | 34 s + latency | dead |

Conclusion for real work: let **PP** (or HSDP's replicate axis, under LoRA)
cross the node boundary; keep TP and CP strictly intra-node. The default rank
placement already does this -- see PLAN_5D_16GPU.md sec 2.

## Step 3 -- NCCL smoke (after the overlay exists)

`nccl_smoke_2node.py` here does rendezvous + a size-swept all-reduce and
prints per-size busbw. Node 0:

```bash
bash phase13_k3like_48b_posttrain/multinode/launch_smoke.sh 0 <OVERLAY_IFACE>
```
Node 1 (same repo checkout; see setup_worker.sh):
```bash
bash phase13_k3like_48b_posttrain/multinode/launch_smoke.sh 1 <OVERLAY_IFACE>
```
`OVERLAY_IFACE` = the new interface from step 1(c) (`ip a` inside the
container; the docs' examples call it eth0 on overlay-provisioned instances,
but on join-ed instances check -- the docker bridge already owns eth0 here).
MASTER_ADDR inside the scripts must then be node 0's OVERLAY address, not
192.168.50.17 (which is the host LAN, fine for the rendezvous but the NCCL
ifname must be the overlay's).

## Step 4 -- the 16-rank 5D run (the actual payload)

```bash
# node 0 / node 1 (identical except the first arg)
bash phase13_k3like_48b_posttrain/multinode/launch_5d.sh 0 <OVERLAY_IFACE>
bash phase13_k3like_48b_posttrain/multinode/launch_5d.sh 1 <OVERLAY_IFACE>
```

dp_shard2 x tp2 x cp2 x pp2 = 16 (+ an EP-folded variant inside the script).
Mesh order is `[pp, dp_replicate, dp_shard, cp, tp]` (parallel_dims.py:231) --
pp outermost -> with default rank placement, node 0 = pp stage 0 and node 1 =
stage 1, so **only the PP stage-boundary P2P crosses the LAN**; tp/cp/dp stay
intra-node. This is the correct placement automatically; do not reorder ranks.

Carrier: `kimi_k3_debugmodel8h` (H=8 clears tp*cp=4 with headroom), seed
42 deterministic, global batch pinned. Gate: descending finite loss +
rank-identical loss/grad_norm across all 16 ranks, compared against the
verified <=8-rank projections in CP_TP_3D_VERIFICATION_2026-07-24.md.

## Worker-node bootstrap

`setup_worker.sh` clones the logbook + submodules at the pinned SHAs and
rebuilds /venv/main exactly per SESSION_HANDOFF_2026-07-24 sec 2 (same
template/image, so torch 2.12.0+cu130 is preinstalled). Optionally add node
0's SSH pubkey to node 1's authorized_keys first so node 0 can drive both ends:

```bash
# on node 1
echo 'ssh-ed25519 AAAAC3NzaC1lZDI1NTE5AAAAIAegwvSEaRefx/eCwUICT4SnGVGVo6p1fUdNTvsuKqKj yiqiaoqiu@hotmail.com' >> ~/.ssh/authorized_keys
# then from node 0
ssh -p 11034 root@192.168.50.19
```
