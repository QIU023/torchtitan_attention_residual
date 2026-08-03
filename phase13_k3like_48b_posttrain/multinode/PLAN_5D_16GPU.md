# 16-GPU two-node plan: full 5D parallelism, full-param + LoRA + QLoRA (2026-07-26)

Closes the last structural item on the 07-24 next-box plan (sec 4 item 5).
Companion docs: [README.md](README.md) (topology, overlay runbook, measured
bandwidth), [NODE1_PROMPT.md](NODE1_PROMPT.md) (worker bootstrap).

## 1. Why 16 ranks -- what this buys and what it does not

As of 2026-07-26 **every <=8-rank projection of the 5D mesh is green**: all
four 3-of-{DP,TP,CP,PP} combinations, each with and without EP, plus HSDP x CP,
cp8, tp8, Interleaved(vp2) x CP x EP, compile x 3D, AC x CP, and LoRA/QLoRA
against individual axes (PACKED_TP_VERIFICATION_2026-07-25 sec 9 and the 07-24
Parts 4-5). What is still missing is the single run where
**FSDP x TP x CP x PP are all > 1 simultaneously**, which needs
`dp_shard2 x tp2 x cp2 x pp2 = 16` ranks and was physically impossible on one
8-GPU box.

Three things this run produces that we cannot get any other way:

1. **The unconditional composition claim.** Today the honest phrasing is
   "every <=8-rank projection verified; the 16-rank single run is untested".
   After this it becomes "5D verified", which is what the RFC and the K3
   support PR want to say about topology coverage.
2. **LoRA and QLoRA under the full mesh.** The post-training recipe's real
   topology. TP for the packed-MXFP4 base was only fixed on 2026-07-25 and has
   never seen PP + CP + DP at the same time.
3. **Axis-to-node placement evidence on commodity Ethernet.** We have two real
   nodes on a measured 1 GbE link (936 Mbit/s symmetric, saturating on a
   single flow). Which parallelism axis you let cross a slow node boundary is
   the single biggest practical decision when reproducing this on rented
   hardware, and nobody in the RFC has a number for it.

**What it does NOT prove, stated up front so no one over-reads it:** nothing
about real-scale throughput or MFU (1 GbE is 2-3 orders below a datacenter
fabric -- any step-time number here is a commodity-Ethernet data point, NOT a
cluster projection), nothing about 48B real weights, nothing about long
context. Debug-scale cells verify MECHANISMS.

## 2. Mesh and placement (verified from the code, not assumed)

`parallel_dims.py:231` builds the mesh as
`["pp", "dp_replicate", "dp_shard", "cp", "tp"]` -- **pp outermost, tp
innermost**. With 8 GPUs per node and global ranks 0-7 on node 0, 8-15 on
node 1:

- `pp2` (outermost) splits at the node boundary: node 0 = stage 0, node 1 = stage 1
- `dp_shard2 x cp2 x tp2 = 8` fits entirely inside each node

So the default rank placement is already optimal: only the PP stage-boundary
P2P crosses the LAN, while TP (2 collectives per layer) and CP (an all-to-all
per attention) stay on-node. **Do not reorder ranks.** Cell 6 measures what the
alternative costs.

Carriers: `kimi_k3_debugmodel8h` (H=8, d=512) for full-param, and the H=4
`kimi_k3_debugmodel_gated_lora` / `..._gated_qlora_mxfp4` for the adapter
cells -- `tp*cp = 4` exactly matches H=4, which is the documented bind.

## 3. Cell matrix

Every cell: `--debug.seed 42 --debug.deterministic`, `--training.global-batch-size`
pinned, bf16, seq 512 unless noted. **Gate for all: descending finite loss AND
rank-identical loss/grad_norm across all 16 ranks** (the Part-3 lesson), plus
step-1 within the mesh-numerics band of the corresponding <=8-rank projection.

| # | cell | flavor | what it proves |
|---|---|---|---|
| 1 | **5D: dp2 x tp2 x cp2 x pp2** | debugmodel8h | the headline -- all four axes at once |
| 2 | **5D + ep2** (folded into dp_shard x cp = 4) | debugmodel8h | the full 6-axis K3 topology; expect losses bit-identical to cell 1 (EP==FSDP reduction parity, which held on every <=8-rank combo) |
| 3 | **5D x LoRA** | debugmodel_gated_lora | the post-train recipe at full mesh; previously only LoRA x fsdp2tp2cp2 (4 axes) |
| 4 | **5D x QLoRA (packed MXFP4)** | debugmodel_gated_qlora_mxfp4 | the 48B-QLoRA path at full mesh. Needs the packed DCP checkpoint on BOTH nodes (9.3 MB, rsync -- see sec 5) |
| 5 | **5D x AC full** | debugmodel8h | 48B SFT will need AC; must not change forward numerics (bit-identical to cell 1, as AC x CP was) |
| 6 | **placement A/B**: (a) pp crossing (default) vs (b) dp_replicate crossing (`dp_replicate2 x [tp2 cp2 pp2 intra-node]`) | debugmodel8h | step-time delta on a measured 1 GbE link -> the axis-to-node guidance |
| 7 | **5D x Interleaved1F1B (vp2)** | debugmodel8h | virtual stages at 16 ranks. **Requires** `pipeline_parallel_layers_per_stage` + first/last_stage_less_layers per the phase3 PP8xVP4 recipe -- without them the default splitter emits non-contiguous P2P buffers (07-24 Part-4 CORRECTION) |
| 8 | **DCP at 16 ranks**: mid-run save + same-mesh resume, then cross-mesh reshard 16 -> 8 | debugmodel8h | SFT recipes reshard at deploy; the 8-rank version of this is verified, the 16-rank mesh is not |

Cells 1-4 are the deliverable; 5-8 are the same-session extras that make the
coverage table complete.

## 4. Risks, in the order I expect them to bite

1. **The AttnRes cross-stage adapter crosses a network for the first time.**
   Its payload rides the PP stage boundary, and `pipeline_adapter.py` reaches
   into `pp_schedule._stages` private attrs. Everything before this ran the
   adapter over PCIe inside one box. This is the highest-risk item; if a cell
   fails, check the adapter before blaming the mesh.
2. **NCCL data plane needs the overlay.** Ranks advertise ephemeral ports on
   their container interface; both nodes' docker bridges are `172.17.0.x` and
   mutually unroutable, and `/dev/net/tun` is absent so no DIY tunnel is
   possible. `NCCL_SOCKET_IFNAME` must name the overlay interface, not `eth0`
   (which is the docker bridge here). Rendezvous alone would work over the LAN
   + a forwarded port, which makes a half-working setup easy to mistake for a
   working one -- cell 0 (the NCCL smoke) exists to catch exactly that.
3. **Cross-node reduction order** differs from the 8-rank projections, so
   step-1 equality is a band, not an identity. Rank-identical-within-a-run is
   the hard gate.
4. **1 GbE + high retransmits** (iperf3 showed ~6k retransmits per 10 s at
   line rate -- shallow buffers). Expect latency jitter; at debug scale the
   default NCCL timeout has ample headroom.
5. **Two agents, one repo.** Node 1 is read-only on git: node 0 owns all
   commits and pushes. Node 1 reports numbers back; node 0 records them.

## 5. Prerequisites and order of operations

```
(a) overlay exists and both instances are joined      <- USER, account API key
(b) node 1 environment built + single-node sanity     <- NODE1_PROMPT.md
(c) rsync the packed ckpt to node 1 (cell 4 only):
      node0$ tar cz -C /workspace packed_mxfp4_ckpt | \
             ssh -p 11034 root@192.168.50.19 'tar xz -C /workspace'
(d) cell 0: nccl_smoke_2node.py  -> records cross-node busbw; MUST pass first
(e) cells 1-4, then 5-8
```

Launchers: `launch_smoke.sh` (cell 0) and `launch_5d.sh` (cells 1-2 today;
3-8 are added as flags once the smoke is green). Both take
`<NODE_RANK> <NCCL_IFACE> <MASTER_ADDR>` and are run once per node.

## 6. Deliverable

`VERIFICATION_5D_16GPU_2026-07-26.md` in this directory: the cell table with
step-1/step-N losses and grad_norms, the cross-node busbw from cell 0, the
cell-6 placement numbers with the 1 GbE caveat attached, and whatever broke.
Then GAPS_TO_K3_SFT B6 and the 07-24 sec-4 item 5 both close.
