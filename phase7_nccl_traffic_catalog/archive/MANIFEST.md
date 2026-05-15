# Phase 7 NCCL trace archive

Generated: 2026-05-03T17:37:11Z
Hostname: b008e969124e
Hardware: 8× RTX 5090 PCIe (Blackwell sm_120, 32 GiB / GPU)
Workspace HEAD: 6fc444d3ea68ac930f6f0c7769e9091b2940fe59 phase7: throughput bottleneck quantification
Submodule HEAD: a2d7ecb3c162005e13dfe0b494633f4f6a5ae8fa kimi_linear, attn_res: A3 (FSDP×PP×TP) end-to-end fix via DSv3 inner_attention pattern

Each per-run tarball is self-contained. Untar and replay with
phase7_nccl_traffic_catalog/extract_collectives.py (re-emit CSV from raw NCCL log).

## Archives

| Archive | Mesh | Tier | GBS | Steps | Size (gz) | sha256 |
|---|---|---|---|---|---|---|
| `b0_fsdp8_alignment.tar.gz` | FSDP=8 PP=1 | tier_c | 16 | 500 | 696K | `b5bb43302a5b…` |
| `b0_fsdp8_tier_b.tar.gz` | FSDP=8 PP=1 | tier_b | 120 | 50 | 96K | `6b8e4b19c4e7…` |
| `b0_fsdp8_tier_a.tar.gz` | FSDP=8 PP=1 | tier_a | 384 | 100 | 1.2M | `5fe17284cc85…` |
| `a2_fsdp2_pp4_alignment.tar.gz` | FSDP=2 PP=4 V=2 | tier_c | 16 | 500 | 2.4M | `8a2bb17036fb…` |
| `a2_fsdp2_pp4_tier_b.tar.gz` | FSDP=2 PP=4 V=2 | tier_b | 120 | 50 (failed) | 32K | `c2cc099a99aa…` |
| `a2_fsdp2_pp4_tier_a.tar.gz` | FSDP=2 PP=4 V=2 | tier_a | 384 | 100 (failed) | 32K | `fd2cca44c8c1…` |
| `a3_fsdp2_pp2_tp2_alignment_SNAPSHOT.tar.gz` | FSDP=2 PP=2 TP=2 V=2 | tier_c | 16 | 500 (in flight) | 11M | `979408db0fb5…` |
| `v10_fsdp8_pretrain_PARTIAL.tar.gz` | FSDP=8 PP=1 | tier_b | 120 | interrupted | 156K | `054f7f49a521…` |
| `v10_3d_fsdp2pp2tp2_pretrain_SNAPSHOT.tar.gz` | FSDP=2 PP=2 TP=2 V=2 | tier_b | 120 | 5000 (3D) | 17M | `58264c209b4e…` |

## Helper scripts (in tools.tar.gz)

- `phase7_nccl_traffic_catalog/extract_collectives.py` — parses NCCL_DEBUG=INFO logs into structured CSV
- `phase7_nccl_traffic_catalog/build_pattern_catalog.py` — aggregates per-run CSVs into pattern_catalog.md
- `phase7_nccl_traffic_catalog/pattern_catalog.md` — human-readable cross-config histogram (current snapshot)
- `phase7_nccl_traffic_catalog/README.md` — three-tier recording rationale
