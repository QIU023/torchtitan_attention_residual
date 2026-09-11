# PR 4500's CP cells, reproduced on our hardware (2026-09-11)

Kept out of the reply to PR 4312 on purpose. It is about someone else's PR, it is off-topic
in a pipeline-parallelism thread, and "we cannot reproduce your published table" is a political
act, not a technical one. If it is ever raised it belongs under #4500, phrased as a question
about the environment (torch build, Attention Gym commit, the config behind the head) rather
than as a conclusion about the table.

PR 4500 published its table on H100. We had only reproduced it on A100, where it read two orders of magnitude larger than published, and the open question was whether the hardware class explained that. This run answers it: the same code, the same protocol, on H100.

`kimi_k3_debugmodel` CP=1 on spmd_types as the reference, then the two CP=2 recipes from the b200 suite, on the 4500 head `2884d82a9`; seed 42, `--debug.deterministic`, `--metrics.log_freq 1`, 256 tokens per train step, 100 steps, no seed checkpoint — the script is the one archived from the A100 run, with only the paths and the GPU ids changed.

| step | CP=1 loss | CP=2 all-gather (diff) | CP=2 Ulysses (diff) | CP=1 grad norm | all-gather gn (diff) | Ulysses gn (diff) |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 1 | `12.561730` | `12.327760` (-1.863%) | `12.340670` (-1.760%) | `28.8750` | `29.500000` (+2.165%) | `29.125000` (+0.866%) |
| 2 | `11.719530` | `11.186950` (-4.544%) | `11.034520` (-5.845%) | `38.0000` | `39.500000` (+3.947%) | `36.500000` (-3.947%) |
| 5 | `8.713460` | `8.047820` (-7.639%) | `7.766800` (-10.864%) | `15.8125` | `13.687500` (-13.439%) | `15.562500` (-1.581%) |
| 10 | `4.149080` | `4.162870` (+0.332%) | `4.510770` (+8.717%) | `7.3750` | `10.687500` (+44.915%) | `13.937500` (+88.983%) |
| 20 | `3.312170` | `3.241000` (-2.149%) | `3.246450` (-1.984%) | `6.6875` | `5.625000` (-15.888%) | `5.812500` (-13.084%) |
| 50 | `2.928190` | `3.037940` (+3.748%) | `2.992010` (+2.180%) | `3.4688` | `5.000000` (+44.142%) | `4.062500` (+17.115%) |
| 100 | `2.872700` | `2.967340` (+3.294%) | `2.866940` (-0.201%) | `5.3438` | `5.250000` (-1.755%) | `5.687500` (+6.432%) |

Side by side with what exists, steps 1 / 10 / 100 of the loss:

| source | hardware | CP=2 all-gather | CP=2 Ulysses |
| --- | --- | --- | --- |
| PR 4500, as published | H100 | `0.608%` / `0.150%` / `2.72%` | `0.576%` / `0.371%` / `1.34%` |
| our replication | 8 x A100-SXM4-40GB | `-0.115%` / `+8.56%` / `-7.86%` | `-0.26%` / `+9.70%` / `-6.31%` |
| our replication | 2 x H100 PCIe | `-1.863%` / `+0.332%` / `+3.294%` | `-1.760%` / `+8.717%` / `-0.201%` |

Cell by cell, which is the only way this comparison means anything:

- **All-gather, steps 10 and 100: our numbers are in her class.** `+0.332%` against her `0.150%`, and `+3.294%` against her `2.72%`. On the A100 the same cell read `+8.56%` and `-7.86%`, so for this cell the H100 run moves it from an order of magnitude away to the same size.
- **Ulysses, step 10: not in her class.** `+8.717%` against her `0.371%`, and that is the size the A100 read for the same cell (`+9.70%`). At step 100 Ulysses reads `-0.201%` against her `+1.34%`: both small, opposite sign.
- **Step 1, both cells: different in sign and about three times the magnitude.** Hers are `+0.608%` and `+0.576%`; ours are `-1.863%` and `-1.760%`. The A100 read `-0.115%` and `-0.26%`, so the three runs do not agree with each other here either.

So "not reproduced" was too broad. One cell lands in her class at the later steps and one does not, and step 1 disagrees for both. What the run does settle is narrower than we assumed: the hardware class alone does not account for the gap, since the same code on H100 puts all-gather close to her table and leaves Ulysses where the A100 had it.

These are single runs on each box, one seed, no repeats. A step-10 comparison between `+0.332%` and `0.150%` is a comparison of two small numbers from one sample each, and nothing here separates them; only the Ulysses step-10 gap and the step-1 sign difference are larger than what a single run can be trusted to show. Why the environments differ at all is open; the candidates are the torch build, the Attention Gym version, the exact code state behind the head, and the data, and this run settles none of them.

Footnote: 2 x NVIDIA H100 PCIe, capability 9.0; head `2884d82a9` (PR 4500), fetched from `pull/4500/head`; torch `2.15.0.dev20260906+cu130` (CUDA 13.0), triton `3.8.0+gitc01b6774`, spmd-types 0.2.5, torch-remat 0.2.0, nvidia-cutlass-dsl 4.6.0; Attention Gym at upstream main `b16d6d3`. That head's own KDA guard already admits capability 9.0 ("Hopper SM90 or Blackwell SM100/SM103"), so no guard relaxation was applied and `bound_gate` ran its fused CuTeDSL path, which is what an H100 run of hers would take; the A100 run had needed both a guard relaxation and an eager reference gate, neither of which applies here. One local uncommitted change was needed: `torchtitan/distributed/cudagraph.py` imports `torch.cuda._annotate_cuda_graph_trace`, which this torch nightly does not have, so the import moved inside the profiling post-processor that is its only user. Nothing else was patched, and nothing was committed.
