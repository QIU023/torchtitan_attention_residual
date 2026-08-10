# The per-parameter gradient band across parallelisms is KDA's

The band -- a 0.16 to 0.23 relative per-parameter gradient deviation between a parallel
leg and its single-rank reference -- has been carried as unexplained since 2026-08-02,
first as "LoRA is not usable on PP/CP", then withdrawn to "an AttnRes question". It is
neither. Isolated to KDA by one-variable arms.

## The measurement

`tp_trainer_grad_probe.py` inside a real trainer run, dumping every parameter's
materialized gradient norm per rank; `grad_band_compare.py` compares a leg against the
reference sharing its accumulation structure. Two columns, because one of them misleads:
`max_all` over every shared parameter is dominated by parameters whose gradient is ~0 in
both legs, so a claim rests on `max>1%`, restricted to parameters holding more than 1% of
the reference's gradient norm.

Warm checkpoint throughout, same protocol as `lora_vs_fullparam_axes.sh`.

## Arms, one variable at a time

| flavor | differs by | pp2 `max>1%` | cp2 `max>1%` | worst >1% |
|---|---|---|---|---|
| `mini_block_attn_res` | 21L, KDA, MoE, AttnRes | **0.16749** | **0.16260** | `layers.1.attn_res_proj.weight` |
| `diag_no_kda` | the same, minus KDA | 0.00249 | 0.00243 | `shared_experts.up_proj` |
| `diag_21l_mla` | minus KDA and MoE | 0.00079 | 0.00035 | `norm.weight` |
| `diag_4l_mla` | 4L, MLA, dense, AttnRes | 0.00056 | 0.00033 | `norm.weight` |
| `diag_4l_mla_noattnres` | the same, minus AttnRes | 0.00015 | 0.00040 | `embed_tokens.weight` |
| `diag_4l_mla` at fp32 | dtype | 0.00001 | 0.00147 | `mlp_res_proj` |

**Removing KDA takes the band from 0.167 to 0.0025 -- a factor of 67.** MoE accounts for
about 3x of what remains (0.0025 to 0.0008). Nothing else moves it.

## What this rules out, and each of these was the standing explanation at some point

* **LoRA.** Not it, and this was already withdrawn on 2026-08-08. The full-parameter arm
  is in the same band. What LoRA changes is composition: the AttnRes pseudo-queries hold
  10.1% of the gradient norm full-parameter and **76.4%** under LoRA, because the frozen
  base contributes none. Global relative deviation follows exactly that -- full-parameter
  0.9% / 2.2% / 0.9% on pp2 / cp2 / tp2, LoRA 2.7% / 6.6% / 1.8%. LoRA did not make the
  parallelism worse; it removed the well-conditioned majority that was diluting the
  measurement.
* **AttnRes.** Not it. With and without AttnRes at 4 layers the band is absent either way
  (0.00056 against 0.00015). The hypothesis under test was cancellation in the
  zero-initialized pseudo-queries -- `attn_res.py` documents 6x to 15x -- and fp32 does
  collapse what little deviation there is (0.00056 to 0.00001 on pp2), but there is no
  band to explain at that scale.
* **Depth.** Not it. 21 layers, MLA-only, with AttnRes: 0.00079.
* **"We hand-route gradients, so every axis we touch deviates."** Not it as a general
  claim, which is worth saying because the pattern invited it: FSDP and EP+FSDP are exact
  while TP, PP and CP all showed the band, and the one established defect (`o_proj`
  Rowwise bypass) was ours. But PP and CP plumbing on a KDA-free model is clean to 1e-3,
  so the axes are not the problem.

## The trap in the "worst parameter" column

The worst >1% parameter reads `attn_res_proj` in the banded arm, which is what pointed
three earlier documents at AttnRes, and pointed me there too. It is where the
contamination SHOWS, not where it comes from: the same `attn_res_proj` sits at 0.0025 once
KDA is removed. A column that names the loudest parameter cannot distinguish a cause from
a downstream amplifier, and the AttnRes pseudo-queries amplify -- they mix every block's
representation, so anything wrong upstream lands on them weighted by their share of the
norm.

The `max_all` column has the mirror problem: the top five in the full-parameter arm are
all `A_log`, at gradient magnitudes of 5e-5. That is a near-zero artifact by itself, but
in hindsight it was also the first sign this was KDA -- `A_log` is a KDA parameter, and it
was dismissed as noise.

## One number that does not reproduce

`LORA_PARALLELISM_MATRIX_2026-08-02`'s full-parameter `max_all` of 2.36570 (pp2) and
2.45181 (cp2) cannot be produced from the dump files that document names: the largest
`leg/ref` ratio in them is 1.967, so `|a-b|/|ref|` caps at 0.967 and a max-denominator
caps at 1.0. The qualitative claims of that section survive on the data -- the
full-parameter arm is not cleaner (0.187 against LoRA's 0.139 on the weighted column), and
the worst weighted parameter is an AttnRes pseudo-query in both arms -- but those two
numbers should not be cited.

## Where this points

KDA is the only path in this model that runs through a foreign kernel (fla's
`chunk_kda`, a triton scan) AND has its own hand-written CP (`_cp_all_to_all_headseq`, a
Ulysses all-to-all in `model.py`, plus `kcp.py`). The band appears under PP as well as CP,
and PP does not use the all-to-all, so the two have to be separated:

* `diag_1l_kda` under pp2 -- a single KDA layer, no all-to-all involved. A band there puts
  it in KDA's backward.
* the same under cp2 -- adds the all-to-all. A band only here puts it in our CP path.

That is the next arm, and it is the one that decides whether this is our code or fla's.

## Repro

    OUT=/workspace/attnres_band STEPS=6 bash attnres_grad_band_axes.sh   # edit the arm list
    python grad_band_compare.py /workspace/attnres_band <arm> ...

`grad_band_compare.py` was validated before use by reproducing the LoRA arm's recorded
numbers exactly (0.21084 / 0.17790) from the 2026-08-07 dumps.
