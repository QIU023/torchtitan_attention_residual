# MoonEP on the 4 x H100 box, 2026-09-19

Raw material behind `phase13_k3like_48b_posttrain/MOONEP_H100_2026-09-19.md`. Box `115.124.123.240:16923`, 4 x H100 80GB SXM on an NVSwitch fabric, torch 2.15 nightly cu126, CUDA toolkit 12.6, MoonEP `2bd860b` with the cutlass 4.6 rename.

`kit/` is the tooling as it stood at the end of the session. What is new today, against the H200 kit it started from:

    run_cell.py, run_probe.py, make_seed.py   checkpointing moved off the CLI in ec953b36; these
                                              drive it through scripts/_checkpoint_test_config
    h200/grad_probe.py                        rewritten as a copy of Trainer.train_step truncated
                                              before optimizer_step, with a loss gate
    moe_trace_probe.py, moe_trace_diff.py     per-layer router and routed-expert capture, and the
                                              diff that names the first divergence
    moe_first_divergence.py                   characterises one layer: rounding order or dropped tokens
    moonep_vs_std_reference.py                both dispatchers on the same tokens against one fp32
                                              dense reference
    moonep_balance.py, balance_sweep.py       per-rank load for both dispatchers, the second with
                                              shapes as parameters and Zipf routing
    rank_damping.py                           how the standard path's rank imbalance grows as
                                              experts per rank falls (arithmetic, no transport)
    row_occupancy_check.py                    cell 9 from the captured per-row counts
    qb_balance_probe.py, qb_summary.py        QB on/off against each backend, sampled after N full
                                              training steps so the QB bias has moved
    install_deepep.sh                         DeepEP v2 build chain; it builds here and cannot run,
                                              see the logbook section

`results/` is every log, csv, json and txt those produced. The `.pt` activation captures (185 MB) are left on the box: every number derived from them is in the logbook and in `results/`.

`onbox_local_flavors_and_guard_lift.patch` is the box's uncommitted tree state: the local flavors (`_moonep`, `_deepep`, `_noqb`, `_noqb_moonep`, the C4 pairs, `_b7` / `_b1` for the slot-count cells) and the KDA capability guard lift. None of it is upstream-bound; the flavor rule keeps backend-selection flavors off the PR branch.
