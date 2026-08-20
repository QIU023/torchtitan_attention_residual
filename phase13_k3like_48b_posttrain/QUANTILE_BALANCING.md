# Quantile Balancing:为什么解而不是步进

从 `kimi_k3/quantile_balance.py` 的模块 docstring 搬出。原文 59 行,
是 reviewer 打开文件第一眼撞上的东西;两条仓库规矩都要求注释精简、WHY 进文档。
内容未改。

so ``b`` regulates dispatch without altering the mixture weights or the
router's gradients. The original rule nudges the bias by a fixed step,
``b_j += gamma * sign(mean_load - load_j)``, where gamma trades slow adaptation
against load oscillation -- and that trade-off gets worse as the routed pool
grows to 896 experts per layer.

Quantile Balancing removes the step size by SOLVING for the bias rather than
nudging it. Route with Top-(k+1) on the biased score: the first k entries are
the routes actually taken, and the (k+1)-th is the cutoff ``alpha_i`` that an
expert must exceed to enter token i's Top-k. Under a candidate bias the count
routed to expert j is ``sum_i 1[s_ij + b_j > alpha_i]``, monotonically
decreasing in ``-b_j``, so setting that count to the target load
``q = m*k/n`` puts ``-b_j`` at the (q+1)-th largest margin ``s_ij - alpha_i``.
Since ``q/m = k/n``::

    b_hat_j = -quantile_{1-k/n}( s_{:,j} - alpha )                    (Eq. 14)
    b       = b_hat - mean(b_hat)

The second line removes a common offset, which leaves Top-k selection
unchanged. The update takes effect on the NEXT step -- a batch is never routed
with a bias derived from itself -- and the bias is frozen at inference.

One property of Eq. 14 worth stating precisely, because it bounds what a single
update can do. The count identity "routing to expert j equals the target load"
holds only up to TIES at the threshold, and the margin distribution has an atom
exactly at zero: alpha_i is itself one of the scores, so whenever expert j is
token i's (k+1)-th expert, s_ij - alpha_i is exactly 0. The atom's mass is the
number of tokens for which j sits at the cutoff, which is large precisely for
the over-subscribed experts the bias is meant to demote. Measured on a
deliberately skewed router (n=16, k=2, m=4096, target load 512), the most
popular expert had 419 zero margins, and one application of the solved bias put
its count at 273 rather than 512 -- undershooting the target, i.e. erring
toward demoting it further.

So a single update does not land on balance; the per-step iteration is what
converges. On the same skewed setup the load coefficient of variation went
0.607 -> 0.182 -> 0.132 -> 0.105 -> 0.084 -> 0.063 -> 0.053 over 60 updates,
monotonically. That is still unconditionally better behaved than the sign rule,
whose step size gamma has to be tuned against oscillation, but "solves for the
bias" describes the per-step subproblem, not the whole trajectory.

At scale the quantile spans the whole global batch (millions of margins across
ranks and accumulation steps), so gathering them exactly is not viable.
:func:`quantile_balance_bias_histogram` instead reads each expert's quantile
from pooled histograms of its margins: counts are additive, so one all-reduce
of the per-rank bin counts represents the whole global batch regardless of how
tokens are sharded, and the estimate is exact up to the bin width.

Superseded: this module previously shipped a PROVISIONAL rule that took the
quantile of the expert LOAD distribution (a smooth stand-in for the sign rule).
That is a different algorithm -- it still nudges by a coefficient, whereas QB
solves for the bias that hits the target load exactly.
