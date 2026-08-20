# `kimi_k3_debugmodel_bubble_ratio`:为什么把 seq 从 256 改成 4096

从 `config_registry.py` 的函数 docstring 搬出。原文 44 行。内容未改。

pp8xvp4 with an HONEST vision/text cost ratio, so bubble hiding is observable.

Report 5.2.3's bubble hiding is invisible on ``report_arch_pp8vp4`` and that is a
property of the config, not of the implementation. Measured with
``dep_cost_ratio.py`` (logbook, phase13): one ViT forward there costs **r = 14.0
text-stage forwards** -- a number that CANNOT be re-derived today, because that script
stopped running after config-ization (KimiK3Model now takes a wrapper carrying
.kimi_config). Treat r as unverified until it is fixed; the bubble planner's
KIMI_VIT_BUBBLE_COST_RATIO has been a hand-filled value ever since.
One ViT forward there costs r = 14.0 text-stage forwards,
while the hideable share reaches the report's "most" only at **r <= 0.3**
(``dep_hiding_theory.py --sweep``). At r = 14 the theoretical hideable share is
ZERO, so a correct implementation must also measure zero -- which is what three
independent checks found.

The cause is the SEQUENCE, not the model. At ``seq_len=256`` with
``max_patches=1024`` the post-merge visual tokens are 1024/4 = 256, i.e. **100% of
the sequence**, while a text stage holds only 30/32 = 0.94 layers. The tower is not
oversized (4.87M against 2.42M for one text layer); the text side is starved.

So this flavor changes ONE thing: ``seq_len`` 256 -> 4096, giving a text stage 16x the
tokens. The patch budget stays at 1024, so one image is 256 post-merge tokens, **6.2%
of the sequence** -- the regime the report describes, where visual tokens are a small
share of a long context.

That lands **r = 0.493**, which is the point of MAXIMUM observable effect rather than
merely an absorbable one. There is a tension between the two ends: at small r the
bubbles absorb the encode but the encode is a negligible share of the step, and at
large r it is a big share but cannot be absorbed. Latency saving is
``hidden_share(r) * ViT_share_of_step(r)``, which peaks near r = 0.5 at **2.65%** and
falls away on both sides -- 1.19% at r = 0.2, 0% at this flavor's parent r = 14.

seq 4096 is also the longest length shown to fit: it peaks at 7.7% of 15.5 GiB under
the chunked loss, while seq 8192 with plain CE OOMs, so 8192 is not assumed safe.

Deliberately NOT changed:

* **Layer counts.** 30 text layers plus the wrapper's two children is exactly
  32 = 8 x 4, which is what makes pp8 x vp4 expressible at all. Changing depth to
  chase a cost ratio would remove the cell being measured.
* **Vision width.** Shrinking the tower would reach the same r, but the tower is
  already small relative to a text layer and the real MoonViT is LARGER relative to
  its text stack (401M over 27 layers against ~1.12B activated per text layer). A
  config that shrinks vision would move away from the report, not toward it.
