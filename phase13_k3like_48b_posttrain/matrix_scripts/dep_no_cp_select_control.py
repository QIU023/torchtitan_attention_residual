"""Run with the DEP vision stage's CP shard selection disabled -- the finding-60 control.

Finding 60 was that the DEP path exchanged the per-rank sentinel counts and then threw
them away, never calling `_select_cp_shard`, so the stage spliced EVERY rank's visual
features into its own sequence shard. The call was added, but two records disagree about
whether it matters: the code comment says the missing call produced a step-1 forward
0.042 away from the non-DEP path, while OVERNIGHT_REVIEW records that the numbers never
moved on any configuration tried. One of them is wrong.

This restores exactly the pre-fix behaviour, and only on the DEP path: overriding
`_select_cp_shard` on `KimiK3ViTStage` leaves `KimiK3MultimodalModel`'s own call intact,
so a DEP-off arm still selects and remains a valid reference.

Usage: KIMI_VIT_DEP=1 torchrun ... dep_no_cp_select_control.py <the usual train args>
"""

from __future__ import annotations

import runpy


def main() -> None:
    from torchtitan.models.kimi_k3.multimodal_model import KimiK3ViTStage

    calls = []

    def keep_everything(self, features, num_rows, counts):
        # Report engagement. A control that silently never fires reads as "the fix
        # changes nothing", which is the failure this script exists to avoid.
        if not calls:
            rows = (
                features.size(0)
                if hasattr(features, "size")
                else sum(f.size(0) for f in features)
            )
            counts_repr = None if counts is None else counts.tolist()
            print(
                f"CONTROL_ENGAGED rows={rows} num_rows={num_rows} "
                f"counts={counts_repr}",
                flush=True,
            )
        calls.append(1)
        return features

    KimiK3ViTStage._select_cp_shard = keep_everything
    runpy.run_module("torchtitan.train", run_name="__main__")


if __name__ == "__main__":
    main()
