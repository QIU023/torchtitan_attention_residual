# MoonEP dispatcher draft:接口与待验项

从 `kimi_k3/moon_ep_dispatcher.py` 的模块 docstring 搬出。原文 48 行。内容未改。

Our EP is torchtitan's ``AllToAllTokenDispatcher``: reorder, dispatch all-to-all, combine
back. That is correct and it is not what the report describes. Sec 5.2.1 pairs EP at 896
experts with a balanced dispatch, and sec 2.3 says the sparsity K3 runs (896 experts,
top-16) is "beyond the range where the existing auxiliary-loss-free bias update still works
well" -- so the gap is not only throughput, it touches whether the router stays balanced at
all. ``quantile_balance.py`` addresses the router half (it solves for the bias instead of
nudging it); this addresses the transport half.

MoonshotAI released the transport as MoonEP (https://github.com/MoonshotAI/MoonEP).

## Why it is a dispatcher and nothing else

torchtitan already has the seam: ``BaseEPTokenDispatcher`` is an ABC with exactly
``dispatch`` and ``combine``, ``wire_meshes`` installs the EP mesh, and ``init_buffer`` is
the hook a persistent-buffer backend needs. DeepEP is already wired through that seam for
inference (``torchtitan/overrides/moe_token_dispatcher.py``), and ``deep_ep.*`` is already in
``pyproject.toml``'s optional-import list. So MoonEP does not need a new abstraction, a new
dependency mechanism, or any change to the MoE module -- it needs one subclass and one line
in that list.

That is the whole reason this file is small. An earlier instinct was to write an EP path;
the work was to find the seam that already existed.

## Status: DRAFT, never executed

MoonEP needs 8 NVLink-connected GPUs and this box has none of that topology, so nothing
below has run. What IS checked: the interface matches ``BaseEPTokenDispatcher``'s signatures
as of this tree, and the import is optional in the same way fla's and DeepEP's are, so
importing this module on a machine without MoonEP does not break collection.

What a reviewer should NOT read into it: any claim about balance or throughput. The point of
writing it now is that the integration shape is decidable without the hardware, while the
numbers are not.

## The two things to verify first when hardware exists

1. **Token conservation.** ``dispatch`` then ``combine`` must return one row per input row,
   in input order, for every routing pattern including an expert receiving zero tokens.
   ``AllToAllTokenDispatcher`` is the reference: run both on the same inputs and compare.
2. **The backward.** ``combine``'s gradient has to reach ``dispatch``'s input. MoonEP's
   kernels are what own that; if they do not carry a backward, this needs an
   ``autograd.Function`` wrapper and the repo has already paid for getting that wrong once --
   a hand-rolled ``dist.all_gather`` halo in KCP dropped the gradient owed to the left
   neighbour while the forward stayed bit-exact.
