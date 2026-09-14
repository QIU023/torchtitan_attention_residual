# LOCAL PROBE (not committed): identity taps that record gradients at runtime.
import torch

STORE: dict[str, list[torch.Tensor]] = {}


@torch.library.custom_op("probe::tap_bwd", mutates_args=())
def tap_bwd(g: torch.Tensor, name: str) -> torch.Tensor:
    lst = STORE.setdefault(name, [])
    if len(lst) < 2:
        lst.append(g.detach().float().cpu())
    return g.clone()


@tap_bwd.register_fake
def _(g, name):
    return torch.empty_like(g)


@torch.library.custom_op("probe::tap", mutates_args=())
def tap(x: torch.Tensor, name: str) -> torch.Tensor:
    return x.clone()


@tap.register_fake
def _(x, name):
    return torch.empty_like(x)


def _setup(ctx, inputs, output):
    ctx.name = inputs[1]


def _bwd(ctx, g):
    return torch.ops.probe.tap_bwd(g, ctx.name), None


tap.register_autograd(_bwd, setup_context=_setup)

torch._dynamo.config.recompile_limit = 64  # LOCAL PROBE: the extra tap guard variant
