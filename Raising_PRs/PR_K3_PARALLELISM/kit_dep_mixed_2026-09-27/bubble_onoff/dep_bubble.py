"""Local recipes (never committed): the vit_dep cell with the bubble placement on and off."""

from torchtitan.trainer import Trainer

from torchtitan_recipes.tests.b200 import kimi_k3_debugmodel_pp4_vp2_vit_dep


def dep_bubble_on() -> Trainer.Config:
    config = kimi_k3_debugmodel_pp4_vp2_vit_dep()
    assert config.model.vision_dep.bubble
    return config


def dep_bubble_off() -> Trainer.Config:
    config = kimi_k3_debugmodel_pp4_vp2_vit_dep()
    config.model.vision_dep.bubble = False
    return config
