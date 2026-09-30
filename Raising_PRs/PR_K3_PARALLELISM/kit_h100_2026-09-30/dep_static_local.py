"""Local probe recipe (never committed): the DEP ratio configs with dynamo's automatic dynamic shapes off, so the
vision tower and the text attention each get their own static flex_attention compile on every rank."""

import torch._dynamo

import dep_ratio_local as base

torch._dynamo.config.automatic_dynamic_shapes = False
w_dep_off = base.w_dep_off
w_dep_k25 = base.w_dep_k25
w_dep_bubble = base.w_dep_bubble
