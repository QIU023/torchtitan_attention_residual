"""Dump inductor code of the attention residual region: list version (this tree) vs main's cat version (copied body)."""
import sys
import torch
from torch._inductor.utils import run_fw_bw_and_get_code
from torchtitan.models.common import Linear
from torchtitan.models.common.nn_modules import RMSNorm

def main_version(partial, block_residual, projection, norm):
    values_TND = block_residual if partial is None else torch.cat((block_residual, partial.unsqueeze(1)), dim=1)
    values_float = values_TND.float()
    variance = values_float.pow(2).mean(dim=-1, keepdim=True)
    keys_TND = values_float * torch.rsqrt(variance + norm.eps)
    score_weight_D = norm.weight.float() * projection.weight.squeeze(0).float()
    scores_TN = (keys_TND * score_weight_D).sum(dim=-1)
    probs_T1N = torch.softmax(scores_TN, dim=-1).unsqueeze(1)
    output_TD = torch.matmul(probs_T1N, values_float).squeeze(1)
    return output_TD.to(values_TND.dtype)

def list_version(sources, projection, norm):
    values_TND = torch.stack(sources, dim=1)
    values_float = values_TND.float()
    variance = values_float.pow(2).mean(dim=-1, keepdim=True)
    keys_TND = values_float * torch.rsqrt(variance + norm.eps)
    score_weight_D = norm.weight.float() * projection.weight.squeeze(0).float()
    scores_TN = (keys_TND * score_weight_D).sum(dim=-1)
    probs_T1N = torch.softmax(scores_TN, dim=-1).unsqueeze(1)
    output_TD = torch.matmul(probs_T1N, values_float).squeeze(1)
    return output_TD.to(values_TND.dtype)

