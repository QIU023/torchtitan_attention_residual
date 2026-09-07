"""A reward with variance for a random-init policy: the fraction of alphabetic characters in the
response. gsm8k's rule-based score is 0 for every sample of an untrained model, so its GRPO cells never
move the actor and say nothing about the weight sync; this one moves it, so rollout_probs_diff can
separate a live sync from a frozen one (KIMI_GRPO_FREEZE_SYNC=1)."""


def compute_score(data_source, solution_str, ground_truth, extra_info=None, **kwargs):
    if not solution_str:
        return 0.0
    return sum(ch.isalpha() for ch in solution_str) / len(solution_str)
