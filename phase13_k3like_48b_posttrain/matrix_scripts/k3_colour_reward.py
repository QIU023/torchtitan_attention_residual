"""Reward for the colour set: 1 when the response names the colour, plus a tenth of the alphabetic
fraction so a random-init policy still has variance inside a group (the text cells' reward)."""


def compute_score(data_source, solution_str, ground_truth, extra_info=None, **kwargs):
    if not solution_str:
        return 0.0
    hit = 1.0 if str(ground_truth).lower() in solution_str.lower() else 0.0
    return hit + 0.1 * sum(ch.isalpha() for ch in solution_str) / len(solution_str)
