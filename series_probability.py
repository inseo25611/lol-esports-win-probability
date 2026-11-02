
from math import comb


def series_score_probabilities(p: float, best_of: int) -> dict:
    """Calculate exact series score probabilities."""
    if not 0 <= p <= 1:
        raise ValueError("p must be between 0 and 1")

    if best_of not in (3, 5):
        raise ValueError("Only Bo3 and Bo5 are supported")

    wins_needed = best_of // 2 + 1
    scores = {}

    for losses in range(wins_needed):
        # Team A wins the series
        probability = (
            comb(wins_needed - 1 + losses, losses)
            * p**wins_needed
            * (1 - p)**losses
        )
        scores[f"{wins_needed}-{losses}"] = probability

        # Team B wins the series
        probability = (
            comb(wins_needed - 1 + losses, losses)
            * (1 - p)**wins_needed
            * p**losses
        )
        scores[f"{losses}-{wins_needed}"] = probability

    return scores


def series_win_probability(p: float, best_of: int) -> float:
    """Calculate probability that Team A wins the series."""
    scores = series_score_probabilities(p, best_of)
    wins_needed = best_of // 2 + 1

    return sum(
        probability
        for score, probability in scores.items()
        if int(score.split("-")[0]) == wins_needed
    )
