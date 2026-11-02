
import pytest

from series_probability import (
    series_score_probabilities,
    series_win_probability,
)


@pytest.mark.parametrize("best_of", [3, 5])
@pytest.mark.parametrize("p", [0, 0.2, 0.5, 0.6, 0.8, 1])
def test_score_probabilities_sum_to_one(p, best_of):
    scores = series_score_probabilities(p, best_of)
    assert sum(scores.values()) == pytest.approx(1.0)


@pytest.mark.parametrize("best_of", [3, 5])
def test_equal_teams_have_50_percent_probability(best_of):
    assert series_win_probability(0.5, best_of) == pytest.approx(0.5)


def test_bo3_win_probability():
    assert series_win_probability(0.6, 3) == pytest.approx(0.648)


def test_bo5_win_probability():
    assert series_win_probability(0.6, 5) == pytest.approx(0.68256)


@pytest.mark.parametrize("best_of", [3, 5])
def test_invalid_probability(best_of):
    with pytest.raises(ValueError):
        series_win_probability(1.2, best_of)
