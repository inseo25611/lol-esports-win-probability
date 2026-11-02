
import pytest
from elo_model import elo_probability


def test_equal_ratings():
    assert elo_probability(1500, 1500) == pytest.approx(0.5)


def test_probabilities_sum_to_one():
    p_blue = elo_probability(1600, 1500)
    p_red = elo_probability(1500, 1600)
    assert p_blue + p_red == pytest.approx(1.0)


def test_blue_bonus_increases_probability():
    without_bonus = elo_probability(1500, 1500)
    with_bonus = elo_probability(1500, 1500, blue_bonus=20)
    assert with_bonus > without_bonus


def test_blue_bonus_reduces_red_probability():
    without_bonus = 1 - elo_probability(1500, 1500)
    with_bonus = 1 - elo_probability(1500, 1500, blue_bonus=20)
    assert with_bonus < without_bonus


def test_equal_rating_shift():
    original = elo_probability(1600, 1500)
    shifted = elo_probability(1800, 1700)
    assert original == pytest.approx(shifted)


def test_higher_elo_increases_win_probability():
    lower = elo_probability(1500, 1500)
    higher = elo_probability(1600, 1500)
    assert higher > lower
