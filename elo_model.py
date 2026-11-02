
def elo_probability(blue_elo, red_elo, blue_bonus=0):
    """Calculate Blue-side win probability from Elo ratings."""
    return 1 / (1 + 10 ** ((red_elo - blue_elo - blue_bonus) / 400))
