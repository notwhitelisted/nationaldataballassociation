"""Model evaluation and Kelly Criterion bet sizing.

Brings together accuracy metrics, calibration metrics, and profitability
analysis via the Kelly Criterion.
"""

import numpy as np

from app.utils import logger


class KellyCriterion:
    """Kelly Criterion bet sizing from calibrated probabilities.

    The Kelly Criterion determines the optimal fraction of bankroll to wager
    based on the edge (difference between true probability and implied
    probability from odds).

    Kelly fraction = (bp - q) / b
    where:
        b = decimal odds - 1 (net profit per $1 wagered)
        p = estimated probability of winning
        q = 1 - p
    """

    @staticmethod
    def fraction(
        estimated_prob: float,
        decimal_odds: float,
    ) -> float:
        """Compute full Kelly fraction.

        Args:
            estimated_prob: Calibrated probability of the outcome.
            decimal_odds: Decimal odds (e.g. 2.0 for even money).

        Returns:
            Optimal fraction of bankroll to wager. Negative means don't bet.
        """
        if decimal_odds <= 1.0:
            return 0.0
        b = decimal_odds - 1.0
        q = 1.0 - estimated_prob
        kelly = (b * estimated_prob - q) / b
        return kelly

    @staticmethod
    def fractional_kelly(
        estimated_prob: float,
        decimal_odds: float,
        fraction: float = 0.25,
    ) -> float:
        """Compute fractional Kelly (more conservative).

        Most practitioners use 25% or 50% Kelly to reduce variance.

        Args:
            estimated_prob: Calibrated probability.
            decimal_odds: Decimal odds.
            fraction: Fraction of full Kelly (0.25 = quarter Kelly).

        Returns:
            Conservative bet fraction.
        """
        full = KellyCriterion.fraction(estimated_prob, decimal_odds)
        return max(0.0, full * fraction)

    @staticmethod
    def expected_value(estimated_prob: float, decimal_odds: float) -> float:
        """Compute expected value per $1 wagered.

        EV = (probability × payout) - (1 - probability)
        Positive EV = profitable bet in the long run.
        """
        return estimated_prob * decimal_odds - 1.0

    @staticmethod
    def american_to_decimal(american_odds: int) -> float:
        """Convert American odds to decimal odds.

        +150 means $150 profit on $100 bet -> decimal 2.50
        -200 means bet $200 to win $100 -> decimal 1.50
        """
        if american_odds > 0:
            return american_odds / 100.0 + 1.0
        else:
            return 100.0 / abs(american_odds) + 1.0

    @staticmethod
    def implied_probability(decimal_odds: float) -> float:
        """Convert decimal odds to implied probability (no-vig)."""
        if decimal_odds <= 0:
            return 0.0
        return 1.0 / decimal_odds

    @staticmethod
    def bet_recommendation(
        estimated_prob: float,
        decimal_odds: float,
        bankroll: float = 1000.0,
        min_edge: float = 0.02,
        max_bet_pct: float = 0.05,
    ) -> dict:
        """Generate a complete bet recommendation.

        Args:
            estimated_prob: Calibrated probability.
            decimal_odds: Decimal odds offered.
            bankroll: Current bankroll.
            min_edge: Minimum edge required to recommend a bet.
            max_bet_pct: Maximum bet as fraction of bankroll (safety cap).

        Returns:
            Dictionary with recommendation details.
        """
        implied = KellyCriterion.implied_probability(decimal_odds)
        edge = estimated_prob - implied
        ev = KellyCriterion.expected_value(estimated_prob, decimal_odds)
        full_kelly = KellyCriterion.fraction(estimated_prob, decimal_odds)

        # Fractional Kelly amounts
        fractions = {
            "quarter_kelly": KellyCriterion.fractional_kelly(estimated_prob, decimal_odds, 0.25),
            "half_kelly": KellyCriterion.fractional_kelly(estimated_prob, decimal_odds, 0.50),
            "full_kelly": max(0.0, full_kelly),
        }

        # Apply safety cap
        for key in fractions:
            fractions[key] = min(fractions[key], max_bet_pct)

        should_bet = edge >= min_edge and ev > 0

        return {
            "should_bet": should_bet,
            "edge": round(edge, 4),
            "expected_value": round(ev, 4),
            "implied_probability": round(implied, 4),
            "estimated_probability": round(estimated_prob, 4),
            "decimal_odds": decimal_odds,
            "kelly_fractions": {k: round(v, 4) for k, v in fractions.items()},
            "bet_amounts": {
                k: round(v * bankroll, 2) for k, v in fractions.items()
            },
        }


def simulate_betting_strategy(
    y_true: np.ndarray,
    y_prob: np.ndarray,
    odds: np.ndarray,
    bankroll: float = 1000.0,
    kelly_fraction: float = 0.25,
    min_edge: float = 0.02,
) -> dict:
    """Simulate a betting strategy over historical data.

    This is how you backtest whether calibration-optimized predictions
    actually generate better returns than accuracy-optimized ones.

    Args:
        y_true: Actual outcomes (0 or 1).
        y_prob: Predicted probabilities.
        odds: Decimal odds for each game.
        bankroll: Starting bankroll.
        kelly_fraction: Fraction of Kelly to use (0.25 = conservative).
        min_edge: Minimum edge to place a bet.

    Returns:
        Dictionary with simulation results.
    """
    current_bankroll = bankroll
    bets_placed = 0
    bets_won = 0
    bankroll_history = [bankroll]

    for i in range(len(y_true)):
        prob = y_prob[i]
        true_outcome = y_true[i]
        decimal_odds = odds[i]

        implied = KellyCriterion.implied_probability(decimal_odds)
        edge = prob - implied

        if edge < min_edge:
            bankroll_history.append(current_bankroll)
            continue

        bet_frac = KellyCriterion.fractional_kelly(prob, decimal_odds, kelly_fraction)
        bet_frac = min(bet_frac, 0.05)  # safety cap at 5%
        bet_amount = current_bankroll * bet_frac

        if bet_amount < 1.0:  # minimum bet
            bankroll_history.append(current_bankroll)
            continue

        bets_placed += 1

        if true_outcome == 1:
            current_bankroll += bet_amount * (decimal_odds - 1)
            bets_won += 1
        else:
            current_bankroll -= bet_amount

        bankroll_history.append(current_bankroll)

    roi = (current_bankroll - bankroll) / bankroll * 100

    return {
        "starting_bankroll": bankroll,
        "ending_bankroll": round(current_bankroll, 2),
        "roi_pct": round(roi, 2),
        "total_games": len(y_true),
        "bets_placed": bets_placed,
        "bets_won": bets_won,
        "win_rate": round(bets_won / bets_placed * 100, 1) if bets_placed > 0 else 0,
        "bankroll_history": bankroll_history,
    }
