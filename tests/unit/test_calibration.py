"""Tests for calibration metrics, model calibrator, and Kelly Criterion."""

import numpy as np
import pytest

from app.ml.calibration import (
    CalibrationMetrics,
    ModelCalibrator,
)
from app.ml.evaluation import KellyCriterion


class TestCalibrationMetrics:
    """Test ECE, Brier score, and decomposition."""

    def test_perfect_calibration(self):
        """Perfectly calibrated predictions should have ECE ≈ 0."""
        np.random.seed(42)
        n = 1000
        y_prob = np.random.uniform(0, 1, n)
        y_true = (np.random.uniform(0, 1, n) < y_prob).astype(float)

        result = CalibrationMetrics.expected_calibration_error(y_true, y_prob)
        assert result.ece < 0.1

    def test_worst_calibration(self):
        """Completely wrong probabilities should have high ECE."""
        y_true = np.array([1, 1, 1, 1, 1, 0, 0, 0, 0, 0], dtype=float)
        y_prob = np.array([0.1, 0.1, 0.1, 0.1, 0.1, 0.9, 0.9, 0.9, 0.9, 0.9])

        result = CalibrationMetrics.expected_calibration_error(y_true, y_prob)
        assert result.ece > 0.5

    def test_brier_score_perfect(self):
        y_true = np.array([1, 0, 1, 0])
        y_prob = np.array([1.0, 0.0, 1.0, 0.0])
        assert CalibrationMetrics.brier_score(y_true, y_prob) == 0.0

    def test_brier_score_worst(self):
        y_true = np.array([1, 0, 1, 0])
        y_prob = np.array([0.0, 1.0, 0.0, 1.0])
        assert CalibrationMetrics.brier_score(y_true, y_prob) == 1.0

    def test_brier_decomposition_sums(self):
        """Reliability - Resolution + Uncertainty ≈ Brier score."""
        np.random.seed(42)
        y_true = np.random.randint(0, 2, 200).astype(float)
        y_prob = np.clip(y_true + np.random.normal(0, 0.3, 200), 0.01, 0.99)

        decomp = CalibrationMetrics.brier_decomposition(y_true, y_prob)
        brier_direct = CalibrationMetrics.brier_score(y_true, y_prob)
        assert abs(decomp["brier"] - brier_direct) < 0.05


class TestModelCalibrator:
    """Test calibration methods."""

    @pytest.fixture
    def sample_data(self):
        np.random.seed(42)
        n = 500
        true_prob = np.random.uniform(0.3, 0.7, n)
        y_true = (np.random.uniform(0, 1, n) < true_prob).astype(float)
        y_prob = np.clip(true_prob + (true_prob - 0.5) * 0.8, 0.01, 0.99)
        return y_true, y_prob

    @pytest.mark.parametrize("method", ["platt", "isotonic", "temperature"])
    def test_calibrator_fits_and_transforms(self, method, sample_data):
        y_true, y_prob = sample_data
        cal = ModelCalibrator(method=method)
        cal.fit(y_true, y_prob)
        calibrated = cal.transform(y_prob)
        assert len(calibrated) == len(y_prob)
        assert np.all(calibrated >= 0) and np.all(calibrated <= 1)

    def test_invalid_method_raises(self):
        with pytest.raises(ValueError):
            ModelCalibrator(method="invalid")

    def test_transform_before_fit_raises(self):
        cal = ModelCalibrator(method="platt")
        with pytest.raises(RuntimeError):
            cal.transform(np.array([0.5]))


class TestKellyCriterion:
    """Test Kelly Criterion bet sizing."""

    def test_positive_edge(self):
        # 60% prob at even money (2.0) => kelly = (1*0.6 - 0.4)/1 = 0.20
        frac = KellyCriterion.fraction(0.60, 2.0)
        assert frac == pytest.approx(0.20, abs=0.01)

    def test_no_edge(self):
        # 50% at even money = no edge
        frac = KellyCriterion.fraction(0.50, 2.0)
        assert frac == pytest.approx(0.0, abs=0.01)

    def test_negative_edge_no_bet(self):
        # 40% at even money = negative edge
        frac = KellyCriterion.fraction(0.40, 2.0)
        assert frac < 0  # negative = don't bet

    def test_fractional_kelly(self):
        full = KellyCriterion.fraction(0.60, 2.0)
        quarter = KellyCriterion.fractional_kelly(0.60, 2.0, 0.25)
        assert quarter == pytest.approx(full * 0.25, abs=0.01)

    def test_american_to_decimal(self):
        assert KellyCriterion.american_to_decimal(+150) == pytest.approx(2.5)
        assert KellyCriterion.american_to_decimal(-200) == pytest.approx(1.5)
        assert KellyCriterion.american_to_decimal(+100) == pytest.approx(2.0)
        assert KellyCriterion.american_to_decimal(-100) == pytest.approx(2.0)

    def test_implied_probability(self):
        assert KellyCriterion.implied_probability(2.0) == pytest.approx(0.50)
        assert KellyCriterion.implied_probability(1.5) == pytest.approx(0.667, abs=0.01)

    def test_bet_recommendation_structure(self):
        rec = KellyCriterion.bet_recommendation(0.60, 2.0, bankroll=1000.0)
        assert "should_bet" in rec
        assert "edge" in rec
        assert "kelly_fractions" in rec
        assert "bet_amounts" in rec
        assert rec["should_bet"] is True
        assert rec["edge"] > 0

    def test_expected_value(self):
        # 60% at 2.0 odds: EV = 0.6*2.0 - 1 = 0.20
        ev = KellyCriterion.expected_value(0.60, 2.0)
        assert ev == pytest.approx(0.20, abs=0.01)
