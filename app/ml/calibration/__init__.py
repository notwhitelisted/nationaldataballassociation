"""Calibration techniques for sports betting models.

This is the core differentiator of the project. Walsh & Joshi (2024) showed
that calibration-optimized models generated +34.69% ROI while accuracy-optimized
models produced -35.17% ROI — a 69.86% profitability difference.

Implements:
    - Platt scaling (logistic calibration)
    - Isotonic regression
    - Temperature scaling
    - Expected Calibration Error (ECE)
    - Brier Score decomposition
    - Reliability diagram generation

Usage:
    from app.ml.calibration import ModelCalibrator, CalibrationMetrics

    calibrator = ModelCalibrator(method="isotonic")
    calibrator.fit(y_true, y_prob_uncalibrated)
    y_calibrated = calibrator.transform(y_prob_new)

    metrics = CalibrationMetrics.compute(y_true, y_calibrated)
    print(f"ECE: {metrics.ece:.4f}, Brier: {metrics.brier:.4f}")
"""

from dataclasses import dataclass

import numpy as np
from sklearn.calibration import CalibratedClassifierCV, calibration_curve
from sklearn.isotonic import IsotonicRegression
from sklearn.linear_model import LogisticRegression

from app.utils import logger


@dataclass
class CalibrationResult:
    """Results from calibration evaluation."""

    ece: float  # Expected Calibration Error
    brier: float  # Brier Score
    mce: float  # Maximum Calibration Error
    bin_accuracies: np.ndarray  # observed frequency per bin
    bin_confidences: np.ndarray  # mean predicted probability per bin
    bin_counts: np.ndarray  # number of samples per bin

    def summary(self) -> str:
        return (
            f"ECE: {self.ece:.4f} | Brier: {self.brier:.4f} | MCE: {self.mce:.4f}"
        )


class CalibrationMetrics:
    """Compute calibration quality metrics."""

    @staticmethod
    def expected_calibration_error(
        y_true: np.ndarray,
        y_prob: np.ndarray,
        n_bins: int = 10,
    ) -> CalibrationResult:
        """Compute Expected Calibration Error and related metrics.

        ECE measures the average difference between predicted probabilities and
        observed frequencies. Lower is better. A perfectly calibrated model has
        ECE = 0.

        Args:
            y_true: Binary outcomes (0 or 1).
            y_prob: Predicted probabilities for the positive class.
            n_bins: Number of bins for grouping predictions.

        Returns:
            CalibrationResult with all metrics.
        """
        y_true = np.asarray(y_true, dtype=float)
        y_prob = np.asarray(y_prob, dtype=float)

        # Bin predictions
        bin_edges = np.linspace(0.0, 1.0, n_bins + 1)
        bin_accuracies = np.zeros(n_bins)
        bin_confidences = np.zeros(n_bins)
        bin_counts = np.zeros(n_bins, dtype=int)

        for i in range(n_bins):
            low, high = bin_edges[i], bin_edges[i + 1]
            if i == n_bins - 1:
                mask = (y_prob >= low) & (y_prob <= high)
            else:
                mask = (y_prob >= low) & (y_prob < high)

            count = mask.sum()
            bin_counts[i] = count

            if count > 0:
                bin_accuracies[i] = y_true[mask].mean()
                bin_confidences[i] = y_prob[mask].mean()

        # ECE: weighted average of |accuracy - confidence| per bin
        weights = bin_counts / bin_counts.sum()
        gaps = np.abs(bin_accuracies - bin_confidences)
        ece = float(np.sum(weights * gaps))

        # MCE: maximum calibration error across bins
        nonempty = bin_counts > 0
        mce = float(gaps[nonempty].max()) if nonempty.any() else 0.0

        # Brier score
        brier = float(np.mean((y_prob - y_true) ** 2))

        return CalibrationResult(
            ece=ece,
            brier=brier,
            mce=mce,
            bin_accuracies=bin_accuracies,
            bin_confidences=bin_confidences,
            bin_counts=bin_counts,
        )

    @staticmethod
    def brier_score(y_true: np.ndarray, y_prob: np.ndarray) -> float:
        """Compute Brier score (mean squared error of probabilities)."""
        return float(np.mean((np.asarray(y_prob) - np.asarray(y_true)) ** 2))

    @staticmethod
    def brier_decomposition(
        y_true: np.ndarray, y_prob: np.ndarray, n_bins: int = 10
    ) -> dict[str, float]:
        """Decompose Brier score into reliability, resolution, and uncertainty.

        Brier = Reliability - Resolution + Uncertainty

        - Reliability (lower is better): measures calibration quality
        - Resolution (higher is better): measures how much predictions vary
        - Uncertainty: base rate entropy, independent of the model
        """
        y_true = np.asarray(y_true, dtype=float)
        y_prob = np.asarray(y_prob, dtype=float)
        n = len(y_true)
        base_rate = y_true.mean()

        bin_edges = np.linspace(0.0, 1.0, n_bins + 1)
        reliability = 0.0
        resolution = 0.0

        for i in range(n_bins):
            low, high = bin_edges[i], bin_edges[i + 1]
            if i == n_bins - 1:
                mask = (y_prob >= low) & (y_prob <= high)
            else:
                mask = (y_prob >= low) & (y_prob < high)

            count = mask.sum()
            if count == 0:
                continue

            bin_accuracy = y_true[mask].mean()
            bin_confidence = y_prob[mask].mean()

            reliability += count * (bin_confidence - bin_accuracy) ** 2
            resolution += count * (bin_accuracy - base_rate) ** 2

        reliability /= n
        resolution /= n
        uncertainty = base_rate * (1 - base_rate)

        return {
            "reliability": reliability,
            "resolution": resolution,
            "uncertainty": uncertainty,
            "brier": reliability - resolution + uncertainty,
        }


class ModelCalibrator:
    """Calibrates model probabilities using various methods.

    Supports:
        - "platt": Platt scaling (logistic regression on probabilities)
        - "isotonic": Isotonic regression (non-parametric)
        - "temperature": Temperature scaling (single parameter)
    """

    METHODS = ("platt", "isotonic", "temperature")

    def __init__(self, method: str = "isotonic"):
        if method not in self.METHODS:
            raise ValueError(f"Unknown calibration method: {method}. Use one of {self.METHODS}")
        self.method = method
        self._calibrator = None
        self._temperature: float = 1.0
        self._is_fitted = False

    def fit(self, y_true: np.ndarray, y_prob: np.ndarray) -> "ModelCalibrator":
        """Fit the calibrator on validation data.

        Args:
            y_true: True binary labels.
            y_prob: Uncalibrated predicted probabilities.
        """
        y_true = np.asarray(y_true, dtype=float)
        y_prob = np.asarray(y_prob, dtype=float).reshape(-1, 1)

        if self.method == "platt":
            self._calibrator = LogisticRegression(C=1e10, solver="lbfgs")
            self._calibrator.fit(y_prob, y_true)

        elif self.method == "isotonic":
            self._calibrator = IsotonicRegression(
                y_min=0.0, y_max=1.0, out_of_bounds="clip"
            )
            self._calibrator.fit(y_prob.ravel(), y_true)

        elif self.method == "temperature":
            self._temperature = self._fit_temperature(y_true, y_prob.ravel())

        self._is_fitted = True
        logger.info("Calibrator fitted using '{}' method", self.method)
        return self

    def transform(self, y_prob: np.ndarray) -> np.ndarray:
        """Apply calibration to new predictions.

        Args:
            y_prob: Uncalibrated predicted probabilities.

        Returns:
            Calibrated probabilities.
        """
        if not self._is_fitted:
            raise RuntimeError("Calibrator not fitted. Call fit() first.")

        y_prob = np.asarray(y_prob, dtype=float)

        if self.method == "platt":
            return self._calibrator.predict_proba(y_prob.reshape(-1, 1))[:, 1]

        elif self.method == "isotonic":
            return self._calibrator.transform(y_prob)

        elif self.method == "temperature":
            # Apply temperature scaling to logits
            logits = np.log(np.clip(y_prob, 1e-10, 1 - 1e-10) / (1 - np.clip(y_prob, 1e-10, 1 - 1e-10)))
            scaled_logits = logits / self._temperature
            return 1.0 / (1.0 + np.exp(-scaled_logits))

    def fit_transform(self, y_true: np.ndarray, y_prob: np.ndarray) -> np.ndarray:
        """Fit and transform in one step (for convenience, but beware overfitting)."""
        self.fit(y_true, y_prob)
        return self.transform(y_prob)

    @staticmethod
    def _fit_temperature(y_true: np.ndarray, y_prob: np.ndarray) -> float:
        """Find optimal temperature parameter via grid search on log-loss.

        Temperature > 1 softens predictions (moves toward 0.5).
        Temperature < 1 sharpens predictions (moves toward 0 or 1).
        """
        best_temp = 1.0
        best_loss = float("inf")

        for temp in np.linspace(0.1, 5.0, 100):
            logits = np.log(np.clip(y_prob, 1e-10, 1 - 1e-10) / (1 - np.clip(y_prob, 1e-10, 1 - 1e-10)))
            scaled = 1.0 / (1.0 + np.exp(-logits / temp))
            # Negative log-likelihood
            loss = -np.mean(
                y_true * np.log(np.clip(scaled, 1e-10, 1))
                + (1 - y_true) * np.log(np.clip(1 - scaled, 1e-10, 1))
            )
            if loss < best_loss:
                best_loss = loss
                best_temp = temp

        logger.debug("Optimal temperature: {:.3f} (NLL: {:.4f})", best_temp, best_loss)
        return best_temp


def compare_calibration_methods(
    y_true: np.ndarray,
    y_prob: np.ndarray,
) -> dict[str, CalibrationResult]:
    """Compare all calibration methods on the same data.

    Useful for deciding which calibration approach works best for your model.
    Uses a simple 50/50 split to avoid overfitting the calibrator.

    Args:
        y_true: True binary labels.
        y_prob: Uncalibrated predicted probabilities.

    Returns:
        Dict mapping method name to CalibrationResult.
    """
    n = len(y_true)
    split = n // 2

    results = {"uncalibrated": CalibrationMetrics.expected_calibration_error(y_true, y_prob)}

    for method in ModelCalibrator.METHODS:
        cal = ModelCalibrator(method=method)
        cal.fit(y_true[:split], y_prob[:split])
        calibrated = cal.transform(y_prob[split:])
        result = CalibrationMetrics.expected_calibration_error(y_true[split:], calibrated)
        results[method] = result
        logger.info("  {}: {}", method, result.summary())

    return results
