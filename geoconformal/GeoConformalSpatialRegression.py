from typing import Callable
import numpy as np
from .utils import GeoConformalResults
from .geocp.weights import spatial_kernel_weights
from .geocp.utils import weighted_quantile


class GeoConformalSpatialRegression:
    """
    Geographically weighted conformal prediction (GeoCP).

    A thin, stateful wrapper over the unified ``geoconformal.geocp`` engine: it
    weights calibration nonconformity scores by a spatial Gaussian kernel on the
    coordinates and takes a per-test-point weighted quantile. The classic
    ``.analyze()`` / ``GeoConformalResults.to_gpd()`` workflow is preserved.

    Parameters
    ----------
    predict_f : Callable
        Prediction function of the spatial model.
    nonconformity_score_f : Callable, optional
        Custom score(pred, gt) -> scores. Default: absolute residual |pred - gt|.
    miscoverage_level : float
        Miscoverage level alpha (0.1 -> 90% prediction intervals).
    bandwidth : float
        Gaussian kernel bandwidth.
    coord_calib, coord_test : array (n, 2) / (m, 2)
        Calibration / test coordinates.
    X_calib, y_calib, X_test, y_test : arrays
        Calibration / test features and targets.
    include_test_atom : bool, default False
        If True, add the unobserved test point's atom at +infinity with weight
        w(x) (Tibshirani et al. 2019). This restores finite-sample coverage on
        small / sparse calibration sets; where local support is insufficient the
        interval becomes +infinity (an honest abstention rather than silent
        under-coverage). The default ``False`` reproduces the original GeoCP
        behaviour exactly.
    """

    def __init__(self, predict_f: Callable, nonconformity_score_f: Callable = None,
                 miscoverage_level: float = 0.1, bandwidth: float = None,
                 coord_calib: np.ndarray = None, coord_test: np.ndarray = None,
                 X_calib: np.ndarray = None, y_calib: np.ndarray = None,
                 X_test: np.ndarray = None, y_test: np.ndarray = None,
                 include_test_atom: bool = False):
        self.predict_f = predict_f
        self.nonconformity_score_f = nonconformity_score_f
        self.miscoverage_level = miscoverage_level
        self.bandwidth = bandwidth
        self.coord_calib = coord_calib
        self.coord_test = coord_test
        self.X_calib = X_calib
        self.y_calib = y_calib
        self.X_test = X_test
        self.y_test = y_test
        self.include_test_atom = include_test_atom
        self.uncertainty = None
        self.geo_uncertainty = None
        self.geo = None
        self.upper_bound = None
        self.lower_bound = None
        self.predicted_value = None
        self.coverage_proba = None

    def predict_geoconformal_uncertainty(self):
        """Calculate the geographically weighted uncertainty for each sample."""
        if self.nonconformity_score_f is None:
            self.nonconformity_score_f = self._default_nonconformity_score
        y_calib_pred = self.predict_f(self.X_calib)
        nonconformity_scores = np.asarray(
            self.nonconformity_score_f(y_calib_pred, self.y_calib), dtype=float)

        q = 1 - self.miscoverage_level
        coord_calib = np.asarray(self.coord_calib)
        coord_test = np.asarray(self.coord_test)
        weight_fn = spatial_kernel_weights(coord_calib, self.bandwidth)
        weights = weight_fn(coord_test, coord_calib)
        self_weights = weight_fn.self_weight(coord_test) if self.include_test_atom else None
        self.geo_uncertainty = weighted_quantile(
            nonconformity_scores, weights, q, self_weights=self_weights)
        self.uncertainty = float(np.quantile(nonconformity_scores, q))

    def predict_confidence_interval(self):
        """Calculate the confidence interval based on uncertainty."""
        predicted_value = self.predict_f(self.X_test)
        self.predicted_value = predicted_value
        self.upper_bound = predicted_value + self.geo_uncertainty
        self.lower_bound = predicted_value - self.geo_uncertainty

    def coverage_probability(self) -> float:
        """Calculate the coverage probability of the confidence interval."""
        self.coverage_proba = np.mean(
            (self.y_test >= self.lower_bound) & (self.y_test <= self.upper_bound))

    def analyze(self):
        self.predict_geoconformal_uncertainty()
        self.predict_confidence_interval()
        self.coverage_probability()
        return GeoConformalResults(self.geo_uncertainty, self.uncertainty, self.coord_test,
                                   self.predicted_value, self.upper_bound, self.lower_bound,
                                   self.coverage_proba)

    @staticmethod
    def _default_nonconformity_score(pred: np.ndarray, gt: np.ndarray) -> np.ndarray:
        """Default nonconformity score: |predicted - ground_truth|."""
        return np.abs(pred - gt)
