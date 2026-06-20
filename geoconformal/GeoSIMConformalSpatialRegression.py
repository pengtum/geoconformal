from typing import Callable
import numpy as np
from .utils import GeoConformalResults
from .geocp.weights import joint_geo_feature_weights
from .geocp.utils import weighted_quantile


class GeoSIMConformalSpatialRegression:
    """
    Geospatial Similarity Conformal Prediction (GeoSIMCP).

    An extension of GeoCP that jointly considers geographic distance and
    feature-space similarity when estimating local uncertainty, for spatially
    nonstationary processes. A thin, stateful wrapper over the unified
    ``geoconformal.geocp`` engine (the joint kernel lives in
    ``geoconformal.geocp.weights.joint_geo_feature_weights``); the classic
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
        Bandwidth parameter for the Gaussian kernel.
    coord_calib, coord_test : array (n, 2) / (m, 2)
        Calibration / test coordinates.
    X_calib, y_calib, X_test, y_test : arrays
        Calibration / test features and targets (used for prediction).
    lambda_weight : float
        Trade-off between geographic distance (lambda=1, reduces to GeoCP) and
        feature distance (lambda=0). Default 1.0.
    X_calib_weight, X_test_weight : array-like, optional
        Features for computing feature distance. Default: X_calib / X_test.
    distance_metric : str
        'euclidean' or 'mnd' (Minimum Normalized Difference). Default 'euclidean'.
    standardize_weights : bool
        z-score normalize feature inputs for distance (euclidean only). Default True.
    include_test_atom : bool, default False
        If True, add the unobserved test point's atom at +infinity with weight
        w(x) (Tibshirani et al. 2019) for finite-sample validity; where local
        support is insufficient the interval becomes +infinity. The default
        ``False`` reproduces the original GeoSIMCP behaviour.
    """

    def __init__(self, predict_f: Callable, nonconformity_score_f: Callable = None,
                 miscoverage_level: float = 0.1, bandwidth: float = None,
                 coord_calib=None, coord_test=None,
                 X_calib=None, y_calib=None,
                 X_test=None, y_test=None,
                 lambda_weight: float = 1.0,
                 X_calib_weight=None, X_test_weight=None,
                 distance_metric: str = 'euclidean',
                 standardize_weights: bool = True,
                 include_test_atom: bool = False):
        self.predict_f = predict_f
        self.nonconformity_score_f = nonconformity_score_f
        self.miscoverage_level = miscoverage_level
        self.bandwidth = bandwidth
        self.lambda_weight = lambda_weight
        self.distance_metric = distance_metric.lower()
        self.standardize_weights = standardize_weights
        self.include_test_atom = include_test_atom

        self.coord_calib = np.asarray(coord_calib, dtype=float)
        self.coord_test = np.asarray(coord_test, dtype=float)
        self.X_calib = self._to_numpy(X_calib)
        self.X_test = self._to_numpy(X_test)
        self.y_calib = np.asarray(y_calib, dtype=float)
        self.y_test = np.asarray(y_test, dtype=float)

        # Features used for the feature-space distance (default to prediction features)
        self.X_calib_weight = (self._to_numpy(X_calib_weight)
                               if X_calib_weight is not None else self.X_calib.copy())
        self.X_test_weight = (self._to_numpy(X_test_weight)
                              if X_test_weight is not None else self.X_test.copy())

        self.uncertainty = None
        self.geo_uncertainty = None
        self.upper_bound = None
        self.lower_bound = None
        self.predicted_value = None
        self.coverage_proba = None

    @staticmethod
    def _to_numpy(x):
        if x is None:
            return None
        if hasattr(x, 'values'):
            return x.values.astype(float)
        return np.asarray(x, dtype=float)

    def predict_geoconformal_uncertainty(self):
        """Compute geographically and feature-weighted uncertainty per test point."""
        if self.nonconformity_score_f is None:
            self.nonconformity_score_f = self._default_nonconformity_score
        y_calib_pred = self.predict_f(self.X_calib)
        nonconformity_scores = np.asarray(
            self.nonconformity_score_f(y_calib_pred, self.y_calib), dtype=float)

        q = 1 - self.miscoverage_level
        weight_fn = joint_geo_feature_weights(
            self.coord_calib, self.X_calib_weight, bandwidth=self.bandwidth,
            lambda_weight=self.lambda_weight, distance_metric=self.distance_metric,
            standardize=self.standardize_weights)
        # Joint kernel weights on the stacked (coordinate | feature) space.
        test_input = np.hstack([self.coord_test, self.X_test_weight])
        weights = weight_fn(test_input, None)
        self_weights = weight_fn.self_weight(test_input) if self.include_test_atom else None
        self.geo_uncertainty = weighted_quantile(
            nonconformity_scores, weights, q, self_weights=self_weights)
        self.uncertainty = float(np.nanmean(self.geo_uncertainty))

    def predict_confidence_interval(self):
        """Compute prediction intervals based on geo-uncertainty."""
        predicted_value = self.predict_f(self.X_test)
        self.predicted_value = predicted_value
        self.upper_bound = predicted_value + self.geo_uncertainty
        self.lower_bound = predicted_value - self.geo_uncertainty

    def coverage_probability(self):
        """Compute empirical coverage probability."""
        self.coverage_proba = np.mean(
            (self.y_test >= self.lower_bound) & (self.y_test <= self.upper_bound))

    def analyze(self):
        """Run full pipeline: uncertainty -> intervals -> coverage."""
        self.predict_geoconformal_uncertainty()
        self.predict_confidence_interval()
        self.coverage_probability()
        return GeoConformalResults(
            self.geo_uncertainty, self.uncertainty, self.coord_test,
            self.predicted_value, self.upper_bound, self.lower_bound,
            self.coverage_proba)

    @staticmethod
    def _default_nonconformity_score(pred, gt):
        return np.abs(pred - gt)
