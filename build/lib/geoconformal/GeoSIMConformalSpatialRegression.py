from typing import Callable
import numpy as np
from sklearn.preprocessing import StandardScaler
from .utils import GeoConformalResults


class GeoSIMConformalSpatialRegression:
    """
    Geospatial Similarity Conformal Prediction (GeoSIMCP).

    An extension of GeoCP that jointly considers geographic distance and
    feature-space similarity when estimating local uncertainty, designed to
    handle spatially nonstationary processes.

    Parameters
    ----------
    predict_f : Callable
        Prediction function of the spatial model, accepting feature matrix
        and returning predicted values.
    nonconformity_score_f : Callable, optional
        Custom nonconformity score function(pred, gt) -> scores.
        Defaults to absolute residuals |pred - gt|.
    miscoverage_level : float
        Miscoverage level (alpha). E.g. 0.1 for 90% prediction intervals.
    bandwidth : float
        Bandwidth parameter for the Gaussian kernel.
    coord_calib : array-like, shape (n_calib, 2)
        Geographic coordinates of calibration samples.
    coord_test : array-like, shape (n_test, 2)
        Geographic coordinates of test samples.
    X_calib : array-like, shape (n_calib, p)
        Feature matrix for calibration samples (used for prediction).
    y_calib : array-like, shape (n_calib,)
        True values for calibration samples.
    X_test : array-like, shape (n_test, p)
        Feature matrix for test samples (used for prediction).
    y_test : array-like, shape (n_test,)
        True values for test samples.
    lambda_weight : float
        Trade-off between geographic distance (lambda=1) and feature distance
        (lambda=0). Default 1.0 reduces to GeoCP.
    X_calib_weight : array-like, optional
        Feature matrix for computing feature distance (calibration).
        Defaults to X_calib if not provided.
    X_test_weight : array-like, optional
        Feature matrix for computing feature distance (test).
        Defaults to X_test if not provided.
    distance_metric : str
        Feature distance metric: 'euclidean' or 'mnd' (Minimum Normalized
        Difference). Default 'euclidean'.
    standardize_weights : bool
        Whether to z-score normalize feature inputs for distance computation.
        Default True.
    """

    def __init__(self, predict_f: Callable, nonconformity_score_f: Callable = None,
                 miscoverage_level: float = 0.1, bandwidth: float = None,
                 coord_calib=None, coord_test=None,
                 X_calib=None, y_calib=None,
                 X_test=None, y_test=None,
                 lambda_weight: float = 1.0,
                 X_calib_weight=None, X_test_weight=None,
                 distance_metric: str = 'euclidean',
                 standardize_weights: bool = True):
        self.predict_f = predict_f
        self.nonconformity_score_f = nonconformity_score_f
        self.miscoverage_level = miscoverage_level
        self.bandwidth = bandwidth
        self.lambda_weight = lambda_weight
        self.distance_metric = distance_metric.lower()

        self.coord_calib = np.asarray(coord_calib, dtype=float)
        self.coord_test = np.asarray(coord_test, dtype=float)

        self.X_calib = self._to_numpy(X_calib)
        self.X_test = self._to_numpy(X_test)
        self.y_calib = np.asarray(y_calib, dtype=float)
        self.y_test = np.asarray(y_test, dtype=float)

        # Feature inputs for distance computation
        X_cw = self._to_numpy(X_calib_weight) if X_calib_weight is not None else self.X_calib.copy()
        X_tw = self._to_numpy(X_test_weight) if X_test_weight is not None else self.X_test.copy()

        if self.distance_metric == 'mnd':
            # For MND, compute per-feature ranges on calibration set
            self._feat_ranges = np.ptp(X_cw, axis=0)
            self._feat_ranges[self._feat_ranges == 0] = 1e-8
            self.X_calib_weight = X_cw
            self.X_test_weight = X_tw
        else:
            # For Euclidean, optionally standardize
            if standardize_weights:
                stds = np.std(X_cw, axis=0)
                valid_cols = stds > 1e-8
                X_cw = X_cw[:, valid_cols]
                X_tw = X_tw[:, valid_cols]
                scaler = StandardScaler()
                X_cw = scaler.fit_transform(X_cw)
                X_tw = scaler.transform(X_tw)
            self.X_calib_weight = X_cw
            self.X_test_weight = X_tw

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

    def _feature_distance(self, x_test, x_calib):
        """Compute feature-space distance between one test point and all calibration points."""
        if self.distance_metric == 'mnd':
            diffs = np.abs(x_calib - x_test)
            scaled = 1.0 - (diffs / self._feat_ranges)
            similarity = np.min(scaled, axis=1)
            return 1.0 - similarity
        else:
            return np.linalg.norm(x_calib - x_test, axis=1)

    def _kernel_smoothing_joint(self, z_test, z_calib, x_test, x_calib, bandwidth, lam):
        """Compute joint geographic-feature kernel weights."""
        d_geo = np.linalg.norm(z_calib - z_test, axis=1)
        d_feat = self._feature_distance(x_test, x_calib)
        d_feat = d_feat + 1e-8  # avoid exact zeros
        d_joint = np.sqrt(lam * d_geo ** 2 + (1 - lam) * d_feat ** 2)
        weights = np.exp(-0.5 * (d_joint / (bandwidth + 1e-8)) ** 2)
        total = np.sum(weights)
        if total > 0:
            weights = weights / total
        return weights

    def predict_geoconformal_uncertainty(self):
        """Compute geographically and feature-weighted uncertainty for each test point."""
        if self.nonconformity_score_f is None:
            self.nonconformity_score_f = self._default_nonconformity_score

        y_calib_pred = self.predict_f(self.X_calib)
        nonconformity_scores = np.abs(y_calib_pred - self.y_calib)

        uncertainty_list = []
        for i in range(len(self.coord_test)):
            weights = self._kernel_smoothing_joint(
                z_test=self.coord_test[i],
                z_calib=self.coord_calib,
                x_test=self.X_test_weight[i],
                x_calib=self.X_calib_weight,
                bandwidth=self.bandwidth,
                lam=self.lambda_weight
            )
            quantile = self._weighted_quantile(nonconformity_scores, self.miscoverage_level, weights)
            uncertainty_list.append(quantile)

        self.geo_uncertainty = np.array(uncertainty_list)
        self.uncertainty = np.nanmean(self.geo_uncertainty)

    def predict_confidence_interval(self):
        """Compute prediction intervals based on geo-uncertainty."""
        predicted_value = self.predict_f(self.X_test)
        self.predicted_value = predicted_value
        self.upper_bound = predicted_value + self.geo_uncertainty
        self.lower_bound = predicted_value - self.geo_uncertainty

    def coverage_probability(self):
        """Compute empirical coverage probability."""
        self.coverage_proba = np.mean(
            (self.y_test >= self.lower_bound) & (self.y_test <= self.upper_bound)
        )

    def analyze(self):
        """Run full pipeline: uncertainty -> intervals -> coverage."""
        self.predict_geoconformal_uncertainty()
        self.predict_confidence_interval()
        self.coverage_probability()
        return GeoConformalResults(
            self.geo_uncertainty, self.uncertainty, self.coord_test,
            self.predicted_value, self.upper_bound, self.lower_bound,
            self.coverage_proba
        )

    @staticmethod
    def _default_nonconformity_score(pred, gt):
        return np.abs(pred - gt)

    @staticmethod
    def _weighted_quantile(scores, alpha=0.1, weights=None):
        """Compute weighted (1-alpha) quantile of nonconformity scores."""
        if weights is None:
            weights = np.ones(len(scores))

        sorted_indices = np.argsort(scores)
        scores_sorted = scores[sorted_indices]
        weights_sorted = weights[sorted_indices]

        cumsum_weights = np.cumsum(weights_sorted)
        if cumsum_weights[-1] == 0 or np.any(np.isnan(cumsum_weights)):
            return np.nan

        normalized_cumsum = cumsum_weights / (cumsum_weights[-1] + 1e-8)
        idx = np.searchsorted(normalized_cumsum, 1 - alpha)
        return scores_sorted[min(idx, len(scores_sorted) - 1)]
