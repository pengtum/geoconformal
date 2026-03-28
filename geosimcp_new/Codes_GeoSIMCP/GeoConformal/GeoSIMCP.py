from typing import Callable
import geopandas as gpd
import numpy as np
import pandas as pd
from utils import GeoConformalResults
from typing import Callable
from sklearn.preprocessing import StandardScaler





class GeoSIMConformalSpatialRegression:
    def __init__(self, predict_f: Callable, nonconformity_score_f: Callable = None,
                 miscoverage_level: float = 0.1, bandwidth: float = None,
                 coord_calib=None, coord_test=None,
                 X_calib=None, y_calib=None,
                 X_test=None, y_test=None,
                 lambda_weight: float = 1.0,
                 X_calib_weight=None, X_test_weight=None,
                 standardize_weights: bool = True):
        """
        lambda_weight: trade-off between geographic distance (1.0) and feature distance (0.0)
        X_*_weight: used only for distance weighting (optional); defaults to X_* if not given
        standardize_weights: whether to z-score normalize the feature space inputs
        """
        self.predict_f = predict_f
        self.nonconformity_score_f = nonconformity_score_f
        self.miscoverage_level = miscoverage_level
        self.bandwidth = bandwidth
        self.lambda_weight = lambda_weight

        self.coord_calib = np.array(coord_calib)
        self.coord_test = np.array(coord_test)

       # self.X_calib = self._to_numpy(X_calib)
        #self.X_test = self._to_numpy(X_test)

        self.X_calib = X_calib   # 不做 _to_numpy
        self.X_test = X_test     # 不做 _to_numpy

        self.y_calib = np.array(y_calib)
        self.y_test = np.array(y_test)

 


        self.X_calib_weight = self._to_numpy(X_calib_weight) if X_calib_weight is not None else self.X_calib
        self.X_test_weight = self._to_numpy(X_test_weight) if X_test_weight is not None else self.X_test

        if standardize_weights:
            scaler = StandardScaler()
            self.X_calib_weight = scaler.fit_transform(self.X_calib_weight)
            self.X_test_weight = scaler.transform(self.X_test_weight)

        self.uncertainty = None
        self.geo_uncertainty = None
        self.upper_bound = None
        self.lower_bound = None
        self.predicted_value = None
        self.coverage_proba = None

    def _to_numpy(self, x):
        if x is None:
            return None
        if hasattr(x, "values"):
            return x.values.astype(float)
        return np.array(x).astype(float)

    def predict_geoconformal_uncertainty(self):
        if self.nonconformity_score_f is None:
            self.nonconformity_score_f = self._default_nonconformity_score

        y_calib_pred = self.predict_f(self.X_calib)
        nonconformity_scores = np.abs(y_calib_pred - self.y_calib)

        uncertainty_list = []
        for i in range(len(self.coord_test)):
            p = self.coord_test[i]
            x_test_i = self.X_test_weight[i]
            weights = self._kernel_smoothing_joint(
                z_test=p,
                z_calib=self.coord_calib,
                x_test=x_test_i,
                x_calib=self.X_calib_weight,
                bandwidth=self.bandwidth,
                lam=self.lambda_weight
            )
            quantile = self._weighted_quantile(nonconformity_scores, self.miscoverage_level, weights)
            uncertainty_list.append(quantile)

        self.geo_uncertainty = np.array(uncertainty_list)
        self.uncertainty = np.mean(self.geo_uncertainty)  # Updated here to reflect lambda effect

    def _kernel_smoothing_joint(self, z_test, z_calib, x_test, x_calib, bandwidth, lam=1.0):
        d_geo = np.linalg.norm(z_calib - z_test, axis=1)
        d_feat = np.linalg.norm(x_calib - x_test, axis=1)
        d_feat += 1e-8  # Avoid exact zeros
        d_joint = np.sqrt(lam * d_geo**2 + (1 - lam) * d_feat**2)
        return np.exp(-0.5 * (d_joint / bandwidth) ** 2)

    def predict_confidence_interval(self):
        predicted_value = self.predict_f(self.X_test)
        upper_bound = predicted_value + self.geo_uncertainty
        lower_bound = predicted_value - self.geo_uncertainty
        self.predicted_value = predicted_value
        self.upper_bound = upper_bound
        self.lower_bound = lower_bound

    def coverage_probability(self) -> float:
        self.coverage_proba = np.mean((self.y_test >= self.lower_bound) & (self.y_test <= self.upper_bound))

    def analyze(self):
        self.predict_geoconformal_uncertainty()
        self.predict_confidence_interval()
        self.coverage_probability()
        return GeoConformalResults(
            self.geo_uncertainty, self.uncertainty, self.coord_test,
            self.predicted_value, self.upper_bound, self.lower_bound, self.coverage_proba
        )

    def _default_nonconformity_score(self, pred: np.ndarray, gt: np.ndarray) -> np.ndarray:
        return np.abs(pred - gt)

    def _weighted_quantile(self, scores: np.ndarray, alpha: float = 0.1, weights: np.ndarray = None):
        if weights is None:
            weights = np.ones(len(scores))

        sorted_indices = np.argsort(scores)
        scores_sorted = scores[sorted_indices]
        weights_sorted = weights[sorted_indices]

        cumsum_weights = np.cumsum(weights_sorted)
        if cumsum_weights[-1] == 0:
            return np.nan

        normalized_cumsum_weights = cumsum_weights / cumsum_weights[-1]
        idx = np.searchsorted(normalized_cumsum_weights, 1 - alpha)
        return scores_sorted[min(idx, len(scores_sorted) - 1)]

