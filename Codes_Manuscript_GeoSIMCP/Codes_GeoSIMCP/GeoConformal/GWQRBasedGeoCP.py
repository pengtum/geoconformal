from typing import Callable, Union
import numpy as np
import pandas as pd
from sklearn.linear_model import QuantileRegressor
from quantile_forest import RandomForestQuantileRegressor
from utils import GeoConformalResults
from scipy.optimize import minimize, least_squares
from joblib import Parallel, delayed
from tqdm import tqdm


class GWQRBasedGeoConformalSpatialRegression:
    def __init__(self,
                 predict_f: Callable, nonconformity_score_f: Callable = None,
                 k: int = 10, miscoverage_level: float = 0.1, beta: float = 0.01, alpha: float = 1,
                 coord_calib: Union[np.ndarray, pd.DataFrame] = None, coord_test: Union[np.ndarray, pd.DataFrame] = None,
                 x_calib: Union[np.ndarray, pd.DataFrame] = None, y_calib: Union[np.ndarray, pd.Series] = None,
                 x_test: Union[np.ndarray, pd.DataFrame] = None, y_test: Union[np.ndarray, pd.Series] = None
    ):
        self.predict_f = predict_f
        self.nonconformity_score_f = nonconformity_score_f
        self.k = k
        self.beta = beta
        self.alpha = alpha
        self.miscoverage_level = miscoverage_level
        self.alpha = alpha
        self.coord_calib = coord_calib
        self.coord_test = coord_test
        self.x_calib = x_calib
        self.y_calib = y_calib
        self.x_test = x_test
        self.y_test = y_test
        self.uncertainty = None
        self.geo_uncertainty = None
        self.geo = None
        self.upper_bound = None
        self.lower_bound = None
        self.predicted_value = None
        self.coverage_proba = None
        self.ks = None
        self.betas = None
        self.alphas = None
        if isinstance(self.coord_test, pd.DataFrame):
            self.coord_test = self.coord_test.values
        if isinstance(self.coord_calib, pd.DataFrame):
            self.coord_calib = self.coord_calib.values
        if isinstance(self.x_calib, pd.DataFrame):
            self.x_calib = self.x_calib.values
        if isinstance(self.x_test, pd.DataFrame):
            self.x_test = self.x_test.values
        if isinstance(self.y_calib, pd.Series):
            self.y_calib = self.y_calib.values
        if isinstance(self.y_test, pd.Series):
            self.y_test = self.y_test.values

    def _gaussian_kernel(self, d: np.ndarray) -> np.ndarray:
        """
        Gaussian distance decay function
        :param d: distances from test samples to calibration samples
        :return: list of weights for calibration samples
        """
        return np.exp(-0.5 * d ** 2)

    def _k_neighbors(self, distances, k):
        sorted_indices = np.argsort(distances)[:k]
        return sorted_indices

    def _weights(self, distances, indices):
        bandwith = distances[indices[-1]]
        distances = distances[indices]
        weights = self._gaussian_kernel(distances / bandwith)
        normalized_weights = weights / weights.sum()
        return normalized_weights

    def _fit_gwqr(self, x, y, locations, target_location, k, q) -> QuantileRegressor:

        distances = np.sqrt(np.sum(np.square(locations - target_location), axis=1))
        indices = self._k_neighbors(distances, k)
        weights = self._weights(distances, indices)
        model = QuantileRegressor(quantile=q, alpha=1).fit(x[indices, :], y[indices], sample_weight=weights)
        return model

    def _fit_qrf_gwqr(self, x, y, locations, target_location, k) -> RandomForestQuantileRegressor:
        distances = np.sqrt(np.sum(np.square(locations - target_location), axis=1))
        indices = self._k_neighbors(distances, k)
        weights = self._weights(distances, indices)
        model = RandomForestQuantileRegressor(n_jobs=8).fit(x[indices, :], y[indices], sample_weight=weights)
        return model

    def _abs_nonconformity_score(self, pred: np.ndarray, gt: np.ndarray) -> np.ndarray:
        """
        Default equation for computing nonconformity score
        :param pred: predicted values
        :param gt: ground truth values
        :return: list of nonconformity scores
        """
        return np.abs(pred - gt)

    def _diff_nonconformity_score(self, pred: np.ndarray, gt: np.ndarray) -> np.ndarray:
        return gt - pred

    def predict_geoconformal_uncertainty(self):
        if self.nonconformity_score_f is None:
            self.nonconformity_score_f = self._abs_nonconformity_score
        y_calib_pred = self.predict_f(self.x_calib)
        nonconformity_scores = np.array(self.nonconformity_score_f(y_calib_pred, self.y_calib))
        uncertainty_list = []
        for coord, x_ in zip(self.coord_test, self.x_test):
            coord = coord.reshape(1, -1)
            gwqr = self._fit_gwqr(self.x_calib, nonconformity_scores, self.coord_calib, coord, self.k, 1 - self.miscoverage_level)
            uncertainty = gwqr.predict(x_.reshape(1, -1))
            uncertainty_list.append(uncertainty[0])
        self.geo_uncertainty = np.array(uncertainty_list)
        self.uncertainty = np.quantile(nonconformity_scores, 1 - self.miscoverage_level)

    def predict_geoconformal_uncertainty_improved(self):
        if self.nonconformity_score_f is None:
            self.nonconformity_score_f = self._diff_nonconformity_score
        y_calib_pred = self.predict_f(self.x_calib)
        nonconformity_scores = np.array(self.nonconformity_score_f(y_calib_pred, self.y_calib))
        uncertainty_list = []
        lb_list = []
        ub_list = []
        for coord, x_ in zip(self.coord_test, self.x_test):
            coord = coord.reshape(1, -1)
            gwqr_lb = self._fit_gwqr(self.x_calib, nonconformity_scores, self.coord_calib, coord, self.k, self.beta)
            gwqr_ub = self._fit_gwqr(self.x_calib, nonconformity_scores, self.coord_calib, coord, self.k, 1 - self.miscoverage_level + self.beta)
            lb = gwqr_lb.predict(x_.reshape(1, -1))
            ub = gwqr_ub.predict(x_.reshape(1, -1))
            lb_list.append(lb[0])
            ub_list.append(ub[0])
            uncertainty_list.append(ub[0] - lb[0])
        self.geo_uncertainty = np.array(uncertainty_list)
        self.lower_bound = np.array(lb_list)
        self.upper_bound = np.array(ub_list)
        self.uncertainty = np.quantile(nonconformity_scores, 1 - self.miscoverage_level)

    def geoconformal_uncertainty_for_single_point(self, params, *args) -> float:
        k, beta, alpha = params
        k = int(k)
        x_new, coord, y_new = args
        y_new_pred = self.predict_f(x_new.reshape(1, -1))[0]
        y_calib_pred = self.predict_f(self.x_calib)
        if self.nonconformity_score_f is None:
            self.nonconformity_score_f = self._diff_nonconformity_score
        nonconformity_scores = np.array(self.nonconformity_score_f(y_calib_pred, self.y_calib))

        gwqr_lb = self._fit_gwqr(self.x_calib, nonconformity_scores, self.coord_calib, coord, k, beta)
        gwqr_ub = self._fit_gwqr(self.x_calib, nonconformity_scores, self.coord_calib, coord, k,
                                  1 - self.miscoverage_level + beta)
        lb = gwqr_lb.predict(x_new.reshape(1, -1))[0]
        ub = gwqr_ub.predict(x_new.reshape(1, -1))[0]
        # gwqr = self._fit_qrf_gwqr(self.x_calib, nonconformity_scores, self.coord_calib, coord, k)
        # lb, ub = gwqr.predict(x_new.reshape(1, -1), quantiles=[beta, 1 - self.miscoverage_level + beta])[0]
        geo_uncertainty = ub - lb
        y_pred_lb = y_new_pred + lb
        y_pred_ub = y_new_pred + ub
        residual_lower = y_new - y_pred_lb
        residual_upper = y_pred_ub - y_new
        lower_penalty = np.maximum(-residual_lower, 0)
        upper_penalty = np.maximum(-residual_upper, 0)
        penalty = alpha * (lower_penalty**2 + upper_penalty**2)
        return geo_uncertainty + penalty

    def predict_confidence_interval_improved(self):
        predicted_value = self.predict_f(self.x_test)
        self.predicted_value = predicted_value
        self.lower_bound = predicted_value + self.lower_bound
        self.upper_bound = predicted_value + self.upper_bound


    def predict_confidence_interval(self):
        """
        Calculate the confidence interval based on uncertainty
        :return:
        """
        predicted_value = self.predict_f(self.x_test)
        upper_bound = predicted_value + self.geo_uncertainty
        lower_bound = predicted_value - self.geo_uncertainty
        self.predicted_value = predicted_value
        self.upper_bound = upper_bound
        self.lower_bound = lower_bound

    def coverage_probability(self) -> float:
        """
        Calculate the coverage probability of confidence interval
        """
        self.coverage_proba = np.mean((self.y_test >= self.lower_bound) & (self.y_test <= self.upper_bound))

    def optimize_prediction_interval_for_single_point(self, x_new, coord_new, y_new):
        bounds = [(2, self.x_calib.shape[0]), (1e-10, self.miscoverage_level - 1e-10), (0, 10)]
        res = minimize(fun=self.geoconformal_uncertainty_for_single_point, x0=np.array([10, 0.01, 8]), bounds=bounds, method='Powell', args=(x_new, coord_new, y_new))
        return res.x

    def predict_optimized_geoconformal_uncertainty_for_single_point(self, x_new, coord, y_new, nonconformity_scores):
        k, beta, alpha = self.optimize_prediction_interval_for_single_point(x_new, coord, y_new)
        k = int(k)
        # gwqr_lb = self._fit_gwqr(self.x_calib, nonconformity_scores, self.coord_calib, coord, k, beta)
        # gwqr_ub = self._fit_gwqr(self.x_calib, nonconformity_scores, self.coord_calib, coord, k,
        #                          1 - self.miscoverage_level + beta)
        # lb = gwqr_lb.predict(x_new.reshape(1, -1))[0]
        # ub = gwqr_ub.predict(x_new.reshape(1, -1))[0]
        gwqr = self._fit_qrf_gwqr(self.x_calib, nonconformity_scores, self.coord_calib, coord, k)
        lb, ub = gwqr.predict(x_new.reshape(1, -1), quantiles=[beta, 1 - self.miscoverage_level + beta])[0]
        geo_uncertainty = ub - lb
        return lb, ub, geo_uncertainty, k, beta, alpha

    def predict_geoconformal_uncertainty_optimized(self, n_jobs: int = 8):
        if self.nonconformity_score_f is None:
            self.nonconformity_score_f = self._diff_nonconformity_score
        y_calib_pred = self.predict_f(self.x_calib)
        nonconformity_scores = np.array(self.nonconformity_score_f(y_calib_pred, self.y_calib))
        N, _ = self.x_test.shape
        results = Parallel(n_jobs=n_jobs)(delayed(self.predict_optimized_geoconformal_uncertainty_for_single_point)(self.x_test[i, :], self.coord_test[i, :], self.y_test[i], nonconformity_scores) for i in tqdm(range(N)))
        self.lower_bound = np.array([result[0] for result in results])
        self.upper_bound = np.array([result[1] for result in results])
        # print(self.y_calib.shape)
        # print(upper_bound.shape)
        # residuals = np.maximum(self.y_calib - upper_bound - y_calib_pred, y_calib_pred + lower_bound - self.y_calib)
        # conformal_adjustment = np.quantile(residuals, 1 - self.miscoverage_level)
        # self.upper_bound = upper_bound + conformal_adjustment
        # self.lower_bound = lower_bound - conformal_adjustment
        self.geo_uncertainty = np.array([result[2] for result in results])
        self.ks = np.array([result[3] for result in results])
        self.betas = np.array([result[4] for result in results])
        self.alphas = np.array([result[5] for result in results])
        self.uncertainty = np.quantile(nonconformity_scores, 1 - self.miscoverage_level)

    def analyze(self):
        self.predict_geoconformal_uncertainty_optimized()
        self.predict_confidence_interval_improved()
        self.coverage_probability()
        return GeoConformalResults(geo_uncertainty=self.geo_uncertainty, uncertainty=self.uncertainty,
                                   coords=self.coord_test, pred=self.predicted_value,
                                   upper_bound=self.upper_bound, lower_bound=self.lower_bound,
                                   coverage_probability=self.coverage_proba,
                                   ks=self.ks, betas=self.betas, alpha=self.alphas)