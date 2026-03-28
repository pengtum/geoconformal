from typing import Callable, Union
import numpy as np
import pandas as pd
#from mapclassify.classifiers import quantile
from mapclassify import Quantiles ## revision_0426
from sklearn.linear_model import QuantileRegressor
from quantile_forest import RandomForestQuantileRegressor
from utils import GeoConformalResults
from scipy.optimize import minimize, least_squares, minimize_scalar
from joblib import Parallel, delayed
from tqdm import tqdm


class LSCP:
    def __init__(self, predict_f: Callable, nonconformity_score_f: Callable = None,
                 k: int = 10, miscoverage_level: float = 0.1,coord_calib: Union[np.ndarray, pd.DataFrame] = None, coord_test: Union[np.ndarray, pd.DataFrame] = None,
                 x_calib: Union[np.ndarray, pd.DataFrame] = None, y_calib: Union[np.ndarray, pd.Series] = None,
                 x_test: Union[np.ndarray, pd.DataFrame] = None, y_test: Union[np.ndarray, pd.Series] = None):
        self.predict_f = predict_f
        self.nonconformity_score_f = nonconformity_score_f
        self.k = k
        self.miscoverage_level = miscoverage_level
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

    def _diff_nonconformity_score(self, pred: np.ndarray, gt: np.ndarray) -> np.ndarray:
        return gt - pred

    def _k_neighbors(self, target, others, k, is_calib=True):
        distances = np.sqrt(np.sum(np.square(others - target), axis=1))
        if is_calib:
            sorted_indices = np.argsort(distances)[1:k+1]
        else:
            sorted_indices = np.argsort(distances)[:k]
        return sorted_indices

    def _fit_qrf(self, y, x):
        qrf = RandomForestQuantileRegressor(n_jobs=8)
        qrf.fit(x, y)
        return qrf

    def _generate_diff_dataset(self, nonconformity_scores):
        y_list = []
        x_list = []
        for i, (coord, nonconformity_score) in enumerate(zip(self.coord_calib, nonconformity_scores)):
            y_list.append(nonconformity_score)
            k_neighbor_indices = self._k_neighbors(coord, self.coord_calib, self.k)
            x_list.append(nonconformity_scores[k_neighbor_indices])
        y = np.array(y_list)
        x = np.array(x_list)
        return y, x

    def _interval_length_single_point(self, params, *args):
        beta = params[0]
        coord_new, non_conformity_scores, qrf, miscoverage_level = args
        k_neighbor_indices = self._k_neighbors(coord_new, self.coord_calib, self.k, is_calib=False)
        x_new = non_conformity_scores[k_neighbor_indices].reshape(1, -1)
        lb, ub = qrf.predict(x_new, quantiles=[beta, 1 - miscoverage_level + beta])[0]
        return ub - lb

    def predict_geoconformal_uncertainty_for_single_point(self, beta, coord_new, non_conformity_scores, qrf, miscoverage_level):
        k_neighbor_indices = self._k_neighbors(coord_new, self.coord_calib, self.k, is_calib=False)
        x_new = non_conformity_scores[k_neighbor_indices].reshape(1, -1)
        lb, ub = qrf.predict(x_new, quantiles=[beta, 1 - miscoverage_level + beta])[0]
        return lb, ub

    def optimize_geoconformal_uncertainty(self):
        if self.nonconformity_score_f is None:
            self.nonconformity_score_f = self._diff_nonconformity_score
        y_calib_pred = self.predict_f(self.x_calib)
        nonconformity_scores = np.array(self.nonconformity_score_f(y_calib_pred, self.y_calib))
        y, x = self._generate_diff_dataset(nonconformity_scores)
        qrf = self._fit_qrf(y, x)
        N, _ = self.x_test.shape
        results = Parallel(n_jobs=8)(delayed(self.optimize_interval_for_single_point)(self.coord_test[i, :], nonconformity_scores, qrf) for i in tqdm(range(N)))
        beta_list = np.array([result[0] for result in results])
        return beta_list

    def predict_geoconformal_uncertainty(self):
        beta_list = self.optimize_geoconformal_uncertainty()
        if self.nonconformity_score_f is None:
            self.nonconformity_score_f = self._diff_nonconformity_score
        y_calib_pred = self.predict_f(self.x_calib)
        nonconformity_scores = np.array(self.nonconformity_score_f(y_calib_pred, self.y_calib))
        y, x = self._generate_diff_dataset(nonconformity_scores)
        qrf = self._fit_qrf(y, x)
        N, _ = self.x_test.shape
        results = Parallel(n_jobs=8)(delayed(self.predict_geoconformal_uncertainty_for_single_point)(beta_list[i], self.coord_test[i, :], nonconformity_scores, qrf, self.miscoverage_level) for i in tqdm(range(N)))
        lb = np.array([result[0] for result in results])
        ub = np.array([result[1] for result in results])
        self.geo_uncertainty = ub - lb
        self.lower_bound = lb
        self.upper_bound = ub
        self.uncertainty = np.quantile(nonconformity_scores, 1 - self.miscoverage_level)


    def predict_confidence_interval_improved(self):
        predicted_value = self.predict_f(self.x_test)
        self.predicted_value = predicted_value
        self.lower_bound = predicted_value + self.lower_bound
        self.upper_bound = predicted_value + self.upper_bound

    def optimize_interval_for_single_point(self, coord_new, nonconformity_scores, qrf):
        res = minimize(fun=self._interval_length_single_point, x0=0.01, bounds=[(1e-10, self.miscoverage_level - 1e-10)], method='Powell',
                       args=(coord_new, nonconformity_scores, qrf, self.miscoverage_level))
        return res.x

    def coverage_probability(self) -> float:
        """
        Calculate the coverage probability of confidence interval
        """
        self.coverage_proba = np.mean((self.y_test >= self.lower_bound) & (self.y_test <= self.upper_bound))

    def analyze(self):
        self.predict_geoconformal_uncertainty()
        self.predict_confidence_interval_improved()
        self.coverage_probability()
        return GeoConformalResults(self.geo_uncertainty, self.uncertainty, self.coord_test, self.predicted_value,
                                   self.upper_bound, self.lower_bound, self.coverage_proba)



