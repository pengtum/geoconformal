"""
Core class for GeoConformal Prediction (geocp).

This module implements the general GeoCP framework that accepts arbitrary
weight functions. Specific instantiations (spatial, covariate shift, k-NN,
etc.) are achieved by providing different weight functions from weights.py.

References:
    - Lou et al. (2025), "GeoConformal Prediction"
    - Snell & Griffiths (2025), "Conformal Prediction as Bayesian Quadrature"
    - Tibshirani et al. (2019), "Conformal Prediction Under Covariate Shift"
"""

from __future__ import annotations
from typing import Callable
import numpy as np
from numpy.typing import NDArray

from .utils import weighted_quantile, bayesian_weighted_quantile
from .results import GeoCPResults


def abs_nonconformity_score(pred: NDArray, gt: NDArray) -> NDArray:
    """Default nonconformity score: |predicted - ground_truth|."""
    return np.abs(pred - gt)


class GeoConformalPrediction:
    """
    GeoConformal Prediction (GeoCP).

    A general framework for weighted (Bayesian) conformal prediction that
    accepts arbitrary weight functions. This unifies and generalizes:
    - Standard CP (uniform weights)
    - BQ-CP (uniform weights + Bayesian posterior)
    - Weighted CP (non-uniform weights, point-estimate threshold)
    - GeoCP (spatial kernel weights)
    - GeoBCP (spatial kernel weights + Bayesian posterior)
    - Localized CP (k-NN weights)

    Validity in the small-sample regime is restored by including the test
    point's own atom at +infinity with weight w(x) (Tibshirani et al. 2019); the
    Bayesian variant replaces BQ-CP's Dir(1,...,1) with the weighted Dirichlet
    Dir(n_eff * w_1, ..., n_eff * w_n, n_eff * w(x)).

    Parameters
    ----------
    predict_f : Callable
        Prediction function that maps features to predictions.
        Signature: (NDArray) -> NDArray.
    x_calib : NDArray
        Calibration features, shape (n_calib, d).
    y_calib : NDArray
        Calibration ground truth, shape (n_calib,).
    weight_fn : Callable[[NDArray, NDArray], NDArray]
        Weight function that maps (x_test, x_calib) -> weights of shape
        (n_test, n_calib). See geocp.weights for factory functions; prefer a
        geocp.weights.WeightFunction so the test self-weight w(x) is exact.
    miscoverage_level : float
        Desired miscoverage level alpha (default 0.1 for 90% coverage).
    score_fn : Callable, optional
        Nonconformity score function. Default: |pred - true|.

    Examples
    --------
    >>> from geocp import GeoConformalPrediction
    >>> from geocp.weights import spatial_kernel_weights
    >>>
    >>> weight_fn = spatial_kernel_weights(coord_calib, bandwidth=0.15)
    >>> geocp = GeoConformalPrediction(model.predict, x_calib, y_calib, weight_fn)
    >>> results = geocp.bayesian_conformalize(coord_test, y_test, beta=0.9)
    >>> print(results.coverage, results.mean_n_eff)
    """

    def __init__(
        self,
        predict_f: Callable,
        x_calib: NDArray,
        y_calib: NDArray,
        weight_fn: Callable[[NDArray, NDArray], NDArray],
        miscoverage_level: float = 0.1,
        score_fn: Callable = None,
    ):
        self.predict_f = predict_f
        self.x_calib = np.asarray(x_calib)
        self.y_calib = np.asarray(y_calib)
        self.weight_fn = weight_fn
        self.miscoverage_level = miscoverage_level
        self.score_fn = score_fn or abs_nonconformity_score

        # Pre-compute calibration scores
        y_calib_pred = self.predict_f(self.x_calib)
        self._scores = self.score_fn(y_calib_pred, self.y_calib)
        self._n_calib = len(self._scores)

    @property
    def q_level(self) -> float:
        """
        Finite-sample-adjusted quantile level for *unweighted* split CP.

        Used only as the legacy fallback when the +infinity test atom is not
        included. When the test atom is used, the finite-sample correction is
        carried by the atom itself, so the quantile level is simply 1 - alpha
        (see :meth:`_q`).
        """
        return np.ceil((1 - self.miscoverage_level) * (self._n_calib + 1)) / self._n_calib

    def _q(self, include_test_atom: bool) -> float:
        """Quantile level to use for the weighted distribution."""
        if include_test_atom:
            # The delta_{+inf} test atom supplies the (n+1) finite-sample
            # correction, so we take the plain (1 - alpha) quantile.
            return 1 - self.miscoverage_level
        return min(self.q_level, 1.0)

    def _self_weights(self, x_test: NDArray, include_test_atom: bool):
        """
        Test self-weights w(x), shape (n_test,), or None if not applicable.

        Pulls w(x) from the weight function when available; falls back to the
        per-row maximum weight (correct for kernels peaked at zero distance)
        with a warning if the weight function predates the self-weight contract.
        """
        if not include_test_atom:
            return None
        self_weight = getattr(self.weight_fn, 'self_weight', None)
        if callable(self_weight):
            return self_weight(x_test)
        import warnings
        warnings.warn(
            "weight_fn does not expose a self_weight(); falling back to the "
            "per-test-point maximum calibration weight as w(x). Provide a "
            "geocp.weights.WeightFunction for exact finite-sample validity.",
            RuntimeWarning,
        )
        weights = self.weight_fn(x_test, self.x_calib)
        return np.max(weights, axis=1)

    def conformalize(
        self,
        x_test: NDArray,
        y_test: NDArray,
        coord_test: NDArray = None,
        include_test_atom: bool = True,
    ) -> GeoCPResults:
        """
        Weighted conformal prediction (point-estimate threshold).

        Computes per-test-point weighted quantiles of calibration scores
        using the weight function. Produces a single threshold per test point.

        Parameters
        ----------
        x_test : NDArray
            Test features, shape (n_test, d). Passed to ``predict_f``.
        y_test : NDArray
            Test ground truth, shape (n_test,).
        coord_test : NDArray, optional
            Test inputs for the *weight function*, shape (n_test, d_w), when the
            weighting space differs from the prediction features -- e.g. spatial
            GeoCP predicts from features but weights by coordinates. If None
            (default), ``x_test`` is used for weighting too. The matching
            calibration coordinates are supplied to the weight-function factory
            (e.g. ``spatial_kernel_weights(coord_calib, ...)``) and must align
            row-for-row with ``x_calib``.
        include_test_atom : bool
            If True (default), include the test point's atom at +infinity with
            weight w(x), giving finite-sample-valid weighted CP (Tibshirani et
            al. 2019). If False, reproduce the legacy behavior that normalizes
            over calibration weights only (kept for ablation; undercovers at
            small n).

        Returns
        -------
        GeoCPResults
            Results with point-estimate thresholds. Bayesian fields are None.
        """
        x_test = np.asarray(x_test)
        y_test = np.asarray(y_test)
        w_input = x_test if coord_test is None else np.asarray(coord_test)

        # Compute weights and (optionally) the test point's self-weight w(x)
        weights = self.weight_fn(w_input, self.x_calib)
        self_weights = self._self_weights(w_input, include_test_atom)

        # Weighted quantile (with the +infinity test atom when requested)
        q = self._q(include_test_atom)
        uncertainty = weighted_quantile(self._scores, weights, q, self_weights=self_weights)

        # Global (unweighted) quantile for reference
        global_uncertainty = float(np.quantile(self._scores, min(self.q_level, 1.0)))

        # Prediction intervals
        y_pred = self.predict_f(x_test)
        upper_bound = y_pred + uncertainty
        lower_bound = y_pred - uncertainty
        coverage = float(np.mean((y_test >= lower_bound) & (y_test <= upper_bound)))

        return GeoCPResults(
            uncertainty=uncertainty,
            upper_bound=upper_bound,
            lower_bound=lower_bound,
            pred_value=y_pred,
            true_value=y_test,
            coverage=coverage,
            global_uncertainty=global_uncertainty,
        )

    def bayesian_conformalize(
        self,
        x_test: NDArray,
        y_test: NDArray,
        coord_test: NDArray = None,
        num_mc: int = 1000,
        beta: float = 0.9,
        concentration_scale: str = 'neff',
        random_state: int = 42,
        include_test_atom: bool = True,
    ) -> GeoCPResults:
        """
        Bayesian weighted conformal prediction (posterior over threshold).

        For each test point, samples from a weighted Dirichlet posterior to
        produce a full distribution of thresholds. The HPD threshold at
        confidence level beta provides data-conditional coverage guarantees.

        This is the core GeoCP method. It provides:
        - HPD threshold (data-conditional guarantee at confidence beta)
        - Posterior standard deviation (meta-uncertainty about interval width)
        - Effective sample size (diagnostic for weight quality)
        - Full posterior samples (for multi-resolution confidence layers)

        Parameters
        ----------
        x_test : NDArray
            Test features, shape (n_test, d). Passed to ``predict_f``.
        y_test : NDArray
            Test ground truth, shape (n_test,).
        coord_test : NDArray, optional
            Test inputs for the weight function when the weighting space differs
            from the prediction features (e.g. spatial GeoCP weights by
            coordinates). If None, ``x_test`` is used for weighting too.
        num_mc : int
            Number of Monte Carlo Dirichlet samples (default 1000).
        beta : float
            Confidence level for HPD threshold (default 0.9).
        concentration_scale : str
            'neff' (recommended) or 'fixed'.
        random_state : int
            Random seed for reproducibility.
        include_test_atom : bool
            If True (default), include the test point's atom at +infinity with
            weight w(x) in the posterior quantile, giving the finite-sample
            correction. If False, reproduce the legacy behavior (test atom
            dropped before the quantile; undercovers at small n_eff).

        Returns
        -------
        GeoCPResults
            Results with Bayesian posterior fields populated.
        """
        x_test = np.asarray(x_test)
        y_test = np.asarray(y_test)
        w_input = x_test if coord_test is None else np.asarray(coord_test)

        # Compute weights and (optionally) the test point's self-weight w(x)
        weights = self.weight_fn(w_input, self.x_calib)
        self_weights = self._self_weights(w_input, include_test_atom)

        # Bayesian weighted quantile (with the +infinity test atom when requested)
        q = self._q(include_test_atom)
        bq_result = bayesian_weighted_quantile(
            scores=self._scores,
            weights=weights,
            q=q,
            self_weights=self_weights,
            num_mc=num_mc,
            beta=beta,
            concentration_scale=concentration_scale,
            random_state=random_state,
        )

        uncertainty = bq_result['hpd_quantiles']
        global_uncertainty = float(np.quantile(self._scores, min(self.q_level, 1.0)))

        # Prediction intervals
        y_pred = self.predict_f(x_test)
        upper_bound = y_pred + uncertainty
        lower_bound = y_pred - uncertainty
        coverage = float(np.mean((y_test >= lower_bound) & (y_test <= upper_bound)))

        return GeoCPResults(
            uncertainty=uncertainty,
            upper_bound=upper_bound,
            lower_bound=lower_bound,
            pred_value=y_pred,
            true_value=y_test,
            coverage=coverage,
            global_uncertainty=global_uncertainty,
            posterior_mean=bq_result['posterior_mean'],
            posterior_std=bq_result['posterior_std'],
            n_eff=bq_result['n_eff'],
            posterior_samples=bq_result['posterior_samples'],
            prob_infinite=bq_result['prob_infinite'],
            beta=beta,
        )


#: Deprecated alias kept for backward compatibility with the LBCP package.
LocalizedBayesianCP = GeoConformalPrediction
