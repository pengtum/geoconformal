"""
Result containers for GeoConformal Prediction (geocp).
"""

from dataclasses import dataclass, field
from typing import Optional
import numpy as np
from numpy.typing import NDArray


@dataclass
class GeoCPResults:
    """
    Results from GeoCP conformalization.

    Contains prediction intervals, coverage metrics, and (optionally)
    Bayesian posterior diagnostics.

    Attributes
    ----------
    uncertainty : NDArray
        Per-point interval half-width (threshold), shape (n_test,).
    upper_bound : NDArray
        Upper bound of prediction interval, shape (n_test,).
    lower_bound : NDArray
        Lower bound of prediction interval, shape (n_test,).
    pred_value : NDArray
        Predicted values at test points, shape (n_test,).
    true_value : NDArray
        True values at test points, shape (n_test,).
    coverage : float
        Empirical coverage (fraction of test points covered).
    global_uncertainty : float
        Global (unweighted) conformal threshold for reference.
    posterior_mean : NDArray or None
        Posterior mean of threshold at each test point (Bayesian only).
    posterior_std : NDArray or None
        Posterior standard deviation (meta-uncertainty) at each test point.
    n_eff : NDArray or None
        Effective sample size at each test point.
    posterior_samples : NDArray or None
        Full MC posterior samples, shape (n_test, num_mc).
    prob_infinite : NDArray or None
        Posterior probability that the threshold is +infinity at each test
        point (Bayesian only). Large values flag test points the calibration
        set covers too poorly to certify at the requested level.
    beta : float or None
        Bayesian confidence level used.
    """
    uncertainty: NDArray
    upper_bound: NDArray
    lower_bound: NDArray
    pred_value: NDArray
    true_value: NDArray
    coverage: float
    global_uncertainty: float = 0.0
    # Bayesian fields (None for non-Bayesian conformalize)
    posterior_mean: Optional[NDArray] = None
    posterior_std: Optional[NDArray] = None
    n_eff: Optional[NDArray] = None
    posterior_samples: Optional[NDArray] = None
    prob_infinite: Optional[NDArray] = None
    beta: Optional[float] = None

    @property
    def is_bayesian(self) -> bool:
        """Whether this result includes Bayesian posterior information."""
        return self.posterior_samples is not None

    @property
    def frac_infinite(self) -> float:
        """Fraction of test points with an infinite (uncertifiable) interval."""
        return float(np.mean(~np.isfinite(self.uncertainty)))

    @property
    def mean_width(self) -> float:
        """
        Mean prediction interval width (2 * mean half-width).

        Returns inf if any interval is infinite. Use :meth:`mean_width_finite`
        to average only over the certifiable (finite) test points.
        """
        return float(2 * np.mean(self.uncertainty))

    @property
    def mean_width_finite(self) -> float:
        """Mean interval width over test points with finite intervals only."""
        finite = self.uncertainty[np.isfinite(self.uncertainty)]
        if finite.size == 0:
            return float('inf')
        return float(2 * np.mean(finite))

    @property
    def mean_n_eff(self) -> float:
        """Mean effective sample size (Bayesian only)."""
        if self.n_eff is None:
            return float('nan')
        return float(np.mean(self.n_eff))

    @property
    def mean_sigma_post(self) -> float:
        """Mean posterior standard deviation (Bayesian only)."""
        if self.posterior_std is None:
            return float('nan')
        return float(np.mean(self.posterior_std))

    def summary(self) -> dict:
        """Return a summary dictionary of key metrics."""
        d = {
            'coverage': self.coverage,
            'mean_width': self.mean_width,
            'mean_width_finite': self.mean_width_finite,
            'frac_infinite': self.frac_infinite,
            'global_uncertainty': self.global_uncertainty,
        }
        if self.is_bayesian:
            d.update({
                'beta': self.beta,
                'mean_n_eff': self.mean_n_eff,
                'mean_sigma_post': self.mean_sigma_post,
            })
        return d

    def __repr__(self) -> str:
        s = f"GeoCPResults(coverage={self.coverage:.4f}, mean_width={self.mean_width:.4f}"
        if self.is_bayesian:
            s += f", beta={self.beta}, mean_n_eff={self.mean_n_eff:.1f}, mean_sigma_post={self.mean_sigma_post:.4f}"
        s += ")"
        return s


#: Deprecated alias kept for backward compatibility with the LBCP package.
LBCPResults = GeoCPResults
