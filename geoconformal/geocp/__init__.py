"""
GeoConformal Prediction (geocp).

A general framework for weighted (Bayesian) conformal prediction that unifies
and extends:
- Standard Conformal Prediction (CP)
- Bayesian Quadrature CP (BQ-CP, Snell & Griffiths 2025)
- Weighted CP under covariate shift (Tibshirani et al. 2019)
- GeoConformal Prediction (GeoCP, Lou et al. 2025)
- Geographical Bayesian CP (GeoBCP)
- Localized CP (Guan et al. 2023)

Following Tibshirani et al. (2019), the unobserved test point contributes its
own atom at +infinity with weight w(x), which restores finite-sample validity in
the small-sample regime. The Bayesian variant replaces BQ-CP's Dir(1,...,1) with
the weighted Dirichlet Dir(n_eff * w_1, ..., n_eff * w_n, n_eff * w(x)).

Quick start::

    from geocp import GeoConformalPrediction
    from geocp.weights import spatial_kernel_weights

    weight_fn = spatial_kernel_weights(coord_calib, bandwidth=0.15)
    geocp = GeoConformalPrediction(model.predict, x_calib, y_calib, weight_fn)
    results = geocp.bayesian_conformalize(coord_test, y_test, beta=0.9)
    print(results)
"""

from .core import (
    GeoConformalPrediction,
    LocalizedBayesianCP,        # deprecated alias
    abs_nonconformity_score,
)
from .estimators import GeoConformalRegressor
from .results import GeoCPResults, LBCPResults  # LBCPResults: deprecated alias
from .utils import (
    effective_sample_size,
    weighted_quantile,
    bayesian_weighted_quantile,
)
from .weights import WeightFunction
from . import weights

__version__ = "0.1.0"

__all__ = [
    "GeoConformalPrediction",
    "GeoConformalRegressor",
    "GeoCPResults",
    "WeightFunction",
    "abs_nonconformity_score",
    "effective_sample_size",
    "weighted_quantile",
    "bayesian_weighted_quantile",
    "weights",
    # deprecated aliases
    "LocalizedBayesianCP",
    "LBCPResults",
    "__version__",
]
