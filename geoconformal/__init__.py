"""
geoconformal: Geographically Weighted Conformal Prediction Methods.

This package exposes two complementary APIs:

Classic spatial-regression API (stateful, GeoDataFrame-friendly)
----------------------------------------------------------------
- GeoConformalSpatialRegression    : geographically weighted conformal prediction (GeoCP)
- GeoSIMConformalSpatialRegression : geographic + feature-similarity conformal prediction (GeoSIMCP)
- GeoConformalResults              : result container with ``.to_gpd()``

Finite-sample-valid weighted / Bayesian framework (``geoconformal.geocp``)
--------------------------------------------------------------------------
A general, model-agnostic framework that restores finite-sample validity by
including the unobserved test point's atom at +infinity with weight w(x)
(Tibshirani et al. 2019), and adds the Bayesian threshold posterior of
Weighted Bayesian CP / GeoBCP (Lou & Luo 2026).

- GeoConformalPrediction : general weighted (Bayesian) CP with the +inf test atom
- GeoConformalRegressor  : high-level spatial wrapper (point estimate or GeoBCP)
- GeoCPResults           : results with finite-sample / Bayesian diagnostics
- weights                : weight-function factories (spatial, adaptive, covariate
                           shift, k-NN, uniform, RBF)

Quick start (corrected GeoCP / GeoBCP)::

    from geoconformal import GeoConformalRegressor

    reg = GeoConformalRegressor(model.predict, X_calib, y_calib, coord_calib,
                                bandwidth=0.15, miscoverage_level=0.1)
    res = reg.geo_conformalize(X_test, y_test, coord_test)              # corrected GeoCP
    bres = reg.geo_conformalize(X_test, y_test, coord_test, bayesian=True, beta=0.9)  # GeoBCP
"""

# --- Classic spatial-regression API (stateful) ---
from .GeoConformalSpatialRegression import GeoConformalSpatialRegression
from .GeoSIMConformalSpatialRegression import GeoSIMConformalSpatialRegression
from .utils import GeoConformalResults

# --- Finite-sample-valid weighted / Bayesian framework ---
from .geocp import (
    GeoConformalPrediction,
    GeoConformalRegressor,
    GeoCPResults,
    WeightFunction,
    abs_nonconformity_score,
    effective_sample_size,
    weighted_quantile,
    bayesian_weighted_quantile,
    weights,
)

__version__ = "0.3.0"

__all__ = [
    # classic spatial-regression API
    "GeoConformalSpatialRegression",
    "GeoSIMConformalSpatialRegression",
    "GeoConformalResults",
    # finite-sample-valid weighted / Bayesian framework
    "GeoConformalPrediction",
    "GeoConformalRegressor",
    "GeoCPResults",
    "WeightFunction",
    "abs_nonconformity_score",
    "effective_sample_size",
    "weighted_quantile",
    "bayesian_weighted_quantile",
    "weights",
    "__version__",
]
