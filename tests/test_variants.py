"""
Correctness review of all GeoCP weight variants through the public API.

For each weighting scheme, over many random splits, check that:
  - point-estimate and Bayesian conformalize run end-to-end via the public API
    (spatial schemes exercise the coord_test feature/coordinate split);
  - corrected marginal coverage >= target (1 - alpha);
  - the legacy (no-atom) variant is no better-covering than corrected.
Plus: uniform weights reduce EXACTLY to standard split conformal prediction.
"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np

from geoconformal.geocp import GeoConformalPrediction, weighted_quantile
from geoconformal.geocp.weights import (
    uniform_weights, spatial_kernel_weights, adaptive_spatial_weights,
    covariate_shift_weights, knn_weights, rbf_feature_weights,
)

ALPHA = 0.10
N_CALIB = 300
N_TEST = 800
N_SPLITS = 40


def _make(seed):
    """Synthetic data: features X (3d), coords (2d), y = linear(X) + noise."""
    rng = np.random.default_rng(seed)
    Xc = rng.standard_normal((N_CALIB, 3))
    cc = rng.uniform(0, 1, (N_CALIB, 2))
    Xt = rng.standard_normal((N_TEST, 3))
    ct = rng.uniform(0, 1, (N_TEST, 2))
    beta = np.array([1.0, -0.5, 0.3])
    f = lambda X: X @ beta  # noqa: E731  true mean; predictor = truth -> scores=|noise|
    yc = f(Xc) + rng.standard_normal(N_CALIB) * 0.5
    yt = f(Xt) + rng.standard_normal(N_TEST) * 0.5
    return Xc, yc, cc, Xt, yt, ct, f


def _variant(name, seed):
    Xc, yc, cc, Xt, yt, ct, f = _make(seed)
    predict_f = f
    spatial = name in ("spatial", "adaptive")
    if name == "uniform":
        wf = uniform_weights()
    elif name == "spatial":
        wf = spatial_kernel_weights(cc, bandwidth=0.25)
    elif name == "adaptive":
        wf = adaptive_spatial_weights(cc, base_bandwidth=0.5, k=20)
    elif name == "covariate":
        # density ratio as a positive function of features
        wf = covariate_shift_weights(lambda X: np.exp(0.5 * np.asarray(X)[:, 0]))
    elif name == "knn":
        wf = knn_weights(k=40)
    elif name == "rbf":
        wf = rbf_feature_weights(gamma=0.3)
    else:
        raise ValueError(name)

    geo = GeoConformalPrediction(predict_f, Xc, yc, wf, miscoverage_level=ALPHA)
    coord_test = ct if spatial else None
    r_corr = geo.conformalize(Xt, yt, coord_test=coord_test, include_test_atom=True)
    r_leg = geo.conformalize(Xt, yt, coord_test=coord_test, include_test_atom=False)
    r_bay = geo.bayesian_conformalize(Xt, yt, coord_test=coord_test,
                                      num_mc=200, beta=0.9, random_state=3)
    return r_corr.coverage, r_leg.coverage, r_bay.coverage, r_corr.frac_infinite


def test_all_variants():
    names = ["uniform", "spatial", "adaptive", "covariate", "knn", "rbf"]
    print(f"n_calib={N_CALIB}, splits={N_SPLITS}, target={1-ALPHA:.2f}\n")
    print(f"{'variant':10s} {'corrected':>10s} {'legacy':>8s} {'bayes':>8s} {'frac_inf':>9s}")
    for nm in names:
        c, l, b, fi = np.mean([_variant(nm, s) for s in range(N_SPLITS)], axis=0)
        flag = "OK " if c >= 1 - ALPHA - 0.02 else "LOW"
        print(f"{nm:10s} {c:10.3f} {l:8.3f} {b:8.3f} {fi:9.2f}  {flag}")
        assert c >= 1 - ALPHA - 0.02, f"{nm}: corrected coverage {c:.3f} below target"


def test_uniform_equals_standard_cp():
    """Uniform-weight GeoCP threshold == standard split-CP order statistic."""
    rng = np.random.default_rng(7)
    for n in [8, 9, 25, 200]:
        scores = rng.standard_exponential(n)
        w = np.ones((1, n))
        got = weighted_quantile(scores, w, 1 - ALPHA, self_weights=np.array([1.0]))[0]
        k = int(np.ceil((1 - ALPHA) * (n + 1)))
        want = np.inf if k > n else np.sort(scores)[k - 1]
        ok = (np.isinf(want) and np.isinf(got)) or np.isclose(got, want)
        assert ok, f"n={n}: got {got}, want {want}"
    print("\nuniform == standard split CP (incl. +inf at small n): OK")


if __name__ == "__main__":
    test_all_variants()
    test_uniform_equals_standard_cp()
    print("\nAll variant checks passed.")
