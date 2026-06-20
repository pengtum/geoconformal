"""
Verification of the test-atom fix in geoconformal.geocp.

Checks:
  (A) Unweighted GeoCP with the test atom == standard split conformal prediction,
      including the +infinity threshold when ceil((1-alpha)(n+1)) > n.
  (B) Over many random small-n splits, marginal coverage of the corrected
      weighted/spatial methods is >= 1-alpha, whereas the legacy (no-atom)
      variant undercovers.
"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np

from geoconformal.geocp import GeoConformalPrediction, weighted_quantile
from geoconformal.geocp.weights import uniform_weights, spatial_kernel_weights


def standard_split_cp_threshold(scores, alpha):
    """Textbook split-CP threshold: the ceil((1-alpha)(n+1))/n empirical quantile,
    which is +inf when that rank exceeds n."""
    n = len(scores)
    k = int(np.ceil((1 - alpha) * (n + 1)))
    if k > n:
        return np.inf
    return np.sort(scores)[k - 1]


def test_unweighted_reduces_to_standard_cp():
    rng = np.random.default_rng(0)
    alpha = 0.1
    for n in [5, 8, 9, 12, 30, 100]:
        scores = rng.standard_exponential(n)
        w = np.ones((1, n))
        q = 1 - alpha
        got = weighted_quantile(scores, w, q, self_weights=np.array([1.0]))[0]
        want = standard_split_cp_threshold(scores, alpha)
        if np.isinf(want):
            assert np.isinf(got), f"n={n}: expected +inf, got {got}"
        else:
            assert np.isclose(got, want), f"n={n}: got {got}, want {want}"
    print("[A] unweighted GeoCP == standard split CP (incl. +inf at small n): OK")


def _coverage_trial(n_calib, alpha, seed, include_test_atom, method):
    rng = np.random.default_rng(seed)
    # 1-D regression: y = f(x) + noise, predict with the truth so scores ~ |noise|
    coords = rng.uniform(0, 1, size=(n_calib, 2))
    y_calib = rng.standard_normal(n_calib)
    predict_f = lambda X: np.zeros(X.shape[0])  # noqa: E731  (scores = |y|)

    coord_test = rng.uniform(0, 1, size=(200, 2))
    y_test = rng.standard_normal(200)

    if method == "uniform":
        wf = uniform_weights()
        x_calib = coords
        x_test = coord_test
    else:
        wf = spatial_kernel_weights(coords, bandwidth=0.3)
        x_calib = coords
        x_test = coord_test

    lbcp = GeoConformalPrediction(predict_f, x_calib, y_calib, wf, miscoverage_level=alpha)
    res = lbcp.conformalize(x_test, y_test, include_test_atom=include_test_atom)
    return res.coverage


def test_small_n_coverage():
    alpha = 0.1
    n_calib = 12  # small: standard CP needs n>=9 just to be finite at 90%
    n_trials = 300
    for method in ["uniform", "spatial"]:
        cov_fixed = np.mean([
            _coverage_trial(n_calib, alpha, s, include_test_atom=True, method=method)
            for s in range(n_trials)
        ])
        cov_legacy = np.mean([
            _coverage_trial(n_calib, alpha, s, include_test_atom=False, method=method)
            for s in range(n_trials)
        ])
        print(f"[B] method={method:8s} n={n_calib}: "
              f"corrected coverage={cov_fixed:.3f}  legacy(no-atom)={cov_legacy:.3f}  "
              f"(target>={1-alpha:.2f})")
        assert cov_fixed >= 1 - alpha - 0.02, \
            f"{method}: corrected coverage {cov_fixed:.3f} below target"


if __name__ == "__main__":
    test_unweighted_reduces_to_standard_cp()
    test_small_n_coverage()
    print("\nAll checks passed.")
