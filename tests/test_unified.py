"""
Unified-architecture tests: the classic GeoConformalSpatialRegression /
GeoSIMConformalSpatialRegression classes are now thin wrappers over the
geoconformal.geocp engine. These checks ensure:

  (A) with include_test_atom=False they reproduce the ORIGINAL implementations
      (pinned reference numbers) -- no silent change to published behaviour;
  (B) include_test_atom=True does not reduce coverage (finite-sample valid);
  (C) GeoSIMCP with lambda_weight=1.0 reduces exactly to GeoCP.
"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np

from geoconformal import GeoConformalSpatialRegression, GeoSIMConformalSpatialRegression


def _data(seed=0, ncal=80, nte=50):
    rng = np.random.default_rng(seed)
    Xc = rng.standard_normal((ncal, 3)); cc = rng.uniform(0, 1, (ncal, 2))
    Xt = rng.standard_normal((nte, 3));  ct = rng.uniform(0, 1, (nte, 2))
    beta = np.array([1.0, -0.5, 0.3])
    predict = lambda X: X @ beta  # noqa: E731
    yc = Xc @ beta + rng.standard_normal(ncal) * 0.5
    yt = Xt @ beta + rng.standard_normal(nte) * 0.5
    return predict, Xc, yc, cc, Xt, yt, ct


def test_geocp_reproduces_reference():
    """include_test_atom=False must match the original GeoCP implementation."""
    predict, Xc, yc, cc, Xt, yt, ct = _data()
    g = GeoConformalSpatialRegression(
        predict_f=predict, bandwidth=0.5, miscoverage_level=0.1,
        coord_calib=cc, coord_test=ct, X_calib=Xc, y_calib=yc, X_test=Xt, y_test=yt)
    r = g.analyze()
    assert np.allclose(r.geo_uncertainty[:4],
                       [0.724984, 0.724984, 0.699622, 0.699622], atol=1e-5)
    assert np.isclose(r.uncertainty, 0.702158, atol=1e-5)
    assert np.isclose(r.coverage_probability, 0.80, atol=1e-9)
    print("[A] GeoCP wrapper reproduces original implementation: OK")


def test_geosim_reproduces_reference():
    """include_test_atom=False must match the original GeoSIMCP implementation."""
    predict, Xc, yc, cc, Xt, yt, ct = _data()
    gs = GeoSIMConformalSpatialRegression(
        predict_f=predict, bandwidth=0.5, miscoverage_level=0.1,
        coord_calib=cc, coord_test=ct, X_calib=Xc, y_calib=yc, X_test=Xt, y_test=yt,
        lambda_weight=0.5, distance_metric='euclidean', standardize_weights=True)
    r = gs.analyze()
    assert np.allclose(r.geo_uncertainty[:4],
                       [0.577968, 0.815261, 0.580545, 0.632772], atol=1e-5)
    assert np.isclose(r.uncertainty, 0.754896, atol=1e-5)
    assert np.isclose(r.coverage_probability, 0.82, atol=1e-9)
    print("[A] GeoSIMCP wrapper reproduces original implementation: OK")


def test_test_atom_does_not_reduce_coverage():
    """include_test_atom=True should not reduce coverage (finite-sample valid)."""
    predict, Xc, yc, cc, Xt, yt, ct = _data(seed=1, ncal=30, nte=200)
    cov = {}
    for atom in (False, True):
        g = GeoConformalSpatialRegression(
            predict_f=predict, bandwidth=0.3, miscoverage_level=0.1,
            coord_calib=cc, coord_test=ct, X_calib=Xc, y_calib=yc, X_test=Xt, y_test=yt,
            include_test_atom=atom)
        cov[atom] = g.analyze().coverage_probability
    assert cov[True] >= cov[False] - 1e-9
    print(f"[B] include_test_atom: cov False={cov[False]:.3f} True={cov[True]:.3f} (True>=False): OK")


def test_geosim_lambda1_equals_geocp():
    """GeoSIMCP with lambda=1 (geographic only) == GeoCP at the same bandwidth."""
    predict, Xc, yc, cc, Xt, yt, ct = _data(seed=2)
    g = GeoConformalSpatialRegression(
        predict_f=predict, bandwidth=0.4, miscoverage_level=0.1,
        coord_calib=cc, coord_test=ct, X_calib=Xc, y_calib=yc, X_test=Xt, y_test=yt)
    gs = GeoSIMConformalSpatialRegression(
        predict_f=predict, bandwidth=0.4, miscoverage_level=0.1,
        coord_calib=cc, coord_test=ct, X_calib=Xc, y_calib=yc, X_test=Xt, y_test=yt,
        lambda_weight=1.0)
    assert np.allclose(g.analyze().geo_uncertainty, gs.analyze().geo_uncertainty, atol=1e-8)
    print("[C] GeoSIMCP(lambda=1) == GeoCP: OK")


if __name__ == "__main__":
    test_geocp_reproduces_reference()
    test_geosim_reproduces_reference()
    test_test_atom_does_not_reduce_coverage()
    test_geosim_lambda1_equals_geocp()
    print("\nAll unified-architecture checks passed.")
