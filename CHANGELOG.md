# Changelog

## 0.3.0

Integrated the finite-sample-valid weighted / Bayesian conformal framework
(`geoconformal.geocp`) alongside the existing `GeoConformalSpatialRegression`
(GeoCP) and `GeoSIMConformalSpatialRegression` (GeoSIMCP) classes. Based on the
GeoCP framework by Xiayin Lou and Peng Luo.

### Added
- `geoconformal.geocp` subpackage — the unified engine all methods build on:
  - `GeoConformalPrediction` — general weighted (Bayesian) conformal prediction
    accepting arbitrary weight functions.
  - `GeoConformalRegressor` — high-level spatial wrapper (point-estimate GeoCP or
    Bayesian GeoBCP via `bayesian=True`).
  - `GeoCPResults` — results with finite-sample / Bayesian diagnostics
    (`frac_infinite`, `mean_width_finite`, `n_eff`, `posterior_std`,
    `prob_infinite`).
  - `WeightFunction` and weight factories: spatial / adaptive-spatial /
    covariate-shift / k-NN / uniform / RBF, plus `joint_geo_feature_weights`
    (the GeoSIMCP joint geographic + feature kernel).
  - Quantile utilities: `weighted_quantile`, `bayesian_weighted_quantile`,
    `effective_sample_size`.
- `include_test_atom` parameter on `GeoConformalSpatialRegression` and
  `GeoSIMConformalSpatialRegression` (default `False`): set `True` to opt into the
  finite-sample `+∞` test atom.
- The new names are re-exported at the top level (`from geoconformal import ...`).
- Test suites under `tests/` (`test_atom.py`, `test_variants.py`,
  `test_unified.py`) and a `tests` CI workflow; the publish workflow now runs the
  tests before uploading to PyPI.
- Self-contained finite-sample coverage checks under `experiments/` on the Seattle
  sample (`seattle_smalln.py`, `seattle_hist.py`).
- `example/geobcp_tutorial.ipynb` — a GeoBCP walkthrough (uncertainty-about-the-
  uncertainty), plus a motivation- and applicability-focused README organised
  around two dimensions (weighting: GeoCP / GeoSIMCP; estimate: point / GeoBCP).

### Changed (unified architecture)
- `GeoConformalSpatialRegression` (GeoCP) and `GeoSIMConformalSpatialRegression`
  (GeoSIMCP) are now thin wrappers over the `geoconformal.geocp` engine instead of
  re-implementing their own kernel + weighted quantile. Their public API
  (`.analyze()`, `GeoConformalResults.to_gpd()`, constructor signatures) is
  unchanged, and with the default `include_test_atom=False` they reproduce the
  previous numerical output exactly (pinned in `tests/test_unified.py`).
- `GeoSIMConformalSpatialRegression` now honours a custom `nonconformity_score_f`
  (the previous implementation accepted but ignored it, always using `|pred - gt|`).

### Fixed (finite-sample validity)
- The engine includes the unobserved test point's `+∞` atom with weight `w(x)`
  (Tibshirani et al. 2019), restoring finite-sample coverage that distance-weighted
  GeoCP loses on small / sparse calibration sets. With `uniform_weights` it reduces
  exactly to standard split conformal prediction.
- Defaults to the conformal-valid **step** quantile `inf{z : F(z) ≥ q}`; linear
  interpolation is opt-in via `interpolate=True`.

## 0.2.1

- Project URL fix; README badges.

## 0.2.0

- `GeoSIMConformalSpatialRegression` (GeoSIMCP): joint geographic + feature
  similarity weighting.
