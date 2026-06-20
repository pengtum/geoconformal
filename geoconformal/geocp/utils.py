"""
Core computation utilities for GeoConformal Prediction (geocp).

These functions are weight-agnostic: they operate on arbitrary weight matrices
regardless of how the weights were computed (spatial kernels, covariate shift,
k-NN, etc.). Both quantile routines are fully vectorized.

Finite-sample validity (Tibshirani et al. 2019)
------------------------------------------------
The valid weighted conformal threshold is the (1 - alpha) quantile of

    sum_i p_i^w(x) * delta_{S_i}  +  p_{n+1}^w(x) * delta_{+inf},

    p_i^w(x)     = w(X_i) / (sum_j w(X_j) + w(x)),
    p_{n+1}^w(x) = w(x)   / (sum_j w(X_j) + w(x)),

i.e. the unobserved test point contributes its own probability atom at
+infinity with weight w(x) (its self-weight). When that atom carries more than
alpha of the mass -- which happens when the calibration set is small or the test
point lies in a region the calibration set covers poorly -- the quantile is
+infinity, yielding an (honest, conservative, finite-sample-valid) infinite
interval. Both routines accept ``self_weights`` to include it.

References:
    - Tibshirani, Barber, Candes & Ramdas (2019),
      "Conformal Prediction Under Covariate Shift", NeurIPS.
    - Snell & Griffiths (2025), "Conformal Prediction as Bayesian Quadrature", ICML.
    - Kish (1965), "Survey Sampling" (effective sample size).
"""

import numpy as np
from numpy.typing import NDArray

__all__ = [
    "effective_sample_size",
    "weighted_quantile",
    "bayesian_weighted_quantile",
]


def effective_sample_size(weights: NDArray) -> NDArray:
    """
    Kish's effective sample size for importance weights.

    Measures how many "equivalent uniform samples" the weighted set represents.
    Lower n_eff means fewer effective observations support the estimate.

    :param weights: shape (n_test, n_calib) or (n_calib,), importance weights (need not be normalized)
    :return: shape (n_test,) or scalar, effective sample size per test point
    """
    weights = np.asarray(weights, dtype=float)
    if weights.ndim == 1:
        weights = weights.reshape(1, -1)
    sum_w = np.sum(weights, axis=1)
    sum_w2 = np.sum(weights ** 2, axis=1)
    n_eff = sum_w ** 2 / (sum_w2 + 1e-14)
    return n_eff


def weighted_quantile(
    scores: NDArray,
    weights: NDArray,
    q: float,
    self_weights: NDArray = None,
    interpolate: bool = False,
) -> NDArray:
    """
    Vectorized weighted quantile of calibration scores per test point.

    By default this is the conformal-valid *step* quantile
    ``inf{ z : F(z) >= q }`` (the appropriate order statistic), which is the
    definition under which the coverage guarantee holds and which reduces
    exactly to standard split conformal in the unweighted case. Set
    ``interpolate=True`` for the older linearly-interpolated variant (sharper
    but not finite-sample valid -- it can undercover).

    If ``self_weights`` is provided, the test point's own atom at +infinity is
    included with weight w(x), normalizing over (sum_j w(X_j) + w(x)) per
    Tibshirani et al. (2019). A test point whose calibration mass cannot reach q
    -- because the +infinity atom holds the remaining mass -- receives +inf.

    :param scores: nonconformity scores, shape (n_calib,)
    :param weights: importance weights, shape (n_test, n_calib)
    :param q: quantile level (e.g., 0.9 for 90% coverage)
    :param self_weights: test self-weights w(x), shape (n_test,), or None
    :param interpolate: if True, linearly interpolate between order statistics
        (legacy behavior); if False (default), use the valid step quantile.
    :return: weighted quantile per test point, shape (n_test,)
    """
    scores = np.asarray(scores, dtype=float)
    weights = np.asarray(weights, dtype=float)
    if weights.ndim == 1:
        weights = weights.reshape(1, -1)
    n_test, n_calib = weights.shape

    # Sort scores once (shared across all test points)
    sorter = np.argsort(scores)
    sorted_scores = scores[sorter]
    sorted_weights = weights[:, sorter]

    # Total mass per test point: calibration mass (+ test atom mass if given)
    calib_mass = sorted_weights.sum(axis=1, keepdims=True)
    if self_weights is None:
        total_w = calib_mass
    else:
        self_weights = np.asarray(self_weights, dtype=float).reshape(n_test, 1)
        total_w = calib_mass + self_weights

    # Cumulative calibration mass, normalized by total (calib + test atom).
    cumsum_w = np.cumsum(sorted_weights, axis=1)
    normalized = cumsum_w / np.where(total_w > 0, total_w, 1.0)

    # Tolerance so an exact-boundary atom (e.g. uniform weights, where the last
    # calibration mass equals q exactly) counts as reaching q despite rounding.
    q_tol = q - 1e-9

    reaches = normalized >= q_tol
    calib_reaches_q = reaches[:, -1]               # else mass tops out below q
    idx = np.argmax(reaches, axis=1)               # first index reaching q

    # Step quantile (vectorized): the order statistic at idx, or +inf when only
    # the +infinity test atom can reach q.
    quantiles = np.where(calib_reaches_q, sorted_scores[idx], np.inf)

    if interpolate:
        # Linearly interpolate between order statistics (legacy, non-valid).
        ar = np.arange(n_test)
        prev = np.maximum(idx - 1, 0)
        x1 = normalized[ar, idx]
        x0 = normalized[ar, prev]
        y1 = sorted_scores[idx]
        y0 = sorted_scores[prev]
        interp = y0 + (y1 - y0) * (q - x0) / (x1 - x0 + 1e-14)
        do = calib_reaches_q & (idx > 0) & (x1 > q_tol + 1e-12)
        quantiles = np.where(do, interp, quantiles)

    return quantiles


def bayesian_weighted_quantile(
    scores: NDArray,
    weights: NDArray,
    q: float,
    self_weights: NDArray = None,
    num_mc: int = 1000,
    beta: float = 0.9,
    concentration_scale: str = 'neff',
    random_state: int = 42,
) -> dict:
    """
    Bayesian weighted quantile via Dirichlet posterior sampling (vectorized).

    For each test point, samples from Dir(c*w_1, ..., c*w_n, c*w_{n+1}) over the
    n calibration atoms plus the test atom, then computes the weighted quantile
    under each sample -- where the test atom sits at +infinity -- yielding a
    posterior distribution of thresholds. The HPD threshold at confidence level
    beta provides data-conditional coverage guarantees.

    Sampling uses ``numpy.random.Generator.dirichlet`` per test point: the draw
    cost is dominated by the underlying Gamma variates (n_test * num_mc * n_bins
    of them, irreducible), for which the per-point C-level call is both faster
    and lower-memory than a broadcast 3-D draw. The cheap summary statistics are
    vectorized.

    The test atom (the (n+1)-th Dirichlet component) participates in the
    quantile -- earlier code added it to the concentration but discarded it
    before taking the quantile, dropping the finite-sample correction and
    causing undercoverage at small n / low effective sample size.

    :param scores: nonconformity scores, shape (n_calib,)
    :param weights: importance weights, shape (n_test, n_calib)
    :param q: quantile level (e.g., 0.9 for 90% coverage)
    :param self_weights: test self-weights w(x), shape (n_test,). If None, the
        test atom weight defaults to 1/(n_calib+1) per test point (the symmetric
        exchangeable choice).
    :param num_mc: number of Monte Carlo Dirichlet samples (default 1000)
    :param beta: confidence level for HPD threshold (default 0.9).
        Higher beta -> more conservative (wider) intervals.
    :param concentration_scale: 'neff' (Kish effective sample size, recommended)
        or 'fixed' (c=1).
    :param random_state: random seed for reproducibility
    :return: dict with keys 'hpd_quantiles', 'posterior_mean', 'posterior_std',
        'n_eff', 'posterior_samples' (shape (n_test, num_mc)), 'prob_infinite'.
    """
    scores = np.asarray(scores, dtype=float)
    weights = np.asarray(weights, dtype=float)
    if weights.ndim == 1:
        weights = weights.reshape(1, -1)
    n_test, n_calib = weights.shape
    rng = np.random.default_rng(random_state)

    # Sort scores once. The test atom score is +infinity, so it occupies the
    # final (n+1)-th position of the augmented, ascending score vector.
    sorter = np.argsort(scores)
    sorted_scores = scores[sorter]
    augmented_scores = np.append(sorted_scores, np.inf)  # (n_calib + 1,)
    sorted_weights = weights[:, sorter]                  # (n_test, n_calib)

    # Test atom weight w(x) per test point
    if self_weights is None:
        w_self = np.full(n_test, 1.0 / (n_calib + 1))
    else:
        w_self = np.asarray(self_weights, dtype=float).reshape(n_test)

    # Normalize calibration + test atom weights to sum to 1 per test point
    total = sorted_weights.sum(axis=1) + w_self + 1e-14
    sorted_weights_norm = sorted_weights / total[:, None]
    w_self_norm = w_self / total

    # Effective sample size (diagnostic; on calibration weights) and concentration
    n_eff = effective_sample_size(weights)
    if concentration_scale == 'neff':
        c = n_eff
    elif concentration_scale == 'fixed':
        c = np.ones(n_test)
    else:
        raise ValueError(f"Unknown concentration_scale: {concentration_scale}")

    # Dirichlet concentration parameters: (n_test, n_calib + 1)
    alpha_min = 1e-6
    alpha_calib = np.maximum(c[:, None] * sorted_weights_norm, alpha_min)
    alpha_future = np.maximum(c * w_self_norm, alpha_min)
    alpha_full = np.concatenate([alpha_calib, alpha_future[:, None]], axis=1)

    n_bins = n_calib + 1
    posterior_thresholds = np.empty((n_test, num_mc))

    for i in range(n_test):
        # Dirichlet samples over the n_calib + 1 atoms: (num_mc, n_bins).
        u = rng.dirichlet(alpha_full[i], size=num_mc)
        # Cumulative mass over sorted atoms (calibration first, +inf test atom
        # last). No renormalization: a Dirichlet sample already sums to 1, and
        # the test atom must keep its share of the mass.
        cum = np.cumsum(u, axis=1)
        reaches = cum >= q
        idx = np.argmax(reaches, axis=1)            # first atom reaching q
        idx[~reaches[:, -1]] = n_bins - 1           # numerical guard -> +inf atom
        posterior_thresholds[i] = augmented_scores[idx]

    # HPD threshold: empirical step quantile inf{ z : F(z) >= beta }. Use
    # method='inverted_cdf' (selects an actual sample) rather than the default
    # linear interpolation, because interpolating between two +inf samples gives
    # inf - inf = NaN. Finite while < (1-beta) of samples are +inf, else +inf.
    hpd_quantiles = np.quantile(
        posterior_thresholds, beta, axis=1, method='inverted_cdf'
    )

    # Posterior mean/std over the *finite* samples (the conditional threshold
    # posterior given the interval is certifiable); prob_infinite carries the
    # censoring probability separately. Computed without NaN warnings.
    finite = np.isfinite(posterior_thresholds)
    cnt = finite.sum(axis=1)
    safe = np.where(finite, posterior_thresholds, 0.0)
    denom = np.maximum(cnt, 1)
    mean = safe.sum(axis=1) / denom
    var = np.maximum((safe ** 2).sum(axis=1) / denom - mean ** 2, 0.0)
    std = np.sqrt(var)
    all_inf = cnt == 0
    posterior_mean = np.where(all_inf, np.inf, mean)
    posterior_std = np.where(all_inf, np.inf, std)
    prob_infinite = np.mean(~finite, axis=1)

    return {
        'hpd_quantiles': hpd_quantiles,
        'posterior_mean': posterior_mean,
        'posterior_std': posterior_std,
        'n_eff': n_eff,
        'posterior_samples': posterior_thresholds,
        'prob_infinite': prob_infinite,
    }
