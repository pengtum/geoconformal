"""
Weight function factories for GeoConformal Prediction (geocp).

Each factory returns a :class:`WeightFunction` -- a callable that maps
(x_test, x_calib) -> weights of shape (n_test, n_calib), and that additionally
exposes ``self_weight(x_test) -> (n_test,)``: the weight w(x) that the test
point assigns to *itself*.

The self-weight is required for valid weighted conformal prediction. Following
Tibshirani et al. (2019), the conformal threshold is the (1 - alpha) quantile of

    sum_i p_i^w(x) * delta_{V_i}  +  p_{n+1}^w(x) * delta_{+inf},

    p_i^w(x)     = w(X_i) / (sum_j w(X_j) + w(x)),     i = 1, ..., n
    p_{n+1}^w(x) = w(x)   / (sum_j w(X_j) + w(x)),

i.e. the test point contributes its own atom at +infinity with weight w(x).
Omitting this atom (as earlier GeoCP code did) breaks the finite-sample
guarantee and causes coverage to collapse when the calibration set is small or
the test point lies in a region poorly covered by calibration points.

What w(x) is depends on the weighting scheme:

    - spatial_kernel_weights:    GeoCP / GeoBCP (Gaussian kernel on coordinates)
                                 -> w(x) = K(0) = 1 (kernel evaluated at zero distance)
    - adaptive_spatial_weights:  Adaptive GeoCP / Adaptive GeoBCP -> w(x) = K(0) = 1
    - covariate_shift_weights:   Weighted CP under covariate shift (Tibshirani 2019)
                                 -> w(x) = density ratio evaluated at the test point
    - knn_weights:               Localized CP via k-NN (Guan 2023) -> w(x) = 1
    - uniform_weights:           Standard CP / BQ-CP -> w(x) = 1
    - rbf_feature_weights:       RBF kernel on feature space -> w(x) = K(0) = 1
"""

from dataclasses import dataclass
from typing import Callable
import numpy as np
from numpy.typing import NDArray


# ============================================================
# Weight function container
# ============================================================

@dataclass
class WeightFunction:
    """
    A weighting scheme for localized / weighted conformal prediction.

    A ``WeightFunction`` is callable -- ``wf(x_test, x_calib)`` returns the
    (n_test, n_calib) weight matrix exactly as a plain weight function would,
    so this is backward compatible with the old callable contract. In addition
    it carries ``self_weight(x_test)`` returning the test point's own weight
    w(x) of shape (n_test,), needed for the +infinity atom in weighted CP.

    Parameters
    ----------
    weights : Callable[[NDArray, NDArray], NDArray]
        Maps (x_test, x_calib) -> weights, shape (n_test, n_calib).
    self_weight_fn : Callable[[NDArray], NDArray]
        Maps x_test -> self-weights w(x), shape (n_test,). Must be on the same
        (unnormalized) scale as the calibration weights returned by ``weights``.
    """
    weights: Callable[[NDArray, NDArray], NDArray]
    self_weight_fn: Callable[[NDArray], NDArray]

    def __call__(self, x_test: NDArray, x_calib: NDArray) -> NDArray:
        return self.weights(x_test, x_calib)

    def self_weight(self, x_test: NDArray) -> NDArray:
        """Test point's own weight w(x), shape (n_test,)."""
        return np.asarray(self.self_weight_fn(x_test), dtype=float).reshape(-1)


# ============================================================
# Internal kernel functions
# ============================================================

def _gaussian_kernel(d: NDArray) -> NDArray:
    """Gaussian distance decay function: K(d) = exp(-d^2/2). Note K(0) = 1."""
    return np.exp(-0.5 * d ** 2)


def _compute_distances(z_test: NDArray, z_calib: NDArray) -> NDArray:
    """Compute pairwise Euclidean distances between test and calibration points."""
    z_test_norm = np.sum(z_test ** 2, axis=1).reshape(-1, 1)
    z_calib_norm = np.sum(z_calib ** 2, axis=1).reshape(1, -1)
    distances = np.sqrt(
        np.maximum(z_test_norm + z_calib_norm - 2 * np.dot(z_test, z_calib.T), 0)
    )
    return distances


def _ones_self_weight(x_test: NDArray) -> NDArray:
    """Self-weight for kernels whose maximum (at zero distance) is 1."""
    return np.ones(np.asarray(x_test).shape[0])


# ============================================================
# Weight function factories
# ============================================================

def spatial_kernel_weights(coord_calib: NDArray, bandwidth: float) -> WeightFunction:
    """
    Create a spatial Gaussian kernel weight function.

    Equivalent to GeoCP's kernel_smoothing. Weights decrease with geographic
    distance via K(||s - s_i|| / h). The test point's self-weight is K(0) = 1,
    the kernel maximum, since a calibration point coincident with the test point
    would sit at zero distance.

    :param coord_calib: calibration coordinates, shape (n_calib, 2)
    :param bandwidth: kernel bandwidth h > 0
    :return: WeightFunction; calling it with coordinates as x_test yields the
        (n_test, n_calib) weight matrix.

    Note: The returned function uses coord_test (not x_test features) for
    distance computation. Pass coordinates as x_test when calling.
    """
    _coord_calib = np.asarray(coord_calib)
    _bw = float(bandwidth)

    def weight_fn(x_test: NDArray, x_calib: NDArray) -> NDArray:
        # x_test is assumed to be coordinates for spatial weighting
        distances = _compute_distances(np.asarray(x_test), _coord_calib)
        weights = _gaussian_kernel(distances / _bw)
        return weights

    return WeightFunction(weights=weight_fn, self_weight_fn=_ones_self_weight)


def adaptive_spatial_weights(
    coord_calib: NDArray,
    base_bandwidth: float = 0.15,
    k: int = 20
) -> WeightFunction:
    """
    Create an adaptive bandwidth spatial kernel weight function.

    Uses k-NN median distance to compute per-test-point bandwidths,
    then applies a Gaussian kernel with local bandwidths.

    Weights are returned unnormalized (raw kernel values). Per-test-point
    normalization is unnecessary because the downstream weighted-quantile and
    Dirichlet routines renormalize anyway, and -- crucially -- the self-weight
    w(x) = K(0) = 1 must be on the same scale as the calibration weights for the
    +infinity test atom to be correct. Normalizing the calibration weights but
    not the self-weight would spuriously inflate the test atom.

    :param coord_calib: calibration coordinates, shape (n_calib, 2)
    :param base_bandwidth: base bandwidth scaling factor
    :param k: number of nearest neighbors for local bandwidth
    :return: WeightFunction
    """
    from sklearn.neighbors import NearestNeighbors

    _coord_calib = np.asarray(coord_calib)
    _bw0 = float(base_bandwidth)
    _k = min(k, len(_coord_calib))

    def weight_fn(x_test: NDArray, x_calib: NDArray) -> NDArray:
        x_test = np.asarray(x_test)
        distances = _compute_distances(x_test, _coord_calib)

        # Compute local bandwidths via k-NN
        nbrs = NearestNeighbors(n_neighbors=_k, algorithm='auto').fit(_coord_calib)
        knn_dists, _ = nbrs.kneighbors(x_test)
        local_bandwidths = np.maximum(np.median(knn_dists, axis=1) * _bw0, 1e-6)

        # Adaptive Gaussian kernel: each test point has its own bandwidth.
        # Returned unnormalized so that w(x) = K(0) = 1 stays on the same scale.
        bw = local_bandwidths[:, None]  # (n_test, 1)
        weights = np.exp(-0.5 * (distances / bw) ** 2)
        return weights

    return WeightFunction(weights=weight_fn, self_weight_fn=_ones_self_weight)


def covariate_shift_weights(density_ratio_fn: Callable) -> WeightFunction:
    """
    Create a covariate shift weight function (Tibshirani et al., 2019).

    Weights are importance ratios: w_i = p_test(X_i) / p_calib(X_i). The test
    point's self-weight is the *same* density ratio evaluated at the test point,
    w(x) = density_ratio_fn(x_test) -- exactly w(x) in Tibshirani Eq. (6).

    :param density_ratio_fn: function that computes density ratios,
        Callable[[NDArray], NDArray], maps features to ratios (shape (n,)).
    :return: WeightFunction
    """
    def weight_fn(x_test: NDArray, x_calib: NDArray) -> NDArray:
        # Density ratios for calibration points (independent of test point)
        ratios = np.asarray(density_ratio_fn(x_calib)).reshape(-1)  # (n_calib,)
        n_test = np.asarray(x_test).shape[0]
        # Same weight profile for all test points
        weights = np.tile(ratios, (n_test, 1))  # (n_test, n_calib)
        return weights

    def self_weight_fn(x_test: NDArray) -> NDArray:
        # Test point's own importance weight w(x) = dP_test/dP_calib (x_test)
        return np.asarray(density_ratio_fn(x_test)).reshape(-1)

    return WeightFunction(weights=weight_fn, self_weight_fn=self_weight_fn)


def knn_weights(k: int = 20) -> WeightFunction:
    """
    Create a k-NN binary weight function for localized conformal prediction.

    Each test point assigns weight 1 to its k nearest calibration neighbors
    and weight 0 to all others (Guan et al., 2023). The test point is its own
    nearest neighbor, so its self-weight is w(x) = 1, matching the unit weight
    given to the selected calibration neighbors.

    :param k: number of nearest neighbors
    :return: WeightFunction
    """
    from sklearn.neighbors import NearestNeighbors

    def weight_fn(x_test: NDArray, x_calib: NDArray) -> NDArray:
        x_test = np.asarray(x_test)
        x_calib = np.asarray(x_calib)
        _k = min(k, x_calib.shape[0])
        nbrs = NearestNeighbors(n_neighbors=_k, algorithm='auto').fit(x_calib)
        _, indices = nbrs.kneighbors(x_test)

        n_test = x_test.shape[0]
        n_calib = x_calib.shape[0]
        weights = np.zeros((n_test, n_calib))
        rows = np.repeat(np.arange(n_test), _k)
        weights[rows, indices.ravel()] = 1.0
        return weights

    return WeightFunction(weights=weight_fn, self_weight_fn=_ones_self_weight)


def uniform_weights() -> WeightFunction:
    """
    Create a uniform weight function (standard CP / BQ-CP).

    All calibration points receive equal weight 1, and the test point's
    self-weight is also w(x) = 1. With the +infinity test atom included, this
    recovers standard split conformal prediction exactly: the (1 - alpha)
    quantile over n+1 equally weighted atoms is S_(ceil((1-alpha)(n+1))), and
    +infinity whenever ceil((1-alpha)(n+1)) > n.

    :return: WeightFunction
    """
    def weight_fn(x_test: NDArray, x_calib: NDArray) -> NDArray:
        n_test = np.asarray(x_test).shape[0]
        n_calib = np.asarray(x_calib).shape[0]
        return np.ones((n_test, n_calib))

    return WeightFunction(weights=weight_fn, self_weight_fn=_ones_self_weight)


def rbf_feature_weights(gamma: float = 1.0) -> WeightFunction:
    """
    Create an RBF kernel weight function on feature space.

    Weights are computed as K(x_test, x_calib) = exp(-gamma * ||x - x'||^2).
    The test point's self-weight is K(x, x) = 1, the kernel maximum.

    :param gamma: RBF kernel parameter (inverse of length scale squared)
    :return: WeightFunction
    """
    def weight_fn(x_test: NDArray, x_calib: NDArray) -> NDArray:
        distances_sq = _compute_distances(np.asarray(x_test), np.asarray(x_calib)) ** 2
        weights = np.exp(-gamma * distances_sq)
        return weights

    return WeightFunction(weights=weight_fn, self_weight_fn=_ones_self_weight)


def joint_geo_feature_weights(
    coord_calib: NDArray,
    feat_calib: NDArray,
    bandwidth: float,
    lambda_weight: float = 1.0,
    distance_metric: str = 'euclidean',
    standardize: bool = True,
) -> WeightFunction:
    """
    Joint geographic + feature-similarity kernel weights (GeoSIMCP).

    Combines geographic distance and feature-space distance into a single
    Gaussian kernel::

        d_joint = sqrt( lambda * d_geo^2 + (1 - lambda) * d_feat^2 )
        w       = exp( -0.5 * (d_joint / bandwidth)^2 )

    ``lambda_weight = 1`` reduces to :func:`spatial_kernel_weights` (geographic
    only); ``0`` uses feature distance only. The test point's self-weight is
    K(0) = 1 (a calibration point coincident in BOTH space and features sits at
    zero joint distance), matching :func:`spatial_kernel_weights`.

    Because the kernel acts on a JOINT (coordinate + feature) space, call the
    returned function with the test coordinates and distance-features stacked
    horizontally::

        weight_fn(np.hstack([coord_test, feat_test]), x_calib_ignored)

    The first ``coord_calib.shape[1]`` columns are treated as coordinates and the
    rest as features. ``GeoSIMConformalSpatialRegression`` does this for you.

    :param coord_calib: calibration coordinates, shape (n_calib, d_geo)
    :param feat_calib: calibration features used for the distance, shape (n_calib, p)
    :param bandwidth: Gaussian kernel bandwidth h > 0
    :param lambda_weight: trade-off in [0, 1] (1 = geographic only, 0 = feature only)
    :param distance_metric: 'euclidean' or 'mnd' (minimum normalized difference)
    :param standardize: z-score the features (euclidean only), fit on calibration
    :return: WeightFunction
    """
    coord_calib = np.asarray(coord_calib, dtype=float)
    feat_calib = np.asarray(feat_calib, dtype=float)
    n_geo = coord_calib.shape[1]
    bw = float(bandwidth)
    lam = float(lambda_weight)
    metric = distance_metric.lower()

    # Pre-fit the feature transform on the calibration set.
    if metric == 'mnd':
        feat_ranges = np.ptp(feat_calib, axis=0)
        feat_ranges[feat_ranges == 0] = 1e-8
        _feat_calib = feat_calib
        valid_cols = mu = sd = None
    else:
        if standardize:
            valid_cols = np.std(feat_calib, axis=0) > 1e-8
            _feat_calib = feat_calib[:, valid_cols]
            mu = _feat_calib.mean(axis=0)
            sd = _feat_calib.std(axis=0)
            sd[sd == 0] = 1.0
            _feat_calib = (_feat_calib - mu) / sd
        else:
            valid_cols = mu = sd = None
            _feat_calib = feat_calib
        feat_ranges = None

    def _feat_distance(ft: NDArray) -> NDArray:
        """Feature distance, test (n_test, p_raw) vs calibration -> (n_test, n_calib)."""
        if metric == 'mnd':
            diffs = np.abs(_feat_calib[None, :, :] - ft[:, None, :])      # (n_test, n_calib, p)
            scaled = 1.0 - diffs / feat_ranges[None, None, :]
            similarity = scaled.min(axis=2)                               # (n_test, n_calib)
            return 1.0 - similarity
        f = ft
        if valid_cols is not None:
            f = f[:, valid_cols]
        if mu is not None:
            f = (f - mu) / sd
        return _compute_distances(f, _feat_calib)

    def weight_fn(x_test: NDArray, x_calib: NDArray) -> NDArray:
        x_test = np.asarray(x_test, dtype=float)
        coord_test = x_test[:, :n_geo]
        feat_test = x_test[:, n_geo:]
        d_geo = _compute_distances(coord_test, coord_calib)
        d_feat = _feat_distance(feat_test) + 1e-8
        d_joint = np.sqrt(lam * d_geo ** 2 + (1.0 - lam) * d_feat ** 2)
        return np.exp(-0.5 * (d_joint / (bw + 1e-8)) ** 2)               # unnormalized

    return WeightFunction(weights=weight_fn, self_weight_fn=_ones_self_weight)
