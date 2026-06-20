"""
High-level GeoConformal estimators (geocp).

These wrappers mirror the published ``GeoConformalPrediction`` API (Lou, Luo &
Meng 2025) -- ``GeoConformalRegressor(..., coord_calib, bandwidth,
miscoverage_level).geo_conformalize(x_test, y_test, coord_test)`` -- on top of
the finite-sample-valid core (:class:`geocp.GeoConformalPrediction`), which
includes the test point's +infinity atom (Tibshirani et al. 2019). They are a
thin convenience layer: features drive the prediction, coordinates drive the
spatial kernel weights.
"""

from __future__ import annotations
from typing import Callable
import numpy as np
from numpy.typing import NDArray

from .core import GeoConformalPrediction
from .results import GeoCPResults
from .weights import spatial_kernel_weights, adaptive_spatial_weights

__all__ = ["GeoConformalRegressor"]


class GeoConformalRegressor:
    """
    GeoConformal regression with a spatial Gaussian kernel.

    Parameters
    ----------
    predict_f : Callable
        Fitted point predictor, features -> prediction. E.g. ``model.predict``.
    x_calib : NDArray
        Calibration features, shape (n_calib, d).
    y_calib : NDArray
        Calibration targets, shape (n_calib,).
    coord_calib : NDArray
        Calibration coordinates, shape (n_calib, 2), row-aligned with x_calib.
    bandwidth : float
        Gaussian kernel bandwidth h (in coordinate units). Default 0.15.
    miscoverage_level : float
        Target miscoverage alpha (default 0.1 for 90% coverage).
    score_fn : Callable, optional
        Nonconformity score; default |pred - true|.
    adaptive : bool
        If True, use a k-NN adaptive-bandwidth kernel instead of a fixed one.
    k : int
        Neighbors for the adaptive bandwidth (only if ``adaptive=True``).

    Examples
    --------
    >>> from geocp import GeoConformalRegressor
    >>> reg = GeoConformalRegressor(model.predict, X_calib, y_calib, loc_calib,
    ...                             bandwidth=0.15, miscoverage_level=0.1)
    >>> res = reg.geo_conformalize(X_test, y_test, loc_test)
    >>> res.coverage, res.mean_width_finite, res.frac_infinite
    """

    def __init__(
        self,
        predict_f: Callable,
        x_calib: NDArray,
        y_calib: NDArray,
        coord_calib: NDArray,
        bandwidth: float = 0.15,
        miscoverage_level: float = 0.1,
        score_fn: Callable = None,
        adaptive: bool = False,
        k: int = 20,
    ):
        coord_calib = np.asarray(coord_calib)
        if adaptive:
            weight_fn = adaptive_spatial_weights(coord_calib, base_bandwidth=bandwidth, k=k)
        else:
            weight_fn = spatial_kernel_weights(coord_calib, bandwidth=bandwidth)
        self.bandwidth = float(bandwidth)
        self.coord_calib = coord_calib
        self._cp = GeoConformalPrediction(
            predict_f, x_calib, y_calib, weight_fn,
            miscoverage_level=miscoverage_level, score_fn=score_fn,
        )

    @property
    def miscoverage_level(self) -> float:
        return self._cp.miscoverage_level

    def geo_conformalize(
        self,
        x_test: NDArray,
        y_test: NDArray,
        coord_test: NDArray,
        bayesian: bool = False,
        include_test_atom: bool = True,
        **kwargs,
    ) -> GeoCPResults:
        """
        Produce spatially-varying prediction intervals at the test points.

        Predictions use ``x_test``; the spatial kernel uses ``coord_test``.

        :param x_test: test features, shape (n_test, d)
        :param y_test: test targets, shape (n_test,)
        :param coord_test: test coordinates, shape (n_test, 2)
        :param bayesian: if True, use the Bayesian posterior (GeoBCP) and return
            HPD intervals; extra kwargs (num_mc, beta, ...) are forwarded.
        :param include_test_atom: keep the +infinity test atom (default True;
            set False only for ablation against the legacy, undercovering form).
        :return: GeoCPResults
        """
        if bayesian:
            return self._cp.bayesian_conformalize(
                x_test, y_test, coord_test=coord_test,
                include_test_atom=include_test_atom, **kwargs,
            )
        return self._cp.conformalize(
            x_test, y_test, coord_test=coord_test,
            include_test_atom=include_test_atom, **kwargs,
        )
