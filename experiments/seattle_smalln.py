"""
Small-sample validation of the finite-sample (+inf test atom) fix in
geoconformal.geocp, on the Seattle house-price sample
(example/seattle_sample_3k.csv).

Design
------
1. Fit the regressor ONCE on a fixed train pool (disjoint from everything used
   for conformal), so predictor quality is constant across conditions.
2. For each random split, hold a fixed test set, then draw calibration sets of
   increasing size n from the remaining pool.
3. Run spatial GeoCP (Gaussian kernel on standardized UTM coordinates) and compare:
     - legacy   : point estimate WITHOUT the test atom (the old GeoCP)
     - corrected: point estimate WITH the +inf test atom (Tibshirani fix)
     - bayes    : GeoBCP WITH the test atom (HPD at beta)
   plus an unweighted split-CP reference.
4. Report marginal coverage (target 1-alpha=0.90), fraction of infinite
   intervals, and mean finite width.

Run:
    python experiments/seattle_smalln.py
"""
import os
import sys
import time
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from geoconformal.geocp import GeoConformalPrediction
from geoconformal.geocp.weights import spatial_kernel_weights, uniform_weights

DATA = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
    "example", "seattle_sample_3k.csv",
)

ALPHA = 0.10
BANDWIDTH = 0.4          # in standardized-coordinate units
N_CALIB_GRID = [20, 40, 80, 160, 320]
N_TEST = 500
N_SPLITS_POINT = 100     # splits for cheap point-estimate methods
N_SPLITS_BAYES = 20      # splits for the (slower) Bayesian method
NUM_MC = 300
BETA = 0.90
TRAIN_POOL = 1500
FEATURES = ['bathrooms', 'sqft_living', 'sqft_lot', 'grade', 'condition',
            'waterfront', 'view', 'age']


def load():
    df = pd.read_csv(DATA)
    y = df['log_price'].to_numpy(float)          # already log-transformed
    X = df[FEATURES].to_numpy(float)
    coord = df[['UTM_X', 'UTM_Y']].to_numpy(float)
    return X, y, coord


def main():
    X, y, coord = load()
    n = len(y)
    rng0 = np.random.default_rng(0)
    perm = rng0.permutation(n)
    tr = perm[:TRAIN_POOL]                 # fixed train pool for the regressor
    pool = perm[TRAIN_POOL:]               # everything else -> calib + test

    # Standardize coordinates using the train pool, then fit the regressor once.
    cmu, csd = coord[tr].mean(0), coord[tr].std(0)
    coordz = (coord - cmu) / csd
    reg = RandomForestRegressor(n_estimators=200, n_jobs=-1, random_state=0)
    reg.fit(X[tr], y[tr])
    predict_f = lambda Xq: reg.predict(Xq)  # noqa: E731

    def one_split(seed, n_calib, do_bayes):
        rng = np.random.default_rng(1000 + seed)
        idx = rng.permutation(pool)
        test_idx = idx[:N_TEST]
        calib_idx = idx[N_TEST:N_TEST + n_calib]

        Xc, yc, cz = X[calib_idx], y[calib_idx], coordz[calib_idx]
        Xt, yt, tz = X[test_idx], y[test_idx], coordz[test_idx]

        out = {}
        # spatial GeoCP: predict from features, weight by coordinates (coord_test)
        wf = spatial_kernel_weights(cz, bandwidth=BANDWIDTH)
        geo = GeoConformalPrediction(predict_f, Xc, yc, wf, miscoverage_level=ALPHA)
        r_corr = geo.conformalize(Xt, yt, coord_test=tz, include_test_atom=True)
        r_leg = geo.conformalize(Xt, yt, coord_test=tz, include_test_atom=False)
        out['leg_cov'] = r_leg.coverage
        out['corr_cov'] = r_corr.coverage
        out['corr_inf'] = r_corr.frac_infinite
        out['corr_w'] = r_corr.mean_width_finite

        # unweighted split-CP reference (= standard split CP)
        uni = GeoConformalPrediction(predict_f, Xc, yc, uniform_weights(),
                                     miscoverage_level=ALPHA)
        r_uni = uni.conformalize(Xt, yt, include_test_atom=True)
        out['uni_cov'] = r_uni.coverage
        out['uni_inf'] = r_uni.frac_infinite

        if do_bayes:
            rb = geo.bayesian_conformalize(Xt, yt, coord_test=tz, num_mc=NUM_MC,
                                           beta=BETA, random_state=7)
            out['bayes_cov'] = rb.coverage
            out['bayes_inf'] = rb.frac_infinite
            out['bayes_w'] = rb.mean_width_finite
        return out

    print(f"data n={n}, train pool={TRAIN_POOL}, pool={len(pool)}, "
          f"alpha={ALPHA}, bandwidth={BANDWIDTH}, target coverage={1-ALPHA:.2f}\n")

    rows = []
    for n_calib in N_CALIB_GRID:
        t0 = time.time()
        pt = [one_split(s, n_calib, do_bayes=False) for s in range(N_SPLITS_POINT)]
        agg = {k: np.mean([d[k] for d in pt]) for k in
               ['leg_cov', 'corr_cov', 'corr_inf', 'corr_w', 'uni_cov', 'uni_inf']}
        bz = [one_split(s, n_calib, do_bayes=True) for s in range(N_SPLITS_BAYES)]
        for k in ['bayes_cov', 'bayes_inf', 'bayes_w']:
            agg[k] = np.mean([d[k] for d in bz])
        agg['n_calib'] = n_calib
        rows.append(agg)
        print(f"n_calib={n_calib:4d} | "
              f"GeoCP legacy cov={agg['leg_cov']:.3f}  "
              f"corrected cov={agg['corr_cov']:.3f} (inf={agg['corr_inf']:.2f}, "
              f"w={agg['corr_w']:.2f}) | "
              f"GeoBCP cov={agg['bayes_cov']:.3f} (inf={agg['bayes_inf']:.2f}) | "
              f"split-CP cov={agg['uni_cov']:.3f}  [{time.time()-t0:.0f}s]")

    df_out = pd.DataFrame(rows)[
        ['n_calib', 'leg_cov', 'corr_cov', 'corr_inf', 'corr_w',
         'bayes_cov', 'bayes_inf', 'bayes_w', 'uni_cov', 'uni_inf']]
    out_csv = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                           "seattle_smalln_results.csv")
    df_out.to_csv(out_csv, index=False)
    print(f"\nsaved {out_csv}")
    _plot(df_out)


def _plot(df):
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except Exception:
        return
    fig, ax = plt.subplots(1, 2, figsize=(11, 4.2))
    x = df['n_calib']
    ax[0].axhline(1 - ALPHA, ls='--', c='k', lw=1, label='target 0.90')
    ax[0].plot(x, df['leg_cov'], 'o-', c='tab:red', label='GeoCP legacy (no atom)')
    ax[0].plot(x, df['corr_cov'], 's-', c='tab:green', label='GeoCP corrected')
    ax[0].plot(x, df['bayes_cov'], '^-', c='tab:blue', label='GeoBCP corrected')
    ax[0].plot(x, df['uni_cov'], 'd:', c='gray', label='split-CP reference')
    ax[0].set_xscale('log'); ax[0].set_xlabel('calibration size n')
    ax[0].set_ylabel('marginal coverage'); ax[0].set_title('Coverage vs n')
    ax[0].legend(fontsize=8); ax[0].grid(alpha=.3)
    ax[1].plot(x, df['corr_inf'], 's-', c='tab:green', label='GeoCP corrected')
    ax[1].plot(x, df['bayes_inf'], '^-', c='tab:blue', label='GeoBCP corrected')
    ax[1].set_xscale('log'); ax[1].set_xlabel('calibration size n')
    ax[1].set_ylabel('fraction infinite intervals')
    ax[1].set_title('Honest abstention vs n'); ax[1].legend(fontsize=8); ax[1].grid(alpha=.3)
    fig.tight_layout()
    p = os.path.join(os.path.dirname(os.path.abspath(__file__)), "seattle_smalln.png")
    fig.savefig(p, dpi=140)
    print(f"saved {p}")


if __name__ == "__main__":
    main()
