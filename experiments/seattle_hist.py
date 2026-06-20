"""
Coverage-distribution view (Tibshirani et al. 2019, Fig 1 style) on the Seattle
house-price sample (example/seattle_sample_3k.csv).

Fixes a small calibration size and plots the histogram of per-split marginal
coverage for legacy GeoCP (no test atom) vs corrected GeoCP / GeoBCP, over many
random splits. Legacy mass sits left of the 0.90 target (collapse); corrected
mass sits at/above it.

Run:
    python experiments/seattle_hist.py
"""
import os
import sys
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from geoconformal.geocp import GeoConformalPrediction
from geoconformal.geocp.weights import spatial_kernel_weights

DATA = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
    "example", "seattle_sample_3k.csv",
)
FEATURES = ['bathrooms', 'sqft_living', 'sqft_lot', 'grade', 'condition',
            'waterfront', 'view', 'age']
ALPHA = 0.10
BANDWIDTH = 0.4

N_CALIB = 100
N_TEST = 500
N_SPLITS = 200
TRAIN_POOL = 1500


def main():
    df = pd.read_csv(DATA)
    y = df['log_price'].to_numpy(float)
    X = df[FEATURES].to_numpy(float)
    coord = df[['UTM_X', 'UTM_Y']].to_numpy(float)
    n = len(y)

    perm = np.random.default_rng(0).permutation(n)
    tr, pool = perm[:TRAIN_POOL], perm[TRAIN_POOL:]
    cmu, csd = coord[tr].mean(0), coord[tr].std(0)
    coordz = (coord - cmu) / csd
    reg = RandomForestRegressor(n_estimators=200, n_jobs=-1, random_state=0).fit(X[tr], y[tr])
    predict_f = lambda Xq: reg.predict(Xq)  # noqa: E731

    leg, corr, bayes = [], [], []
    for s in range(N_SPLITS):
        rng = np.random.default_rng(2000 + s)
        idx = rng.permutation(pool)
        ti, ci = idx[:N_TEST], idx[N_TEST:N_TEST + N_CALIB]
        Xc, yc, cz = X[ci], y[ci], coordz[ci]
        Xt, yt, tz = X[ti], y[ti], coordz[ti]
        geo = GeoConformalPrediction(predict_f, Xc, yc,
                                     spatial_kernel_weights(cz, BANDWIDTH),
                                     miscoverage_level=ALPHA)
        r_leg = geo.conformalize(Xt, yt, coord_test=tz, include_test_atom=False)
        r_corr = geo.conformalize(Xt, yt, coord_test=tz, include_test_atom=True)
        r_b = geo.bayesian_conformalize(Xt, yt, coord_test=tz, num_mc=300,
                                        beta=0.90, random_state=7)
        leg.append(r_leg.coverage)
        corr.append(r_corr.coverage)
        bayes.append(r_b.coverage)

    leg, corr, bayes = map(np.array, (leg, corr, bayes))
    print(f"n_calib={N_CALIB}, splits={N_SPLITS}, target={1-ALPHA:.2f}")
    for name, a in [('legacy   ', leg), ('corrected', corr), ('GeoBCP   ', bayes)]:
        print(f"  {name}: mean cov={a.mean():.3f}  P(cov>=0.90)={np.mean(a >= 0.90):.2f}  "
              f"min={a.min():.3f}")

    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        fig, ax = plt.subplots(figsize=(7, 4.2))
        bins = np.linspace(0.6, 1.001, 45)
        ax.hist(leg, bins=bins, alpha=.55, color='tab:red', label=f'GeoCP legacy (mean {leg.mean():.3f})')
        ax.hist(corr, bins=bins, alpha=.55, color='tab:green', label=f'GeoCP corrected (mean {corr.mean():.3f})')
        ax.hist(bayes, bins=bins, alpha=.45, color='tab:blue', label=f'GeoBCP (mean {bayes.mean():.3f})')
        ax.axvline(1 - ALPHA, ls='--', c='k', lw=1.2, label='target 0.90')
        ax.set_xlabel('per-split marginal coverage'); ax.set_ylabel('frequency')
        ax.set_title(f'Seattle, n_calib={N_CALIB} ({N_SPLITS} splits)')
        ax.legend(fontsize=8)
        fig.tight_layout()
        p = os.path.join(os.path.dirname(os.path.abspath(__file__)), "seattle_hist.png")
        fig.savefig(p, dpi=140)
        print(f"saved {p}")
    except Exception as e:
        print("plot skipped:", e)


if __name__ == "__main__":
    main()
