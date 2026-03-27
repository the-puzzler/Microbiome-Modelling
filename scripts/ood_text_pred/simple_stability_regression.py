#!/usr/bin/env python3
"""Quick regressions to predict stability from evenness and count.

Uses `data/ood_text_pred/stability_vs_evenness.tsv`, which is produced by
`scripts/ood_text_pred/plot_stability_vs_evenness.py`.
"""

import argparse
import csv
from pathlib import Path

import numpy as np
from sklearn.ensemble import HistGradientBoostingRegressor, RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.model_selection import train_test_split


IN_TSV = Path("data/ood_text_pred/stability_vs_evenness.tsv")
TEST_SIZE = 0.2
RANDOM_STATE = 42


def load_rows(path: Path):
    rows = []
    with path.open() as f:
        reader = csv.DictReader(f, delimiter="\t")
        for row in reader:
            try:
                rows.append(
                    {
                        "sampleid": row["sampleid"],
                        "stability_score": float(row["stability_score"]),
                        "evenness_pielou": float(row["evenness_pielou"]),
                        "count": float(row["count"]),
                    }
                )
            except (KeyError, TypeError, ValueError):
                continue
    return rows


def _linear_overall_f_pvalue(r2: float, n: int, p: int):
    """Overall F-test p-value for linear regression (vs intercept-only).

    This is the standard in-sample F-test under usual linear-model assumptions.
    """
    if not (0.0 <= r2 <= 1.0):
        return None
    df1 = int(p)
    df2 = int(n - p - 1)
    if df1 <= 0 or df2 <= 0:
        return None
    if r2 == 1.0:
        return 0.0
    f_stat = (r2 / df1) / ((1.0 - r2) / df2)
    if not np.isfinite(f_stat) or f_stat < 0:
        return None
    try:
        from scipy.stats import f as f_dist  # type: ignore
    except Exception:
        return None
    return float(f_dist.sf(f_stat, df1, df2))


def _permutation_pvalue_for_test_r2(model, x_train, y_train, x_test, y_test, observed_r2, n_perm, rng):
    """Permutation test for test R2 by shuffling y_train and refitting."""
    if n_perm <= 0:
        return None
    null_ge = 0
    for _ in range(int(n_perm)):
        y_perm = rng.permutation(y_train)
        m = model.__class__(**model.get_params())
        m.fit(x_train, y_perm)
        r2 = r2_score(y_test, m.predict(x_test))
        if r2 >= observed_r2:
            null_ge += 1
    # Add-one smoothing so p-value is never exactly 0.
    return (null_ge + 1.0) / (n_perm + 1.0)


def fit_and_report(model, x, y, label, *, n_perm: int, perm_seed: int | None):
    x_train, x_test, y_train, y_test = train_test_split(
        x, y, test_size=TEST_SIZE, random_state=RANDOM_STATE
    )

    model.fit(x_train, y_train)

    train_pred = model.predict(x_train)
    test_pred = model.predict(x_test)

    train_rmse = mean_squared_error(y_train, train_pred) ** 0.5
    test_rmse = mean_squared_error(y_test, test_pred) ** 0.5

    train_r2 = r2_score(y_train, train_pred)
    test_r2 = r2_score(y_test, test_pred)

    print(f"\nModel: {model.__class__.__name__} | Target: {label}")
    if isinstance(model, LinearRegression):
        coef_str = ", ".join(f"{c:.6f}" for c in model.coef_.ravel().tolist())
        print(f"  coefs = [{coef_str}]")
        print(f"  intercept = {float(model.intercept_):.6f}")
        p = int(x_train.shape[1])
        n = int(x_train.shape[0])
        f_p = _linear_overall_f_pvalue(train_r2, n=n, p=p)
        if f_p is None:
            print("  f_test_p = (unavailable; needs scipy)")
        elif f_p == 0.0:
            # This can happen with very large n where scipy underflows to 0.
            print("  f_test_p < 1e-300")
        else:
            print(f"  f_test_p = {f_p:.6g}")

    print(f"  train_r2 = {train_r2:.6f}")
    print(f"  test_r2  = {test_r2:.6f}")
    print(f"  train_rmse = {train_rmse:.6f}")
    print(f"  test_rmse  = {test_rmse:.6f}")

    if n_perm > 0:
        rng = np.random.default_rng(perm_seed)
        p_perm = _permutation_pvalue_for_test_r2(
            model,
            x_train,
            y_train,
            x_test,
            y_test,
            observed_r2=test_r2,
            n_perm=n_perm,
            rng=rng,
        )
        print(f"  perm_test_p(test_r2) = {p_perm:.6g} (n_perm={int(n_perm)})")


def main():
    p = argparse.ArgumentParser()
    p.add_argument(
        "--model",
        choices=["linear", "hgb", "rf", "both", "all"],
        default="both",
        help="Which model(s) to run.",
    )
    p.add_argument(
        "--log-count",
        action="store_true",
        help="Use log1p(count) as the count feature.",
    )
    p.add_argument(
        "--n-perm",
        type=int,
        default=100,
        help="Number of permutations for perm_test_p(test_r2). Set 0 to disable.",
    )
    p.add_argument(
        "--perm-seed",
        type=int,
        default=0,
        help="Seed for permutation test RNG.",
    )
    args = p.parse_args()

    if not IN_TSV.exists():
        raise FileNotFoundError(
            f"Missing input file: {IN_TSV}. Run scripts/ood_text_pred/plot_stability_vs_evenness.py first."
        )

    rows = load_rows(IN_TSV)
    if len(rows) < 5:
        raise RuntimeError(f"Not enough rows in {IN_TSV} to run a train/test split.")

    print(f"Loaded {len(rows)} samples from {IN_TSV}")
    print("Features: evenness_pielou, " + ("log1p(count)" if args.log_count else "count"))

    count_feat = np.log1p(np.asarray([row["count"] for row in rows], dtype=float)) if args.log_count else np.asarray(
        [row["count"] for row in rows], dtype=float
    )
    x = np.asarray(
        [[row["evenness_pielou"], count_feat[i]] for i, row in enumerate(rows)],
        dtype=float,
    )
    y_stability = np.asarray([row["stability_score"] for row in rows], dtype=float)

    if args.model in ("linear", "both", "all"):
        fit_and_report(
            LinearRegression(),
            x,
            y_stability,
            "stability_score",
            n_perm=args.n_perm,
            perm_seed=args.perm_seed,
        )

    if args.model in ("hgb", "both", "all"):
        # Small, fast nonlinear baseline with mild regularization.
        fit_and_report(
            HistGradientBoostingRegressor(
                max_depth=3,
                learning_rate=0.05,
                max_iter=300,
                min_samples_leaf=80,
                l2_regularization=1e-3,
                random_state=RANDOM_STATE,
            ),
            x,
            y_stability,
            "stability_score",
            n_perm=args.n_perm,
            perm_seed=args.perm_seed,
        )

    if args.model in ("rf", "all"):
        # Modest RF to keep runtime reasonable on ~90k samples.
        fit_and_report(
            RandomForestRegressor(
                n_estimators=200,
                max_depth=12,
                min_samples_leaf=50,
                n_jobs=-1,
                random_state=RANDOM_STATE,
            ),
            x,
            y_stability,
            "stability_score",
            n_perm=args.n_perm,
            perm_seed=args.perm_seed,
        )


if __name__ == "__main__":
    main()
