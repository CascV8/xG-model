"""Fit the xG model, evaluate it on a held-out season, write reports/."""

from __future__ import annotations

import json
import logging

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.inspection import permutation_importance
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import brier_score_loss, log_loss, roc_auc_score

from . import collect, plots
from .features import FEATURES, build_shots

log = logging.getLogger("xg_model.train")

MODELS = collect.ROOT / "models"
REPORTS = collect.ROOT / "reports"
MODEL_PATH = MODELS / "xg.joblib"


def make_model() -> HistGradientBoostingClassifier:
    return HistGradientBoostingClassifier(
        learning_rate=0.05,
        max_iter=1000,
        max_leaf_nodes=31,
        min_samples_leaf=200,
        l2_regularization=1.0,
        categorical_features="from_dtype",
        early_stopping=True,
        validation_fraction=0.1,
        n_iter_no_change=30,
        random_state=0,
    )


def shots_for(seasons) -> pd.DataFrame:
    return build_shots(collect.load(seasons))


def metrics(y, p) -> dict:
    return {
        "shots": int(len(y)),
        "goals": int(y.sum()),
        "xg": round(float(p.sum()), 1),
        "log_loss": round(log_loss(y, p), 4),
        "auc": round(roc_auc_score(y, p), 4),
        "brier": round(brier_score_loss(y, p), 5),
    }


def moneypuck_xg(test: pd.DataFrame, season) -> pd.Series:
    """MoneyPuck's xG for the same shots, matched on game, game time and shooter."""
    import puckscrape as ps

    cache = collect.DATA / f"moneypuck_shots_{season}.parquet"
    if cache.exists():
        mp = pd.read_parquet(cache)
    else:
        mp = ps.moneypuck.shots(season)
        mp.to_parquet(cache)
    mp = mp.loc[mp["isPlayoffGame"] == 0, ["game_id", "time", "shooterPlayerId", "xGoal"]]
    mp = mp.assign(game_key=mp["game_id"].astype("int64"), time=mp["time"].astype("int64"),
                   shooter_id=mp["shooterPlayerId"].astype("float64"))
    mp = mp.drop_duplicates(["game_key", "time", "shooter_id"])[["game_key", "time", "shooter_id", "xGoal"]]

    keys = pd.DataFrame({
        "game_key": (test["game_id"] % 1_000_000).astype("int64"),
        "time": test["game_seconds"].astype("int64"),
        "shooter_id": test["shooter_id"].astype("float64"),
    })
    return keys.merge(mp, how="left", on=["game_key", "time", "shooter_id"])["xGoal"].set_axis(test.index)


def train(train_seasons, test_season, benchmark: bool = True) -> dict:
    log.info("Building features: train %s, test %s", train_seasons, test_season)
    train_df = shots_for(train_seasons)
    test_df = shots_for(test_season)
    log.info("%d training shots (%.1f%% goals), %d test shots",
             len(train_df), 100 * train_df["is_goal"].mean(), len(test_df))

    X, y = train_df[FEATURES], train_df["is_goal"]
    Xt, yt = test_df[FEATURES], test_df["is_goal"]

    log.info("Fitting gradient boosting model")
    model = make_model().fit(X, y)
    log.info("Stopped after %d iterations", model.n_iter_)
    p = model.predict_proba(Xt)[:, 1]

    log.info("Fitting distance + angle baseline")
    base_cols = ["distance", "angle"]
    baseline = LogisticRegression(max_iter=1000).fit(X[base_cols], y)
    p_base = baseline.predict_proba(Xt[base_cols])[:, 1]

    report = {
        "train_seasons": [str(s) for s in train_seasons],
        "test_season": str(test_season),
        "test": {"model": metrics(yt, p), "baseline_distance_angle": metrics(yt, p_base)},
    }

    mp = None
    if benchmark:
        mp = moneypuck_xg(test_df, test_season)
        matched = mp.notna()
        log.info("Matched %.1f%% of test shots to MoneyPuck", 100 * matched.mean())
        report["moneypuck_benchmark"] = {
            "matched_share": round(float(matched.mean()), 4),
            "model": metrics(yt[matched], p[matched]),
            "moneypuck": metrics(yt[matched], mp[matched]),
        }

    log.info("Permutation importance on a 40k-shot sample")
    sample = test_df.sample(min(40_000, len(test_df)), random_state=0)
    imp = permutation_importance(model, sample[FEATURES], sample["is_goal"], scoring="neg_log_loss",
                                 n_repeats=3, random_state=0)
    importance = pd.Series(imp.importances_mean, index=FEATURES).sort_values(ascending=False)
    report["feature_importance_log_loss"] = importance.round(5).to_dict()

    MODELS.mkdir(exist_ok=True)
    REPORTS.mkdir(exist_ok=True)
    joblib.dump({"model": model, "features": FEATURES, "trained_on": report["train_seasons"]}, MODEL_PATH)
    (REPORTS / "metrics.json").write_text(json.dumps(report, indent=2))

    plots.calibration(yt, p, mp, REPORTS / "calibration.png")
    plots.importance(importance, REPORTS / "feature_importance.png")
    plots.xg_map(model, REPORTS / "xg_map.png")

    test_df.assign(xg=p, xg_moneypuck=mp).to_parquet(REPORTS / f"test_shots_{test_season}.parquet")
    log.info("Saved %s and reports/:\n%s", MODEL_PATH, json.dumps(report["test"], indent=2))
    if benchmark:
        log.info("vs MoneyPuck:\n%s", json.dumps(report["moneypuck_benchmark"], indent=2))
    return report
