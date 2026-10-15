import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import puckscrape.nhl as nhl

from xg_model.collect import slim
from xg_model.features import FEATURES, build_shots

FIXTURE = Path(__file__).parent / "fixtures" / "pbp_2024020001.json"


@pytest.fixture(scope="module")
def shots():
    data = json.loads(FIXTURE.read_text(encoding="utf-8"))
    original = nhl._http.get_json
    nhl._http.get_json = lambda *a, **k: data
    try:
        return build_shots(slim(nhl.play_by_play(2024020001)))
    finally:
        nhl._http.get_json = original


def test_only_unblocked_shots(shots):
    assert set(shots["event_type"]) <= {"goal", "shot-on-goal", "missed-shot"}
    assert shots["is_goal"].sum() == 4  # NJD 4, BUF 1, minus the empty-net goal


def test_shots_are_rotated_toward_positive_x(shots):
    # Nearly every shot comes from the attacking half once rotated
    assert (shots["x"] > 0).mean() > 0.95
    near = shots[shots["distance"] < 20]
    assert (near["x"] > 60).all()


def test_score_state_is_from_shooters_perspective(shots):
    goals = shots[shots["is_goal"] == 1].sort_values("game_seconds")
    # NJD go 1-0, 2-0, 3-0; BUF score at 0-3 (NJD's 4-1 goal was into an empty net)
    assert goals["score_diff"].tolist() == [0, 1, 2, -3]
    assert goals["event_team"].tolist() == ["NJD", "NJD", "NJD", "BUF"]


def test_geometry(shots):
    assert np.allclose(shots["distance"], np.hypot(89 - shots["x"], shots["y_abs"]))
    assert shots["angle"].between(0, 180).all()


def test_feature_frame_is_model_ready(shots):
    X = shots[FEATURES]
    assert isinstance(X["shot_type"].dtype, pd.CategoricalDtype)
    assert X.drop(columns=["shot_type", "prev_event_type"]).dtypes.eq("float64").all()


def test_empty_net_shots_excluded(shots):
    # The model is trained only on shots at a goalie; the empty_net feature is gone
    assert "empty_net" not in FEATURES
    assert len(shots) == 89  # 90 unblocked shots, one of them at an empty net
