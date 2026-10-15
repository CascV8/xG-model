"""Turn play-by-play into one row per unblocked shot with model features.

Coordinates are rotated so every shooter attacks toward +x, with the target
net at (89, 0). Blocked shots are excluded (the NHL records the block
location, not the shot location), as are shootouts, penalty shots and
shots at an empty net (near-certain goals that would only flatter the metrics).
"""

from __future__ import annotations

import numpy as np
import pandas as pd

NET_X = 89.0
BLUE_LINE_X = 25.0
UNBLOCKED = ("shot-on-goal", "missed-shot", "goal")

NUMERIC = [
    "distance", "angle", "x", "y_abs",
    "period", "period_seconds", "score_diff",
    "shooter_skaters", "opp_skaters", "own_goalie_pulled", "is_home",
    "secs_since_prev", "prev_x", "prev_y_abs", "dist_from_prev", "speed_from_prev",
    "prev_same_team", "rebound", "rebound_angle_change", "rush",
]
CATEGORICAL = ["shot_type", "prev_event_type"]
FEATURES = NUMERIC + CATEGORICAL

# Fixed category levels so train and test encode identically.
SHOT_TYPES = ["wrist", "snap", "slap", "backhand", "tip-in", "deflected", "wrap-around",
              "poke", "bat", "between-legs", "cradle"]
PREV_EVENTS = ["faceoff", "shot-on-goal", "missed-shot", "blocked-shot", "goal", "hit",
               "giveaway", "takeaway", "penalty", "stoppage", "delayed-penalty", "period-start"]


def _angle(dx, dy):
    """Degrees off the net's centre line; >90 means behind the goal line."""
    return np.degrees(np.arctan2(np.abs(dy), dx))


def build_shots(pbp: pd.DataFrame) -> pd.DataFrame:
    df = pbp.sort_values(["game_id", "sort_order"]).reset_index(drop=True)
    # NA-aware string dtype makes `==` return NA; plain objects compare False.
    df = df.astype({c: object for c in df.select_dtypes(["string", "str"]).columns})
    df = df[df["period_type"] != "SO"]

    # --- score before each event --------------------------------------------
    goal = df["event_type"] == "goal"
    home_goal = (goal & (df["event_team"] == df["home_team"])).astype(int)
    away_goal = (goal & (df["event_team"] == df["away_team"])).astype(int)
    g = df["game_id"]
    df["home_score_before"] = home_goal.groupby(g).cumsum() - home_goal
    df["away_score_before"] = away_goal.groupby(g).cumsum() - away_goal

    # --- rotate every event into its own team's attacking frame -------------
    is_home = df["event_team"] == df["home_team"]
    home_attacks_right = df["home_team_defending_side"] == "left"
    sign = np.where(is_home == home_attacks_right, 1.0, -1.0)
    df["is_home"] = is_home.astype(int)
    df["sign"] = sign

    # --- previous event in the same game and period -------------------------
    grp = df.groupby(["game_id", "period"], sort=False)
    df["prev_event_type"] = grp["event_type"].shift()
    df["prev_team"] = grp["event_team"].shift()
    df["prev_seconds"] = grp["period_seconds"].shift()
    prev_x_raw = grp["x_coord"].shift()
    prev_y_raw = grp["y_coord"].shift()

    shots = df[df["event_type"].isin(UNBLOCKED) & df["x_coord"].notna() & df["y_coord"].notna()].copy()

    # --- strength from the situation code "AGAS HSHG" -> away goalie, away skaters, home skaters, home goalie
    code = shots["situation_code"].str.zfill(4)
    away_g, away_s = pd.to_numeric(code.str[0]), pd.to_numeric(code.str[1])
    home_s, home_g = pd.to_numeric(code.str[2]), pd.to_numeric(code.str[3])
    home = shots["is_home"] == 1
    shots["shooter_skaters"] = np.where(home, home_s, away_s)
    shots["opp_skaters"] = np.where(home, away_s, home_s)
    empty_net = np.where(home, away_g, home_g) == 0
    shots["own_goalie_pulled"] = (np.where(home, home_g, away_g) == 0).astype(int)
    # Penalty shots (1 skater) and malformed codes
    shots = shots[(shots["shooter_skaters"] >= 3) & (shots["opp_skaters"] >= 3) & ~empty_net]

    # --- location -----------------------------------------------------------
    s = shots["sign"]
    shots["x"] = shots["x_coord"] * s
    y = shots["y_coord"] * s
    shots["y_abs"] = y.abs()
    shots["distance"] = np.hypot(NET_X - shots["x"], y)
    shots["angle"] = _angle(NET_X - shots["x"], y)

    # --- what happened just before ------------------------------------------
    px = prev_x_raw.loc[shots.index] * s
    py = prev_y_raw.loc[shots.index] * s
    shots["prev_x"] = px
    shots["prev_y_abs"] = py.abs()
    shots["secs_since_prev"] = (shots["period_seconds"] - shots["prev_seconds"]).astype("float")
    shots["dist_from_prev"] = np.hypot(shots["x"] - px, y - py)
    shots["speed_from_prev"] = shots["dist_from_prev"] / shots["secs_since_prev"].clip(lower=1)
    shots["prev_same_team"] = (shots["prev_team"] == shots["event_team"]).astype(int)

    prev_was_shot = shots["prev_event_type"].isin(UNBLOCKED + ("blocked-shot",))
    quick = shots["secs_since_prev"] <= 3
    shots["rebound"] = (prev_was_shot & quick).astype(int)
    shots["rebound_angle_change"] = np.where(
        shots["rebound"] == 1, np.abs(np.degrees(np.arctan2(y, NET_X - shots["x"])) - np.degrees(np.arctan2(py, NET_X - px))), 0.0)
    shots["rush"] = ((shots["secs_since_prev"] <= 4) & (px < BLUE_LINE_X) & (shots["rebound"] == 0)).astype(int)

    # --- game state -----------------------------------------------------------
    diff = shots["home_score_before"] - shots["away_score_before"]
    shots["score_diff"] = np.where(shots["is_home"] == 1, diff, -diff).clip(-3, 3)

    shots["shot_type"] = pd.Categorical(shots["shot_type"], categories=SHOT_TYPES)
    shots["prev_event_type"] = pd.Categorical(shots["prev_event_type"], categories=PREV_EVENTS)
    shots["is_goal"] = (shots["event_type"] == "goal").astype(int)
    shots["shooter_id"] = shots["shooting_player_id"].fillna(shots["scoring_player_id"])
    shots["shooter"] = shots["shooting_player_name"].fillna(shots["scoring_player_name"])

    for col in NUMERIC:
        shots[col] = shots[col].astype("float64")

    keep = ["game_id", "season", "game_type", "event_id", "period_type", "game_seconds",
            "event_team", "home_team", "away_team", "shooter_id", "shooter", "goalie_in_net_id",
            "event_type", "is_goal"] + FEATURES
    return shots[keep].reset_index(drop=True)
