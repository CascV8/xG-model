"""Score shots with the trained model."""

from __future__ import annotations

import joblib
import pandas as pd
import puckscrape as ps

from .collect import slim
from .features import build_shots
from .train import MODEL_PATH

_bundle = None


def load_model():
    global _bundle
    if _bundle is None:
        if not MODEL_PATH.exists():
            raise FileNotFoundError(f"{MODEL_PATH} not found; run `xg-model train` first")
        _bundle = joblib.load(MODEL_PATH)
    return _bundle


def score(pbp: pd.DataFrame) -> pd.DataFrame:
    """Unblocked shots from a puckscrape play-by-play frame, with an ``xg`` column."""
    bundle = load_model()
    shots = build_shots(slim(pbp)) if len(pbp) else pd.DataFrame(columns=bundle["features"])
    shots["xg"] = bundle["model"].predict_proba(shots[bundle["features"]])[:, 1] if len(shots) else []
    return shots


def game_report(game_id: int) -> None:
    shots = score(ps.nhl.play_by_play(game_id))
    if shots.empty:
        print(f"No shots recorded for game {game_id} yet (not started, or no play-by-play).")
        return
    away, home = shots["away_team"].iloc[0], shots["home_team"].iloc[0]
    print(f"\n{away} @ {home}  (game {game_id})\n")

    teams = shots.groupby("event_team").agg(goals=("is_goal", "sum"), xg=("xg", "sum"), shots=("xg", "size"))
    print(teams.loc[[away, home]].round(2).to_string(), "\n")

    print("Best chances:")
    cols = ["event_team", "shooter", "shot_type", "distance", "rebound", "rush", "event_type", "xg"]
    print(shots.nlargest(8, "xg")[cols].round(2).to_string(index=False), "\n")

    print("Shooters:")
    players = shots.groupby(["event_team", "shooter"]).agg(shots=("xg", "size"), goals=("is_goal", "sum"), xg=("xg", "sum"))
    print(players.sort_values("xg", ascending=False).head(10).round(2).to_string())
