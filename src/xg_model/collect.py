"""Download and cache NHL play-by-play with puckscrape.

Each game is cached as its own parquet file under data/pbp/<season>/, so an
interrupted download resumes where it left off.
"""

from __future__ import annotations

import logging
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import pandas as pd
import puckscrape as ps
from puckscrape._util import start_year

log = logging.getLogger("xg_model.collect")

# Paths are anchored to the project folder, so commands work from any directory.
ROOT = Path(__file__).resolve().parents[2]
DATA = ROOT / "data"

# Everything the feature builder needs; the rest of the play-by-play is dropped.
KEEP = {
    "game_id": "int64", "season": "int64", "game_type": "int64",
    "away_team": "string", "home_team": "string",
    "event_id": "int64", "sort_order": "int64",
    "period": "int64", "period_type": "string", "period_seconds": "Int64", "game_seconds": "Int64",
    "situation_code": "string", "home_team_defending_side": "string",
    "event_type": "string", "event_team": "string", "zone_code": "string",
    "x_coord": "float64", "y_coord": "float64", "shot_type": "string",
    "shooting_player_id": "Int64", "scoring_player_id": "Int64", "goalie_in_net_id": "Int64",
    "shooting_player_name": "string", "scoring_player_name": "string",
}


def slim(pbp: pd.DataFrame) -> pd.DataFrame:
    """Keep the columns the model uses, with stable dtypes across games."""
    out = pd.DataFrame(index=pbp.index)
    for col, dtype in KEEP.items():
        out[col] = (pbp[col] if col in pbp else pd.Series(pd.NA, index=pbp.index)).astype(dtype)
    return out


def game_ids(season, game_types=(2,)) -> list[int]:
    """Completed games in a season (regular season by default)."""
    y = start_year(season)
    path = DATA / f"schedule_{y}.parquet"
    if path.exists():
        sched = pd.read_parquet(path)
    else:
        sched = ps.nhl.schedule(f"{y}-09-15", f"{y + 1}-06-30")
        sched = sched[sched["season"] == int(f"{y}{y + 1}")].drop_duplicates("game_id")
        # Only cache once the season is over, so reruns mid-season pick up new games.
        if sched["game_state"].isin(["OFF", "FINAL"]).all():
            path.parent.mkdir(parents=True, exist_ok=True)
            sched.to_parquet(path)
    done = sched["game_state"].isin(["OFF", "FINAL"]) & sched["game_type"].isin(game_types)
    return sorted(sched.loc[done, "game_id"].astype(int))


def _fetch(game_id: int, path: Path) -> int:
    df = slim(ps.nhl.play_by_play(game_id))
    df.to_parquet(path)
    return len(df)


def collect(season, workers: int = 8, game_types=(2,)) -> Path:
    """Download every completed game of a season that isn't cached yet."""
    y = start_year(season)
    folder = DATA / "pbp" / str(y)
    folder.mkdir(parents=True, exist_ok=True)
    todo = [g for g in game_ids(y, game_types) if not (folder / f"{g}.parquet").exists()]
    log.info("%d-%02d: %d games to download", y, (y + 1) % 100, len(todo))

    failed = []
    with ThreadPoolExecutor(workers) as pool:
        futures = {pool.submit(_fetch, g, folder / f"{g}.parquet"): g for g in todo}
        for i, fut in enumerate(as_completed(futures), 1):
            try:
                fut.result()
            except Exception as e:  # one bad game shouldn't sink the season
                failed.append(futures[fut])
                log.warning("game %s failed: %s", futures[fut], e)
            if i % 100 == 0:
                log.info("%d-%02d: %d/%d", y, (y + 1) % 100, i, len(todo))
    if failed:
        log.warning("%d games failed; rerun to retry: %s", len(failed), failed[:10])
    return folder


def load(seasons) -> pd.DataFrame:
    """All cached play-by-play for the given seasons."""
    frames = []
    for season in ps._util.as_list(seasons):
        files = sorted((DATA / "pbp" / str(start_year(season))).glob("*.parquet"))
        if not files:
            raise FileNotFoundError(f"no cached games for {season}; run `xg-model collect {season}` first")
        frames.extend(pd.read_parquet(f) for f in files)
    return pd.concat(frames, ignore_index=True)
