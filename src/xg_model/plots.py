"""Report charts. Colors follow the reference data-viz palette (light mode)."""

from __future__ import annotations

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.colors import LinearSegmentedColormap

from .features import FEATURES, NET_X, PREV_EVENTS, SHOT_TYPES

SURFACE = "#fcfcfb"
INK = "#0b0b0b"
INK_2 = "#52514e"
MUTED = "#898781"
GRID = "#e1e0d9"
AXIS = "#c3c2b7"
MODEL = "#2a78d6"       # categorical slot 1
MONEYPUCK = "#eb6834"   # categorical slot 2
BLUES = LinearSegmentedColormap.from_list(
    "blues", ["#cde2fb", "#9ec5f4", "#6da7ec", "#3987e5", "#256abf", "#184f95", "#0d366b"])

plt.rcParams.update({
    "figure.facecolor": SURFACE, "axes.facecolor": SURFACE, "savefig.facecolor": SURFACE,
    "font.family": ["Segoe UI", "DejaVu Sans", "sans-serif"], "font.size": 10,
    "text.color": INK, "axes.labelcolor": INK_2, "axes.edgecolor": AXIS,
    "xtick.color": MUTED, "ytick.color": MUTED, "axes.titlesize": 13, "axes.titleweight": "semibold",
    "axes.titlelocation": "left", "axes.spines.top": False, "axes.spines.right": False,
    "grid.color": GRID, "grid.linewidth": 0.8,
})


def _binned(y, p, bins: int = 20) -> pd.DataFrame:
    df = pd.DataFrame({"y": np.asarray(y), "p": np.asarray(p)})
    df["bin"] = pd.qcut(df["p"], bins, duplicates="drop")
    return df.groupby("bin", observed=True).agg(pred=("p", "mean"), obs=("y", "mean"), n=("y", "size"))


def calibration(y, p, p_moneypuck, path) -> None:
    """Predicted vs observed goal rate in 20 equal-count bins."""
    fig, ax = plt.subplots(figsize=(7, 6))
    hi = 0.0
    series = [("This model", p, MODEL)]
    if p_moneypuck is not None:
        m = p_moneypuck.notna().to_numpy()
        series = [("This model", np.asarray(p)[m], MODEL), ("MoneyPuck", p_moneypuck.to_numpy()[m], MONEYPUCK)]
        y = np.asarray(y)[m]
    for label, probs, color in series:
        b = _binned(y, probs)
        ax.plot(b["pred"], b["obs"], color=color, lw=2, marker="o", ms=5,
                markeredgecolor=SURFACE, markeredgewidth=1.5, label=label, zorder=3)
        ax.annotate(label, (b["pred"].iloc[-1], b["obs"].iloc[-1]), xytext=(8, 0),
                    textcoords="offset points", va="center", color=INK_2, fontsize=9)
        hi = max(hi, b["pred"].max(), b["obs"].max())
    lim = hi * 1.08
    ax.plot([0, lim], [0, lim], color=MUTED, lw=1, ls=(0, (4, 3)), zorder=1)
    ax.annotate("Perfect calibration", (lim * 0.62, lim * 0.62), rotation=0, xytext=(6, -14),
                textcoords="offset points", color=MUTED, fontsize=9)
    ax.set(xlim=(0, lim * 1.18), ylim=(0, lim), xlabel="Predicted goal probability (bin mean)",
           ylabel="Actual goal rate")
    ax.grid(True)
    ax.legend(frameon=False, loc="upper left", labelcolor=INK_2)
    ax.set_title("Calibration on the held-out season")
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def importance(imp: pd.Series, path) -> None:
    """Permutation importance: how much log loss worsens when a feature is shuffled."""
    imp = imp.sort_values()
    fig, ax = plt.subplots(figsize=(7, 0.32 * len(imp) + 1.2))
    ax.barh(imp.index, imp.to_numpy(), height=0.6, color=MODEL)
    ax.axvline(0, color=AXIS, lw=1)
    ax.grid(True, axis="x")
    ax.set_axisbelow(True)
    ax.tick_params(axis="y", colors=INK_2, length=0)
    ax.spines["left"].set_visible(False)
    ax.set_xlabel("Increase in log loss when shuffled")
    ax.set_title("Which features the model leans on")
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def xg_map(model, path) -> None:
    """The model's xG surface for a 5v5 wrist shot with no rebound or rush."""
    xs = np.arange(25, 100, 1.0)
    ys = np.arange(-42, 43, 1.0)
    gx, gy = np.meshgrid(xs, ys)
    grid = pd.DataFrame({"x": gx.ravel(), "y": gy.ravel()})
    dx = NET_X - grid["x"]
    row = {
        "distance": np.hypot(dx, grid["y"]), "angle": np.degrees(np.arctan2(grid["y"].abs(), dx)),
        "x": grid["x"], "y_abs": grid["y"].abs(), "period": 2.0, "period_seconds": 600.0, "score_diff": 0.0,
        "shooter_skaters": 5.0, "opp_skaters": 5.0, "own_goalie_pulled": 0.0, "is_home": 1.0,
        "secs_since_prev": 15.0, "prev_x": 0.0, "prev_y_abs": 20.0, "dist_from_prev": np.hypot(grid["x"], grid["y"] - 20),
        "prev_same_team": 0.0, "rebound": 0.0, "rebound_angle_change": 0.0, "rush": 0.0,
    }
    X = pd.DataFrame(row)
    X["speed_from_prev"] = X["dist_from_prev"] / 15.0
    X["shot_type"] = pd.Categorical(["wrist"] * len(X), categories=SHOT_TYPES)
    X["prev_event_type"] = pd.Categorical(["faceoff"] * len(X), categories=PREV_EVENTS)
    z = model.predict_proba(X[FEATURES])[:, 1].reshape(gx.shape)

    fig, ax = plt.subplots(figsize=(7.5, 6.6))
    mesh = ax.pcolormesh(gx, gy, z, cmap=BLUES, shading="nearest", vmin=0, vmax=np.quantile(z, 0.995))
    # Rink markings: blue line, goal line, crease, net, boards
    ax.axvline(25, color=MUTED, lw=1.5)
    ax.axvline(NET_X, color=INK_2, lw=1)
    ax.add_patch(plt.Rectangle((NET_X, -3), 3.3, 6, fill=False, ec=INK, lw=1.5))
    ax.add_patch(matplotlib.patches.Arc((NET_X, 0), 12, 12, theta1=90, theta2=270, ec=INK_2, lw=1))
    ax.set(xlim=(25, 100), ylim=(-42.5, 42.5), aspect="equal")
    ax.set_xticks([]), ax.set_yticks([])
    for s in ax.spines.values():
        s.set_visible(False)
    cb = fig.colorbar(mesh, ax=ax, fraction=0.035, pad=0.02)
    cb.set_label("Expected goals per shot", color=INK_2)
    cb.outline.set_visible(False)
    ax.set_title("xG by location: 5v5 wrist shot, no rebound or rush")
    ax.annotate("Blue line", (25, 42.5), xytext=(4, -12), textcoords="offset points", color=INK_2, fontsize=9)
    ax.annotate("Goal line", (NET_X, 42.5), xytext=(-52, -12), textcoords="offset points", color=INK_2, fontsize=9)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)
