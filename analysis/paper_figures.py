#!/usr/bin/env python3
"""Trade-off and cold-start figures as PDF + 300dpi PNG in analysis/figures/.

Usage:
    python analysis/paper_figures.py
"""
from __future__ import annotations

import csv
import os
import statistics
import sys
from collections import defaultdict
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
# FGA_SUMMARY picks another summary file, e.g. matrix_summary_final.csv
SUMMARY = Path(os.environ.get("FGA_SUMMARY", REPO_ROOT / "federatedgeneticalgorithm" / "artifacts" / "matrix_summary.csv"))
RUNS_DIR = REPO_ROOT / "federatedgeneticalgorithm" / "artifacts" / "runs"
OUT_DIR = REPO_ROOT / "analysis" / "figures"

sys.path.insert(0, str(Path(__file__).resolve().parent))
from fitness_bias import infer_missing_round1, load_windows, parse_traces  # noqa: E402

REGIMES = {
    "ga_perclient_cifar": dict(label="Per-client GA (zero coupling)", color="#2a78d6", ls="-"),
    "ga_surrogate_cifar": dict(label="Surrogate-aided (medium)", color="#eb6834", ls="-"),
    "ga_broadcast_cifar": dict(label="Server-broadcast GA (high)", color="#1baf7a", ls="-"),
}
EXPERT = "fixed_expert_cifar"
GRAY = "#6b6b66"
GRID = dict(color="#e7e7e2", linewidth=0.6)

plt.rcParams.update({
    "font.size": 8.5, "axes.titlesize": 9, "axes.labelsize": 8.5,
    "xtick.labelsize": 8, "ytick.labelsize": 8, "legend.fontsize": 7.5,
    "axes.spines.top": False, "axes.spines.right": False,
    "axes.edgecolor": "#6b6b66", "axes.linewidth": 0.8,
    "figure.dpi": 110, "savefig.bbox": "tight",
    "pdf.fonttype": 42, "ps.fonttype": 42,  # TrueType; IEEE PDF eXpress rejects Type 3 fonts
})


def rows_by_scenario() -> dict[str, list[dict]]:
    out: dict[str, list[dict]] = defaultdict(list)
    with SUMMARY.open(encoding="utf-8") as fh:
        for r in csv.DictReader(fh):
            if r.get("status") == "ok":
                out[r["scenario_name"]].append(r)
    return out


def eval_curve(run_id: str) -> list[float | None]:
    path = RUNS_DIR / run_id / "server_aggregated_rounds.csv"
    by_round: dict[int, float | None] = {}
    if path.exists():
        with path.open(encoding="utf-8") as fh:
            for row in csv.DictReader(fh):
                if row.get("phase") != "evaluate":
                    continue
                try:
                    by_round[int(row["server_round"])] = float(row.get("eval-acc") or "")
                except ValueError:
                    by_round[int(row["server_round"])] = None
    return [by_round.get(i) for i in range(1, 21)]


def curves_for(scn: str, rows: dict) -> np.ndarray:
    cs = [eval_curve(r["run_id"]) for r in rows.get(scn, [])]
    return np.array([[np.nan if v is None else v for v in c] for c in cs], dtype=float)


def drop_rounds(curve: list[float | None]) -> list[tuple[int, float]]:
    """(round, accuracy after the drop) for every >10 pp single-round drop."""
    out, prev = [], None
    for i, v in enumerate(curve):
        if v is None:
            continue
        if prev is not None and prev - v > 0.10:
            out.append((i + 1, v))
        prev = v
    return out


DESIGNS = [  # scenario, label, color, marker, filled (top to bottom in panel d)
    ("fixed_expert_cifar", "expert (baseline)", GRAY, "D", True),
    ("ga_perclient_cifar_r40", "per-client GA, 40 rounds", "#2a78d6", "o", False),
    ("ga_perclient_cifar", "per-client GA", "#2a78d6", "o", True),
    ("ga_surrogate_nopool_cifar", "surrogate GA, no shared pool", "#eb6834", "x", True),
    ("ga_surrogate_cifar", "surrogate GA", "#eb6834", "o", True),
    ("ga_broadcast_cifar_r40", "broadcast GA, 40 rounds", "#1baf7a", "o", False),
    ("ga_broadcast_cifar", "broadcast GA", "#1baf7a", "o", True),
    ("ga_broadcast_noelite_cifar", "broadcast GA, no elitism", "#1baf7a", "x", True),
    ("tpe_broadcast_cifar", "broadcast TPE", "#1baf7a", "s", True),
    ("rs_broadcast_cifar", "broadcast RS", "#1baf7a", "^", True),
    ("fixed_naive_cifar", "naive (baseline)", GRAY, "D", False),
]


def fig2(rows: dict) -> None:
    fig = plt.figure(figsize=(7.16, 2.75))
    outer = fig.add_gridspec(1, 2, width_ratios=[3.0, 2.35], wspace=0.62, left=0.065, right=0.925, bottom=0.17, top=0.86)
    left = outer[0, 0].subgridspec(1, 3, wspace=0.16)
    x = np.arange(1, 21)
    expert = np.nanmean(curves_for(EXPERT, rows) * 100, axis=0)
    panels = [("ga_perclient_cifar", "(a) per-client"), ("ga_surrogate_cifar", "(b) surrogate"),
              ("ga_broadcast_cifar", "(c) broadcast")]
    first = None
    for k, (scn, title) in enumerate(panels):
        ax = fig.add_subplot(left[0, k], sharey=first) if first else fig.add_subplot(left[0, k])
        first = first or ax
        color = REGIMES[scn]["color"]
        ax.plot(x, expert, color=GRAY, ls="--", lw=0.9, zorder=2)
        n_drops, seeds_hit = 0, 0
        for r in sorted(rows[scn], key=lambda r: int(r["seed"])):
            c = eval_curve(r["run_id"])
            ax.plot(x, [np.nan if v is None else v * 100 for v in c], color=color, lw=0.9, alpha=0.85, zorder=3)
            d = drop_rounds(c)
            n_drops += len(d); seeds_hit += bool(d)
            if d:
                ax.scatter([a for a, _ in d], [b * 100 for _, b in d], marker="v", s=16, color=color,
                           edgecolor="white", linewidth=0.6, zorder=4)
        ax.text(1.6, 97, f"{n_drops} drops, {seeds_hit}/{len(rows[scn])} seeds", ha="left", va="top",
                fontsize=6.4, color="#3a3a37")
        ax.set_title(title, loc="left", fontsize=7.8)
        ax.set_xlim(1, 20)
        ax.set_ylim(0, 100)
        ax.set_yticks([0, 20, 40, 60, 80])
        ax.set_xticks([5, 10, 15, 20])
        ax.set_xlabel("Round", fontsize=7.6)
        ax.grid(axis="y", **GRID)
        if k == 0:
            ax.set_ylabel("Eval accuracy (%)")
        else:
            plt.setp(ax.get_yticklabels(), visible=False)
        if k == 2:
            ax.text(20, expert[-1] + 1.5, "expert", ha="right", va="bottom", fontsize=6.4, color=GRAY, style="italic")

    ax = fig.add_subplot(outer[0, 1])
    n = len(DESIGNS)
    for i, (scn, label, color, marker, filled) in enumerate(DESIGNS):
        y = n - 1 - i
        rs = rows.get(scn, [])
        peaks = [float(r["peak_eval_acc"]) * 100 for r in rs]
        walls = [float(r["wall_seconds"]) / 60 for r in rs]
        face = color if filled else "white"
        ax.scatter(peaks, [y] * len(peaks), marker=marker, s=12, facecolor=face, edgecolor=color, linewidth=0.6,
                   alpha=0.6, zorder=3)
        m = statistics.fmean(peaks)
        ax.plot([m, m], [y - 0.32, y + 0.32], color=color if color != GRAY else "#3a3a37", lw=1.6, zorder=4)
        ax.text(1.03, y, f"{statistics.fmean(walls):.0f} min", transform=ax.get_yaxis_transform(), fontsize=6.4,
                va="center", ha="left", color="#3a3a37")
    ax.set_yticks(range(n))
    ax.set_yticklabels([f"{d[1]} ({len(rows.get(d[0], []))})" for d in reversed(DESIGNS)], fontsize=6.6)
    ax.set_ylim(-0.6, n - 0.4)
    ax.set_xlim(66, 87)
    ax.set_xlabel("Peak accuracy (%)")
    ax.grid(axis="x", **GRID)
    ax.tick_params(axis="y", length=0)
    ax.set_title("(d) peak per seed, bar = mean", loc="left", fontsize=7.8)
    for ext in ("pdf", "png"):
        fig.savefig(OUT_DIR / f"fig2_tradeoff.{ext}", dpi=300)
    plt.close(fig)


def gen0_candidates() -> dict[tuple[str, int], list[tuple[float, bool]]]:
    """Fitness (%) of the 4 gen-0 candidates per condition and seed, flagged True for the expert."""
    path = REPO_ROOT / "federatedgeneticalgorithm" / "artifacts" / "expert_position.csv"
    out: dict[tuple[str, int], list[tuple[float, bool]]] = defaultdict(list)
    with path.open(encoding="utf-8") as fh:
        for r in csv.DictReader(fh):
            cond = "sequential" if r["phase"] == "sequential" else r["checkpoint"]
            out[(cond, int(r["seed"]))].append((float(r["val_acc"]) * 100, r["is_expert"] == "1"))
    return out


def fig3() -> None:
    windows = load_windows()
    traces = parse_traces(windows)
    infer_missing_round1(traces)
    fig, (ax, bx) = plt.subplots(2, 1, figsize=(3.5, 5.4), gridspec_kw={"height_ratios": [1, 1.45], "hspace": 0.55})
    seeded_first, series = [], []
    for (scn, seed), tr in sorted(traces.items()):
        if scn == "ga_broadcast_deltafitness_cifar":
            continue
        gen0 = [r["fitness"] * 100 for r in tr if r["gen"] == 0][:4]
        if len(gen0) < 4:
            continue
        series.append(gen0)
        ax.plot(range(1, 5), gen0, color="#b9b9b3", lw=0.8, zorder=2)
        if scn == "ga_broadcast_cifar":
            seeded_first.append(gen0[0])
    mean = np.mean(np.array(series), axis=0)
    ax.plot(range(1, 5), mean, color="#1baf7a", lw=2.0, zorder=4, marker="o", ms=4.5)
    ax.text(4.08, mean[3], f"mean of\n{len(series)} runs", color="#15875f", fontsize=6.8, fontweight="bold", va="center")
    ax.scatter([1] * len(seeded_first), seeded_first, marker="D", s=20, color="#eb6834", edgecolor="white",
               linewidth=0.5, zorder=5)
    ax.text(1.12, 1.5, "expert (seeded, position 1)", color="#c4541f", fontsize=6.8, va="bottom")
    ax.set_xticks([1, 2, 3, 4])
    ax.set_xlim(0.8, 4.7)
    ax.set_xlabel("Evaluation position in generation 0", fontsize=7.8)
    ax.set_ylabel("Fitness (%)")
    ax.set_ylim(0, 80)
    ax.grid(axis="y", **GRID)
    ax.set_title("(a) gen-0 fitness by evaluation order", loc="left", fontsize=8.2)

    cands = gen0_candidates()
    conds = [("sequential", "evaluated in sequence, as in the GA"),
             ("cold", "each trained from the initial model"),
             ("warm", "each trained from the model after gen. 0")]
    ordinal = {1: "1st", 2: "2nd", 3: "3rd", 4: "4th"}
    y, yticks, ylabels = 0.0, [], []
    for key, header in conds:
        bx.text(0, y - 0.85, header, fontsize=7.2, color="#3a3a37", va="center")
        for seed in sorted(s for c, s in cands if c == key):
            vals = cands[(key, seed)]
            expert = next(v for v, is_exp in vals if is_exp)
            others = [v for v, is_exp in vals if not is_exp]
            bx.plot([min(v for v, _ in vals), max(v for v, _ in vals)], [y, y], color="#d9d9d4", lw=0.9, zorder=2)
            bx.scatter(others, [y] * len(others), s=16, color="#a3a39d", edgecolor="white", linewidth=0.5, zorder=3)
            bx.scatter([expert], [y], marker="D", s=26, color="#eb6834", edgecolor="white", linewidth=0.5, zorder=4)
            rank = 1 + sum(v > expert for v, _ in vals)
            bx.text(87, y, ordinal[rank], fontsize=7, va="center", ha="center", color="#3a3a37")
            yticks.append(y)
            ylabels.append(f"seed {seed}")
            y += 1
        y += 1.3
    bx.text(87, -0.85, "rank", fontsize=6.8, ha="center", va="center", color="#6b6b66", style="italic")
    bx.set_yticks(yticks)
    bx.set_yticklabels(ylabels, fontsize=6.8)
    bx.tick_params(axis="y", length=0)
    bx.set_ylim(y - 1.8, -1.5)
    bx.set_xlim(0, 92)
    bx.set_xticks([0, 20, 40, 60, 80])
    bx.set_xlabel("Fitness (%)", fontsize=7.8)
    bx.grid(axis="x", **GRID)
    bx.spines["left"].set_visible(False)
    bx.legend(handles=[
        plt.Line2D([], [], marker="D", ls="", color="#eb6834", markeredgecolor="white", ms=5, label="expert (its rank at right)"),
        plt.Line2D([], [], marker="o", ls="", color="#a3a39d", markeredgecolor="white", ms=4.5, label="other gen-0 candidates"),
    ], loc="upper center", bbox_to_anchor=(0.45, -0.2), ncol=2, frameon=False, fontsize=7, handletextpad=0.3)
    bx.set_title("(b) the same four candidates, evaluated three ways", loc="left", fontsize=8.2, pad=6)
    for ext in ("pdf", "png"):
        fig.savefig(OUT_DIR / f"fig3_coldstart.{ext}", dpi=300)
    plt.close(fig)


def main() -> int:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    rows = rows_by_scenario()
    fig2(rows)
    fig3()
    print(f"[figures] wrote fig2_tradeoff + fig3_coldstart into {OUT_DIR}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
