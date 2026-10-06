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
from matplotlib.lines import Line2D

REPO_ROOT = Path(__file__).resolve().parents[1]
# the runs the paper reports; FGA_SUMMARY picks another summary file, e.g. matrix_summary.csv
SUMMARY = Path(os.environ.get("FGA_SUMMARY", REPO_ROOT / "federatedgeneticalgorithm" / "artifacts" / "matrix_summary_final.csv"))
RUNS_DIR = REPO_ROOT / "federatedgeneticalgorithm" / "artifacts" / "runs"
OUT_DIR = REPO_ROOT / "analysis" / "figures"

sys.path.insert(0, str(Path(__file__).resolve().parent))
from fitness_bias import infer_missing_round1, load_windows, parse_traces  # noqa: E402

# categorical hues checked for color-vision deficiency; the expert is dark gray in every figure
BLUE, ORANGE, GREEN, PURPLE, MAGENTA = "#2a78d6", "#eb6834", "#1baf7a", "#6a4c93", "#c2185b"
GRAY, DARK, MID, LIGHT, INK = "#6b6b66", "#3a3a37", "#9a9a94", "#b4b4ad", "#2b2b28"
COLORS = {"ga_perclient_cifar": BLUE, "ga_surrogate_cifar": ORANGE, "ga_broadcast_cifar": GREEN, "fedex_cifar": PURPLE}
EXPERT = "fixed_expert_cifar"
GRID = dict(color="#e7e7e2", linewidth=0.6)

plt.rcParams.update({
    "font.family": "STIXGeneral", "mathtext.fontset": "stix",  # Times-like, ships with matplotlib
    "font.size": 8, "axes.titlesize": 8.5, "axes.labelsize": 8.5,
    "xtick.labelsize": 8, "ytick.labelsize": 8, "legend.fontsize": 7.8,
    "axes.spines.top": False, "axes.spines.right": False,
    "axes.edgecolor": GRAY, "axes.linewidth": 0.8,
    "figure.dpi": 110, "savefig.bbox": "tight",
    "pdf.fonttype": 42, "ps.fonttype": 42,  # TrueType; IEEE PDF eXpress rejects Type 3 fonts
})


def save(fig, name: str) -> None:
    fig.savefig(OUT_DIR / f"{name}.pdf", metadata={"CreationDate": None})  # no timestamp: reruns give the same file
    fig.savefig(OUT_DIR / f"{name}.png", dpi=300)
    plt.close(fig)


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


# panel (e): scenario, label, color, marker, filled, label position (min, %) and alignment.
# Label positions are placed by hand for these runs; hollow marks are the 40-round runs.
TRADEOFF = [
    ("fixed_expert_cifar", "Expert", DARK, "D", True, (43, 84.4), "right"),
    ("fixed_expert_cifar_r40", None, DARK, "D", False, None, None),
    ("fedex_cifar", "FedEx", PURPLE, "o", True, (43, 81.1), "right"),
    ("ga_surrogate_cifar", "Surrogate GA", ORANGE, "o", True, (70, 79.7), "left"),
    ("ga_broadcast_cifar", "Broadcast GA", GREEN, "o", True, (70, 78.5), "left"),
    ("fixed_naive_cifar", "Naive", MID, "D", True, (70, 77.3), "left"),
    ("tpe_broadcast_cifar", "TPE", GREEN, "s", True, (70, 76.1), "left"),
    ("rs_broadcast_cifar", "RS", GREEN, "^", True, (59, 74.4), "left"),
    ("ga_broadcast_cifar_r40", None, GREEN, "o", False, None, None),
    ("ga_perclient_cifar", "Per-client GA", BLUE, "o", True, (141.6, 81.2), "center"),
    ("ga_perclient_cifar_r40", None, BLUE, "o", False, None, None),
    ("ga_surrogate_longeval_cifar", "Surrogate GA,\nlong evaluation", ORANGE, "s", False, (140, 86.9), "left"),
    ("fedpop_cifar", "FedPop", MAGENTA, "o", True, (249.2, 81.7), "center"),
]
HORIZON = [("fixed_expert_cifar", "fixed_expert_cifar_r40"), ("ga_perclient_cifar", "ga_perclient_cifar_r40"),
           ("ga_broadcast_cifar", "ga_broadcast_cifar_r40")]


def fig2(rows: dict) -> None:
    fig = plt.figure(figsize=(7.16, 2.85))
    outer = fig.add_gridspec(1, 2, width_ratios=[4.15, 2.65], wspace=0.3, left=0.06, right=0.99, bottom=0.2, top=0.88)
    left = outer[0, 0].subgridspec(1, 4, wspace=0.12)
    x = np.arange(1, 21)
    expert = np.nanmean(curves_for(EXPERT, rows) * 100, axis=0)
    panels = [("ga_perclient_cifar", "(a) Per-client GA"), ("ga_surrogate_cifar", "(b) Surrogate GA"),
              ("ga_broadcast_cifar", "(c) Broadcast GA"), ("fedex_cifar", "(d) FedEx")]
    axes = []
    for k, (scn, title) in enumerate(panels):
        ax = fig.add_subplot(left[0, k], sharey=axes[0] if axes else None)
        axes.append(ax)
        color = COLORS[scn]
        ax.plot(x, expert, color=DARK, ls=(0, (3, 1.5)), lw=0.9, zorder=2)
        n_drops, seeds_hit = 0, 0
        for r in sorted(rows[scn], key=lambda r: int(r["seed"])):
            c = eval_curve(r["run_id"])
            ax.plot(x, [np.nan if v is None else v * 100 for v in c], color=color, lw=0.9, alpha=0.85, zorder=3)
            d = drop_rounds(c)
            n_drops += len(d); seeds_hit += bool(d)
            if d:
                ax.scatter([a for a, _ in d], [b * 100 for _, b in d], marker="v", s=20, color=color,
                           edgecolor="white", linewidth=0.4, zorder=4)
        ax.text(1.8, 99, f"{n_drops} drop{'' if n_drops == 1 else 's'}\n{seeds_hit}/{len(rows[scn])} seeds",
                ha="left", va="top", fontsize=7.8, color=INK, linespacing=1.05)
        ax.set_title(title, loc="left")
        ax.set_xlim(1, 20)
        ax.set_ylim(0, 100)
        ax.set_yticks([0, 20, 40, 60, 80])
        ax.set_xticks([5, 10, 15, 20])
        ax.grid(axis="y", **GRID)
        if k:
            plt.setp(ax.get_yticklabels(), visible=False)
    axes[0].set_ylabel("Eval accuracy (%)")
    axes[0].legend(handles=[Line2D([], [], color=DARK, ls=(0, (3, 1.5)), lw=0.9, label="Expert"),
                            Line2D([], [], marker="v", ls="", color=GRAY, mec="white", mew=0.4, ms=5,
                                   label="Drop > 10 pp")],
                   loc="lower right", frameon=False, handlelength=1.1, handletextpad=0.3, borderaxespad=0.05,
                   borderpad=0.1, labelspacing=0.25, fontsize=7.6)
    p0, p3 = axes[0].get_position(), axes[-1].get_position()
    fig.text((p0.x0 + p3.x1) / 2, p0.y0 - 0.1, "Round", ha="center", va="top", fontsize=8.5)

    ax = fig.add_subplot(outer[0, 1])
    pos = {}
    for scn, label, color, marker, filled, at, ha in TRADEOFF:
        peaks = [float(r["peak_eval_acc"]) * 100 for r in rows[scn]]
        walls = [float(r["wall_seconds"]) / 60 for r in rows[scn]]
        px, py = statistics.fmean(walls), statistics.fmean(peaks)
        pos[scn] = (px, py)
        ax.errorbar(px, py, yerr=statistics.stdev(peaks), fmt="none", ecolor=color, elinewidth=0.7, alpha=0.4, zorder=2)
        ax.scatter([px], [py], marker=marker, s=24, facecolor=color if filled else "white", edgecolor=color,
                   linewidth=1.0, zorder=4)
        if label is None:
            continue
        ax.text(*at, label, ha=ha, va="center", fontsize=7.8, color=INK, linespacing=0.95)
        if at[1] - py > 2:  # label set above the mark: a short line up to it
            ax.plot([px, px], [py + 0.5, at[1] - 1.05], color=LIGHT, lw=0.5, zorder=1)
        elif ha != "center" and (abs(at[1] - py) > 0.3 or abs(at[0] - px) > 10):  # label set beside, away from the mark
            side = -1 if ha == "right" else 1
            ax.plot([px + side * 2.5, at[0] - side * 1.5], [py, at[1]], color=LIGHT, lw=0.5, zorder=1)
    for a, b in HORIZON:  # the broadcast arrow bends to pass clear of the surrogate mark
        bend = "arc3,rad=-0.3" if a == "ga_broadcast_cifar" else "arc3,rad=0"
        ax.annotate("", xy=pos[b], xytext=pos[a], zorder=1,
                    arrowprops=dict(arrowstyle="-|>", lw=0.7, color=MID, shrinkA=4.5, shrinkB=4.5, mutation_scale=6,
                                    connectionstyle=bend))
    ax.legend(handles=[Line2D([], [], marker="o", ls="", mfc="white", mec=GRAY, mew=1.0, ms=4.5,
                              label="40 rounds (arrow from 20)")],
              loc="lower right", frameon=False, handletextpad=0.2, borderaxespad=0.1)
    ax.set_xlim(0, 280)
    ax.set_ylim(69, 89)
    ax.set_xticks([0, 50, 100, 150, 200, 250])
    ax.set_yticks([70, 75, 80, 85])
    ax.set_xlabel("Wall-time per run (min)")
    ax.set_ylabel("Peak accuracy (%)")
    ax.grid(**GRID)
    ax.set_title("(e) Peak accuracy and wall-time", loc="left")
    save(fig, "fig2_tradeoff")


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
    fig, (ax, bx) = plt.subplots(2, 1, figsize=(3.72, 4.07), gridspec_kw={"height_ratios": [1, 1.3], "hspace": 0.62})
    seeded_first, series = [], []
    for (scn, seed), tr in sorted(traces.items()):
        if scn == "ga_broadcast_deltafitness_cifar":
            continue
        gen0 = [r["fitness"] * 100 for r in tr if r["gen"] == 0][:4]
        if len(gen0) < 4:
            continue
        series.append(gen0)
        ax.plot(range(1, 5), gen0, color=LIGHT, lw=0.8, zorder=2)
        if scn == "ga_broadcast_cifar":
            seeded_first.append(gen0[0])
    mean = np.mean(np.array(series), axis=0)
    ax.plot(range(1, 5), mean, color=GREEN, lw=1.8, zorder=4, marker="o", ms=4)
    ax.text(4.1, mean[3], f"Mean of\n{len(series)} runs", color=INK, fontsize=7.8, va="center", linespacing=0.95)
    ax.scatter([1] * len(seeded_first), seeded_first, marker="D", s=16, color=DARK, edgecolor="white",
               linewidth=0.4, zorder=5)
    ax.text(1.12, 2, "Expert, seeded at position 1", color=INK, fontsize=7.8, va="bottom")
    ax.set_xticks([1, 2, 3, 4])
    ax.set_xlim(0.8, 4.75)
    ax.set_xlabel("Evaluation position in generation 0")
    ax.set_ylabel("Fitness (%)")
    ax.set_ylim(0, 80)
    ax.grid(axis="y", **GRID)
    ax.set_title("(a) Fitness by evaluation position", loc="left")

    cands = gen0_candidates()
    conds = [("sequential", "Evaluated in sequence, as in the GA"),
             ("cold", "Each trained from the initial model"),
             ("warm", "Each trained from the model after generation 0")]
    ordinal = {1: "1st", 2: "2nd", 3: "3rd", 4: "4th"}
    y, yticks, ylabels = 0.0, [], []
    for key, header in conds:
        bx.text(0, y - 0.9, header, fontsize=7.8, color=INK, va="center")
        for seed in sorted(s for c, s in cands if c == key):
            vals = cands[(key, seed)]
            expert = next(v for v, is_exp in vals if is_exp)
            others = [v for v, is_exp in vals if not is_exp]
            bx.plot([min(v for v, _ in vals), max(v for v, _ in vals)], [y, y], color="#dcdcd6", lw=0.9, zorder=2)
            bx.scatter(others, [y] * len(others), s=14, color=MID, edgecolor="white", linewidth=0.4, zorder=3)
            bx.scatter([expert], [y], marker="D", s=22, color=DARK, edgecolor="white", linewidth=0.4, zorder=4)
            rank = 1 + sum(v > expert for v, _ in vals)
            bx.text(88, y, ordinal[rank], fontsize=7.8, va="center", ha="center", color=INK)
            yticks.append(y)
            ylabels.append(f"seed {seed}")
            y += 1
        y += 1.2
    bx.text(88, -0.9, "rank", fontsize=7.8, ha="center", va="center", color=GRAY, style="italic")
    bx.set_yticks(yticks)
    bx.set_yticklabels(ylabels, fontsize=7.8)
    bx.tick_params(axis="y", length=0)
    bx.set_ylim(y - 1.7, -1.5)
    bx.set_xlim(0, 93)
    bx.set_xticks([0, 20, 40, 60, 80])
    bx.set_xlabel("Fitness (%)")
    bx.grid(axis="x", **GRID)
    bx.spines["left"].set_visible(False)
    bx.legend(handles=[
        Line2D([], [], marker="D", ls="", color=DARK, markeredgecolor="white", ms=4.5, label="Expert (rank at right)"),
        Line2D([], [], marker="o", ls="", color=MID, markeredgecolor="white", ms=4, label="Other generation-0 candidates"),
    ], loc="upper center", bbox_to_anchor=(0.45, -0.24), ncol=2, frameon=False, handletextpad=0.2, columnspacing=1.0)
    bx.set_title("(b) The same four candidates, evaluated three ways", loc="left", pad=5)
    save(fig, "fig3_coldstart")


def main() -> int:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    rows = rows_by_scenario()
    fig2(rows)
    fig3()
    print(f"[figures] wrote fig2_tradeoff + fig3_coldstart into {OUT_DIR}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
