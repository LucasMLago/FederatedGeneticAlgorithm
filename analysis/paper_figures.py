#!/usr/bin/env python3
"""Paper-quality figures (vector PDF + 300dpi PNG) for the coupling-regimes paper.

Fig 2, main trade-off (CIFAR-10/α=0.5, N=5 paired):
  (a) per-round eval-acc mean ± sd per coupling regime, expert baseline as
      dashed reference;
  (b) peak accuracy × wall-time with sd error bars (no regime Pareto-dominates);
  (c) severe single-round drops (>10pp) per regime.

Fig 3, cold-start bias mechanism: generation-0 fitness by evaluation position
  across all 8 broadcast-GA runs; the seeded expert HP always sits at position 1.

Fig 4, extreme-heterogeneity boundary (CIFAR-10/α=0.1, N=3): per-round eval-acc
  for the expert anchor, per-client GA, and broadcast GA; the exploration toll
  grows with coupling and the broadcast arm never stabilizes.

Colors: validated 3-slot categorical palette (dataviz skill, all checks pass);
identity is never color-alone (direct labels + line style differentiate).

Usage:
    python analysis/paper_figures.py            # writes paper/figures/*.{pdf,png}
"""
from __future__ import annotations

import csv
import statistics
import sys
from collections import defaultdict
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
SUMMARY = REPO_ROOT / "federatedgeneticalgorithm" / "artifacts" / "matrix_summary.csv"
RUNS_DIR = REPO_ROOT / "federatedgeneticalgorithm" / "artifacts" / "runs"
OUT_DIR = REPO_ROOT / "analysis" / "figures"

sys.path.insert(0, str(Path(__file__).resolve().parent))
from fitness_bias import infer_missing_round1, load_windows, parse_traces  # noqa: E402

# Fixed entity → color assignment (categorical slots 1-3, light mode, validated)
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


def fig2(rows: dict) -> None:
    fig, axes = plt.subplots(1, 3, figsize=(9.6, 2.9), gridspec_kw={"width_ratios": [2.1, 1.15, 0.95]})
    ax = axes[0]
    x = np.arange(1, 21)
    exp_curves = curves_for(EXPERT, rows) * 100
    ax.plot(x, np.nanmean(exp_curves, axis=0), color=GRAY, ls="--", lw=1.4, zorder=2)
    ax.annotate("expert fixed HP", (20.2, np.nanmean(exp_curves, axis=0)[-1] + 1.2),
                color=GRAY, fontsize=7.5, ha="right", style="italic")
    for scn, style in REGIMES.items():
        cs = curves_for(scn, rows) * 100
        mean, sd = np.nanmean(cs, axis=0), np.nanstd(cs, axis=0, ddof=1)
        ax.fill_between(x, mean - sd, mean + sd, color=style["color"], alpha=0.14, lw=0)
        ax.plot(x, mean, color=style["color"], lw=1.8, ls=style["ls"], zorder=3)
    # direct labels, staggered at the left knee where the curves separate
    ax.annotate("per-client", (6.1, 76.5), color=REGIMES["ga_perclient_cifar"]["color"], fontsize=7.5, fontweight="bold")
    ax.annotate("broadcast", (7.5, 61.5), color=REGIMES["ga_broadcast_cifar"]["color"], fontsize=7.5, fontweight="bold")
    ax.annotate("surrogate", (3.4, 36.0), color=REGIMES["ga_surrogate_cifar"]["color"], fontsize=7.5, fontweight="bold")
    ax.set_xlabel("Server round")
    ax.set_ylabel("Global eval accuracy (%)")
    ax.set_xlim(1, 20.4)
    ax.set_ylim(5, 92)
    ax.set_xticks([1, 5, 10, 15, 20])
    ax.grid(axis="y", **GRID)
    ax.set_title("(a) Mean ± sd across 5 paired seeds", loc="left")

    ax = axes[1]
    for scn, style in REGIMES.items():
        rs = rows[scn]
        peaks = [float(r["peak_eval_acc"]) * 100 for r in rs]
        walls = [float(r["wall_seconds"]) / 60 for r in rs]
        ax.errorbar(statistics.fmean(walls), statistics.fmean(peaks),
                    xerr=statistics.stdev(walls), yerr=statistics.stdev(peaks),
                    fmt="o", ms=6, color=style["color"], capsize=3, lw=1.2)
    ax.annotate("per-client", (108, 82.15), color=REGIMES["ga_perclient_cifar"]["color"], fontsize=7.5, ha="right", fontweight="bold")
    ax.annotate("surrogate", (67, 79.2), color=REGIMES["ga_surrogate_cifar"]["color"], fontsize=7.5, fontweight="bold")
    ax.annotate("broadcast", (54, 82.6), color=REGIMES["ga_broadcast_cifar"]["color"], fontsize=7.5, fontweight="bold")
    ax.invert_xaxis()
    ax.set_xlabel("Wall-time per run (min) — better →")
    ax.set_ylabel("Peak accuracy (%)")
    ax.grid(axis="both", **GRID)
    ax.set_title("(b) No regime dominates", loc="left")

    ax = axes[2]
    names, drops, fracs = [], [], []
    for scn, style in REGIMES.items():
        rs = rows[scn]
        per_run = []
        for r in rs:
            c = [v for v in eval_curve(r["run_id"]) if v is not None]
            per_run.append(sum(1 for i in range(1, len(c)) if c[i - 1] - c[i] > 0.10))
        names.append(style["label"].split(" (")[0].replace("Server-broadcast GA", "broadcast").replace("Per-client GA", "per-client").replace("Surrogate-aided", "surrogate"))
        drops.append(sum(per_run))
        fracs.append(f"{sum(1 for d in per_run if d)}/{len(per_run)} seeds")
    bars = ax.bar(names, drops, width=0.58, color=[s["color"] for s in REGIMES.values()], zorder=3)
    for b, d, f in zip(bars, drops, fracs):
        ax.annotate(f"{d}\n{f}", (b.get_x() + b.get_width() / 2, d + 0.12),
                    ha="center", va="bottom", fontsize=6.8, color="#3a3a37")
    ax.set_ylabel("Severe drops (>10 pp / round)")
    ax.set_ylim(0, max(drops) + 1.6)
    ax.grid(axis="y", **GRID)
    ax.set_title("(c) Robustness cost", loc="left")
    ax.tick_params(axis="x", labelsize=7.5)

    fig.tight_layout(w_pad=1.6)
    for ext in ("pdf", "png"):
        fig.savefig(OUT_DIR / f"fig2_tradeoff.{ext}", dpi=300)
    plt.close(fig)


def fig3() -> None:
    windows = load_windows()
    traces = parse_traces(windows)
    infer_missing_round1(traces)
    fig, ax = plt.subplots(figsize=(3.7, 2.7))
    seeded_first, series = [], []
    for (scn, seed), tr in sorted(traces.items()):
        if scn == "ga_broadcast_deltafitness_cifar":
            continue
        gen0 = [r["fitness"] * 100 for r in tr if r["gen"] == 0][:4]
        if len(gen0) < 4:
            continue
        series.append(gen0)
        ax.plot(range(1, 5), gen0, color="#b9b9b3", lw=0.9, zorder=2)
        if scn == "ga_broadcast_cifar":
            seeded_first.append(gen0[0])
    mean = np.mean(np.array(series), axis=0)
    ax.plot(range(1, 5), mean, color="#1baf7a", lw=2.2, zorder=4, marker="o", ms=5)
    ax.annotate("mean of 8 runs", (3.02, mean[2] - 9), color="#199e70", fontsize=7.5, fontweight="bold")
    ax.scatter([1] * len(seeded_first), seeded_first, marker="D", s=26, color="#eb6834", zorder=5)
    ax.annotate("seeded expert HP\n(always position 1)", (1.1, 3.5),
                color="#d95926", fontsize=7.2)
    ax.set_xticks([1, 2, 3, 4])
    ax.set_xlabel("Evaluation position within generation 0\n(model warms as rounds pass)", fontsize=8)
    ax.set_ylabel("Fitness credited (%)")
    ax.set_ylim(0, 80)
    ax.grid(axis="y", **GRID)
    fig.tight_layout()
    for ext in ("pdf", "png"):
        fig.savefig(OUT_DIR / f"fig3_coldstart.{ext}", dpi=300)
    plt.close(fig)


def fig4(rows: dict) -> None:
    """Per-round curves at α=0.1: same entity→color mapping as fig2."""
    fig, ax = plt.subplots(figsize=(3.7, 2.7))
    x = np.arange(1, 21)
    exp_curves = curves_for("fixed_expert_alpha01", rows) * 100
    ax.plot(x, np.nanmean(exp_curves, axis=0), color=GRAY, ls="--", lw=1.4, zorder=2)
    arms = {
        "ga_perclient_alpha01": REGIMES["ga_perclient_cifar"]["color"],
        "ga_surrogate_alpha01": REGIMES["ga_surrogate_cifar"]["color"],
        "ga_broadcast_alpha01": REGIMES["ga_broadcast_cifar"]["color"],
    }
    for scn, color in arms.items():
        cs = curves_for(scn, rows) * 100
        mean, sd = np.nanmean(cs, axis=0), np.nanstd(cs, axis=0, ddof=1)
        ax.fill_between(x, mean - sd, mean + sd, color=color, alpha=0.14, lw=0)
        ax.plot(x, mean, color=color, lw=1.8, zorder=3)
    ax.annotate("expert fixed HP", (19.8, np.nanmean(exp_curves, axis=0)[-1] + 3),
                color=GRAY, fontsize=7.5, ha="right", style="italic")
    ax.annotate("per-client", (12.0, 63.0), color=arms["ga_perclient_alpha01"], fontsize=7.5, fontweight="bold")
    ax.annotate("broadcast", (14.2, 51.0), color=arms["ga_broadcast_alpha01"], fontsize=7.5, fontweight="bold")
    ax.annotate("surrogate", (15.5, 24.0), color=arms["ga_surrogate_alpha01"], fontsize=7.5, fontweight="bold")
    ax.set_xlabel("Server round")
    ax.set_ylabel("Global eval accuracy (%)")
    ax.set_xlim(1, 20.4)
    ax.set_ylim(0, 82)
    ax.set_xticks([1, 5, 10, 15, 20])
    ax.grid(axis="y", **GRID)
    fig.tight_layout()
    for ext in ("pdf", "png"):
        fig.savefig(OUT_DIR / f"fig4_alpha01.{ext}", dpi=300)
    plt.close(fig)


def main() -> int:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    rows = rows_by_scenario()
    fig2(rows)
    fig3()
    fig4(rows)
    print(f"[figures] wrote fig2_tradeoff + fig3_coldstart + fig4_alpha01 into {OUT_DIR}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
