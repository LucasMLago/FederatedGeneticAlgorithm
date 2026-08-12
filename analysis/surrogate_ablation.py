"""Paired analysis of the surrogate ablation (Stage D vs Stage C).

Compares the 5 seeds of `ga_surrogate_cifar` (treatment, surrogate=ON)
against the 5 seeds of `ga_perclient_cifar` (control, surrogate=OFF), both
sharing identical config except `ENABLE_SURROGATE_GA`.

Outputs:
  - Mean / sd / CI95% / IQR for peak and final eval-acc
  - Mann-Whitney U (two-sided, exact) for peak and final
  - Wall-time comparison (mean ± sd, ratio OFF/ON)
  - Per-seed peak table
  - Round-by-round mean ± sd plot for both arms
  - Catastrophic-crash detection (Δeval-acc < -10pp in a single round) per seed

Saves:
  - analysis/surrogate_ablation.png
  - analysis/surrogate_ablation_report.md
"""
from __future__ import annotations

import csv
import statistics
from collections import defaultdict
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from scipy import stats

ROOT = Path(__file__).resolve().parents[1]
ARTIFACTS = ROOT / "federatedgeneticalgorithm" / "artifacts"
MATRIX = ARTIFACTS / "matrix_summary.csv"
OUT_PNG = Path(__file__).parent / "surrogate_ablation.png"
OUT_MD = Path(__file__).parent / "surrogate_ablation_report.md"

CONTROL = "ga_perclient_cifar"            # surrogate=OFF (Stage C)
TREATMENT = "ga_surrogate_cifar"  # surrogate=ON  (Stage D)


def load_matrix():
    by_scen = defaultdict(list)
    for row in csv.DictReader(MATRIX.open()):
        if row["status"] != "ok":
            continue
        if not (row.get("peak_eval_acc") and row.get("wall_seconds")):
            continue  # skip incomplete rows (e.g. killed FEMNIST runs)
        by_scen[row["scenario_name"]].append(
            {
                "seed": int(row["seed"]),
                "run_id": row["run_id"],
                "peak": float(row["peak_eval_acc"]),
                "final": float(row["final_eval_acc"]),
                "wall_s": float(row["wall_seconds"]),
            }
        )
    return by_scen


def load_eval_curve(run_id: str):
    path = ARTIFACTS / "runs" / run_id / "server_aggregated_rounds.csv"
    out = []
    with path.open() as f:
        for row in csv.DictReader(f):
            if row.get("phase") != "evaluate" or row.get("run_id") != run_id:
                continue
            try:
                out.append((int(row["server_round"]), float(row["eval-acc"])))
            except (KeyError, ValueError):
                continue
    return sorted(out)


def ci95_half(values):
    n = len(values)
    if n < 2:
        return float("nan")
    sd = statistics.stdev(values)
    # Use t critical value for small N (more honest than 1.96)
    t = stats.t.ppf(0.975, n - 1)
    return t * sd / np.sqrt(n)


def describe(arr):
    a = np.asarray(arr, dtype=float)
    return {
        "n": int(a.size),
        "mean": float(a.mean()),
        "sd": float(a.std(ddof=1)) if a.size > 1 else float("nan"),
        "ci95": float(ci95_half(a.tolist())),
        "median": float(np.median(a)),
        "iqr": float(np.percentile(a, 75) - np.percentile(a, 25)),
        "min": float(a.min()),
        "max": float(a.max()),
    }


def fmt_pct(x, digits=2):
    if x is None or (isinstance(x, float) and (np.isnan(x))):
        return "—"
    return f"{x * 100:.{digits}f}"


def detect_crashes(curve, threshold_pp=10.0):
    """Return list of (round, drop_pp) where eval-acc fell >threshold_pp in a single round."""
    crashes = []
    for (r0, a0), (r1, a1) in zip(curve, curve[1:]):
        drop_pp = (a0 - a1) * 100.0
        if drop_pp >= threshold_pp:
            crashes.append((r1, drop_pp))
    return crashes


def main():
    by_scen = load_matrix()
    ctrl = sorted(by_scen[CONTROL], key=lambda r: r["seed"])
    trt = sorted(by_scen[TREATMENT], key=lambda r: r["seed"])

    assert len(ctrl) == 5 and len(trt) == 5, f"Expected 5+5 seeds, got {len(ctrl)}+{len(trt)}"

    # ============ Stats ============
    ctrl_peak = describe([r["peak"] for r in ctrl])
    trt_peak = describe([r["peak"] for r in trt])
    ctrl_final = describe([r["final"] for r in ctrl])
    trt_final = describe([r["final"] for r in trt])
    ctrl_wall = describe([r["wall_s"] for r in ctrl])
    trt_wall = describe([r["wall_s"] for r in trt])

    # Mann-Whitney U (two-sided, exact for small N)
    u_peak = stats.mannwhitneyu(
        [r["peak"] for r in ctrl], [r["peak"] for r in trt], alternative="two-sided", method="exact"
    )
    u_final = stats.mannwhitneyu(
        [r["final"] for r in ctrl], [r["final"] for r in trt], alternative="two-sided", method="exact"
    )
    u_wall = stats.mannwhitneyu(
        [r["wall_s"] for r in ctrl], [r["wall_s"] for r in trt], alternative="two-sided", method="exact"
    )

    delta_peak = ctrl_peak["mean"] - trt_peak["mean"]
    delta_final = ctrl_final["mean"] - trt_final["mean"]
    delta_wall_min = (ctrl_wall["mean"] - trt_wall["mean"]) / 60.0
    wall_ratio = ctrl_wall["mean"] / trt_wall["mean"]

    # ============ Round-by-round curves & crash detection ============
    fig, axes = plt.subplots(1, 2, figsize=(14, 5.5))
    panels = [
        ("Surrogate OFF (control, Stage C)", ctrl, "#1f77b4", axes[0]),
        ("Surrogate ON (treatment, Stage D)", trt, "#d62728", axes[1]),
    ]
    crashes_per_seed = {}
    for label, rows, color, ax in panels:
        per_seed = []
        for r in rows:
            curve = load_eval_curve(r["run_id"])
            rounds = [rr for rr, _ in curve]
            accs = [a * 100 for _, a in curve]
            per_seed.append((r["seed"], rounds, accs))
            ax.plot(rounds, accs, color=color, alpha=0.35, lw=1.0, label=f"seed {r['seed']}")
            crashes_per_seed[(label, r["seed"])] = detect_crashes(curve, threshold_pp=10.0)
        all_rounds = sorted({rr for _, rs, _ in per_seed for rr in rs})
        means, sds = [], []
        for rr in all_rounds:
            vals = [a for _, rs, accs in per_seed for r0, a in zip(rs, accs) if r0 == rr]
            means.append(statistics.mean(vals))
            sds.append(statistics.stdev(vals) if len(vals) > 1 else 0.0)
        means_arr = np.array(means)
        sds_arr = np.array(sds)
        ax.plot(all_rounds, means_arr, color=color, lw=3.0, label="mean")
        ax.fill_between(
            all_rounds, means_arr - sds_arr, means_arr + sds_arr, color=color, alpha=0.18, label="±1 sd"
        )
        peak_mean = max(means)
        peak_sd = sds[int(np.argmax(means))]
        final_mean = means[-1]
        final_sd = sds[-1]
        ax.set_title(
            f"{label}\nN=5 · peak {peak_mean:.2f}±{peak_sd:.2f}% · final {final_mean:.2f}±{final_sd:.2f}%",
            fontsize=11, fontweight="bold",
        )
        ax.set_xlabel("Server round")
        ax.set_ylabel("eval-acc (%)")
        ax.set_xlim(0, 21)
        ax.set_ylim(0, 90)
        ax.set_xticks(range(0, 21, 2))
        ax.grid(alpha=0.3)
        ax.legend(loc="lower right", fontsize=8, ncol=2)

    fig.suptitle(
        "Surrogate comparison (paired): per-client GA, CIFAR-10 α=0.5, N=5+5 seeds",
        fontsize=13, fontweight="bold", y=1.02,
    )
    plt.tight_layout()
    plt.savefig(OUT_PNG, dpi=130, bbox_inches="tight")

    # ============ Markdown report ============
    lines = []
    lines.append("# Surrogate comparison (paired): with vs without surrogate\n")
    lines.append(
        f"**Setup**: 5 seeds × 2 variants (surrogate ON vs OFF) of per-client GA on "
        f"CIFAR-10, Dirichlet α=0.5, 20 rounds. Only `ENABLE_SURROGATE_GA` differs.\n"
    )
    lines.append("## 1. Peak eval-acc\n")
    lines.append("| Variant | N | Mean (%) | SD (%) | CI95% ± (%) | Median (%) | IQR (%) | Min–Max (%) |")
    lines.append("|---|---:|---:|---:|---:|---:|---:|---:|")
    for label, s in [("Surrogate OFF (control)", ctrl_peak), ("Surrogate ON (treatment)", trt_peak)]:
        lines.append(
            f"| {label} | {s['n']} | {fmt_pct(s['mean'])} | {fmt_pct(s['sd'])} | "
            f"{fmt_pct(s['ci95'])} | {fmt_pct(s['median'])} | {fmt_pct(s['iqr'])} | "
            f"{fmt_pct(s['min'])}–{fmt_pct(s['max'])} |"
        )
    lines.append("")
    lines.append(
        f"**Δ (OFF − ON) = {delta_peak*100:+.2f} pp** · Mann-Whitney U "
        f"= {u_peak.statistic:.1f}, p = {u_peak.pvalue:.4f} (two-sided, exact)\n"
    )

    lines.append("## 2. Final eval-acc (round 20)\n")
    lines.append("| Variant | N | Mean (%) | SD (%) | CI95% ± (%) | Median (%) | IQR (%) | Min–Max (%) |")
    lines.append("|---|---:|---:|---:|---:|---:|---:|---:|")
    for label, s in [("Surrogate OFF (control)", ctrl_final), ("Surrogate ON (treatment)", trt_final)]:
        lines.append(
            f"| {label} | {s['n']} | {fmt_pct(s['mean'])} | {fmt_pct(s['sd'])} | "
            f"{fmt_pct(s['ci95'])} | {fmt_pct(s['median'])} | {fmt_pct(s['iqr'])} | "
            f"{fmt_pct(s['min'])}–{fmt_pct(s['max'])} |"
        )
    lines.append("")
    lines.append(
        f"**Δ (OFF − ON) = {delta_final*100:+.2f} pp** · Mann-Whitney U "
        f"= {u_final.statistic:.1f}, p = {u_final.pvalue:.4f} (two-sided, exact)\n"
    )

    lines.append("## 3. Wall-time\n")
    lines.append("| Variant | N | Mean (min) | SD (min) | Min–Max (min) |")
    lines.append("|---|---:|---:|---:|---:|")
    for label, s in [("Surrogate OFF (control)", ctrl_wall), ("Surrogate ON (treatment)", trt_wall)]:
        lines.append(
            f"| {label} | {s['n']} | {s['mean']/60:.1f} | {s['sd']/60:.1f} | "
            f"{s['min']/60:.1f}–{s['max']/60:.1f} |"
        )
    lines.append("")
    lines.append(
        f"**Δ wall (OFF − ON) = {delta_wall_min:+.1f} min** · ratio OFF/ON = "
        f"**{wall_ratio:.2f}×** · Mann-Whitney U "
        f"= {u_wall.statistic:.1f}, p = {u_wall.pvalue:.4f}\n"
    )

    lines.append("## 4. Per-seed peak (paired)\n")
    lines.append("| Seed | OFF peak (%) | ON peak (%) | Δ (OFF−ON) pp |")
    lines.append("|---:|---:|---:|---:|")
    for c, t in zip(ctrl, trt):
        assert c["seed"] == t["seed"]
        diff_pp = (c["peak"] - t["peak"]) * 100
        lines.append(f"| {c['seed']} | {fmt_pct(c['peak'])} | {fmt_pct(t['peak'])} | {diff_pp:+.2f} |")
    lines.append("")

    # Paired (Wilcoxon signed-rank, valid since same seed = same data partition)
    paired_diff = [c["peak"] - t["peak"] for c, t in zip(ctrl, trt)]
    try:
        w = stats.wilcoxon(paired_diff, alternative="two-sided", method="exact")
        lines.append(
            f"**Wilcoxon signed-rank (paired by seed)**: W = {w.statistic:.1f}, "
            f"p = {w.pvalue:.4f}\n"
        )
    except ValueError as exc:
        lines.append(f"_Wilcoxon failed: {exc}_\n")

    lines.append("## 5. Catastrophic crashes (Δeval-acc < -10pp in 1 round)\n")
    any_crash = False
    for label, rows, _color, _ax in panels:
        for r in rows:
            cs = crashes_per_seed[(label, r["seed"])]
            if cs:
                any_crash = True
                desc = ", ".join(f"R{rr} (−{dp:.1f}pp)" for rr, dp in cs)
                lines.append(f"- **{label} · seed {r['seed']}**: {desc}")
    if not any_crash:
        lines.append("_None detected. Neither variant exhibits the single-round catastrophic crash"
                     " characteristic of FedGA._")
    lines.append("")

    lines.append("## 6. Plot\n")
    lines.append(f"![surrogate comparison]({OUT_PNG.name})\n")

    OUT_MD.write_text("\n".join(lines), encoding="utf-8")
    print("\n".join(lines))
    print(f"\nWrote {OUT_PNG}")
    print(f"Wrote {OUT_MD}")


if __name__ == "__main__":
    main()
