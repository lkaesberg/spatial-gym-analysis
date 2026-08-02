"""
Rebuttal experiment: controlled SFT vs RLVF (RLHF-equivalent) comparison
within the OLMo 32B family on Spatial-Gym with backtracking.

Two metrics, both reusing the paper's existing definitions:
  - Accuracy benefit from backtracking: Delta acc = acc(traceback) - acc(non-traceback),
    using the "Correctly Solved" % (same as plot_traceback_diff.py).
  - Backtracking ratio (Figure 5): per puzzle steps_taken / path_edges, where
    path_edges = len(clean_path(extracted_path)) - 1; keep steps < 100 and path_edges > 0;
    report the per-model median.

Outputs (results/rebuttal/):
  - rebuttal_backtracking.{png,pdf}   2-panel figure (Delta acc bars + ratio boxplot)
  - rebuttal_backtracking.md          markdown table

Scope: the four OLMo 32B variants (two SFT-vs-RLVF pairs: Think and Instruct).
The 7B Instruct runs are intentionally excluded. Backtracking runs cover <500 puzzles for
some variants; metrics are computed over the available puzzles.
"""
import numpy as np
from pathlib import Path

import matplotlib.pyplot as plt

from plot_config import setup_plot_style, TEXT_WIDTH_INCHES, get_model_color
from plot_traceback_steps_vs_path import extract_traceback_steps_vs_path_per_model
from plot_traceback_diff import extract_accuracy_from_stats


def _tint(hexcolor, t=0.55):
    """Lighten a hex color by blending it t-fraction toward white."""
    h = hexcolor.lstrip("#")
    r, g, b = int(h[0:2], 16), int(h[2:4], 16), int(h[4:6], 16)
    r = int(r + (255 - r) * t)
    g = int(g + (255 - g) * t)
    b = int(b + (255 - b) * t)
    return f"#{r:02x}{g:02x}{b:02x}"


# OLMo family palette (matches MODEL_COLORS["OLMo 3.1 32B"]).
OLMO_STRONG = get_model_color("OLMo 3.1 32B", warn_on_missing=False)  # #BE185D
OLMO_TINT = _tint(OLMO_STRONG, 0.55)

# Controlled pairs within the OLMo 32B family (shared base + SFT data; differ in preference tuning).
# (model_name on disk, display name, short chart label, tuning label, is_rlhf, pair)
MODELS = [
    ("allenai_Olmo-3-32B-Think-SFT",      "OLMo 3 32B Think-SFT",      "OLMo 3 32B\nThink\nSFT",     "SFT only", False, "Think"),
    ("allenai_Olmo-3.1-32B-Think",        "OLMo 3.1 32B Think",        "OLMo 3 32B\nThink\nRLVF",    "RLVF",     True,  "Think"),
    ("allenai_Olmo-3.1-32B-Instruct-SFT", "OLMo 3.1 32B Instruct-SFT", "OLMo 3 32B\nInstruct\nSFT",  "SFT only", False, "Instruct"),
    ("allenai_Olmo-3.1-32B-Instruct",     "OLMo 3.1 32B Instruct",     "OLMo 3 32B\nInstruct\nRLVF", "RLVF",     True,  "Instruct"),
]


def compute_ratio_stats(results_dir, filter_max_steps=True):
    """Per-model backtracking-ratio stats (only models with a traceback jsonl)."""
    per_model = extract_traceback_steps_vs_path_per_model(results_dir)
    out = {}
    for model_name, data in per_model.items():
        s, pe = data["steps"], data["path_edges"]
        if filter_max_steps:
            m = s < 100
            s, pe = s[m], pe[m]
        valid = pe > 0
        if valid.sum() == 0:
            continue
        ratios = s[valid] / pe[valid]
        out[model_name] = {
            "ratios": ratios, "n": int(valid.sum()),
            "median": float(np.median(ratios)), "mean": float(ratios.mean()),
        }
    return out


def acc_from(results_dir, model_name, suffix):
    f = Path(results_dir) / f"{model_name}{suffix}_stats.csv"
    return extract_accuracy_from_stats(f) if f.exists() else None


def collect(results_dir):
    ratio_stats = compute_ratio_stats(results_dir)
    rows = []
    for name, disp, short, tuning, is_rlhf, pair in MODELS:
        base_acc = acc_from(results_dir, name, "_gym")
        tb_acc = acc_from(results_dir, name, "_gym_traceback")
        delta = (tb_acc - base_acc) if (tb_acc is not None and base_acc is not None) else None
        rs = ratio_stats.get(name)
        rows.append({
            "name": name, "disp": disp, "short": short, "tuning": tuning, "is_rlhf": is_rlhf, "pair": pair,
            "base_acc": base_acc, "tb_acc": tb_acc, "delta": delta,
            "ratios": rs["ratios"] if rs else None,
            "median": rs["median"] if rs else None,
            "mean": rs["mean"] if rs else None,
            "n": rs["n"] if rs else None,
            "has_tb": tb_acc is not None,
        })
    return rows


def _grouped_positions(tb_rows, gap=0.6):
    """X positions that visually separate the two pairs (Think | Instruct)."""
    positions = []
    cur = 0.0
    prev_pair = None
    for r in tb_rows:
        if prev_pair is not None and r["pair"] != prev_pair:
            cur += 1.0 + gap
        else:
            cur += 1.0 if prev_pair is not None else 0.0
        positions.append(cur)
        prev_pair = r["pair"]
    return np.array(positions)


def create_figure(rows, output_path):
    setup_plot_style(use_latex=True)
    tb_rows = [r for r in rows if r["has_tb"]]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(TEXT_WIDTH_INCHES, 1.9))

    # --- Panel (a): non-traceback vs traceback accuracy, with Delta label ---
    x = _grouped_positions(tb_rows)
    width = 0.38
    base_vals = [r["base_acc"] for r in tb_rows]
    tb_vals = [r["tb_acc"] for r in tb_rows]
    b1 = ax1.bar(x - width / 2, base_vals, width, color=OLMO_TINT,
                 edgecolor="black", linewidth=0.5, label="No backtracking")
    b2 = ax1.bar(x + width / 2, tb_vals, width, color=OLMO_STRONG,
                 edgecolor="black", linewidth=0.5, label="Backtracking")
    for bars, vals in ((b1, base_vals), (b2, tb_vals)):
        for bar, v in zip(bars, vals):
            ax1.annotate(f"{v:.1f}\\%", xy=(bar.get_x() + bar.get_width() / 2, bar.get_height()),
                         xytext=(0, 1.5), textcoords="offset points", ha="center", va="bottom",
                         fontsize=5, fontweight="bold")
    for xi, r in zip(x, tb_rows):
        ax1.annotate(f"$\\Delta={r['delta']:+.1f}$", xy=(xi, max(r['base_acc'], r['tb_acc'])),
                     xytext=(0, 9), textcoords="offset points", ha="center", va="bottom",
                     fontsize=6.5, fontweight="bold", color=OLMO_STRONG)
    ax1.set_xticks(x)
    ax1.set_xticklabels([r["short"] for r in tb_rows], fontsize=5.5)
    ax1.set_ylabel("Accuracy (\\%)")
    ax1.set_ylim(0, max(max(base_vals), max(tb_vals)) * 1.5)
    ax1.spines["top"].set_visible(False)
    ax1.spines["right"].set_visible(False)
    ax1.yaxis.grid(True, linestyle="--", alpha=0.3)
    ax1.set_axisbelow(True)
    ax1.legend(loc="upper right", framealpha=0.9, fontsize=6.5)
    ax1.set_title("(a) Accuracy With vs. Without Backtracking", fontweight="bold")

    # --- Panel (b): backtracking ratio boxplot ---
    box_data = [r["ratios"] for r in tb_rows]
    bp = ax2.boxplot(box_data, positions=x, widths=0.55, patch_artist=True, showfliers=False,
                     medianprops=dict(color="black", linewidth=1.5),
                     whiskerprops=dict(color="black", linewidth=0.8),
                     capprops=dict(color="black", linewidth=0.8))
    for patch, r in zip(bp["boxes"], tb_rows):
        patch.set_facecolor(OLMO_STRONG if r["is_rlhf"] else OLMO_TINT)
        patch.set_edgecolor("black")
        patch.set_linewidth(0.5)
    ax2.set_xticks(x)
    ax2.set_xticklabels([r["short"] for r in tb_rows], fontsize=5.5)
    ax2.set_ylabel("Steps / Path Edges")
    ax2.spines["top"].set_visible(False)
    ax2.spines["right"].set_visible(False)
    ax2.yaxis.grid(True, linestyle="--", alpha=0.3)
    ax2.set_axisbelow(True)
    ax2.set_title("(b) Backtracking Ratio by Variant", fontweight="bold")

    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches="tight")
    print(f"Figure saved to: {output_path}")
    plt.close(fig)


def write_markdown_table(rows, output_path):
    L = []
    L.append("## Backtracking & RLHF: controlled comparison within the OLMo 32B family (Spatial-Gym)")
    L.append("")
    L.append("Δ Acc. = acc(backtracking) − acc(no backtracking) (\"Correctly Solved\" %). "
             "Backtracking ratio = median(steps_taken / path_edges) (Figure 5 metric; steps < 100, path_edges > 0).")
    L.append("")
    L.append("| Model | Tuning | No-BT Acc. (%) | BT Acc. (%) | Δ Acc. (pp) | Median ratio | Mean ratio | N (BT valid)$^a$ |")
    L.append("|---|---|---:|---:|---:|---:|---:|---:|")
    for r in rows:
        disp = r["disp"].replace("\n", " ")
        base = f"{r['base_acc']:.1f}" if r["base_acc"] is not None else "n/a"
        if r["has_tb"]:
            tb = f"{r['tb_acc']:.1f}"
            delta = f"{r['delta']:+.1f}"
            med = f"{r['median']:.2f}"
            mean = f"{r['mean']:.2f}"
            n = f"{r['n']}"
        else:
            tb = delta = med = mean = n = "—$^a$"
        L.append(f"| {disp} | {r['tuning']} | {base} | {tb} | {delta} | {med} | {mean} | {n} |")
    L.append("")
    L.append("**Notes.**")
    L.append("- $^a$ Backtracking runs cover fewer than 500 puzzles for some variants "
             "(Think-SFT 360, Instruct-SFT 436, Instruct 320); accuracies are run-level percentages "
             "and the median ratio is over puzzles with a valid path.")
    L.append("- The 7B Instruct models are excluded per the 32B-only scope.")
    L.append("")
    L.append("**Observation.** In both pairs the SFT-only (non-RLHF) variant backtracks at least as much as, "
             "and is helped more by backtracking than, its RLVF sibling. The effect is clearest for "
             "the Instruct pair: the SFT model *gains* from backtracking (+2.2 pp) and backtracks far more "
             "(median ratio 2.94), while the RLVF model *loses* accuracy (−1.5 pp) and backtracks "
             "much less (1.39). For the Think pair both variants decline, but the RLVF one declines "
             "more (−1.8 vs −0.8 pp). This is consistent with the RLHF hypothesis, but with only two pairs that "
             "differ in other ways it remains supporting context for a hypothesis rather than a tested causal claim.")
    L.append("")
    Path(output_path).write_text("\n".join(L))
    print(f"Markdown table saved to: {output_path}")


def main():
    base = Path(__file__).parent
    results_dir = base / "results" / "rebuttal"
    rows = collect(results_dir)

    print("\nOLMo 32B controlled comparison:")
    print("-" * 84)
    for r in rows:
        disp = r["disp"].replace("\n", " ")
        base_s = f"{r['base_acc']:.1f}%" if r["base_acc"] is not None else "n/a"
        if r["has_tb"]:
            print(f"  {disp:<26s} noBT={base_s:>6s}  BT={r['tb_acc']:.1f}%  "
                  f"dAcc={r['delta']:+.1f}pp  median={r['median']:.2f}  mean={r['mean']:.2f}  n={r['n']}")
        else:
            print(f"  {disp:<26s} noBT={base_s:>6s}  BT=—  (no traceback run)")
    print("-" * 84)

    for ext in ("png", "pdf"):
        create_figure(rows, results_dir / f"rebuttal_backtracking.{ext}")
    write_markdown_table(rows, results_dir / "rebuttal_backtracking.md")


if __name__ == "__main__":
    main()
