"""
Rebuttal experiment: vision ablation across model families.

Extends the single-family Qwen3-VL vision drop to Gemma 3 27B, Gemma 4 31B (gemma-4-31B-it),
and Magistral Small, to test whether the text -> vision accuracy drop is family-specific or general.

Accuracy is the "Correctly Solved" percentage from the stats CSVs (same extractor as
plot_vision_comparison.py). Vision drop = accuracy(vision) - accuracy(text).

Outputs (results/rebuttal/):
  - rebuttal_vision.{png,pdf}   grouped bar chart (text vs vision) per model with delta labels
  - rebuttal_vision.md          markdown table

NOTE: the Magistral vision run is version 2509 while the only text baseline in the pool is 2507;
this is compared with a footnote caveat.
"""
import numpy as np
from pathlib import Path

import matplotlib.pyplot as plt

from plot_config import setup_plot_style, TEXT_WIDTH_INCHES
from plot_vision_comparison import extract_stats_from_csv


def _tint(hexcolor, t=0.55):
    """Lighten a hex color by blending it t-fraction toward white."""
    h = hexcolor.lstrip("#")
    r, g, b = int(h[0:2], 16), int(h[2:4], 16), int(h[4:6], 16)
    return f"#{int(r + (255 - r) * t):02x}{int(g + (255 - g) * t):02x}{int(b + (255 - b) * t):02x}"


# Single condition color-pair across all models (text = light tint, vision = full),
# matching the binary-condition convention used in plot_reasoning_comparison.py.
VISION_COLOR = "#1976D2"          # Spatial-Gym blue (VARIANT_COLORS["gym"])
TEXT_COLOR = _tint(VISION_COLOR, 0.55)

# Each entry: display name, color, text stats path, vision stats path, note flag
BASE = Path(__file__).parent
REB = BASE / "results" / "rebuttal"
MAIN = BASE / "results" / "spatial_gym"

MODELS = [
    {
        "display": "Qwen3-VL 32B",
        "color": "#C026D3",
        "text": MAIN / "Qwen_Qwen3-VL-32B-Thinking_gym_stats.csv",
        "vision": MAIN / "Qwen_Qwen3-VL-32B-Thinking_gym_visual_stats.csv",
        "note": "ref",
    },
    {
        "display": "Gemma 3 27B",
        "color": "#4285F4",
        "text": REB / "google_gemma-3-27b-it_gym_stats.csv",
        "vision": REB / "google_gemma-3-27b-it_gym_visual_visual_stats.csv",
        "note": None,
    },
    {
        "display": "Gemma 4 31B",
        "color": "#1A73E8",
        "text": REB / "google_gemma-4-31B-it_gym_stats.csv",
        "vision": REB / "google_gemma-4-31B-it_gym_visual_visual_stats.csv",
        "note": None,
    },
    {
        "display": "Magistral Small",
        "color": "#EA580C",
        "text": MAIN / "mistralai_Magistral-Small-2507_gym_stats.csv",
        "vision": REB / "mistralai_Magistral-Small-2509_gym_visual_visual_stats.csv",
        "note": "version",
    },
]


def collect():
    rows = []
    for m in MODELS:
        text_acc = extract_stats_from_csv(m["text"]).get("accuracy")
        vision_acc = extract_stats_from_csv(m["vision"]).get("accuracy")
        rows.append({**m, "text_acc": text_acc, "vision_acc": vision_acc,
                     "delta": vision_acc - text_acc})
    return rows


def create_chart(rows, output_path):
    setup_plot_style(use_latex=True)
    fig, ax = plt.subplots(figsize=(TEXT_WIDTH_INCHES * 0.82, 1.7))

    n = len(rows)
    x = np.arange(n)
    width = 0.38

    text_vals = [r["text_acc"] for r in rows]
    vision_vals = [r["vision_acc"] for r in rows]

    bars_text = ax.bar(x - width / 2, text_vals, width,
                       color=TEXT_COLOR, edgecolor="black", linewidth=0.5, label="Text")
    bars_vis = ax.bar(x + width / 2, vision_vals, width,
                      color=VISION_COLOR, edgecolor="black", linewidth=0.5, label="Vision")

    for bars, vals in ((bars_text, text_vals), (bars_vis, vision_vals)):
        for bar, v in zip(bars, vals):
            ax.annotate(f"{v:.1f}\\%",
                        xy=(bar.get_x() + bar.get_width() / 2, bar.get_height()),
                        xytext=(0, 1.5), textcoords="offset points",
                        ha="center", va="bottom", fontsize=5, fontweight="bold")

    # Delta labels centered above each pair
    ymax = max(max(text_vals), max(vision_vals))
    for xi, r in zip(x, rows):
        ax.annotate(f"$\\Delta={r['delta']:+.1f}$",
                    xy=(xi, max(r['text_acc'], r['vision_acc'])),
                    xytext=(0, 9), textcoords="offset points",
                    ha="center", va="bottom", fontsize=6.5, fontweight="bold",
                    color="#333333")

    labels = [r["display"] + ("$^\\dagger$" if r["note"] == "version" else "") for r in rows]
    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.set_ylabel("Accuracy (\\%)")
    ax.set_ylim(0, ymax * 1.45)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.yaxis.grid(True, linestyle="--", alpha=0.3)
    ax.set_axisbelow(True)
    ax.legend(loc="upper right", framealpha=0.9)
    ax.set_title("Text vs. Vision Accuracy Across Model Families", fontweight="bold")

    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches="tight")
    print(f"Figure saved to: {output_path}")
    plt.close(fig)


def write_markdown_table(rows, output_path):
    lines = []
    lines.append("## Vision ablation across model families (Spatial-Gym)")
    lines.append("")
    lines.append("Accuracy = \"Correctly Solved\" %. Δ = vision − text (negative = vision hurts).")
    lines.append("")
    lines.append("| Model | Text Acc. (%) | Vision Acc. (%) | Δ (pp) |")
    lines.append("|---|---:|---:|---:|")
    for r in rows:
        disp = r["display"] + ("$^\\dagger$" if r["note"] == "version" else "")
        if r["note"] == "ref":
            disp += " (ref.)"
        lines.append(f"| {disp} | {r['text_acc']:.1f} | {r['vision_acc']:.1f} | {r['delta']:+.1f} |")
    lines.append("")
    lines.append("**Notes.**")
    lines.append("- (ref.) Qwen3-VL 32B is the original single-family ablation, shown for reference.")
    lines.append("- $^\\dagger$ The Magistral vision run is **Magistral-Small-2509**, compared against the "
                 "**Magistral-Small-2507** text baseline in our pool (version mismatch).")
    lines.append("")
    lines.append("**Observation.** The text→vision accuracy drop is **not** specific to the Qwen family: "
                 "every model loses accuracy under the visual setting, with the largest drop for the "
                 "strongest text model (Gemma 4 31B).")
    lines.append("")
    Path(output_path).write_text("\n".join(lines))
    print(f"Markdown table saved to: {output_path}")


def main():
    rows = collect()
    print("\nVision ablation (text vs vision):")
    print("-" * 60)
    for r in rows:
        print(f"  {r['display']:<18s} text={r['text_acc']:5.1f}%  "
              f"vision={r['vision_acc']:5.1f}%  delta={r['delta']:+.1f}pp")
    print("-" * 60)

    for ext in ("png", "pdf"):
        create_chart(rows, REB / f"rebuttal_vision.{ext}")
    write_markdown_table(rows, REB / "rebuttal_vision.md")


if __name__ == "__main__":
    main()
