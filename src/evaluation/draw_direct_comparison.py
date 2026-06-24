"""
Direct Mode vs Standard Mode: Sequence Quality Comparison Graph
gpt-5-direct (LLM Mapping) vs Standard models (Rule-based Engine)
3 metrics: Step Completeness (F1), Parameter Accuracy (Exact Match), Step Order Accuracy
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import os

OUTPUT_DIR = "graphs_output"
os.makedirs(OUTPUT_DIR, exist_ok=True)

# ── Data ─────────────────────────────────────────────────────────────────────
# Standard models: based on claude evaluator, before phase, model_comparison.csv
# gpt-5-direct:    based on evaluation_results/model_comparison.csv
DATA = {
    "gpt-5":            {"completeness": 0.9994, "param_matched": 1.0000, "order": 1.0000},
    "gpt-4.1":          {"completeness": 0.9946, "param_matched": 0.9433, "order": 1.0000},
    "o4-mini":          {"completeness": 0.9688, "param_matched": 0.7647, "order": 1.0000},
    "gpt-4.1-mini":     {"completeness": 0.9393, "param_matched": 0.6447, "order": 1.0000},
    "llama-4-maverick": {"completeness": 0.7875, "param_matched": 0.5661, "order": 0.9982},
    "llama-3.3-70b":    {"completeness": 0.7052, "param_matched": 0.3557, "order": 0.9627},
    "gpt-5-direct":     {"completeness": 0.9095, "param_matched": 0.2656, "order": 0.9960},
}

MODEL_COLORS = {
    "gpt-5":             "#4472C4",
    "gpt-4.1":           "#2E9E6B",
    "o4-mini":           "#C8A84B",
    "gpt-4.1-mini":      "#E694C0",
    "llama-4-maverick":  "#C0504D",
    "llama-3.3-70b":     "#7F7F7F",
    "gpt-5-direct":      "#FF6B00",   # orange highlight
}

MODEL_LABELS = {
    "gpt-5":             "GPT-5",
    "gpt-4.1":           "GPT-4.1",
    "o4-mini":           "o4-mini",
    "gpt-4.1-mini":      "GPT-4.1-mini",
    "llama-4-maverick":  "Llama-4-Maverick",
    "llama-3.3-70b":     "Llama-3.3-70B",
    "gpt-5-direct":      "GPT-5 (Direct)",
}

MODEL_ORDER = [
    "gpt-5", "gpt-4.1", "o4-mini", "gpt-4.1-mini",
    "llama-4-maverick", "llama-3.3-70b",
    "gpt-5-direct",   # direct comes last
]

METRICS      = ["completeness", "param_matched", "order"]

GRAPH_CONFIG = {
    "en": {
        "metric_labels": [
            "Step Completeness\n(F1)",
            "Parameter Accuracy\n(Exact Match)",
            "Step Order\nAccuracy",
        ],
        "title":   "Sequence Quality: Standard (Rule-based) vs Direct (LLM Mapping)\n[Validation Agent: Claude Sonnet 4.6]",
        "ylabel":  "Score",
        "legend_standard": "Standard Pipeline\n(Rule-based Engine)",
        "legend_direct":   "Direct Pipeline\n(LLM Mapping)",
        "note":    "Parameter Accuracy = exact match between generated and reference sequence parameters",
        "filename": "direct_vs_standard_seq_en.png",
    },
    "ko": {
        "metric_labels": [
            "Step Completeness\n(F1)",
            "Parameter Accuracy\n(Exact Match)",
            "Step Order\nAccuracy",
        ],
        "title":   "Sequence Quality: Standard (Rule-based) vs Direct (LLM Mapping)\n[Validation Agent: Claude Sonnet 4.6]",
        "ylabel":  "Score",
        "legend_standard": "Standard Pipeline\n(Rule-based Engine)",
        "legend_direct":   "Direct Pipeline\n(LLM Mapping)",
        "note":    "Parameter Accuracy = exact match between generated and reference sequence parameters",
        "filename": "direct_vs_standard_seq_ko.png",
    },
}


def draw(lang: str = "en"):
    cfg = GRAPH_CONFIG[lang]

    if lang == "ko":
        plt.rcParams["font.family"] = "Malgun Gothic"
    else:
        plt.rcParams["font.family"] = "DejaVu Sans"
    plt.rcParams["axes.unicode_minus"] = False

    n_models  = len(MODEL_ORDER)
    n_metrics = len(METRICS)
    bar_w     = 0.10          # bar width
    group_gap = 0.35          # gap between metric groups
    sep_gap   = 0.06          # extra gap before the direct mode

    # Center X coordinate of each metric group
    group_centers = np.arange(n_metrics) * (n_models * bar_w + group_gap)

    fig, ax = plt.subplots(figsize=(13, 6))
    fig.patch.set_facecolor("white")
    ax.set_facecolor("white")

    # ── Draw bars ──
    for g, metric in enumerate(METRICS):
        for m_idx, model in enumerate(MODEL_ORDER):
            is_direct = (model == "gpt-5-direct")
            # The direct mode is shifted slightly further to the right
            extra = sep_gap if is_direct else 0.0

            x = group_centers[g] + m_idx * bar_w + extra - (n_models * bar_w) / 2
            val = DATA[model][metric]
            color = MODEL_COLORS[model]

            bar = ax.bar(
                x, val, width=bar_w,
                color=color,
                hatch="///" if is_direct else "",
                edgecolor="white" if not is_direct else "#AA3300",
                linewidth=0.5,
                alpha=0.92,
                zorder=3,
            )

    # ── X-axis labels (metric names) ──
    ax.set_xticks(group_centers)
    ax.set_xticklabels(cfg["metric_labels"], fontsize=12)

    # ── Y-axis ──
    ax.set_ylim(0, 1.08)
    ax.set_ylabel(cfg["ylabel"], fontsize=12, labelpad=8)
    ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda v, _: f"{v:.1f}"))
    ax.tick_params(axis="y", labelsize=10)

    # ── Grid ──
    ax.yaxis.grid(True, linestyle="--", alpha=0.35, color="#AAAAAA")
    ax.set_axisbelow(True)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    # ── Title ──
    ax.set_title(cfg["title"], fontsize=12, fontweight="bold", pad=12)

    # ── Legend (per model) ──
    legend_handles = []
    for model in MODEL_ORDER:
        is_direct = (model == "gpt-5-direct")
        patch = mpatches.Patch(
            facecolor=MODEL_COLORS[model],
            hatch="///" if is_direct else "",
            edgecolor="#AA3300" if is_direct else "white",
            label=MODEL_LABELS[model],
        )
        legend_handles.append(patch)

    # Empty handle used as a separator
    separator = mpatches.Patch(facecolor="none", edgecolor="none", label="")

    # Standard vs Direct group headers
    std_header  = mpatches.Patch(facecolor="none", edgecolor="none",
                                  label=f"── {cfg['legend_standard']}")
    dir_header  = mpatches.Patch(facecolor="none", edgecolor="none",
                                  label=f"── {cfg['legend_direct']}")

    full_handles = (
        [std_header] +
        legend_handles[:-1] +   # standard models
        [separator, dir_header] +
        [legend_handles[-1]]     # direct
    )

    ax.legend(
        handles=full_handles,
        loc="center left", bbox_to_anchor=(1.01, 0.5),
        fontsize=9, frameon=False,
        handlelength=1.4, handleheight=1.0,
    )

    # ── Annotation: direct mode separator line ──
    # Vertical dotted line before the direct bar in each group
    for g in range(n_metrics):
        direct_idx = MODEL_ORDER.index("gpt-5-direct")
        x_sep = group_centers[g] + direct_idx * bar_w + sep_gap - (n_models * bar_w) / 2 - bar_w * 0.6
        ax.axvline(x=x_sep, ymin=0, ymax=0.95, color="#AA3300",
                   linestyle=":", linewidth=1.2, alpha=0.6, zorder=2)

    plt.tight_layout()
    out_path = os.path.join(OUTPUT_DIR, cfg["filename"])
    plt.savefig(out_path, dpi=180, bbox_inches="tight")
    plt.close()
    print(f"[OK] saved: {out_path}")


if __name__ == "__main__":
    draw("en")
    draw("ko")
    print("Done.")
