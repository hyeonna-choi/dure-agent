"""
Graph generation script: Protocol Translation Accuracy across Validation Iterations
Generates a version per validator (claude / gpt-5 / llama-4-maverick)
"""

import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import numpy as np
import os

# ── Configuration ─────────────────────────────────────────────
EVAL_FOLDERS = {
    "claude-sonnet-4-6":  "claude_evaluation_results",
    "gpt-5":              "gpt-5_evaluation_results",
    "llama-4-maverick":   "llama-4-maverick_evaluation_results",
}

# Per-model colors (similar to the original graph colors)
MODEL_COLORS = {
    "gpt-5":             "#4472C4",   # blue
    "gpt-4.1":           "#2E9E6B",   # green
    "o4-mini":           "#C8A84B",   # gold
    "llama-4-maverick":  "#C0504D",   # red
    "gpt-4.1-mini":      "#E694C0",   # pink
    "llama-3.3-70b":     "#7F7F7F",   # gray
    "gpt-4.1-nano":      "#CCCCCC",   # light gray
}

MODEL_LABELS = {
    "gpt-5":             "GPT-5",
    "gpt-4.1":           "GPT-4.1",
    "o4-mini":           "o4-mini",
    "llama-4-maverick":  "llama-4-maverick",
    "gpt-4.1-mini":      "GPT-4.1-mini",
    "llama-3.3-70b":     "llama-3.3-70b",
    "gpt-4.1-nano":      "GPT-4.1-nano",
}

# Legend order (same as the original)
LEGEND_ORDER = ["gpt-5", "gpt-4.1", "o4-mini", "llama-4-maverick",
                "gpt-4.1-mini", "llama-3.3-70b", "gpt-4.1-nano"]

PHASES = ["before", "attempt_1", "attempt_2", "attempt_3"]
X_LABELS = ["before", "attempt 1", "attempt 2", "attempt 3"]

# ── Graph titles / axis labels (per validator, Korean/English versions) ──
GRAPH_CONFIG = {
    "claude-sonnet-4-6": {
        "en": {
            "title": "Protocol Accuracy per Validation\n[Claude Sonnet 4.6]",
            "xlabel": "Validation Round",
            "ylabel": "Parameter Accuracy\n(Steps & Values)",
            "filename": "accuracy_claude-sonnet-4-6_en.png",
        },
        "ko": {
            "title": "Protocol Accuracy per Validation Round\n[Claude Sonnet 4.6]",
            "xlabel": "Validation Round",
            "ylabel": "Parameter Accuracy\n(Steps & Values)",
            "filename": "accuracy_claude-sonnet-4-6_ko.png",
        },
    },
    "gpt-5": {
        "en": {
            "title": "Protocol Accuracy per Validation\n[GPT-5]",
            "xlabel": "Validation Round",
            "ylabel": "Parameter Accuracy\n(Steps & Values)",
            "filename": "accuracy_gpt-5_en.png",
        },
        "ko": {
            "title": "Protocol Accuracy per Validation Round\n[GPT-5]",
            "xlabel": "Validation Round",
            "ylabel": "Parameter Accuracy\n(Steps & Values)",
            "filename": "accuracy_gpt-5_ko.png",
        },
    },
    "llama-4-maverick": {
        "en": {
            "title": "Protocol Accuracy per Validation\n[Llama-4-Maverick]",
            "xlabel": "Validation Round",
            "ylabel": "Parameter Accuracy\n(Steps & Values)",
            "filename": "accuracy_llama-4-maverick_en.png",
        },
        "ko": {
            "title": "Protocol Accuracy per Validation Round\n[Llama-4-Maverick]",
            "xlabel": "Validation Round",
            "ylabel": "Parameter Accuracy\n(Steps & Values)",
            "filename": "accuracy_llama-4-maverick_ko.png",
        },
    },
}

OUTPUT_DIR = "graphs_output"
os.makedirs(OUTPUT_DIR, exist_ok=True)


def draw_accuracy_graph(validator_name: str, lang: str = "en"):
    """lang: 'en' or 'ko'"""
    folder = EVAL_FOLDERS[validator_name]
    csv_path = os.path.join(folder, "validation_loop_metrics_by_phase.csv")

    if not os.path.exists(csv_path):
        print(f"[SKIP] {csv_path} not found")
        return

    df = pd.read_csv(csv_path)
    pivot = df.pivot_table(index="Model", columns="Phase", values="Overall_Accuracy")

    cfg = GRAPH_CONFIG[validator_name][lang]

    # Korean font setting (for ko)
    if lang == "ko":
        plt.rcParams["font.family"] = "Malgun Gothic"
    else:
        plt.rcParams["font.family"] = "DejaVu Sans"
    plt.rcParams["axes.unicode_minus"] = False

    x_labels = ["before", "attempt 1", "attempt 2", "attempt 3"] if lang == "en" \
                else ["before", "attempt 1", "attempt 2", "attempt 3"]

    fig, ax = plt.subplots(figsize=(9, 6))
    fig.patch.set_facecolor("white")
    ax.set_facecolor("white")

    x = np.arange(len(PHASES))

    for model in LEGEND_ORDER:
        if model not in pivot.index:
            continue
        y = [pivot.loc[model, p] if p in pivot.columns else np.nan for p in PHASES]
        color = MODEL_COLORS.get(model, "#333333")
        label = MODEL_LABELS.get(model, model)
        ax.plot(x, y, color=color, linewidth=2.2, marker="o",
                markersize=6, label=label, zorder=3)

    # Axis settings
    ax.set_xticks(x)
    ax.set_xticklabels(x_labels, fontsize=12)
    ax.set_ylim(0, 1.05)
    ax.yaxis.set_major_formatter(mticker.FormatStrFormatter("%.1f"))
    ax.tick_params(axis="y", labelsize=11)

    # Grid
    ax.yaxis.grid(True, linestyle="--", alpha=0.4, color="#AAAAAA")
    ax.set_axisbelow(True)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    # Title / axis labels
    ax.set_title(cfg["title"], fontsize=13, fontweight="bold", pad=14)
    ax.set_xlabel(cfg["xlabel"], fontsize=12, labelpad=8)
    ax.set_ylabel(cfg["ylabel"], fontsize=12, labelpad=8)

    # Legend
    ax.legend(loc="center left", bbox_to_anchor=(1.01, 0.5),
              fontsize=10, frameon=False)

    plt.tight_layout()
    out_path = os.path.join(OUTPUT_DIR, cfg["filename"])
    plt.savefig(out_path, dpi=180, bbox_inches="tight")
    plt.close()
    print(f"[OK] saved: {out_path}")


if __name__ == "__main__":
    for v in EVAL_FOLDERS:
        for lang in ["en", "ko"]:
            draw_accuracy_graph(v, lang)
    print("Done.")
