# A Dual-Agent Framework for Translating Biological Protocols in Self-Driving Laboratories

> A **Parser Agent + rule-based mapping engine + heterogeneous Validation Agent** framework that
> translates unstructured natural-language biological protocols into executable equipment commands,
> with **3-axis cross-verification** (completeness, parameter accuracy, execution order) and iterative
> **self-correction** — deployed end-to-end on the **KIMM BioForge-1** self-driving laboratory.

🔗 **Project page:** https://hyeonna-choi.github.io/dure-agent/
📄 **[Full Paper (PDF)](docs/paper.pdf)** · 🖼️ **[Poster (PDF)](docs/poster.pdf)** · 🎬 **Demo video**

> Presented at the **KSME Bio-Engineering Division Spring Conference, 2026** (Poster).

![architecture](docs/static/images/architecture.png)

## Overview

Biological protocols are written in natural language, whereas automation systems rely on predefined
control commands — a semantic gap that limits autonomous execution. Microplate experiments are
especially hard because well mapping, sample–reagent combinations, replicate placement, and parallel
dispensing must be controlled at once. We bridge this gap with a hybrid LLM + rule architecture:

1. **Parser Agent** — structures the raw protocol into a tagged step schema
   (`<KIT_TITLE>`, `<MANUAL>`, `<INSTRUMENT>`), separating user-performed from instrument-executed steps.
2. **Rule-Based Mapping Engine** — deterministically expands each structured step into device-level
   commands, enforcing physical constraints (tip usage, transfer paths, coordinate transforms). A
   single *Wash 4×* step expands into 34 device commands.
3. **Validation Agent** — a heterogeneous LLM cross-verifies on three axes
   (**completeness, parameter accuracy, execution order**) and returns PASS/FAIL with structured feedback.
4. **Auto-Regenerate (≤3×)** — on FAIL, the framework self-corrects and regenerates before hardware execution.

## Key Findings

- A **7 Parser × 3 Validator** sweep over 30 ELISA protocols (sampled from 1,000 collected) shows that
  the effect of cross-model verification depends not on heterogeneity alone but on the **Validator's
  critical verification capability**.
- **Claude Sonnet 4.6** produced structured feedback that drove the largest accuracy recovery — small
  Parsers climbed from ~0.4 toward 0.7–0.8 through iteration; liberal validators yielded no gains.
- **Small models recovered toward large-model accuracy** using inference-time validation and
  regeneration alone — no extra training or fine-tuning.
- The **rule-based engine outperformed LLM end-to-end mapping** on both accuracy and latency.
- End-to-end **Bradford total-protein quantification** was executed on **KIMM BioForge-1** directly from
  a natural-language protocol.

## Repository structure

```
.
├── docs/                      # GitHub Pages project page
│   ├── index.html
│   ├── paper.pdf
│   ├── poster.pdf
│   └── static/{images,videos,css}
└── README.md
```

## Authors

**Hyeonna Choi**¹, Jung Yup Kim², Hyuneui Lim¹˒†, Seunggyu Jeon¹˒†
¹ Dept. of Bionic Machinery, Research Institute of AI Robot, KIMM
² Nano-convergence Manufacturing Research Division, KIMM
† Corresponding authors

## Status

🚧 Conference work (2026). **Source code release pending institutional review.**
