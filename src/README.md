# Dual-Agent Protocol Translation — Source Code

Reference implementation of the dual-agent framework that translates unstructured
natural-language biological protocols into executable commands for the KIMM
BioForge-1 robotic laboratory platform, with a deterministic rule-based mapping
engine and heterogeneous cross-model verification.

- **Project page:** https://hyeonna-choi.github.io/dure-agent/
- **arXiv:** https://arxiv.org/abs/2606.20120

This directory contains the **core framework** and the **evaluation harness**. The
interactive desktop GUI is not included in this release.

## Architecture

The pipeline is `Parser Agent → Rule Engine → Validation Agent → self-correction`:

| Component | Module | Role |
|---|---|---|
| Parser Agent | `bioforge/core/pdf_processor.py` | Structures a natural-language protocol (PDF/text) into a tagged representation (`<KIT_TITLE>`, `<MANUAL>`, `<INSTRUMENT>`). |
| Rule Engine | `bioforge/core/sequence_builder.py` | Deterministically compiles the structured protocol into a device-level command sequence under the platform's physical constraints. |
| Validation Agent | `bioforge/core/validator_claude.py` | Heterogeneous-LLM verifier checking completeness, parameter accuracy, and execution order; emits PASS/FAIL with structured feedback. |
| Direct baseline | `bioforge/core/direct_sequence_generator.py` | LLM end-to-end mapping baseline (no rule engine) used for comparison. |
| Data models | `bioforge/models/` | `protocol.py`, `sequence.py` — typed representations. |
| Configuration | `bioforge/config/settings.py` | Paths, model lists, defaults, command whitelist, deck/reservoir/handler mappings. |
| Evaluation | `evaluation/evaluator.py`, `evaluation/evaluator_core/` | Metric computation (parameter accuracy, pass rate) and report generation for the Parser×Validator sweep. |
| Figures | `evaluation/draw_accuracy_graph.py`, `evaluation/draw_direct_comparison.py` | Reproduce the accuracy/comparison plots. |

## Layout

```
src/
├── bioforge/                 # core framework
│   ├── config/               # settings (paths, models, rules)
│   ├── core/                 # parser, rule engine, validator, baseline
│   ├── models/               # protocol / sequence data models
│   └── utils/                # text, excel, file helpers
├── evaluation/               # evaluation harness and plotting
│   ├── evaluator.py
│   └── evaluator_core/
├── data/                     # rules (included) + your protocols (not included)
├── validate_standalone.py    # CLI for the validation stage alone
├── requirements.txt
└── .env.example
```

## Installation

```bash
python -m venv .venv && source .venv/bin/activate   # optional
pip install -r requirements.txt
cp .env.example .env        # then fill in your API keys
```

No API keys are hard-coded; every LLM client reads its key from an environment
variable (`ANTHROPIC_API_KEY` / `CLAUDE_API_KEY`, `OPENAI_API_KEY`,
`LLAMA_API_KEY`). See `.env.example`.

## Usage

**Standalone validation** of a structured protocol and/or command sequence:

```bash
python validate_standalone.py --mode all \
    --original   path/to/original_protocol.txt \
    --structured path/to/StructuredProtocol.txt \
    --sequence   path/to/Seq.xlsx \
    --rules      data/rules/bioforge_commands_and_rules.xlsx \
    --output     ./output
```

**Evaluation harness** (computes parameter accuracy and pass-rate tables once
model outputs and ground truth are in place; see `data/README.md`):

```bash
python evaluation/evaluator.py
```

## Note on language

Documentation, comments, and user-facing messages are in English. A number of
string literals remain in Korean **by design**: the structured-protocol DSL and
the keyword/regex patterns that parse it encode the BioForge-1 protocol format,
which is Korean-based. These are functional tokens — translating them would change
the parsing behavior — and are therefore preserved intentionally.

## Status

Conference and preprint work (2026). Provided for transparency and reproducibility;
a formal, packaged release is pending institutional review.

## Citation

```bibtex
@inproceedings{choi2026dualagent,
  author    = {Choi, Hyeonna and Kim, Jung Yup and Lim, Hyuneui and Jeon, Seunggyu},
  title     = {A Dual-Agent Framework for Translating Unstructured Biological
               Experimental Protocols with Cross-Verification in Self-Driving Laboratories},
  booktitle = {KSME Bio-Engineering Division Spring Conference},
  year      = {2026}
}
```
