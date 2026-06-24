# Data

The framework reads input protocols and rule definitions from this directory and
writes results to `../output/`.

```
data/
├── rules/
│   └── bioforge_commands_and_rules.xlsx   # device command set and mapping rules
├── input/
│   └── pdfs/                              # your protocol PDFs (not included)
└── protocol_gt/                          # evaluation ground truth (not included)
    ├── structuredProtocol/               # *_StructuredProtocol.txt
    └── sequences/                        # *_Seq.xlsx
```

- `rules/bioforge_commands_and_rules.xlsx` is included: it defines the BioForge-1
  command whitelist and the rule-engine mapping used to compile structured
  protocols into device-level commands.
- `input/pdfs/` and `protocol_gt/` are **not** distributed. The evaluation in the
  paper used commercial ELISA kit manuals, which are third-party copyrighted
  documents. Place your own protocol PDFs in `input/pdfs/`, and your own ground
  truth in `protocol_gt/`, to run the pipeline and the evaluation harness.
