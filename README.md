# TcadGPT Core Release (Elmer + TCAD)

This release bundle contains the core scripts used for the Elmer and TCAD experiments described in the paper. It excludes large datasets, model weights, and generated outputs. API keys are redacted and must be provided by the user.

## Contents

- `elmer/`
  - `scripts/`: QA generation from documentation (keyword extraction + Alpaca QA)
  - `IR_DPO_ELMER/`: IR → DPO pipeline for Elmer `.sif`
  - `QA_test/`: Elmer QA test set and model outputs
- `tcad/`
  - `scripts/`: TCAD QA generation (keyword extraction + Alpaca QA)
  - `code_test/`: TCAD instruction-to-code examples (`.txt` + `.cmd`)
  - `QA_test/`: TCAD QA test set and model outputs

## External Dependencies (not included)

- **ElmerFEM repo** (for meshes/examples): https://github.com/ElmerCSC/elmerfem
- **ElmerSolver** binary (build from ElmerFEM or install locally)
- Any proprietary TCAD toolchain referenced by the TCAD scripts
- Model checkpoints / datasets used in training

## Notes on Evaluation

- For Elmer code execution, model outputs should be cleaned to remove non-`.sif` text (e.g., explanations/Markdown). Mesh references should point to existing meshes (Mesh DB corrected or generated via ElmerGrid if needed).
- QA training data is produced using the Pipeline-2 QA synthesis approach from documentation; the 100-question QA test set is written manually from documentation and consists of single-fact questions.

## API Keys

All API keys in this release have been redacted (shown as `sk-REDACTED`). Set your keys via environment variables or edit the scripts as needed.

## Quick Pointers

- Elmer QA generation: `elmer/scripts/kaywords_gen_V6.py`, `elmer/scripts/data_gen_from_keywords_v4-Deepseek.py`
- Elmer IR → DPO pipeline: `elmer/IR_DPO_ELMER/`
- Elmer QA test set: `elmer/QA_test/Elmer_QA_testset.txt`
- TCAD QA generation: `tcad/scripts/kaywords_gen_V6.py`, `tcad/scripts/data_gen_from_keywords_v4-Deepseek.py`, `tcad/scripts/data_gen_parallel_v6-general.py`
- TCAD code examples: `tcad/code_test/`
- TCAD QA test set: `tcad/QA_test/TCAD_QA_testset.xlsx`
