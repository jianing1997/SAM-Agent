# Archive Manifest

This file summarizes large data and model artifacts that are intentionally excluded from the GitHub source-code repository and should be deposited in a long-term external archive with a DOI.

Inventory source: local full project directory `F:\projects\SAM_Agent_`

Inventory date: 2026-07-02

Inventory baseline code commit: `985c199`

## Summary

The public GitHub repository contains code, model training/evaluation scripts, configuration files, and small example notebooks. It excludes trained weights, generated molecular libraries, processed datasets, pretrained third-party model files, and runtime artifacts.

Estimated archive candidates:

| Class | Count | Size |
| --- | ---: | ---: |
| Large/data/model candidate files | 13,403 | 25.132 GB |

## Major Local Artifact Groups

| Local path | Files | Size | Archive role |
| --- | ---: | ---: | --- |
| `models/fine_tuned_predictor/` | 13,482 | 12.229 GB | Property-prediction training datasets, checkpoints, and analysis artifacts |
| `SAM_AGENT/AI_agents/tools/retrosynthesis_planner/` | 202 | 7.498 GB | Retrosynthesis datasets and one-step model resources |
| `SAM_AGENT/AI_agents/tools/property_predictor/` | 106 | 3.006 GB | Uni-Mol/property-predictor weights and related processed artifacts |
| `SAM_AGENT/AI_agents/tools/literature_extractor/` | 74 | 0.903 GB | DECIMER and image-extraction pretrained model files |
| `data/` | 2 | 0.726 GB | Generated molecular libraries |
| `models/structure_constrained_molecular_generator/` | 61 | 0.512 GB | Molecular-generator datasets and artifacts |
| `models/chemical_vae/` | 50 | 0.095 GB | Chemical VAE resources |
| `models/device_evaluator/` | 12 | 0.032 GB | Device-evaluator tabular/model resources |

## File-Type Summary

| Extension | Count | Size |
| --- | ---: | ---: |
| `.pth` | 75 | 13.234 GB |
| `.pt` | 19 | 7.458 GB |
| `.csv` | 89 | 2.474 GB |
| `.ckpt` | 1 | 0.733 GB |
| `.data-00000-of-00001` | 2 | 0.596 GB |
| `.h5` | 6 | 0.285 GB |
| `.pkl` | 6 | 0.142 GB |
| `.dat` | 1 | 0.083 GB |
| `.zip` | 1 | 0.074 GB |
| `.pb` | 2 | 0.052 GB |
| `.sdf` | 13,199 | less than 0.001 GB total |

## Recommended Archive Contents

Deposit the following as a versioned archive package:

- `data/generated/generated_molcule_1.csv`
- `data/generated/generated_molcule_2.csv`
- `SAM_AGENT/AI_agents/tools/retrosynthesis_planner/retro_star/retro_data/`
- `SAM_AGENT/AI_agents/tools/property_predictor/` model-weight folders and processed predictor artifacts
- `SAM_AGENT/AI_agents/tools/literature_extractor/image_extraction/core/decimer_*` pretrained model files
- `models/fine_tuned_predictor/` large training datasets, `.pt/.pth` checkpoints, and prediction artifacts
- `models/structure_constrained_molecular_generator/train/datasets/`
- Any final RAG knowledge-card/vector-index files required to reproduce the manuscript workflow

Do not archive local-only or unrelated material such as `.git/`, `.env`, editor caches, Python caches, or unrelated JACS manuscript/PDF files.

## Suggested Manuscript Wording

Code and model-training scripts are available at GitHub. Processed datasets, generated molecular libraries, trained model weights, and other large artifacts are available from the external archive at DOI: `to be added`.
