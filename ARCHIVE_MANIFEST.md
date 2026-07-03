# Archive Manifest

This file summarizes large data and model artifacts that are intentionally excluded from the GitHub source-code repository and deposited in the Zenodo archive with DOI `10.5281/zenodo.21132629`.

Inventory source: local full project directory `F:\projects\SAM_Agent_`

Inventory date: 2026-07-02

Zenodo DOI: `10.5281/zenodo.21132629`

Zenodo URL: `https://doi.org/10.5281/zenodo.21132629`

Zenodo all-versions DOI: `10.5281/zenodo.21132628`

Zenodo record version: `v20260703`

Archive file version: `v20260702`

## Summary

The public GitHub repository contains code, model training/evaluation scripts, configuration files, and small example notebooks. It excludes trained weights, generated molecular libraries, processed datasets, pretrained third-party model files, and runtime artifacts.

Published Zenodo archive package:

| Class | Count / files | Size |
| --- | ---: | ---: |
| Files staged in `overlay/` | 13,626 | 24.991 GB |
| Upload-ready tar packages | 7 tar files plus 2 checksum/manifest files | 25.001 GB |

## Upload-Ready Tar Packages

| Package | Size | Contents |
| --- | ---: | --- |
| `sam-agent-metadata-v20260702.tar` | 0.004 GB | Archive README, file-level manifest, checksums, copy log, and repository metadata template |
| `sam-agent-generated-data-v20260702.tar` | 0.726 GB | Generated SAM molecular libraries |
| `sam-agent-retrosynthesis-resources-v20260702.tar` | 7.488 GB | Retrosynthesis datasets and one-step model resources |
| `sam-agent-property-predictor-resources-v20260702.tar` | 3.006 GB | Property-predictor/Uni-Mol resources and weights |
| `sam-agent-decimer-resources-v20260702.tar` | 0.902 GB | DECIMER/image-extraction pretrained model resources |
| `sam-agent-fine-tuned-predictor-resources-v20260702.tar` | 12.235 GB | Fine-tuned predictor datasets, checkpoints, and analysis artifacts |
| `sam-agent-generator-vae-device-resources-v20260702.tar` | 0.639 GB | Structure-constrained generator, chemical VAE, and device-evaluator resources |

The local package also includes `archives_manifest.csv` and `archives_sha256.txt` for package-level size and SHA256 verification.

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

## Archived Contents

The Zenodo archive contains the following versioned artifact groups:

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

Code and model-training scripts are available at GitHub. Processed datasets, generated molecular libraries, trained model weights, and other large artifacts are available from the Zenodo archive at DOI: `10.5281/zenodo.21132629`.
