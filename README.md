# SAM-Agent

SAM-Agent is a multi-agent AI framework for end-to-end discovery of self-assembled monolayer (SAM) molecules for perovskite optoelectronics. It integrates literature knowledge extraction, structure-constrained molecular generation, molecular property prediction, synthetic-feasibility assessment, device-level evaluation, and a web interface.

This public repository contains the code release for the manuscript:

**Multi-Agent AI Enables End-to-End Discovery of Functional Molecules for Optoelectronics**

Model training/evaluation scripts are included in this GitHub repository. Large model weights, generated molecular libraries, processed datasets, and other bulky artifacts are not stored here. They will be archived in an external data repository with a DOI.

## Repository Contents

```text
SAM-Agent/
├── SAM_AGENT/
│   ├── AI_agents/
│   │   ├── SAMAgent.py
│   │   ├── config/
│   │   └── tools/
│   │       ├── molecular_generator/
│   │       ├── property_predictor/
│   │       ├── device_evaluator/
│   │       ├── retrosynthesis_planner/
│   │       ├── literature_extractor/
│   │       ├── molecular_informatics_tools/
│   │       └── RAG/
│   ├── webapp/
│   └── run_webapp.py
├── models/
│   ├── chemical_vae/
│   ├── device_evaluator/
│   ├── fine_tuned_predictor/
│   └── structure_constrained_molecular_generator/
├── experiments/
│   └── screening_workflow.ipynb
├── environment.yml
├── LICENSE
└── README.md
```

## Data and Model Availability

This GitHub repository intentionally excludes large files, including trained model weights, generated molecular libraries, retrosynthesis datasets, RAG vector indexes, and full processed datasets.

After archival, please cite and download the full data/model package from:

```text
DOI: to be added
```

Expected external artifacts include:

- trained structure-constrained molecular generator weights
- Uni-Mol/property-prediction weights and processed datasets
- device-evaluator models and tabular datasets
- retrosynthesis resources used by the workflow
- generated SAM candidate libraries and screening outputs
- RAG knowledge cards and vector indexes

## Installation

Create the conda environment from the project root:

```bash
conda env create -f environment.yml
conda activate sam-agent
```

If the environment already exists:

```bash
conda env update -f environment.yml --prune
conda activate sam-agent
```

## API Keys

Copy the environment template and fill in your own keys:

```bash
cd SAM_AGENT
cp .env.example .env
```

On Windows PowerShell:

```powershell
cd SAM_AGENT
Copy-Item .env.example .env
```

Required and optional keys:

```env
OPENAI_API_KEY=your_openai_or_chatanywhere_key
TAVILY_API_KEY=your_tavily_key
DeepSeek_API_KEY=your_deepseek_key
MOLPORT_API_KEY=your_molport_key
```

`OPENAI_API_KEY` is required for ChatGPT-compatible LLM calls and RAG embeddings. `TAVILY_API_KEY`, `DeepSeek_API_KEY`, and `MOLPORT_API_KEY` are optional and only required for the corresponding tools.

## Launch the Web App

```bash
cd SAM_AGENT
python run_webapp.py
```

Then open:

- `http://localhost:8000`
- `http://localhost:8000/docs`

## Minimal Usage

```python
import os
from AI_agents.SAMAgent import SAMMultiAIAgent

agent = SAMMultiAIAgent(
    open_ai_key=os.environ.get("OPENAI_API_KEY"),
    deepseek_key=os.environ.get("DeepSeek_API_KEY"),
    tavily_key=os.environ.get("TAVILY_API_KEY"),
    llm_model="chatgpt",
)

response = agent.invoke(
    "Summarize the interfacial role of MeO-2PACz in perovskite solar cells with literature support.",
    session_id="demo",
)

print(response["output"])
```

## Notes

- The molecular generator expects scaffold SMILES inputs rather than molecule names.
- Large model checkpoints and generated files should be downloaded from the external data archive before running full reproduction workflows.
- The `models/` directory contains training, evaluation, and analysis scripts only; weights and large datasets are excluded from GitHub.
- Local runtime files are written to `SAM_AGENT/runtime/` and are ignored by Git.
- Do not commit `.env` or other files containing API keys.

## Citation

Citation details will be added after manuscript publication or preprint release.

## License

This project is released under the MIT License.
