from pathlib import Path


AI_AGENTS_DIR = Path(__file__).resolve().parents[1]
SAM_AGENT_DIR = AI_AGENTS_DIR.parent
PROJECT_ROOT = SAM_AGENT_DIR.parent

TOOLS_DIR = AI_AGENTS_DIR / "tools"
RUNTIME_DIR = SAM_AGENT_DIR / "runtime"
RUNTIME_DIR.mkdir(parents=True, exist_ok=True)

SMILES_INPUT_CSV = RUNTIME_DIR / "smiles_data.csv"
GENERATED_DATA_CSV = RUNTIME_DIR / "generated_data.csv"

RAG_DIR = TOOLS_DIR / "RAG"
RAG_KNOWLEDGE_CARDS_DIR = RAG_DIR / "knowledge_cards"
RAG_VECTOR_DIR = RAG_DIR / "SAM_knowledge_cards_vector"
