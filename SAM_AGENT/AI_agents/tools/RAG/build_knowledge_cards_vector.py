import argparse
import json
import os
from pathlib import Path
from typing import Any, Dict, Iterable, List

from AI_agents.config.llm import RAG_EMBEDDING_MODEL
from AI_agents.config.paths import RAG_KNOWLEDGE_CARDS_DIR, RAG_VECTOR_DIR


SKIP_KEYS = {"_comment", "_instruction", "instruction"}
EMPTY_VALUES = {"", "N/A", "NA", "n/a", "None", "none", None}


def is_meaningful(value: Any) -> bool:
    if isinstance(value, str):
        return value.strip() not in EMPTY_VALUES
    if isinstance(value, (list, tuple, set, dict)):
        return bool(value)
    if value in EMPTY_VALUES:
        return False
    return True


def flatten_fields(value: Any, prefix: str = "") -> Iterable[str]:
    if isinstance(value, dict):
        for key, sub_value in value.items():
            if key in SKIP_KEYS:
                continue
            label = f"{prefix}.{key}" if prefix else key
            yield from flatten_fields(sub_value, label)
    elif isinstance(value, list):
        meaningful_items = [item for item in value if is_meaningful(item)]
        if meaningful_items:
            normalized = []
            for item in meaningful_items:
                if isinstance(item, (dict, list)):
                    normalized.append(json.dumps(item, ensure_ascii=False))
                else:
                    normalized.append(str(item))
            yield f"{prefix}: {'; '.join(normalized)}"
    elif is_meaningful(value):
        yield f"{prefix}: {value}"


def card_to_document(card: Dict[str, Any]) -> str:
    metadata = card.get("metadata", {})
    molecule = card.get("molecule_identity", {})
    header = [
        "Knowledge card for self-assembled monolayers in perovskite solar cells.",
        f"DOI: {metadata.get('DOI', 'N/A')}",
        f"Paper title: {metadata.get('paper_title', 'N/A')}",
        f"Journal: {metadata.get('journal', 'N/A')}",
        f"Year: {metadata.get('year', 'N/A')}",
        f"System ID: {metadata.get('system_id', 'N/A')}",
        f"System name: {metadata.get('system_name', 'N/A')}",
        f"System role: {metadata.get('system_role', 'N/A')}",
        f"Molecule full name: {molecule.get('full_name', 'N/A')}",
        f"Molecule abbreviation: {molecule.get('abbreviation', 'N/A')}",
        f"SMILES: {molecule.get('smiles', 'N/A')}",
    ]

    body = []
    for section_name in [
        "intrinsic_structure_features",
        "electronic_and_chemical_properties",
        "interface_effects",
        "processing_conditions",
        "device_performance",
        "practicality",
        "design_insights",
        "extraction_metadata",
    ]:
        section = card.get(section_name)
        if not section:
            continue
        section_lines = list(flatten_fields(section, section_name))
        if section_lines:
            body.append(f"\n[{section_name}]")
            body.extend(section_lines)

    return "\n".join(header + body)


def load_cards(input_path: Path) -> List[Dict[str, Any]]:
    with input_path.open("r", encoding="utf-8") as file:
        cards = json.load(file)
    if not isinstance(cards, list):
        raise ValueError("Expected a JSON array of knowledge cards.")
    return cards


def main() -> None:
    parser = argparse.ArgumentParser(description="Build a RAG vector store from SAM knowledge cards.")
    parser.add_argument(
        "--input",
        default=str(RAG_KNOWLEDGE_CARDS_DIR / "extracted_knowledge_cards_v3_20251124_with_smiles_manual_verification.json"),
        help="Path to the knowledge-card JSON file.",
    )
    parser.add_argument(
        "--output",
        default=str(RAG_VECTOR_DIR),
        help="Output directory for document.json and vectors.json.",
    )
    parser.add_argument(
        "--embedding-model",
        default=RAG_EMBEDDING_MODEL,
        help="OpenAI embedding model name.",
    )
    parser.add_argument(
        "--documents-only",
        action="store_true",
        help="Only write document.json without calling the embedding API.",
    )
    args = parser.parse_args()

    input_path = Path(args.input).resolve()
    output_path = Path(args.output).resolve()

    cards = load_cards(input_path)
    documents = [card_to_document(card) for card in cards]

    if args.documents_only:
        output_path.mkdir(parents=True, exist_ok=True)
        with (output_path / "document.json").open("w", encoding="utf-8") as file:
            json.dump(documents, file, ensure_ascii=False)
        print(f"Wrote {len(documents)} documents to {output_path / 'document.json'}")
        return

    from AI_agents.tools.RAG.vector_storage import VectorStore

    vector_store = VectorStore(document=documents)
    api_key = os.environ["OPENAI_API_KEY"]
    from AI_agents.tools.RAG.Embedding import OpenAIEmbedding_model

    embedding_model = OpenAIEmbedding_model(model_type=args.embedding_model, api_key=api_key)
    vector_store.get_vector(embedding_model)
    vector_store.persist(str(output_path))

    print(f"Built vector store with {len(documents)} documents at {output_path}")


if __name__ == "__main__":
    main()
