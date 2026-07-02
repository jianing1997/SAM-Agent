from copy import deepcopy
from typing import Any, Dict


CHAT_MODEL_CONFIGS: Dict[str, Dict[str, Any]] = {
    "chatgpt": {
        "model_name": "gpt-4.1",
        "base_url": "https://api.chatanywhere.tech/v1",
        "temperature": 0.3,
        "max_tokens": None,
        "timeout": None,
        "max_retries": 8,
    },
    "deepseek": {
        "model_name": "deepseek-chat",
        "base_url": "https://api.deepseek.com",
        "temperature": 0.3,
        "max_tokens": None,
        "timeout": None,
        "max_retries": 8,
    },
}

RAG_CHAT_MODEL_OVERRIDES: Dict[str, Dict[str, Any]] = {
    "chatgpt": {
        "model_name": "gpt-4.1-mini",
        "temperature": 0,
    },
    "deepseek": {
        "temperature": 0,
    },
}

RAG_EMBEDDING_MODEL = "text-embedding-3-large"


def get_chat_model_config(engine: str, *, purpose: str = "agent") -> Dict[str, Any]:
    normalized_engine = engine.lower()
    if normalized_engine not in CHAT_MODEL_CONFIGS:
        raise ValueError("Unsupported LLM model. Please choose 'chatgpt' or 'deepseek'.")

    config = deepcopy(CHAT_MODEL_CONFIGS[normalized_engine])
    if purpose == "rag":
        config.update(RAG_CHAT_MODEL_OVERRIDES.get(normalized_engine, {}))
    elif purpose != "agent":
        raise ValueError("Unsupported LLM config purpose. Please choose 'agent' or 'rag'.")
    return config
