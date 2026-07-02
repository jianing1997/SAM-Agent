from AI_agents.config.llm import RAG_EMBEDDING_MODEL, get_chat_model_config
from AI_agents.config.paths import RAG_VECTOR_DIR
from AI_agents.tools.RAG.Embedding import OpenAIEmbedding_model
from AI_agents.tools.RAG.vector_storage import VectorStore
from langchain_core.messages import HumanMessage, SystemMessage
from langchain_openai import ChatOpenAI
from pathlib import Path
import os



class RetrievalQA:
    def __init__(self,llm = 'chatgpt',path = None, open_ai_key=None, deepseek_key=None):
        self.data = VectorStore()
        self.path = path or RAG_VECTOR_DIR
        self.llm_name = llm
        self.open_ai_key = open_ai_key or os.environ.get('OPENAI_API_KEY')
        self.deepseek_key = deepseek_key or os.environ.get('DeepSeek_API_KEY')
        self.embedding_model = None
        self.llm = None

    def _get_embedding_model(self):
        if self.embedding_model is None:
            if not self.open_ai_key:
                raise ValueError("OPENAI_API_KEY is required for RAG embeddings.")
            self.embedding_model = OpenAIEmbedding_model(model_type=RAG_EMBEDDING_MODEL,api_key=self.open_ai_key)
        return self.embedding_model

    def _get_llm(self):
        if self.llm is not None:
            return self.llm
        if self.llm_name == 'deepseek':
            if not self.deepseek_key:
                raise ValueError("DeepSeek_API_KEY is required for DeepSeek RAG answers.")
            self.llm = ChatOpenAI(**get_chat_model_config(self.llm_name, purpose="rag"), api_key=self.deepseek_key)
        elif self.llm_name == 'chatgpt':
            if not self.open_ai_key:
                raise ValueError("OPENAI_API_KEY is required for ChatGPT RAG answers.")
            self.llm = ChatOpenAI(**get_chat_model_config(self.llm_name, purpose="rag"), api_key=self.open_ai_key)
        else:
            raise ValueError("Invalid LLM name. Choose either 'deepseek' or 'chatgpt'.")
        return self.llm

    @staticmethod
    def _resolve_vector_path(path) -> str:
        vector_path = Path(path)
        candidates = [vector_path] if vector_path.is_absolute() else [
            Path(__file__).resolve().parents[3] / vector_path,
            Path.cwd() / vector_path,
            Path(__file__).resolve().parent / vector_path,
        ]

        for candidate in candidates:
            if candidate.is_dir() and (candidate / "document.json").is_file():
                return str(candidate)

        checked_paths = ", ".join(str(candidate) for candidate in candidates)
        raise FileNotFoundError(
            f"Vector store document.json not found under one of: {checked_paths}"
        )

    def run(self, query: str, top_k: int = 1) -> str:
        try:
            vector_path = self._resolve_vector_path(self.path)
        except FileNotFoundError as exc:
            return str(exc)

        vector_file = Path(vector_path) / "vectors.json"
        if not vector_file.is_file():
            return (
                "RAG vector store is not ready: vectors.json is missing. "
                "Please run AI_agents.tools.RAG.build_knowledge_cards_vector without --documents-only "
                "after configuring OPENAI_API_KEY."
            )

        if query:
            self.data.load_vector(vector_path)
        context = ""
        embedding_model = self._get_embedding_model()
        for string in self.data.query(query, embedding_model, k= top_k):
            context += string + '\n'
        # Prepare the prompt
        messages = [
            SystemMessage(content="You are a helpful assistant. You will be given a section of an article and a question. Please answer the question based on the provided article section and provide the references."),
            HumanMessage(f'\n\narticle section:\n"""\n{context}\n"""'),
            HumanMessage(content=query),
        ]
            
        # Get the response from the LLM
        response = self._get_llm()(messages)
        return response.content