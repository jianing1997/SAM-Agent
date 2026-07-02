import os
import token
from typing import Dict, List, Optional, Tuple, Union
import json
from AI_agents.tools.RAG.Embedding import OpenAIEmbedding_model
import numpy as np
from tqdm import tqdm
import tiktoken

class VectorStore:
    def __init__(self, document: Optional[List[str]] = None):
        self.document = document
        self.vectors = []
        
    # Count how many tokens cost for the input
    def count_tokens(self, text:str):
        """Returns the number of tokens in a text string."""
        encoding = tiktoken.get_encoding("cl100k_base")
        num_tokens = len(encoding.encode(text))
        return num_tokens
    
    # Split the content into batches by token limit
    def batch_by_token_limit(self, content: List[str], max_tokens: int = 250000):
        batch = []
        total_tokens = 0
        content = [text.strip().replace("\n", " ") for text in content]
        token_list = [self.count_tokens(i) for i in content]
        for idx, tokens in enumerate(token_list):
            if total_tokens + tokens <= max_tokens:
                batch.append(content[idx])
                total_tokens += tokens
            else:
                yield batch
                batch = [content[idx]]
                total_tokens = tokens
        if batch:
            yield batch
        
    def get_vector(self, EmbeddingModel) -> List[List[float]]:
        '''
        Calculate the embedding of the a list of documents
        
        args:
            EmbeddingModel: the embedding model -> Here is the OpenAIEmbedding : 1. text-embedding-3-small 2. text-embedding-3-large 3. text-embedding-ada-002
            https://platform.openai.com/docs/guides/embeddings
        '''
        for batch in tqdm(self.batch_by_token_limit(self.document, max_tokens=300000), desc="Batching embeddings"):
            batch_embeddings = EmbeddingModel.document_embedding(batch)
            self.vectors.extend(batch_embeddings)
        return self.vectors

    # Write the vectors and document to a json file
    def persist(self, path: str = 'storage'):
        if not os.path.exists(path):
            os.makedirs(path)
        with open(f"{path}/document.json", 'w', encoding='utf-8') as f:
            json.dump(self.document, f, ensure_ascii=False)
        if self.vectors:
            with open(f"{path}/vectors.json", 'w', encoding='utf-8') as f:
                json.dump(self.vectors, f)

    def load_vector(self, path: str = 'storage'):
        with open(f"{path}/vectors.json", 'r', encoding='utf-8') as f:
            self.vectors = json.load(f)
        with open(f"{path}/document.json", 'r', encoding='utf-8') as f:
            self.document = json.load(f)

    def get_similarity(self, vector1: List[float], vector2: List[float], EmbeddingModel) -> float:
        return EmbeddingModel.cosine_similarity(vector1, vector2)

    def query(self, query: str, EmbeddingModel, k: int = 1) -> List[str]:
        query_vector = EmbeddingModel.query_embedding(query)
        result = np.array([self.get_similarity(query_vector, vector, EmbeddingModel)
                          for vector in self.vectors])
        return np.array(self.document)[result.argsort()[-k:][::-1]].tolist()