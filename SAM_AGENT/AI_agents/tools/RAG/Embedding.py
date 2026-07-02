from langchain_openai import OpenAIEmbeddings
from typing import List
import numpy as np

class OpenAIEmbedding_model:
    def __init__(self,model_type,api_key):
        self.api=api_key
        self.model=OpenAIEmbeddings(model=model_type,openai_api_base='https://api.chatanywhere.tech',openai_api_key=api_key)
     
    # document embedding, it receives a list of strings and returns a list of vectors, details could be found in https://python.langchain.com/docs/how_to/embed_text/
    def document_embedding(self,content:List[str]):
        if self.api:
            content = [text.strip().replace("\n", " ") for text in content]
            return self.model.embed_documents(content)
        else:
            raise NotImplementedError
        
    # query embedding, it receives a string and returns a vector    
    def query_embedding(self,content:str):
        if self.api:
            content = content.replace("\n", " ")
            return self.model.embed_query(content)
        else:
            raise NotImplementedError
    # Allow multiple queries to be embedded at once    
    def multi_query_embedding(self,content:List[str]):
        if self.api:
            content = [text.strip().replace("\n", " ") for text in content]
            return self.model.embed_query(content)
        else:
            raise NotImplementedError
        
    @staticmethod
    def cosine_similarity(vector1: List[float], vector2: List[float]) -> float:
        """
        calculate cosine similarity between two vectors
        """
        dot_product = np.dot(vector1, vector2)
        magnitude = np.linalg.norm(vector1) * np.linalg.norm(vector2)
        if not magnitude:
            return 0
        return dot_product / magnitude
        
        