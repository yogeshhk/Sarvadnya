import asyncio
from typing import List
from sentence_transformers import SentenceTransformer
from llama_index.core.embeddings import BaseEmbedding
from pydantic.v1 import PrivateAttr

class MiniLMEmbedding(BaseEmbedding):
    _model: SentenceTransformer = PrivateAttr()

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self._model = SentenceTransformer("all-MiniLM-L6-v2")

    def _get_text_embedding(self, text: str) -> List[float]:
        return self._model.encode(text).tolist()

    def _get_query_embedding(self, query: str) -> List[float]:
        return self._model.encode(query).tolist()

    async def _aget_query_embedding(self, query: str) -> List[float]:
        return await asyncio.to_thread(self._get_query_embedding, query)
