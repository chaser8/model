from pydantic import BaseModel, Extra
from langchain.embeddings.base import Embeddings
from typing import Any, List
from modelscope.pipelines import pipeline
from modelscope.utils.constant import Tasks

class ModelScopeEmbeddings(BaseModel, Embeddings):
    embed: Any
    model_id: str = ""
    """Model name to use."""

    def __init__(self, **kwargs: Any):
        """Initialize the modelscope"""
        super().__init__(**kwargs)
        try:
            self.embed = pipeline(Tasks.sentence_embedding, model=self.model_id)

        except ImportError as e:
            raise ValueError(
                "Could not import some python packages." "Please install it with `pip install modelscope`."
            ) from e

    class Config:
        extra = Extra.forbid

    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        texts = list(map(lambda x: x.replace("\n", " "), texts))
        inputs = {"source_sentence": texts}
        embeddings = self.embed(input=inputs)['text_embedding']
        return embeddings

    def embed_query(self, text: str) -> List[float]:
        text = text.replace("\n", " ")
        inputs = {"source_sentence": [text]}
        embedding = self.embed(input=inputs)['text_embedding'][0]
        return embedding
