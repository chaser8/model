import os

from langchain_community.vectorstores.pgvector import PGVector

from demo.model_scope_embeddings import ModelScopeEmbeddings

# os.system('pip install tiktoken')
# os.system('pip install tiktoken')
# os.system('pip install tiktoken')
# os.system('pip install tiktoken')
# os.system('pip install tiktoken')
VECTOR_DB_CONNECTION_ARGS = "postgresql+psycopg2://knowledge:knowledge@192.168.10.159:5432/knowledge"

embeddings = ModelScopeEmbeddings(model_id="damo/nlp_corom_sentence-embedding_chinese-base")

db = PGVector(embedding_function=embeddings, collection_name="knowledge", connection_string=VECTOR_DB_CONNECTION_ARGS)

search = db.search(query="",search_type="mmr")
for doc in search:
    print(doc)
