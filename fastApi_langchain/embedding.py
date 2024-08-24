from decouple import config
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_groq import ChatGroq
from langchain_core.documents import Document
from langchain_postgres import PGVector
from langchain_postgres.vectorstores import PGVector
from .db import DATABASE_URL

LLAMA_KEY = config("LLAMA_API_KEY")
LLM_MODEL = config("LLM_MODEL")
EMBEDDING_MODEL = config("EMBEDDING_MODEL_2")
llm = ChatGroq( model=LLM_MODEL, api_key=LLAMA_KEY)

embeddings = HuggingFaceEmbeddings(model=EMBEDDING_MODEL)

collection_name = "my_docs"


vector_store = PGVector(
    embeddings=embeddings,
    collection_name=collection_name,
    connection=DATABASE_URL,
    use_jsonb=True,
)