from decouple import config
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_groq import ChatGroq
from langchain_core.documents import Document
from langchain_postgres import PGVector
from langchain_postgres.vectorstores import PGVector

DATABASE_URL = config("DATABASE_URL")
LLAMA_KEY = config("LLAMA_API_KEY")
LLM_MODEL = config("LLM_MODEL")
EMBEDDING_MODEL = config("EMBEDDING_MODEL_L")
llm = ChatGroq( model=LLM_MODEL, api_key=LLAMA_KEY)

# embeddings = HuggingFaceEmbeddings()
model_name = EMBEDDING_MODEL
model_kwargs = {'device': 'cpu'}
encode_kwargs = {'normalize_embeddings': False}
hf = HuggingFaceEmbeddings(
    model_name=model_name,
    model_kwargs=model_kwargs,
    encode_kwargs=encode_kwargs
)

collection_name = "vector_db"

vector_store = PGVector(
    embeddings=hf,
    collection_name=collection_name,
    connection=DATABASE_URL,
    use_jsonb=True,
)