from collections.abc import AsyncGenerator
from decouple import config
from sqlalchemy import create_engine, text
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker, create_async_engine


DATABASE_URL = config("DATABASE_URL")
VECTOR_DB_TABLE_NAME = ""
VECTOR_DB_NAME = "langchain"
engine = create_async_engine(DATABASE_URL, echo=True)
async_session_maker = async_sessionmaker(engine, expire_on_commit=False)

def init_vector_db():
    global DATABASE_URL
    db_url = DATABASE_URL
    vector_db_name = VECTOR_DB_NAME
    vector_db_name = VECTOR_DB_TABLE_NAME
    engine = create_engine(db_url, isolation_level="AUTOCOMMIT")
    with engine.connect() as connection:
        result = connection.execute(text("SELECT 1 FROM pg_database WHERE datname = :db_name"),
                                         {"db_name": vector_db_name})
        db_exists = result.scalar() == 1
        if not db_exists:
            connection.execute(text('CREATE EXTENSION IF NOT EXISTS vector'))
            connection.execute(text(f"CREATE DATABASE {vector_db_name}"))


async def get_async_session() -> AsyncGenerator[AsyncSession, None]:
    async with async_session_maker() as session:
        yield session

