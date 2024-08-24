
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser
from .embedding import vector_store, llm

message = """
Answer this question using the provided context only.

{question}

Context:
{context}
"""

prompt = ChatPromptTemplate.from_messages([("human", message)])

retriever = vector_store.as_retriever(
    search_type='similarity',
    search_kwargs={'k': 2}
)

parser = StrOutputParser()


def get_chain():
    return {"context": retriever, "question": RunnablePassthrough()} | prompt | llm | parser


