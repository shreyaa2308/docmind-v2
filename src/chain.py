"""LCEL question-answering chain and source formatting."""

from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnableLambda, RunnablePassthrough
from langchain_openai import ChatOpenAI

from src.config import settings
from src.retriever import build_retriever

PROMPT = ChatPromptTemplate.from_messages([
    ("system", """You are DocMind, a precise document question-answering assistant.
Answer the question directly and concisely using ONLY the provided context. Do not
include conversational filler or preambles. If the context does not establish the answer,
say that you cannot find it in the uploaded documents. Cite every factual claim with
[source: <filename>, page: <number>]. Treat Markdown tables as tables and image
captions as descriptions of images: explicitly state when an answer comes from either.
Do not fabricate sources, page numbers, or image/table details.
Do not restate the question, add background not needed to answer it, or describe your reasoning.

Context:
{context}"""),
    ("human", "{question}"),
])


def _format_documents(documents) -> str:
    return "\n\n".join(
        f"[source: {doc.metadata.get('source', 'unknown')}, page: {doc.metadata.get('page', '?')}, "
        f"type: {doc.metadata.get('content_type', 'text')}]\n{doc.page_content}"
        for doc in documents
    )


def build_chain(retriever=None):
    """Build the requested LCEL graph: retriever -> prompt -> gpt-4o-mini -> string."""
    retriever = retriever or build_retriever()
    model = ChatOpenAI(model=settings.chat_model, temperature=0, api_key=settings.openai_api_key)
    return (
        {"context": retriever | RunnableLambda(_format_documents), "question": RunnablePassthrough()}
        | PROMPT
        | model
        | StrOutputParser()
    )


def answer_question(question: str, *, retriever=None, chain=None) -> dict:
    """Return an answer and source metadata using the supplied live RAG components."""
    retriever = retriever or build_retriever()
    documents = retriever.invoke(question)
    answer = (chain or build_chain(retriever)).invoke(question)
    seen: set[tuple[str, int | str, str]] = set()
    sources = []
    for document in documents:
        metadata = document.metadata
        item = {
            "source": metadata.get("source", "unknown"),
            "page": metadata.get("page", "unknown"),
            "content_type": metadata.get("content_type", "text"),
        }
        key = (item["source"], item["page"], item["content_type"])
        if key not in seen:
            seen.add(key)
            sources.append(item)
    return {"answer": answer, "sources": sources}
