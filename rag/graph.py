"""LangGraph RAG pipeline: retrieve context, then generate an answer."""

from __future__ import annotations

import os
from typing import TypedDict

from langchain_core.prompts import ChatPromptTemplate
from langchain_ollama import ChatOllama
from langgraph.graph import END, START, StateGraph

from rag.store import DEFAULT_TOP_K, VectorStore

DEFAULT_CHAT_MODEL = "llama3.2"
DEFAULT_OLLAMA_BASE_URL = "http://localhost:11434"

RAG_PROMPT = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            "You are a helpful assistant. Answer the user's question using only "
            "the provided context. If the answer is not in the context, say "
            "you don't know based on the uploaded documents.",
        ),
        (
            "human",
            "Context:\n{context}\n\nQuestion: {question}",
        ),
    ]
)


class RAGState(TypedDict):
    question: str
    context: str
    answer: str


def _chat_model() -> ChatOllama:
    return ChatOllama(
        model=os.getenv("OLLAMA_CHAT_MODEL", DEFAULT_CHAT_MODEL),
        base_url=os.getenv("OLLAMA_BASE_URL", DEFAULT_OLLAMA_BASE_URL),
        temperature=0.2,
    )


def build_retrieve_node(store: VectorStore, top_k: int = DEFAULT_TOP_K):
    """Return a LangGraph node that retrieves relevant chunks."""

    def retrieve(state: RAGState) -> dict[str, str]:
        docs = store.similarity_search(state["question"], k=top_k)
        context = "\n\n".join(doc.page_content for doc in docs)
        return {"context": context}

    return retrieve


def build_generate_node():
    """Return a LangGraph node that calls Ollama with retrieved context."""

    def generate(state: RAGState) -> dict[str, str]:
        if not state.get("context", "").strip():
            return {
                "answer": (
                    "No documents have been indexed yet. Upload one or more "
                    ".txt files and try again."
                )
            }

        chain = RAG_PROMPT | _chat_model()
        response = chain.invoke(
            {"context": state["context"], "question": state["question"]}
        )
        return {"answer": response.content}

    return generate


def build_rag_graph(store: VectorStore, top_k: int = DEFAULT_TOP_K):
    """Compile a linear retrieve-then-generate LangGraph."""
    graph = StateGraph(RAGState)
    graph.add_node("retrieve", build_retrieve_node(store, top_k=top_k))
    graph.add_node("generate", build_generate_node())
    graph.add_edge(START, "retrieve")
    graph.add_edge("retrieve", "generate")
    graph.add_edge("generate", END)
    return graph.compile()
