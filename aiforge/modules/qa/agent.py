"""LangGraph StateGraph agent for document Q&A with agentic refinement.

The graph implements a classify → retrieve → reason → respond pipeline with
a conditional loop: if the *reason* node deems the retrieved context
insufficient, control flows back to *retrieve* with a refined query (up to
``max_iterations`` times).

Usage::

    graph = build_qa_graph(llm_fn=my_llm, retrieve_fn=my_retriever)
    result = graph.invoke(initial_state)
    print(result["answer"])
"""

from __future__ import annotations

import operator
from typing import Annotated, Any, Callable, TypedDict

from langgraph.graph import END, StateGraph

from aiforge.modules.qa.prompts import (
    CLASSIFY_PROMPT,
    REASON_PROMPT,
    RESPOND_PROMPT,
)


# ---------------------------------------------------------------------------
# Agent state
# ---------------------------------------------------------------------------


class QAAgentState(TypedDict):
    """State passed through the QA agent graph.

    Fields annotated with ``operator.add`` use LangGraph's *reducer*
    semantics: node return values are **accumulated** rather than replaced.
    """

    query: str
    repo_filter: str | None
    intent: str
    retrieved_chunks: Annotated[list[dict[str, Any]], operator.add]
    current_query: str
    iteration: int
    max_iterations: int
    trace: Annotated[list[dict[str, Any]], operator.add]
    answer: str
    sources: list[str]
    confidence: float
    decision: str
    total_input_tokens: Annotated[int, operator.add]
    total_output_tokens: Annotated[int, operator.add]
    model: str


# ---------------------------------------------------------------------------
# Graph builder
# ---------------------------------------------------------------------------


def build_qa_graph(
    llm_fn: Callable[[str], Any],
    retrieve_fn: Callable[..., list[dict[str, Any]]],
    retrieval_k: int = 6,
) -> Any:
    """Build and compile the RAG agent graph.

    Args:
        llm_fn:      ``(prompt: str) → CompletionResult`` — wraps the module's
                     ``_call_llm`` method.
        retrieve_fn: ``(query, top_k, filter) → list[dict]`` — embeds the query
                     and returns vector store hits.
        retrieval_k: Number of chunks to retrieve per iteration.

    Returns:
        A compiled LangGraph ``Runnable``.
    """

    # -- nodes ---------------------------------------------------------------

    def classify_node(state: QAAgentState) -> dict[str, Any]:
        prompt = CLASSIFY_PROMPT.format(query=state["query"])
        result = llm_fn(prompt)
        intent = result.text.strip().lower()
        return {
            "intent": intent,
            "trace": [{"step": "classify", "intent": intent}],
            "total_input_tokens": result.input_tokens,
            "total_output_tokens": result.output_tokens,
            "model": result.model,
        }

    def retrieve_node(state: QAAgentState) -> dict[str, Any]:
        q = state["current_query"]
        filt: dict[str, Any] | None = None
        if state.get("repo_filter"):
            filt = {"source": {"$contains": state["repo_filter"]}}

        hits = retrieve_fn(q, top_k=retrieval_k, filter=filt)

        new_chunks: list[dict[str, Any]] = []
        new_sources = list(state["sources"])
        for h in hits:
            src = h.get("metadata", {}).get("source", "unknown")
            new_chunks.append({"text": h["text"], "source": src})
            if src not in new_sources:
                new_sources.append(src)

        return {
            "retrieved_chunks": new_chunks,
            "sources": new_sources,
            "iteration": state["iteration"] + 1,
            "trace": [
                {
                    "step": "retrieve",
                    "query": q,
                    "chunks_found": len(hits),
                    "iteration": state["iteration"] + 1,
                }
            ],
        }

    def reason_node(state: QAAgentState) -> dict[str, Any]:
        recent = state["retrieved_chunks"][-retrieval_k:]
        context = "\n\n---\n\n".join(c["text"] for c in recent) if recent else ""

        prompt = REASON_PROMPT.format(
            query=state["query"],
            intent=state["intent"],
            iteration=state["iteration"],
            max_iterations=state["max_iterations"],
            context=context,
        )
        result = llm_fn(prompt)
        raw = result.text.strip()

        updates: dict[str, Any] = {
            "total_input_tokens": result.input_tokens,
            "total_output_tokens": result.output_tokens,
            "model": result.model,
        }

        if raw.upper().startswith("SUFFICIENT"):
            try:
                conf = float(raw.split(":")[1].strip())
                conf = max(0.0, min(1.0, conf))
            except (IndexError, ValueError):
                conf = 0.7
            updates["confidence"] = conf
            updates["decision"] = "sufficient"
            updates["trace"] = [
                {"step": "reason", "result": raw, "decision": "sufficient"}
            ]
        else:
            updates["decision"] = "insufficient"
            if ":" in raw:
                updates["current_query"] = raw.split(":", 1)[1].strip()
            updates["trace"] = [
                {"step": "reason", "result": raw, "decision": "insufficient"}
            ]

        return updates

    def respond_node(state: QAAgentState) -> dict[str, Any]:
        context = (
            "\n\n---\n\n".join(c["text"] for c in state["retrieved_chunks"])
            if state["retrieved_chunks"]
            else "No relevant documents found."
        )
        prompt = RESPOND_PROMPT.format(query=state["query"], context=context)
        result = llm_fn(prompt)
        return {
            "answer": result.text,
            "total_input_tokens": result.input_tokens,
            "total_output_tokens": result.output_tokens,
            "model": result.model,
            "trace": [{"step": "respond"}],
        }

    # -- routing -------------------------------------------------------------

    def should_continue(state: QAAgentState) -> str:
        if state["iteration"] >= state["max_iterations"]:
            return "respond"
        if state["decision"] == "sufficient":
            return "respond"
        return "retrieve"

    # -- assemble graph ------------------------------------------------------

    graph = StateGraph(QAAgentState)

    graph.add_node("classify", classify_node)
    graph.add_node("retrieve", retrieve_node)
    graph.add_node("reason", reason_node)
    graph.add_node("respond", respond_node)

    graph.set_entry_point("classify")
    graph.add_edge("classify", "retrieve")
    graph.add_edge("retrieve", "reason")
    graph.add_conditional_edges(
        "reason",
        should_continue,
        {"retrieve": "retrieve", "respond": "respond"},
    )
    graph.add_edge("respond", END)

    return graph.compile()
