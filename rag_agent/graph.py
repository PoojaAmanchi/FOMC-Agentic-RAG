"""
THE GRAPH -- wiring the 4 nodes into one agentic system
===========================================================
"""
from langgraph.graph import END, StateGraph

from rag_agent.nodes.retriever import retriever_node
from rag_agent.nodes.router import router_node
from rag_agent.nodes.synthesizer import synthesizer_node
from rag_agent.nodes.verifier import verifier_node
from rag_agent.state import RAGState


def route_after_verification(state: RAGState) -> str:
    if state.get("final_answer") is not None:
        return END
    return "retriever"


def build_graph() -> StateGraph:
    graph = StateGraph(RAGState)

    graph.add_node("router", router_node)
    graph.add_node("retriever", retriever_node)
    graph.add_node("synthesizer", synthesizer_node)
    graph.add_node("verifier", verifier_node)

    graph.set_entry_point("router")

    graph.add_edge("router", "retriever")
    graph.add_edge("retriever", "synthesizer")
    graph.add_edge("synthesizer", "verifier")

    graph.add_conditional_edges(
        "verifier",
        route_after_verification,
        {END: END, "retriever": "retriever"},
    )

    return graph


def compile_graph():
    return build_graph().compile()


app = compile_graph()


def run_query(query: str) -> dict:
    initial_state: RAGState = {
        "query": query,
        "query_type": None,
        "relevant_dates": [],
        "retrieved_chunks": [],
        "draft_answer": None,
        "cited_sources": [],
        "is_grounded": None,
        "confidence_score": None,
        "retry_count": 0,
        "final_answer": None,
    }
    return app.invoke(initial_state)


if __name__ == "__main__":
    result = run_query(
        "Did the Fed's language about inflation change between January and June 2026?"
    )
    print("ANSWER:\n", result["final_answer"])
    print("\nGrounded:", result["is_grounded"])
    print("Confidence:", result["confidence_score"])
    print("Retries used:", result["retry_count"])
    print("Sources cited:", result["cited_sources"])
