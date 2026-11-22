from functools import partial
from typing import Any

from langgraph.graph import END, START, StateGraph

from app.graph import edges, nodes
from app.graph.nodes import GraphDeps
from app.graph.state import RAGState


def build_graph(deps: GraphDeps) -> Any:
    graph = StateGraph(RAGState)
    for name in (
        "validate_input",
        "retrieve",
        "evidence_gate",
        "rewrite_query",
        "generate",
        "verify_output",
        "fallback",
        "audit_and_finalize",
    ):
        graph.add_node(name, partial(getattr(nodes, name), deps=deps))

    graph.add_edge(START, "validate_input")
    graph.add_conditional_edges("validate_input", edges.after_validate)
    graph.add_conditional_edges("retrieve", edges.after_retrieve)
    graph.add_conditional_edges("evidence_gate", edges.after_gate)
    graph.add_conditional_edges("rewrite_query", edges.after_rewrite)
    graph.add_conditional_edges("generate", edges.after_generate)
    graph.add_conditional_edges("verify_output", edges.after_verify)
    graph.add_edge("fallback", "audit_and_finalize")
    graph.add_edge("audit_and_finalize", END)
    return graph.compile()
