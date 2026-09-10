from functools import lru_cache

from langgraph.graph import END, START, StateGraph

from lib.retriever import hybrid_retrieve
from llm import get_llm
from prompts.answer_generation import get_answer_prompt
from prompts.query_analysis import QueryAnalysis, get_query_analysis_prompt
from workflows.states import GraphRAGState


def build_workflow(client, llm=None):
    model = llm or get_llm()

    def analyze_question(state: GraphRAGState) -> GraphRAGState:
        analysis = model.with_structured_output(QueryAnalysis).invoke(get_query_analysis_prompt().invoke({"question": state["user_question"]}))
        return {"rewritten_question": analysis.rewritten_question, "extracted_entities": analysis.entities}

    def retrieve(state: GraphRAGState) -> GraphRAGState:
        result = hybrid_retrieve(state["rewritten_question"], client, state.get("extracted_entities", []))
        return {"vector_results": result.vector_results, "graph_results": result.graph_results, "combined_context": result.combined_context, "sources": result.sources, "errors": result.errors}

    def generate_answer(state: GraphRAGState) -> GraphRAGState:
        if not state.get("combined_context", "").strip():
            return {"final_answer": "I could not find supporting evidence in the indexed documents."}
        response = model.invoke(get_answer_prompt().invoke({"question": state["user_question"], "context": state["combined_context"]}))
        return {"final_answer": response.content}

    graph = StateGraph(GraphRAGState)
    graph.add_node("analyze_question", analyze_question)
    graph.add_node("retrieve", retrieve)
    graph.add_node("generate_answer", generate_answer)
    graph.add_edge(START, "analyze_question")
    graph.add_edge("analyze_question", "retrieve")
    graph.add_edge("retrieve", "generate_answer")
    graph.add_edge("generate_answer", END)
    return graph.compile()
