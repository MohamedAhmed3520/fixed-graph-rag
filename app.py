import streamlit as st

from config.settings import get_settings
from evaluation.datasets import load_dataset
from evaluation.evaluator import evaluate_case, summarize
from ingestion.graph_writer import GraphWriter
from ingestion.pipeline import IngestionPipeline
from neo4j.connection import get_neo4j_client
from neo4j.schema import initialize_schema
from workflows.graph_rag_workflow import build_workflow


st.set_page_config(page_title="Local GraphRAG", page_icon="G", layout="wide")
settings = get_settings()
st.title(settings.app_name)
st.caption("Local document ingestion, vector retrieval, and knowledge-graph grounded answers")


def service_status() -> tuple[object | None, str]:
    try:
        client = get_neo4j_client()
        client.verify_connectivity()
        initialize_schema(client)
        return client, "Connected"
    except Exception as exc:
        return None, str(exc)


@st.cache_resource
def get_pipeline(_client):
    return IngestionPipeline(GraphWriter(_client), _client)


with st.sidebar:
    st.subheader("Services")
    st.write("Neo4j: checking...")
    client, connection_message = service_status()
    st.write("Neo4j: Connected" if client else f"Neo4j: unavailable ({connection_message})")
    missing = settings.missing_services()
    st.write("Configuration: ready" if not missing else f"Configuration needs: {', '.join(missing)}")
    st.divider()
    st.subheader("Ingest")
    uploads = st.file_uploader("Documents", type=["pdf", "txt", "md", "docx"], accept_multiple_files=True)
    allow_duplicate = st.checkbox("Allow explicit re-ingestion", value=False)
    ingest_clicked = st.button("Ingest documents", disabled=not uploads or not client, use_container_width=True)
    st.divider()
    st.subheader("Evaluation")
    evaluate_clicked = st.button("Run local evaluation", disabled=not client or not settings.openrouter_api_key, use_container_width=True)

if ingest_clicked and client:
    pipeline = get_pipeline(client)
    for upload in uploads:
        with st.status(f"Ingesting {upload.name}"):
            result = pipeline.ingest_bytes(upload.name, upload.getvalue(), allow_duplicate=allow_duplicate)
        st.write({"document": result.filename, "status": result.status, "chunks": result.chunks_created, "entities": result.entities_created, "relationships": result.relationships_created, "embeddings": result.embeddings_created, "warnings": result.warnings, "errors": result.errors})

if evaluate_clicked and client:
    cases = [case for case in load_dataset("data/evaluation/dataset.json") if case.get("question")]
    if not cases:
        st.info("Not evaluated: add verified questions to data/evaluation/dataset.json.")
    else:
        workflow = build_workflow(client)
        results = [evaluate_case(case, workflow.invoke({"user_question": case["question"]})) for case in cases]
        st.session_state.evaluation_report = summarize(results)

if st.session_state.get("evaluation_report"):
    st.subheader("Evaluation results")
    st.json(st.session_state.evaluation_report)
else:
    st.caption("Evaluation: Not evaluated")

if "messages" not in st.session_state:
    st.session_state.messages = []
for message in st.session_state.get("messages", []):
    with st.chat_message(message["role"]):
        st.markdown(message["content"])
        if message.get("sources"):
            with st.expander("Sources"):
                st.json(message["sources"])
        if message.get("debug"):
            with st.expander("Debug / Trace"):
                st.json(message["debug"])

question = st.chat_input("Ask about your indexed documents")
if question:
    st.session_state.messages.append({"role": "user", "content": question})
    with st.chat_message("user"):
        st.markdown(question)
    with st.chat_message("assistant"):
        if not client:
            answer = "Neo4j is unavailable. Configure the local database before asking questions."
            st.markdown(answer)
        elif not settings.openrouter_api_key:
            answer = "OPENROUTER_API_KEY is not configured."
            st.markdown(answer)
        else:
            try:
                workflow = build_workflow(client)
                output = workflow.invoke({"user_question": question})
                answer = output.get("final_answer", "No answer was produced.")
                st.markdown(answer)
                if output.get("sources"):
                    with st.expander("Sources"):
                        st.json(output["sources"])
                with st.expander("Debug / Trace"):
                    st.json({"entities": output.get("extracted_entities", []), "vector_results": len(output.get("vector_results", [])), "graph_results": len(output.get("graph_results", [])), "errors": output.get("errors", [])})
                st.session_state.messages.append({"role": "assistant", "content": answer, "sources": output.get("sources", []), "debug": output.get("debug_info", {})})
            except Exception as exc:
                answer = f"The workflow failed: {exc}"
                st.error(answer)
        if not st.session_state.messages or st.session_state.messages[-1].get("content") != answer:
            st.session_state.messages.append({"role": "assistant", "content": answer})
