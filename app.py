"""JusticeMitra - AI-Powered Legal Information Assistant for Women in India.

Modernized presentation tier with Gemma generation, MongoDB Atlas Vector Search,
Render readiness, and Sentry Agent Tracing.
"""

import os
import json
import logging
from pathlib import Path
import streamlit as st

from justicemitra.config import config
from justicemitra.core.sanitizer import EMERGENCY_HELPLINES
from justicemitra.core.rag import JusticeMitraRAGPipeline, RAGResponse
from justicemitra.services.embedding import get_embedding_service
from justicemitra.services.database import get_mongo_client, get_legal_chunks_collection, check_mongo_connection
from justicemitra.services.retriever import AtlasVectorRetriever, LocalFallbackRetriever
from justicemitra.services.generation import GemmaOpenAIGateway
from justicemitra.observability.telemetry import setup_sentry_tracing, is_sentry_active

# Configure logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(name)s: %(message)s")
logger = logging.getLogger("justicemitra.app")

# Initialize Sentry Observability (disabled if SENTRY_DSN is absent)
setup_sentry_tracing()


@st.cache_resource
def get_cached_embedder():
    """Caches the singleton embedding model in memory."""
    return get_embedding_service(config.embedding_model_name)


@st.cache_resource
def load_fallback_corpus():
    """Loads offline legal corpus for development or offline testing fallback."""
    corpus_path = Path(__file__).resolve().parent / "data" / "legal_corpus.json"
    if corpus_path.exists():
        try:
            with open(corpus_path, "r", encoding="utf-8") as f:
                return json.load(f)
        except Exception as e:
            logger.error(f"Error loading local corpus JSON: {e}")
    return []


def initialize_pipeline() -> JusticeMitraRAGPipeline:
    """Initializes the RAG pipeline with MongoDB Atlas or transparent local fallback."""
    embedder = get_cached_embedder()
    client = get_mongo_client()

    retriever = None
    if client and check_mongo_connection(client):
        collection = get_legal_chunks_collection(client)
        if collection is not None:
            retriever = AtlasVectorRetriever(
                collection=collection,
                embedder=embedder,
                index_name=config.mongodb_vector_index,
                candidates_multiplier=config.retrieval_candidates_multiplier,
            )
            logger.info("Retriever initialized with live MongoDB Atlas Vector Search.")

    if retriever is None:
        # Explicit fallback
        fallback_docs = load_fallback_corpus()
        retriever = LocalFallbackRetriever(documents=fallback_docs, embedder=embedder)
        logger.info(f"Retriever initialized with Local In-Memory Fallback ({len(fallback_docs)} chunks).")

    generator = GemmaOpenAIGateway(app_config=config)
    return JusticeMitraRAGPipeline(retriever=retriever, generator=generator, app_config=config)


def render_emergency_banner(helplines=None):
    """Renders prominent emergency assistance card."""
    items = helplines or EMERGENCY_HELPLINES
    st.error("🚨 **Immediate Crisis & Emergency Support Available**")
    cols = st.columns(min(len(items), 3))
    for i, h in enumerate(items):
        with cols[i % len(cols)]:
            st.markdown(f"**{h['name']}**  \n📞 **`{h['number']}`**  \n*{h['description']}*")


def main():
    st.set_page_config(
        page_title="JusticeMitra - Women's Legal Rights Assistant",
        page_icon="⚖️",
        layout="wide",
        initial_sidebar_state="expanded",
    )

    # Initialize session state
    if "previous_searches" not in st.session_state:
        st.session_state.previous_searches = []
    if "pipeline" not in st.session_state:
        st.session_state.pipeline = initialize_pipeline()

    pipeline: JusticeMitraRAGPipeline = st.session_state.pipeline
    active_backend = getattr(pipeline.retriever, "backend_name", "Unknown Backend")

    # Header and description
    st.title("⚖️ JusticeMitra")
    st.markdown(
        "**AI-Powered Legal Rights Information for Women in India**  \n"
        "*Grounded statutory guidance on Domestic Violence (PWDVA 2005), Workplace Harassment (POSH 2013), "
        "Dowry Prohibition, Maintenance, and Free Legal Aid.*"
    )

    # Sidebar: Telemetry, Architecture Status & Search History
    with st.sidebar:
        st.header("System Status")
        if "Atlas" in active_backend:
            st.success(f"🟢 **Vector Store:** {active_backend}")
        else:
            st.warning(f"🟡 **Vector Store:** {active_backend}")
            st.caption("MongoDB Atlas URI not connected; using verified local statutory corpus.")

        st.info(f"🧠 **Model Gateway:** Gemma (`{config.gemma_model_name}`)")

        if is_sentry_active():
            st.success("📡 **Observability:** Sentry Agent Tracing Active")
        else:
            st.caption("📡 **Observability:** Sentry Standby (No DSN)")

        st.markdown("---")
        st.header("Emergency Contacts")
        st.markdown("""
        - **Women Helpline**: `1091`
        - **National Emergency**: `112`
        - **NCW 24/7 Helpline**: `7827170170`
        - **Domestic Violence Helpline**: `181`
        - **NALSA Free Legal Aid**: `1516`
        """)

        st.markdown("---")
        st.header("Search History")
        if st.session_state.previous_searches:
            for idx, item in enumerate(reversed(st.session_state.previous_searches[-8:]), 1):
                st.caption(f"{idx}. {item}")
        else:
            st.caption("No previous inquiries.")

    # Main user query interface
    query = st.text_input(
        "Please describe your situation or ask your legal question:",
        placeholder="e.g. Can a woman stay in her matrimonial home after filing a domestic violence complaint?",
    )

    # Example query suggestions
    with st.expander("💡 Click to view example legal scenarios"):
        col1, col2 = st.columns(2)
        with col1:
            st.markdown("""
            - *Can a woman stay in her matrimonial home after filing a domestic violence complaint?*
            - *My husband's family is demanding cash and vehicles before our wedding.*
            """)
        with col2:
            st.markdown("""
            - *Where can an indigent woman obtain a free advocate for court representation?*
            - *Can a workplace harassment complaint be filed if the incident occurred five months ago?*
            """)

    if query:
        if query not in st.session_state.previous_searches:
            st.session_state.previous_searches.append(query)

        with st.spinner("Analyzing statutory provisions and retrieving legal evidence..."):
            try:
                response: RAGResponse = pipeline.process_query(query)

                # Emergency banner if acute distress detected
                if response.is_emergency:
                    render_emergency_banner(response.emergency_helplines)

                # Primary answer block
                st.markdown("### 📋 Legal Information")
                st.write(response.answer)

                # Source Citations
                if response.citations:
                    st.markdown("---")
                    st.markdown("### 📚 Statutory Sources & Legal Citations")
                    cit_cols = st.columns(min(len(response.citations), 2))
                    for idx, cit in enumerate(response.citations):
                        with cit_cols[idx % len(cit_cols)]:
                            st.info(
                                f"**{cit.get('statute', 'Statute')}**  \n"
                                f"📌 **{cit.get('section', 'General Section')}**  \n"
                                f"🔗 [Official Legislation Link]({cit.get('url', 'https://www.indiacode.nic.in/')})"
                            )

                # Operational Telemetry Footer
                st.markdown("---")
                col_meta1, col_meta2, col_meta3 = st.columns(3)
                with col_meta1:
                    st.caption(f"⏱️ **Response Latency:** {response.latency_ms:.1f} ms")
                with col_meta2:
                    st.caption(f"🔍 **Backend:** {response.retrieval_backend}")
                with col_meta3:
                    st.caption(f"🤖 **Model:** {response.model_name}")

                # Important Legal Disclaimer
                st.warning(
                    "⚠️ **Legal Disclaimer:** JusticeMitra provides statutory information for educational and awareness "
                    "purposes only. It does not constitute formal legal representation or legal advice. If you require legal "
                    "counsel or court representation, please contact a qualified advocate or your District Legal Services Authority (DLSA)."
                )

            except Exception as e:
                logger.error(f"Application error processing query: {e}", exc_info=True)
                st.error(
                    "An error occurred while processing your legal request. "
                    "If you are facing an emergency, please use the helpline numbers on the sidebar immediately."
                )


if __name__ == "__main__":
    main()
