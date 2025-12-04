import streamlit as st
import sys
from pathlib import Path

sys.path.append(str(Path(__file__).parent.parent))

from services.rag_service.rag_engine import RAGEngine
from services.llm_service.claude_client import ClaudeClient


@st.cache_resource
def load_rag_engine():
    rag = RAGEngine()
    rag.load_chunks("data/chunks.json")
    rag.load_index("data/faiss_index/index.faiss")
    return rag


@st.cache_resource
def load_llm_client():
    return ClaudeClient()


def main():
    st.set_page_config(
        page_title="Мастер и Маргарита - RAG",
        page_icon="📚",
        layout="wide"
    )

    st.title("📚 Чат-бот по роману 'Мастер и Маргарита'")
    st.markdown("Задавайте вопросы о сюжете, персонажах и событиях романа")

    if "messages" not in st.session_state:
        st.session_state.messages = []

    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])
            if "sources" in message:
                with st.expander("Источники"):
                    for src in message["sources"]:
                        st.markdown(f"**Глава {src['chapter']}**")
                        st.text(src['text'][:200] + "...")

    if query := st.chat_input("Задайте вопрос о книге"):
        st.session_state.messages.append({"role": "user", "content": query})

        with st.chat_message("user"):
            st.markdown(query)

        with st.chat_message("assistant"):
            with st.spinner("Ищу ответ в книге..."):
                rag = load_rag_engine()
                llm = load_llm_client()

                context, sources = rag.get_context_for_llm(query, top_k=3)
                answer = llm.generate_answer(query, context, sources)

                st.markdown(answer)

                with st.expander("📖 Источники из книги"):
                    for i, src in enumerate(sources, 1):
                        st.markdown(f"### {i}. Глава {src['chapter']}")
                        st.text(src['text'][:300] + "...")
                        st.markdown("---")

                st.session_state.messages.append({
                    "role": "assistant",
                    "content": answer,
                    "sources": sources
                })


if __name__ == "__main__":
    main()
