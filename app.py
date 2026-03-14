import streamlit as st
import os
import sys
import io

from utils.file_loader import load_documents
from utils.rag_utils import build_vector_store, retrieve_relevant_docs
from utils.web_search import web_search

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))
from models.llm import get_chatgroq_model

chat_model = get_chatgroq_model()
DEMO_DOCS_PATH = "demo_docs"


def load_demo_docs():
    files = []
    if os.path.exists(DEMO_DOCS_PATH):
        for fname in os.listdir(DEMO_DOCS_PATH):
            if fname.endswith((".pdf", ".txt", ".docx")):
                with open(os.path.join(DEMO_DOCS_PATH, fname), "rb") as f:
                    buf = io.BytesIO(f.read())
                    buf.name = fname
                    files.append(buf)
    return files


def rag_prompt(context, query, style):
    return f"""You are MedChat AI, a clinical document intelligence assistant.

Your job is to answer the user's question using ONLY the context extracted from their uploaded medical documents.

RULES:
- Read the context carefully before answering.
- If the answer is clearly present in the context, answer accurately and mention the source file.
- If the context is partially relevant, use what is available and say what is missing.
- If the context has absolutely nothing to do with the question, reply with only this word: INSUFFICIENT_CONTEXT
- Do NOT use outside knowledge. Do NOT hallucinate. Do NOT guess.

{style}

CONTEXT FROM DOCUMENTS:
{context}

USER QUESTION:
{query}

YOUR ANSWER:"""


def web_prompt(web_results, query, style):
    return f"""You are MedChat AI, a clinical document intelligence assistant.

The user's uploaded documents did not contain relevant information for this question.
You are now answering using live web search results.

RULES:
- Answer based on the web results provided.
- Be factual and precise.
- End your answer with: "— answered via web search"

{style}

WEB SEARCH RESULTS:
{web_results}

USER QUESTION:
{query}

YOUR ANSWER:"""


def get_response(messages, mode):
    try:
        query = messages[-1]["content"]
        style = (
            "Be brief and precise (2-3 sentences max)."
            if mode == "Concise"
            else "Be detailed, structured, and use bullet points where helpful."
        )
        source = None

        if st.session_state.indexed:
            docs, sources = retrieve_relevant_docs(query)
            context = "\n\n".join(docs) if docs else ""

            if context.strip() and sources:
                first = sources[0]
                source_detail = f"{first.get('file', '')} {('p.' + str(first['page'])) if first.get('page') else ''}".strip()
                st.session_state["last_source_detail"] = source_detail

                raw = chat_model.invoke(
                    rag_prompt(context, query, style)
                ).content.strip()

                if "INSUFFICIENT_CONTEXT" in raw.upper():
                    source = "web"
                    response = chat_model.invoke(
                        web_prompt(web_search(query), query, style)
                    ).content
                else:
                    source = "rag"
                    response = raw

            else:
                source = "web"
                response = chat_model.invoke(
                    web_prompt(web_search(query), query, style)
                ).content

        else:
            source = "web"
            response = chat_model.invoke(
                web_prompt(web_search(query), query, style)
            ).content

        return response, source

    except Exception as e:
        return f"Error: {str(e)}", None


def source_badge(source):
    if source == "rag":
        detail = st.session_state.get("last_source_detail", "")
        label = f"✅ Source: {detail}" if detail else "✅ Source: Uploaded Documents"
        st.markdown(
            f'<span style="background:#d1fae5;color:#065f46;padding:2px 10px;border-radius:10px;font-size:12px;">{label}</span>',
            unsafe_allow_html=True,
        )
    elif source == "web":
        st.markdown(
            '<span style="background:#fef9c3;color:#713f12;padding:2px 10px;border-radius:10px;font-size:12px;">🌐 Source: Live Web Search</span>',
            unsafe_allow_html=True,
        )


def chat_page():
    st.title("🏥 MedChat AI")
    st.caption(
        "Ask questions from your clinical documents — RAG-powered with live web fallback."
    )
    st.info(
        "⚠️ For informational use only. Always consult a qualified healthcare professional."
    )

    mode = st.radio("Response Mode", ["Concise", "Detailed"], horizontal=True)

    for key in ["indexed", "messages", "sources"]:
        if key not in st.session_state:
            st.session_state[key] = False if key == "indexed" else []

    st.markdown("#### 📂 Knowledge Base")
    col1, col2 = st.columns([3, 1])

    with col1:
        uploaded = st.file_uploader(
            "Upload PDFs, DOCX, or TXT",
            type=["pdf", "docx", "txt"],
            accept_multiple_files=True,
        )

    with col2:
        st.markdown("<br>", unsafe_allow_html=True)
        if st.button("🧪 Load Demo Docs", use_container_width=True):
            demo = load_demo_docs()
            if demo:
                docs = load_documents(demo)
                build_vector_store(docs)
                st.session_state.indexed = True
                st.success(f"{len(demo)} demo file(s) indexed!")
            else:
                st.warning("Add PDFs to /demo_docs folder first.")

    if uploaded and not st.session_state.indexed:
        docs = load_documents(uploaded)
        build_vector_store(docs)
        st.session_state.indexed = True
        st.success(f"{len(docs)} chunks indexed!")

    if st.session_state.indexed:
        st.markdown(
            '<span style="background:#dbeafe;color:#1e3a5f;padding:2px 10px;border-radius:10px;font-size:12px;">📚 Knowledge base ready</span>',
            unsafe_allow_html=True,
        )

    st.divider()

    for i, msg in enumerate(st.session_state.messages):
        with st.chat_message(msg["role"]):
            st.markdown(msg["content"])
            if msg["role"] == "assistant":
                idx = i // 2
                if idx < len(st.session_state.sources):
                    source_badge(st.session_state.sources[idx])

    if prompt := st.chat_input("Ask a clinical question..."):
        st.session_state.messages.append({"role": "user", "content": prompt})
        with st.chat_message("user"):
            st.markdown(prompt)

        with st.chat_message("assistant"):
            with st.spinner("Thinking..."):
                reply, source = get_response(st.session_state.messages, mode)
            st.markdown(reply)
            source_badge(source)

        st.session_state.messages.append({"role": "assistant", "content": reply})
        st.session_state.sources.append(source)


def instructions_page():
    st.title("🏥 MedChat AI — Setup")
    st.markdown(
        """
### Install
```bash
pip install -r requirements.txt
```

### API Key
```bash
export GROQ_API_KEY="your_key"
```

### Demo Mode
Drop any medical PDFs into `/demo_docs` and click **Load Demo Docs** in the app.

### Response Modes
- **Concise** — quick fact lookup (dosage, definitions, contraindications)
- **Detailed** — full reasoning for research or case review

### Source Labels
- ✅ Green = answered from your uploaded documents
- 🌐 Yellow = answered from live web search

### Disclaimer
For informational use only. Not a substitute for professional medical advice.
    """
    )


def main():
    st.set_page_config(page_title="MedChat AI", page_icon="🏥", layout="wide")

    with st.sidebar:
        st.title("🏥 MedChat AI")
        st.caption("Clinical Document Intelligence")
        st.divider()
        page = st.radio("Navigate", ["Chat", "Setup Guide"])
        st.divider()
        if st.button("🗑️ Clear Chat", use_container_width=True):
            st.session_state.messages = []
            st.session_state.indexed = False
            st.session_state.sources = []
            st.session_state.pop("faiss_index", None)
            st.session_state.pop("faiss_docs", None)
            st.rerun()
        st.divider()
        st.caption("Groq · FAISS · SentenceTransformers")

    if page == "Chat":
        chat_page()
    else:
        instructions_page()


if __name__ == "__main__":
    main()
