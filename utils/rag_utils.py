import faiss
import numpy as np
import streamlit as st
from models.embeddings import get_embeddings


def build_vector_store(docs):
    if not docs:
        return

    texts = [doc["content"] for doc in docs]
    embeddings = np.array(get_embeddings(texts)).astype("float32")

    index = faiss.IndexFlatIP(embeddings.shape[1])
    index.add(embeddings)

    st.session_state["faiss_index"] = index
    st.session_state["faiss_docs"] = docs


def retrieve_relevant_docs(query, top_k=5):
    index = st.session_state.get("faiss_index", None)
    documents = st.session_state.get("faiss_docs", [])

    if index is None or not documents:
        return [], []

    try:
        query_embedding = np.array(get_embeddings([query])).astype("float32")
        top_k = min(top_k, len(documents))
        distances, indices = index.search(query_embedding, top_k)

        results = []
        sources = []

        for i, idx in enumerate(indices[0]):
            if idx == -1:
                continue

            doc = documents[idx]
            distance = float(distances[0][i])
            page_ref = f"p.{doc['page']}" if doc.get("page") else ""

            results.append(f"[{doc['file']} {page_ref}]\n{doc['content']}")
            sources.append(
                {
                    "file": doc["file"],
                    "page": doc.get("page"),
                    "distance": round(distance, 3),
                }
            )

        if not sources:
            return [], []

        best = sources[0]["distance"]
        if best < 0.4:
            return [], []

        return results, sources

    except Exception as e:
        print("Retrieval error:", e)
        return [], []
