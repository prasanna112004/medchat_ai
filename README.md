🏥 MedChat AI — Clinical Document Intelligence Assistant

MedChat AI lets healthcare professionals and researchers upload clinical documents and ask questions in plain English — getting precise, cited answers in seconds. If the uploaded documents don't have the answer, the system automatically searches the web and tells you it did so.

## The Problem

Doctors, researchers, and pharmacists work with hundreds of pages of clinical guidelines, drug references, and research papers daily. Finding a specific answer means reading through entire documents manually — slow, error-prone, and impractical in a fast-paced environment.

## The Solution

Upload your clinical PDFs and just ask. MedChat AI finds the relevant section, cites the source file and page number, and gives you a direct answer. When your documents don't cover the question, it falls back to live web search automatically.

## Features

- **RAG Pipeline** — sentence-aware chunking, BGE embeddings, FAISS vector store
- **Two-layer fallback** — confidence score check + LLM self-judgment before triggering web search
- **Source attribution** — every answer shows which file and page it came from
- **Response modes** — Concise for quick lookups, Detailed for research and case review
- **Demo mode** — preloaded clinical documents, no upload needed to test
- **Medical disclaimer** — responsible AI, answers grounded in verified sources only

## Tech Stack

- Streamlit
- Groq LLM (llama-3.1-8b-instant)
- FAISS (IndexFlatIP)
- BAAI/bge-base-en-v1.5 embeddings
- DuckDuckGo search
- PyPDF2, python-docx

## Project Structure
```
config/        — environment config
models/        — LLM and embedding models  
utils/         — RAG, file loading, web search
demo_docs/     — preloaded clinical PDFs
app.py         — main Streamlit app
```

## Setup
```bash
pip install -r requirements.txt
export GROQ_API_KEY="your_key"
streamlit run app.py
```

## Deployed App

[Add your Streamlit URL here]

## Disclaimer

MedChat AI is for informational and research reference only. Always consult a qualified healthcare professional before making clinical decisions.
