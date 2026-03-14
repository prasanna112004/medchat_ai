from PyPDF2 import PdfReader
from docx import Document


def chunk_text(text, chunk_size=500):
    sentences = text.replace("\n", " ").split(". ")
    chunks = []
    current = ""

    for sentence in sentences:
        if len(current) + len(sentence) < chunk_size:
            current += sentence + ". "
        else:
            if current.strip():
                chunks.append(current.strip())
            current = sentence + ". "

    if current.strip():
        chunks.append(current.strip())

    return chunks


def load_documents(uploaded_files):
    documents = []

    for file in uploaded_files:
        filename = file.name.lower()

        try:
            if filename.endswith(".pdf"):
                reader = PdfReader(file)
                for page_num, page in enumerate(reader.pages, start=1):
                    page_text = page.extract_text() or ""
                    if not page_text.strip():
                        continue
                    for chunk in chunk_text(page_text):
                        documents.append(
                            {"file": file.name, "page": page_num, "content": chunk}
                        )

            elif filename.endswith(".docx"):
                doc = Document(file)
                text = "\n".join([p.text for p in doc.paragraphs if p.text.strip()])
                for chunk in chunk_text(text):
                    documents.append(
                        {"file": file.name, "page": None, "content": chunk}
                    )

            elif filename.endswith(".txt"):
                text = file.read().decode("utf-8")
                for chunk in chunk_text(text):
                    documents.append(
                        {"file": file.name, "page": None, "content": chunk}
                    )

        except Exception:
            continue

    return documents
