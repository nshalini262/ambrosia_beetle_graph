from langchain_community.document_loaders import PyPDFLoader, UnstructuredXMLLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter

import os
import pickle
import logging

logging.getLogger("pypdf").setLevel(logging.ERROR)

FILES_FOLDER = "files/"
MIN_PDF_SIZE = 1024  # 1 KB


def is_valid_pdf(path):
    # size check
    if os.path.getsize(path) < MIN_PDF_SIZE:
        return False

    # header check
    try:
        with open(path, "rb") as f:
            return f.read(5) == b"%PDF-"
    except Exception:
        return False


def load_docs(folder):
    docs = []

    for file in os.listdir(folder):
        path = os.path.join(folder, file)

        if file.lower().endswith(".pdf"):
            if not is_valid_pdf(path):
                print(f"Invalid PDF: {file}")
                continue

            try:
                loader = PyPDFLoader(path, strict=False)
                docs.extend(loader.load())
            except Exception as e:
                print(f" PDF failed: {file} ({e})")

        elif file.lower().endswith(".xml"):
            try:
                loader = UnstructuredXMLLoader(path)
                docs.extend(loader.load())
            except Exception as e:
                print(f" XML failed: {file} ({e})")

    return docs

def chunk_documents(docs):
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=150
    )
    return splitter.split_documents(docs)


if __name__ == "__main__":
    docs = load_docs(FILES_FOLDER)
    chunks = chunk_documents(docs)

    with open("chunks.pkl", "wb") as f:
        pickle.dump(chunks, f)

    print(f"Saved {len(chunks)} chunks")
