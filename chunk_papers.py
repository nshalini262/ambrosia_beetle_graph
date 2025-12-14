from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
import os
import pickle
import pypdf

#file containing 4 pdfs
PDF_FOLDER = "selected_papers/"

def load_all_pdfs(folder):
    docs = []
    for file in os.listdir(folder):
        if file.endswith(".pdf"):
            loader = PyPDFLoader(os.path.join(folder, file))
            docs.extend(loader.load())
    return docs

def chunk_documents(docs):
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=150
    )
    return splitter.split_documents(docs)

if __name__ == "__main__":
    docs = load_all_pdfs(PDF_FOLDER)
    chunks = chunk_documents(docs)

    with open("chunks.pkl", "wb") as f:
        pickle.dump(chunks, f)

    print(f"Saved {len(chunks)} chunks to chunks.pkl")
