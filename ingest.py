from pathlib import Path

from langchain_community.document_loaders import TextLoader, PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS

from app.config import EMBEDDING_MODEL, BASE_DIR


DATA_DIR = BASE_DIR / "data"
VECTORSTORE_DIR = BASE_DIR / "vectorstore"


def load_documents():
    documents = []

    for file_path in DATA_DIR.iterdir():

        if file_path.suffix.lower() == ".txt":
            loader = TextLoader(
                str(file_path),
                encoding="utf-8"
            )
            documents.extend(loader.load())

        elif file_path.suffix.lower() == ".pdf":
            loader = PyPDFLoader(str(file_path))
            documents.extend(loader.load())

    return documents


def create_vectorstore(embeddings=None):

    print("Loading documents...")

    documents = load_documents()

    if not documents:
        raise ValueError(
            "No documents found inside the data folder."
        )

    print(f"Loaded {len(documents)} document pages/files.")

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=700,
        chunk_overlap=120,
        separators=[
            "\n\n",
            "\n",
            ". ",
            " ",
            ""
        ]
    )

    chunks = splitter.split_documents(documents)

    print(f"Created {len(chunks)} chunks.")

    if embeddings is None:
        embeddings = HuggingFaceEmbeddings(
            model_name=EMBEDDING_MODEL
        )

    print("Creating FAISS vector store...")

    vectorstore = FAISS.from_documents(
        chunks,
        embeddings
    )

    VECTORSTORE_DIR.mkdir(
        parents=True,
        exist_ok=True
    )

    vectorstore.save_local(
        str(VECTORSTORE_DIR)
    )

    print("FAISS vector store created successfully.")
    return vectorstore


if __name__ == "__main__":
    create_vectorstore()