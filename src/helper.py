from typing import List

from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.documents import Document
from langchain_community.document_loaders import PyPDFLoader, DirectoryLoader
from langchain_huggingface import HuggingFaceEmbeddings


# -------------------- LOAD PDF FILES --------------------
def load_pdf_file(path: str) -> List[Document]:
    """
    Load all PDF files from a directory and return LangChain Document objects.
    """
    loader = DirectoryLoader(
        path,
        glob="*.pdf",
        loader_cls=PyPDFLoader
    )
    documents = loader.load()
    return documents


# -------------------- FILTER METADATA --------------------
def filter_to_minimal_docs(docs: List[Document]) -> List[Document]:
    """
    Keep only page_content and 'source' metadata to reduce noise.
    """
    minimal_docs: List[Document] = []

    for doc in docs:
        src = doc.metadata.get("source")
        minimal_docs.append(
            Document(
                page_content=doc.page_content,
                metadata={"source": src}
            )
        )

    return minimal_docs


# -------------------- SPLIT TEXT INTO CHUNKS --------------------
def text_split(docs: List[Document]) -> List[Document]:
    """
    Split documents into smaller overlapping chunks.
    """
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=500,
        chunk_overlap=20
    )
    text_chunks = text_splitter.split_documents(docs)
    return text_chunks


# -------------------- EMBEDDINGS --------------------
def download_embeddings():
    """
    Download Hugging Face sentence-transformer embeddings.
    """
    embeddings = HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2"
    )
    return embeddings
