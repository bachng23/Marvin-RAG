import os
import shutil
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma

from src.config import DATA_RAW_DIR, DB_PATH, EMBEDDING_MODEL_NAME, CHUNK_SIZE, CHUNK_OVERLAP
from src.utils import clean_directory

def ingest_pdf(pdf_file_path):
    """
    Progress: PDF file -> Chunking -> Embedding -> Save into DB
    """
    
    clean_directory(DB_PATH)

    #Read file pdf
    if not os.path.exists(pdf_file_path):
        print(f"Error: Can't find the file in {pdf_file_path}")
        return None
    
    loader = PyPDFLoader(pdf_file_path)
    documents = loader.load()
    print(f"Documents ({len(documents)}) are read")

    #Chunking
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size = CHUNK_SIZE,
        chunk_overlap = CHUNK_OVERLAP,
        separators=["\n\n", "\n", " ", ""]
    )
    chunks = text_splitter.split_documents(documents)
    print(f"Divided into {len(chunks)} chunks")

    #Embedding
    print(f"Initializing the Embedding Model...")
    embedding_model = HuggingFaceEmbeddings(
        model_name = EMBEDDING_MODEL_NAME
    ) 

    #Save into DB
    print(f"Saving into Chroma DB...")
    vector_db = Chroma.from_documents(
        documents=chunks,
        embedding=embedding_model,
        persist_directory=DB_PATH
    )

    print(f"Complete! All data is saved in {DB_PATH}")
    return vector_db

if __name__ == "__main__":
    pass