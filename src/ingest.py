import os
import shutil
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma

current_dir = os.path.dirname(os.path.abspath(__file__))
DATA_PATH = os.path.join(current_dir, "..", "data", "raw")
DB_PATH = os.path.join(current_dir, "..", "chroma_db")

def ingest_pdf(pdf_file_path):
    """
    Progress: PDF file -> Chunking -> Embedding -> Save into DB
    """
    
    if os.path.exists(DB_PATH):
        shutil.rmtree(DB_PATH) # Xóa toàn bộ thư mục DB cũ
        print("🧹 Clean previous data.")

    #Read file pdf
    if not os.path.exists(pdf_file_path):
        print(f"Error: Can't find the file in {pdf_file_path}")
        return None
    
    loader = PyPDFLoader(pdf_file_path)
    documents = loader.load()
    print(f"Documents ({len(documents)}) are read")

    #Chunking
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size = 500,
        chunk_overlap = 50,
        separators=["\n\n", "\n", " ", ""]
    )
    chunks = text_splitter.split_documents(documents)
    print(f"Divided into {len(chunks)} chunks")

    #Embedding
    print(f"Initializing the Embedding Model...")
    embedding_model = HuggingFaceEmbeddings(
        model_name = "sentence-transformers/all-mpnet-base-v2"
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
    test_pdf_name = "test.pdf"
    full_path = os.path.join(DATA_PATH, test_pdf_name)

    ingest_pdf(full_path)