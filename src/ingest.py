import os

from langchain_community.document_loaders import PyPDFLoader
from langchain_experimental.text_splitter import SemanticChunker
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma
from typing import List

from src.config import DATA_RAW_DIR, DB_PATH, EMBEDDING_MODEL_NAME, CHUNK_SIZE, CHUNK_OVERLAP
from src.utils import clean_directory

def ingest_files(pdf_file_paths: List[str]):
    """
    Progress: PDF file -> Chunking -> Embedding -> Save into DB
    """
    
    #Delete old database
    clean_directory(DB_PATH)

    #Build semantic splitter
    embedding_model = HuggingFaceEmbeddings(model_name=EMBEDDING_MODEL_NAME)

    text_splitter = SemanticChunker(
        embeddings=embedding_model,
        breakpoint_threshold_type="percentile" #percentile = cut in the most different point
    )

    all_chunks = []

    for file_path in pdf_file_paths:
        #Error check when trying to read pdf file
        if not os.path.exists(file_path):
            print(f"Skip: Can't find this file {file_path}")
            continue

        print(f"Reading...: {os.path.basename(file_path)}")
        loader = PyPDFLoader(file_path)
        documents = loader.load()
        
        #Chunking
        chunks = text_splitter.split_documents(documents)
        all_chunks.extend(chunks)

    print(f" -> Total: {len(all_chunks)} chunks from {len(pdf_file_paths)} file.")
    
    
    #Embedding & Saving into DB
    if len(all_chunks) > 0:
        print(f"Initializing the Embedding Model:{EMBEDDING_MODEL_NAME}")
        embedding_model = HuggingFaceEmbeddings(
            model_name = EMBEDDING_MODEL_NAME
        ) 

        print(f"Creating and saving into Chroma DB...")
        vector_db = Chroma.from_documents(
            documents=all_chunks,
            embedding=embedding_model,
            persist_directory=DB_PATH
        )
        print(f"Complete! All data is saved in {DB_PATH}")
        return vector_db
    
    else:
        print(f"There is no data to save.")
        return None
    
#Testing
if __name__ == "__main__":
    pass