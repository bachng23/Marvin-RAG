import os

SRC_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.abspath(os.path.join(SRC_DIR, ".."))

DATA_RAW_DIR = os.path.join(PROJECT_ROOT, "data", "raw")
DB_PATH = os.path.join(PROJECT_ROOT, "chroma_db")

EMBEDDING_MODEL_NAME = "sentence-transformers/all-mpnet-base-v2"
LLM_MODEL_NAME = "llama3.1"
RERANKING_MODEL_NAME = "ms-marco-MiniLM-L-12-v2"

CHUNK_SIZE = 500
CHUNK_OVERLAP = 50

RETRIEVAL_K = 10