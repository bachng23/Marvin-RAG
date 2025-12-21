from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma
from langchain_ollama import ChatOllama
from langchain_core.prompts import ChatPromptTemplate
from langchain_classic.chains.combine_documents import create_stuff_documents_chain
from langchain_classic.chains import create_retrieval_chain
from langchain_classic.chains import create_history_aware_retriever
from langchain_core.documents import Document

# --- CÁC IMPORT MỚI CHO ADVANCED RAG ---
from langchain_community.retrievers import BM25Retriever
from langchain_classic.retrievers import EnsembleRetriever
from langchain_classic.retrievers.contextual_compression import ContextualCompressionRetriever
from langchain_community.document_compressors import FlashrankRerank

from src.config import DB_PATH, EMBEDDING_MODEL_NAME, LLM_MODEL_NAME, RETRIEVAL_K, RERANKING_MODEL_NAME
from src.prompts import contextualize_q_prompt, qa_prompt

def get_rag_chain():
    """
    Connecting to Vector DB and Llama 3.1 
    """
    #Load Vector DB
    print(f"Loading data from Chroma DB")
    embedding_model = HuggingFaceEmbeddings(
        model_name = EMBEDDING_MODEL_NAME
    )
    vector_db = Chroma(
        persist_directory=DB_PATH,
        embedding_function=embedding_model
    )

    #Building Hybrid Search
    print("⚡ Building Hybrid Search (BM25 + Vector)...")
    
    #Create Basic Vector Retriever
    vector_retriever = vector_db.as_retriever(search_kwargs={"k": RETRIEVAL_K})

    #Create BM25 Retriever
    db_data = vector_db.get()

    if len(db_data['documents']) == 0:
        #If empty db
        print("⚠️ Warning: Empty Database, can't create BM25.")
        return None

    #Transfrom raw data to Documents
    docs_list = [
        Document(page_content=txt, metadata=meta)
        for txt, meta in zip(db_data['documents'], db_data["metadatas"])
    ]

    bm25_retriever = BM25Retriever.from_documents(docs_list)
    bm25_retriever.k = RETRIEVAL_K

    #Ensemble
    ensemble_retriever = EnsembleRetriever(
        retrievers=[bm25_retriever, vector_retriever],
        weights=[0.5, 0.5]
    )    

    #Reranking
    print("Initializing FlashRank Reranker...")
    compressor = FlashrankRerank(model=RERANKING_MODEL_NAME)

    #Create final retriever
    compression_retriever = ContextualCompressionRetriever(
        base_compressor=compressor,
        base_retriever=ensemble_retriever
    )


    #Load Model LLM
    print(f"Connecting to {LLM_MODEL_NAME}...")
    llm = ChatOllama(
        model=LLM_MODEL_NAME,
        temperature=0.3
    )

    #Create history aware retriever
    history_aware_retriever = create_history_aware_retriever(
        llm,
        compression_retriever,
        contextualize_q_prompt
    )

    #RAG Chain
    question_answer_chain = create_stuff_documents_chain(llm, qa_prompt)
    rag_chain = create_retrieval_chain(history_aware_retriever, question_answer_chain)

    return rag_chain