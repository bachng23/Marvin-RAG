import os
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma
from langchain_ollama import ChatOllama
from langchain_core.prompts import ChatPromptTemplate
from langchain_classic.chains.combine_documents import create_stuff_documents_chain
from langchain_classic.chains import create_retrieval_chain
from langchain_classic.chains import create_history_aware_retriever


from src.config import DB_PATH, EMBEDDING_MODEL_NAME, LLM_MODEL_NAME, RETRIEVAL_K
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

    #Create Retriever
    retriever = vector_db.as_retriever(search_kwargs={"k": RETRIEVAL_K})

    #Load Model LLM
    print(f"Connecting to Llama 3.1...")
    llm = ChatOllama(
        model=LLM_MODEL_NAME,
        temperature=0.3
    )

    #Create history aware retriever
    history_aware_retriever = create_history_aware_retriever(
        llm,
        retriever,
        contextualize_q_prompt
    )

    #RAG Chain
    question_answer_chain = create_stuff_documents_chain(llm, qa_prompt)
    rag_chain = create_retrieval_chain(history_aware_retriever, question_answer_chain)

    return rag_chain