import os
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma
from langchain_ollama import ChatOllama
from langchain_core.prompts import ChatPromptTemplate
from langchain_classic.chains.combine_documents import create_stuff_documents_chain
from langchain_classic.chains import create_retrieval_chain

current_dir = os.path.dirname(os.path.abspath(__file__))
DB_PATH = os.path.join(current_dir, "..", "chroma_db")

def get_rag_chain():
    """
    Connecting to Vector DB and Llama 3.1 
    """
    #Load Vector DB
    print(f"Loading data from Chroma DB")
    embedding_model = HuggingFaceEmbeddings(
        model_name = "sentence-transformers/all-mpnet-base-v2"
    )

    vector_db = Chroma(
        persist_directory=DB_PATH,
        embedding_function=embedding_model
    )

    #Create Retriever
    retriever = vector_db.as_retriever(search_kwargs={"k": 1})

    #Load Model LLM
    print(f"Connecting to Llama 3.1...")
    llm = ChatOllama(
        model="llama3.1",
        temperature=0.3
    )

    #Prompt
    system_prompt = (
        "You are an intelligent assistant designed to help with document analysis. "
        "Use the retrieved context below to answer the user's question accurately. "
        "If the answer is not in the context, clearly state that you don't know. "
        "Keep your answer concise and professional."
        "\n\n"
        "Context:\n{context}"
    )

    prompt = ChatPromptTemplate([
        ("system", system_prompt),
        ("human", "{input}")
    ])

    #RAG Chain
    question_answer_chain = create_stuff_documents_chain(llm, prompt)
    rag_chain = create_retrieval_chain(retriever, question_answer_chain)

    return rag_chain