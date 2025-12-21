from langchain_core.prompts import ChatPromptTemplate

RAG_SYSTEM_PROMPT = (
    "You are an intelligent assistant designed to help with document analysis. "
    "Use the retrieved context below to answer the user's question accurately. "
    "If the answer is not in the context, clearly state that you don't know. "
    "Keep your answer concise and professional."
    "\n\n"
    "Context:\n{context}"
)

def get_rag_prompt_template():
    prompt = ChatPromptTemplate.from_messages([
        ("system", RAG_SYSTEM_PROMPT),
        ("human", "{input}"),
    ])
    return prompt