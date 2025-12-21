from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder

#Build Context
contextualize_q_system_prompt = (
    "Given a chat history and the latest user question "
    "which might reference context in the chat history, "
    "formulate a standalone question which can be understood "
    "without the chat history. Do NOT answer the question, "
    "just reformulate it if needed and otherwise return it as is."
)

contextualize_q_prompt = ChatPromptTemplate([
    ("system", contextualize_q_system_prompt),
    MessagesPlaceholder("chat_history"),
    ("human", "{input}")
])

RAG_SYSTEM_PROMPT = (
    "You are an intelligent assistant designed to help with document analysis. "
    "Use the retrieved context below to answer the user's question accurately. "
    "If the answer is not in the context, clearly state that you don't know. "
    "Keep your answer concise and professional."
    "\n\n"
    "Context:\n{context}"
)

#QnA 
qa_prompt = ChatPromptTemplate([
    ("system", RAG_SYSTEM_PROMPT),
    MessagesPlaceholder("chat_history"),
    ("human", "{input}")        
])