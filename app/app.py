import chainlit as cl
import os
import sys
from langchain_core.messages import HumanMessage, AIMessage

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.ingest import ingest_files
from src.generation import get_rag_chain
from src.config import DATA_RAW_DIR

@cl.on_chat_start
async def on_chat_start():
    #Initializing memory
    cl.user_session.set("chat_history", [])

    files = None

    while files == None:
        files = await cl.AskFileMessage(
            content="Welcome to Marvin! Please upload a PDF file to begin.",
            accept=["application/pdf"],
            max_size_mb=20,
            max_files=10,
            timeout=1800
        ).send()

    #Process each file
    msg = cl.Message(content=f"Processing {len(files)} files...")
    await msg.send()

    os.makedirs(DATA_RAW_DIR, exist_ok=True)
   
    saved_file_paths = []

    for file in files:
        #Create path for saving 
        save_path = os.path.join(DATA_RAW_DIR, file.name)
        saved_file_paths.append(save_path)

        with open(file.path, "rb") as f_src:
            with open(save_path, "wb") as f_dest:
                f_dest.write(f_src.read())

    await cl.make_async(ingest_files)(saved_file_paths)

    #Initializing chain
    chain = get_rag_chain()
    cl.user_session.set("chain", chain)

    file_names = ', '.join([f.name for f in files])
    msg.content = f"Done! Knowledge base loaded with: '{file_names}'."
    await msg.update()


@cl.on_message
async def main(message: cl.Message):
    chain = cl.user_session.get("chain")
    
    #Take history data
    chat_history = cl.user_session.get("chat_history")

    if not chain:
        await cl.Message(content="Please upload a PDF file first!").send()
        return 
    
    msg = cl.Message(content="")
    await msg.send()

    res = await chain.ainvoke({
        "input": message.content,
        "chat_history": chat_history
    })

    answer = res["answer"]
    source_documents = res["context"]

    text_elements = []
    if source_documents:
        for i, doc in enumerate(source_documents):
            page_num = int(doc.metadata.get('page', 0)) + 1
            source_name = f"Page {page_num}"
            text_elements.append(
                cl.Text(content=doc.page_content, name=source_name, display="inline")
            )

    msg.content = answer
    msg.elements = text_elements
    await msg.update()        

    #Update history data 
    chat_history.append(HumanMessage(content=message.content))
    chat_history.append(AIMessage(content=answer))

    cl.user_session.set("chat_history", chat_history)