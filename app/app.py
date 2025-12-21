import chainlit as cl
import os
import sys

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.ingest import ingest_pdf
from src.generation import get_rag_chain

@cl.on_chat_start

async def on_chat_start():
    files = None

    while files == None:
        files = await cl.AskFileMessage(
            content="Welcome to Marvin! Please upload a PDF file to begin.",
            accept=["application/pdf"],
            max_size_mb=20,
            timeout=18000
        ).send()

    file = files[0]

    msg = cl.Message(content=f"Processing '{file.name}'...")
    await msg.send()

    temp_dir = "data/raw"
    os.makedirs(temp_dir, exist_ok=True)
    save_path = os.path.join(temp_dir, file.name)

    with open(file.path, "rb") as f:
        file_content = f.read()


    with open(save_path, "wb") as f:
        f.write(file_content)

    await cl.make_async(ingest_pdf)(save_path)

    chain = get_rag_chain()
    cl.user_session.set("chain", chain)

    msg.content = f"Done! You can now ask questions about '{file.name}'"
    await msg.update()


@cl.on_message
async def main(message: cl.Message):
    chain = cl.user_session.get("chain")

    if not chain:
        await cl.Message(content="Please upload a PDF file first!").send()
        return 
    
    msg = cl.Message(content="")
    await msg.send()

    res = await chain.ainvoke({"input": message.content})

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