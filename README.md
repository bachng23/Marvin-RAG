# Marvin RAG 📚

Local RAG system powered by Llama 3.1, ChromaDB, and Chainlit.

## Features
- **Offline Privacy:** Runs 100% locally using Ollama.
- **Interactive UI:** ChatGPT-like interface built with Chainlit.
- **Smart Retrieval:** Uses `all-mpnet-base-v2` for high-quality embeddings.

## Installation
1. Install Ollama & Pull Model: `ollama pull llama3.1`
2. Install Python packages: `pip install -r requirements.txt`
3. Ingest Data: `python src/ingest.py`
4. Run App: `chainlit run app/cl_app.py -w`