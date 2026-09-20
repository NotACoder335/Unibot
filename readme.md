# UniBot

This project is a Retrieval-Augmented Generation (RAG) academic chatbot that answers questions using a local knowledge base built from academic and admission-related files.

## 🏗️ Built With

![Python](https://img.shields.io/badge/Python-3776AB?style=for-the-badge&logo=python&logoColor=white)
![FastAPI](https://img.shields.io/badge/FastAPI-009688?style=for-the-badge&logo=fastapi&logoColor=white)
![Ollama](https://img.shields.io/badge/Ollama-000000?style=for-the-badge&logo=ollama&logoColor=white)
![LangChain](https://img.shields.io/badge/LangChain-1C3C3C?style=for-the-badge&logo=chainlink&logoColor=white)
![FAISS](https://img.shields.io/badge/FAISS-5B2C6F?style=for-the-badge&logo=sqlite&logoColor=white)
![Pydantic](https://img.shields.io/badge/Pydantic-FF6F00?style=for-the-badge&logo=pydantic&logoColor=white)

## 📌 Project Overview

The system combines:
- 🗂️ a local document database built from CSV, TXT, and PDF files
-  text chunking and embedding generation
-  a FAISS vector store for semantic retrieval
-  an Ollama-based LLM for answer generation
-  a FastAPI API for the chatbot backend
-  a simple web interface for chatting

## 📁 Main Files

- `backend/rag.py` - loads the knowledge files, creates embeddings, builds the FAISS index, and answers questions using the local model.
- `backend/server.py` - exposes the chatbot through a FastAPI API at `/ask`.
- `frontend/academic_chatbot.html` - browser-based chat UI that sends requests to the API.
- `data/` - source documents used to build the knowledge base.
- `Academic info/` - saved FAISS vector store created from the processed documents.

## 🔄 How the RAG Flow Works

1.  The app reads files from the `data` folder.
2.  It detects file type using the extension:
   - `.pdf` -> `PyPDFLoader`
   - `.csv` -> `CSVLoader`
   - `.txt` -> `TextLoader`
3.  Documents are split into chunks using `RecursiveCharacterTextSplitter`.
4.  The chunks are embedded using `OllamaEmbeddings(model="bge-m3")`.
5.  A FAISS vector database is created and saved locally.
6.  When a user asks a question, the app retrieves the most relevant chunks.
7.  The retrieved context is passed to `ChatOllama` to generate a grounded answer.

## 🧰 Tech Stack

- Python
- FastAPI
- Pydantic
- LangChain Community
- LangChain Text Splitters
- LangChain Ollama
- LangChain Core
- FAISS
- Ollama

## ⚙️ Setup Instructions

Run every command below from the `UniBot` folder. The repository already contains the FAISS index, so rebuilding it is optional for a normal installation.

### Prerequisites

- Python 3.10 or newer
- [Ollama](https://ollama.com/download) installed and running
- At least 6 GB of available disk space for the local models

### 1. 🐍 Create a virtual environment

```bash
python -m venv .venv
```

### 2. 📦 Install dependencies

```bash
pip install -r requirements.txt
```

### 3. 🤖 Install the required Ollama models

Start Ollama if it is not already running, then download the exact models used by the backend:

```bash
ollama serve
ollama pull bge-m3
ollama pull llava-v1.5-7b-q4:latest
```

On Windows, Ollama may already be running in the system tray. In that case, leave `ollama serve` stopped and run only the two `ollama pull` commands.

### 4. 🏗️ Optional: rebuild the vector database

Skip this step when using the index included in the repository. To recreate it from the files in `data/`, run:

```bash
python -m backend.rag --build
```

This recreates the local vector database in `Academic info/`.

### 5. 🚀 Run the FastAPI backend

From the `UniBot` folder:

```bash
python -m uvicorn backend.server:app --reload
```

This starts the API server and exposes the `/ask` endpoint.

### 6. 🌐 Open the frontend

Open `frontend/academic_chatbot.html` in a browser.

The frontend sends requests to:

```text
http://127.0.0.1:8000/ask
```

## 📡 API Endpoint

### POST `/ask`

Request body:

```json
{
  "question": "What are the admission requirements?"
}
```

Response:

```json
{
  "answer": "..."
}
```

## 📝 Notes

- The chatbot only answers questions based on the retrieved context from the knowledge base.
- If the required information is missing, the system is designed to respond that it does not have the information and suggests contacting the admission office.
- The current project is focused on academic and admission information extracted from the dataset in the `data` folder.

## 📜 License

This project is for academic and educational use only.

