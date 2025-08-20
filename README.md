# crystal-chatbot
Celira Crystal Healing Chatbot

An AI-powered FastAPI application that provides personalized crystal healing guidance using Retrieval-Augmented Generation (RAG) and semantic search. It leverages FAISS for vector embeddings, Sentence Transformers for embedding generation, and integrates with Jinja2 templates for web UI rendering.

Features

REST API built with FastAPI

Semantic search over crystal data using FAISS

RAG Engine (CrystalRAG) to generate personalized recommendations

Template-driven UI with Jinja2

Asynchronous HTTP calls using HTTPX

Environment configuration via .env

Requirements

Python 3.10+ (3.12 support may be limited for FAISS)

faiss-cpu

fastapi

uvicorn[standard]

httpx

python-dotenv

pandas

sentence-transformers

numpy

jinja2

All dependencies are listed in requirements.txt.

Usage

Run the application with Uvicorn:

uvicorn main:app --reload

The API will be available at http://127.0.0.1:8000.

API docs: http://127.0.0.1:8000/docs

Redoc: http://127.0.0.1:8000/redoc
