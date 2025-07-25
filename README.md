# Multylanguage-_RAG
# Multilingual Retrieval-Augmented Generation (RAG) System

This project implements a Retrieval-Augmented Generation system capable of answering queries in English and Bangla using PDF documents as the knowledge source.

---

## Setup Guide

### Prerequisites
- Python 3.8+
- OpenAI API key (with available quota)
- `.env` file containing:

### Install Dependencies

```bash
pip install python-dotenv PyPDF2 numpy tiktoken openai
python main.py
| Tool / Library    | Purpose                                    |
| ----------------- | ------------------------------------------ |
| PyPDF2            | Extract text from PDF                      |
| tiktoken          | Tokenize text for chunking                 |
| numpy             | Vector math (cosine similarity)            |
| OpenAI Python SDK | Embeddings generation and chat completions |
| python-dotenv     | Load environment variables                 |
