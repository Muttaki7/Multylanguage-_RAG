import os
from dotenv import load_dotenv
from PyPDF2 import PdfReader
import numpy as np
import tiktoken
from openai import OpenAI
load_dotenv(r"C:\Users\Muttaki\anaconda3\pythonProject\.env")
OPENAI_API_KEY = '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'
PDF_PATH = os.getenv("PDF_PATH", "C:/Users/Muttaki/Downloads/AI Engineer (Level-1) Assessment.pdf")

if not OPENAI_API_KEY:
    raise ValueError("Missing OPENAI_API_KEY in environment variables or .env file")
if not PDF_PATH:
    raise ValueError("Missing PDF_PATH in environment variables or .env file")

client = OpenAI(api_key=OPENAI_API_KEY)
tokenizer = tiktoken.encoding_for_model("gpt-3.5-turbo")

def load_pdf_text(path):
    reader = PdfReader(path)
    text = ""
    for page in reader.pages:
        text += page.extract_text() + "\n"
    return text

def num_tokens(text):
    return len(tokenizer.encode(text))

def chunk_text(text, max_tokens=500):
    paragraphs = text.split("\n")
    chunks, current = [], ""
    for para in paragraphs:
        if num_tokens(current + para) <= max_tokens:
            current += para + "\n"
        else:
            chunks.append(current.strip())
            current = para + "\n"
    if current:
        chunks.append(current.strip())
    return chunks

def create_embeddings(chunks):
    embeddings = []
    for chunk in chunks:
        try:
            emb = client.embeddings.create(
                model="text-embedding-3-large",
                input=chunk,
            ).data[0].embedding
            embeddings.append((chunk, emb))
        except Exception as e:
            print(f"Embedding error for chunk: {e}")
    return embeddings

def cosine_similarity(a, b):
    a = np.array(a)
    b = np.array(b)
    return np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b))

def retrieve_relevant_chunks(query, embeddings, top_k=3):
    try:
        query_emb = client.embeddings.create(
            model="text-embedding-3-large",
            input=query
        ).data[0].embedding
    except Exception as e:
        print(f"Embedding error for query: {e}")
        return []

    scored = [(chunk, cosine_similarity(query_emb, emb)) for chunk, emb in embeddings]
    top_chunks = sorted(scored, key=lambda x: x[1], reverse=True)[:top_k]
    return [chunk for chunk, _ in top_chunks]

def generate_answer(chunks, query):
    context = "\n\n".join(chunks)
    try:
        response = client.chat.completions.create(
            model="gpt-4",
            messages=[
                {"role": "system", "content": "You are an expert AI answering questions based on provided context."},
                {"role": "user", "content": f"Context:\n{context}\n\nQuestion: {query}"}
            ],
            temperature=0.3,
        )
        return response.choices[0].message.content.strip()
    except Exception as e:
        print(f"OpenAI chat error: {e}")
        return "Sorry, I could not generate an answer at this time."

if __name__ == "__main__":
    print("Loading PDF...")
    full_text = load_pdf_text(PDF_PATH)

    print("Chunking text...")
    chunks = chunk_text(full_text)
    print("Creating embeddings...")
    embedded_chunks = create_embeddings(chunks)
    query = input("💬 Ask your question about the document: ")
    print("📚 Retrieving relevant chunks...")
    top_chunks = retrieve_relevant_chunks(query, embedded_chunks)

    print("Generating answer...")
    answer = generate_answer(top_chunks, query)
    print("\nAnswer:\n", answer)
