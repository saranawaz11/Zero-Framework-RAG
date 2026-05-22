import os
from dotenv import load_dotenv
from google import genai
from google.genai import types
import math
import pymupdf

load_dotenv()
api_key = os.getenv("GEMINI_API_KEY")
if not api_key:
    raise ValueError("API key not found.")

client = genai.Client(api_key=api_key)

# functions
def dot_product(a: list[float], b: list[float]) -> float :
    return sum(x * y for x, y in zip(a, b))

def magnitude(v: list[float]) -> float:
    return math.sqrt(sum(x ** 2 for x in v))

def cosine_similarity(a: list[float], b: list[float]) -> float:
    return dot_product(a, b) / (magnitude(a) * magnitude(b))

def get_embedding(text: str) -> list[float]:
    response = client.models.embed_content(
        model="gemini-embedding-001",
        contents=text,
        config=types.EmbedContentConfig(task_type="SEMANTIC_SIMILARITY"),
    )
    return response.embeddings[0].values

def extract_pdf_chunks(pdf_path: str, chunk_size: int = 400) -> list[str]:
    doc = pymupdf.open(pdf_path)
    chunks = []
    for page in doc:
        text = page.get_text()
        words = text.split()
        for i in range(0, len(words), chunk_size):
            chunk = " ".join(words[i:i + chunk_size])
            if chunk.strip():
                chunks.append(chunk)
    return chunks

documents = extract_pdf_chunks("kindness.pdf")
print(f"Loaded {len(documents)} chunks from PDF")

doc_embeddings = [get_embedding(doc) for doc in documents]

query = "Can the person who is abusive with tongue consider kind?"
query_embedding = get_embedding(query)

scores = [cosine_similarity(query_embedding, doc_emb) for doc_emb in doc_embeddings]

top_k = 3
top_indices = sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)[:top_k]
retrieved_context = "\n\n".join([documents[i] for i in top_indices])

response = client.models.generate_content(
    model="gemini-2.5-flash",
    config=types.GenerateContentConfig(
        system_instruction="""You are Dr. Sophia, a warm and insightful psychology expert with 20 years of experience.

Your response style:
- Write in a reflective, thoughtful tone like a well-researched article
- Use clear sections with a brief intro, main insights, and a closing reflection
- Be honest about what you know vs what falls outside the provided context
- If the context doesn't fully answer the question, say: "Based on the available material..." and note the gap
- Keep responses between 150-200 words
- Use empathetic, accessible language — avoid jargon unless you explain it
- End with a thought-provoking closing sentence
"""
    ),
    contents=f"""Using the context below from a psychology paper, answer the question in a reflective, article-style response.
Context:{retrieved_context} Question: {query} If the context is insufficient, acknowledge what is known and what remains unclear."""
)

print(response.text)