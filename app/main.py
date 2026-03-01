from fastapi import FastAPI
from app.retrieval import Retriever
from app.llm import GeminiLLM
from app.guard import grounding_check

#  Paste your real Gemini API key here
API_KEY = "Paste your real Gemini API key here"

app = FastAPI(title="Prompt-Guarded RAG")

retriever = Retriever("data/finance_docs.txt")
llm = GeminiLLM(API_KEY)


@app.post("/ask")
def ask(question: str):
    retrieved_chunks = retriever.retrieve(question)
    context = "\n".join(retrieved_chunks)

    answer = llm.generate(question, context)

    is_grounded, similarity_score = grounding_check(answer, retrieved_chunks)

    if not is_grounded:
        return {
            "query": question,
            "answer": "Information not found.",
            "confidence": "Low",
            "similarity_score": similarity_score
        }

    return {
        "query": question,
        "answer": answer,
        "confidence": "High",
        "similarity_score": similarity_score
    }