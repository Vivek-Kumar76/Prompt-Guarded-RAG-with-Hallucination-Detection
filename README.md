Prompt-Guarded RAG with Hallucination Detection

A safety-aware Retrieval-Augmented Generation (RAG) system that reduces hallucinations using embedding-based grounding validation.

This project enhances a standard RAG pipeline by introducing a semantic similarity guard layer that validates whether the generated response is supported by retrieved context.

 Features

 FAISS-based semantic retrieval

 Gemini LLM for response generation

 Embedding-based hallucination detection

 Cosine similarity-based confidence scoring

 FastAPI REST API deployment

 Architecture

User Query
↓
Vector Retrieval (FAISS + SentenceTransformers)
↓
Gemini LLM Generation
↓
Semantic Grounding Check (Cosine Similarity)
↓
Confidence-Based Response Filtering

 Project Structure
guarded_rag/
│
├── app/
│   ├── main.py
│   ├── retrieval.py
│   ├── llm.py
│   ├── guard.py
│
├── data/
│   └── finance_docs.txt
│
├── requirements.txt
└── README.md
 Tech Stack

Python

FastAPI

FAISS (Vector Search)

SentenceTransformers (all-MiniLM-L6-v2)

Google Gemini API (google-genai SDK)

NumPy

Scikit-learn (Cosine Similarity)

🛠 Installation
1️ Clone the repository
git clone <your-repo-url>
cd guarded_rag
2️ Install dependencies
pip install -r requirements.txt
Configure API Key

Open:

app/main.py

Replace:

API_KEY = "PASTE_YOUR_REAL_API_KEY"

With your actual Gemini API key.

▶️ Run the Server
uvicorn app.main:app --reload

Open in browser:

http://127.0.0.1:8000/docs
🧪 Example Queries
✅ In-Domain Query

Input:

What happens after 60 days?

Output:

{
  "query": "What happens after 60 days?",
  "answer": "After 60 days, accounts are escalated to collections.",
  "confidence": "High",
  "similarity_score": 0.84
}
 Out-of-Domain Query

Input:

What is employee leave policy?

Output:

{
  "query": "What is employee leave policy?",
  "answer": "Information not found.",
  "confidence": "Low",
  "similarity_score": 0.31
}
🛡 Hallucination Guard Mechanism

After generating an answer, the system:

Embeds the generated response

Embeds retrieved context

Computes cosine similarity

Rejects response if similarity < threshold (default = 0.6)

This reduces unsupported or hallucinated outputs.

 Key Learning Outcomes

Implementing retrieval-augmented generation

Reducing hallucinations using embedding similarity

Designing safety-aware LLM pipelines

Deploying LLM systems with FastAPI

Working with vector databases and semantic search

 

Multi-language support

Logging and monitoring dashboard
