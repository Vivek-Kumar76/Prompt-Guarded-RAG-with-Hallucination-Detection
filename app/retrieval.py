import faiss
import numpy as np
from sentence_transformers import SentenceTransformer

class Retriever:
    def __init__(self, file_path):
        self.model = SentenceTransformer("all-MiniLM-L6-v2")

        with open(file_path, "r") as f:
            self.docs = f.readlines()

        self.docs = [doc.strip() for doc in self.docs]

        embeddings = self.model.encode(self.docs)
        dimension = embeddings.shape[1]

        self.index = faiss.IndexFlatL2(dimension)
        self.index.add(np.array(embeddings))

    def retrieve(self, query, k=2):
        query_embedding = self.model.encode([query])
        D, I = self.index.search(np.array(query_embedding), k)
        return [self.docs[i] for i in I[0]]