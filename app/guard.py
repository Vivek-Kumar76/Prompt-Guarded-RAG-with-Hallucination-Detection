import numpy as np
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity

model = SentenceTransformer("all-MiniLM-L6-v2")

def grounding_check(answer, context_chunks, threshold=0.6):
    answer_embedding = model.encode([answer])
    context_embedding = model.encode([" ".join(context_chunks)])

    similarity = float(
        cosine_similarity(answer_embedding, context_embedding)[0][0]
    )

    if similarity < threshold:
        return False, similarity

    return True, similarity