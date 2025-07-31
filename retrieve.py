import numpy as np
import faiss
import pickle
from sentence_transformers import SentenceTransformer

MODEL_NAME = "all-MiniLM-L6-v2"

def load_index(index_file="faiss_index.bin"):
    return faiss.read_index(index_file)

def load_corpus(corpus_file="corpus.pkl"):
    with open(corpus_file, 'rb') as f:
        return pickle.load(f)

def retrieve_answer(query, model, corpus, index, k=1, threshold=0.20):
    """Original function for backward compatibility"""
    query_embedding = model.encode([query], normalize_embeddings=True)
    scores, indices = index.search(np.array(query_embedding, dtype=np.float32), k)
    
    best_score = scores[0][0]
    best_idx = indices[0][0]
    
    if best_score < threshold or best_idx >= len(corpus):
        return ["Not enough information available. Please contact [TLC Consultations](https://teaching.ucla.edu/services/consultations/) for further support."]
    
    return [corpus[best_idx]["answer"]]

def enhanced_search(query, model, corpus, index, k=5, threshold=0.15):
    """Enhanced search function that returns multiple results with metadata"""
    query_embedding = model.encode([query], normalize_embeddings=True)
    scores, indices = index.search(np.array(query_embedding, dtype=np.float32), k)
    
    results = []
    for i, (score, idx) in enumerate(zip(scores[0], indices[0])):
        if score >= threshold and idx < len(corpus):
            results.append({
                "question": corpus[idx]["question"],
                "answer": corpus[idx]["answer"],
                "category": corpus[idx].get("category", "General"),
                "relevance_score": float(score),
                "rank": i + 1
            })
    
    return results

def search_by_category(query, model, corpus, index, category, k=3):
    """Search within a specific category"""
    # Filter corpus by category
    category_corpus = [item for item in corpus if item.get("category") == category]
    
    if not category_corpus:
        return []
    
    # Create temporary index for category
    model_temp = SentenceTransformer(MODEL_NAME)
    texts = [item["question"] + " " + item["answer"] for item in category_corpus]
    embeddings = model_temp.encode(texts, normalize_embeddings=True)
    
    dim = embeddings.shape[1]
    temp_index = faiss.IndexFlatIP(dim)
    temp_index.add(np.array(embeddings, dtype=np.float32))
    
    # Search
    query_embedding = model.encode([query], normalize_embeddings=True)
    scores, indices = temp_index.search(np.array(query_embedding, dtype=np.float32), k)
    
    results = []
    for i, (score, idx) in enumerate(zip(scores[0], indices[0])):
        if idx < len(category_corpus):
            results.append({
                "question": category_corpus[idx]["question"],
                "answer": category_corpus[idx]["answer"],
                "category": category_corpus[idx].get("category", category),
                "relevance_score": float(score),
                "rank": i + 1
            })
    
    return results
