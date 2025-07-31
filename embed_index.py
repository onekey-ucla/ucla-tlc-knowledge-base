import json
import pickle
import faiss
import numpy as np
from sentence_transformers import SentenceTransformer

MODEL_NAME = "all-MiniLM-L6-v2"

def load_corpus(file_path):
    with open(file_path, 'r', encoding='utf-8') as f:
        return [json.loads(line) for line in f]

def create_faiss_index(corpus, index_file="faiss_index.bin", corpus_file="corpus.pkl"):
    model = SentenceTransformer(MODEL_NAME)
    
    # Create embeddings from question + answer text
    texts = [item["question"] + " " + item["answer"] for item in corpus]
    embeddings = model.encode(texts, normalize_embeddings=True)

    dim = embeddings.shape[1]
    index = faiss.IndexFlatIP(dim)
    index.add(np.array(embeddings, dtype=np.float32))

    faiss.write_index(index, index_file)
    with open(corpus_file, 'wb') as f:
        pickle.dump(corpus, f)

    print("✅ FAISS index and corpus saved successfully.")
    print(f"📊 Indexed {len(corpus)} Q&A pairs")
    
    # Print category statistics if available
    if corpus and "category" in corpus[0]:
        categories = {}
        for item in corpus:
            cat = item.get("category", "Unknown")
            categories[cat] = categories.get(cat, 0) + 1
        
        print("\n📋 Category breakdown:")
        for cat, count in sorted(categories.items()):
            print(f"  - {cat}: {count} items")

if __name__ == "__main__":
    # Try enhanced corpus first, fall back to original
    try:
        corpus = load_corpus("enhanced_corpus.jsonl")
        print("📁 Using enhanced corpus with metadata")
    except FileNotFoundError:
        corpus = load_corpus("qna_corpus.jsonl")
        print("📁 Using original corpus")
    
    create_faiss_index(corpus)
