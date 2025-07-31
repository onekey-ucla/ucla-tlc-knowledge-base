from flask import Flask, request, jsonify
from flask_cors import CORS
from sentence_transformers import SentenceTransformer
from retrieve import load_index, load_corpus, retrieve_answer

app = Flask(__name__)
CORS(app)

index = load_index("faiss_index.bin")
corpus = load_corpus("corpus.pkl")
model = SentenceTransformer("all-MiniLM-L6-v2", device="cpu")

@app.route("/ask", methods=["POST"])
def ask_question():
    data = request.get_json()
    query = data.get("question", "").strip().lower()

    # ✅ Use FAISS retrieval
    results = retrieve_answer(query, model, corpus, index, k=1)
    answer = results[0] if results else (
        "Not enough information available. "
        "Please contact [TLC Consultations](https://teaching.ucla.edu/services/consultations/) for further support."
    )
    return jsonify({"answer": answer})

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5001)
