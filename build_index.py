import json
import numpy as np
from sentence_transformers import SentenceTransformer
import faiss

DB_PATH = "model_db.json"
INDEX_PATH = "model_index.faiss"
EMB_PATH = "model_embeddings.npy"
META_PATH = "model_metadata.json"

def load_db(path=DB_PATH):
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)

def build_corpus_entry(model_obj):
    fields = [
        model_obj.get("id", ""),
        model_obj.get("family", ""),
        model_obj.get("task", ""),
        " ".join(model_obj.get("input_type", [])),
        " ".join(model_obj.get("domain", [])),
        model_obj.get("description", ""),
        " ".join(model_obj.get("best_for", [])),
        " ".join(model_obj.get("limitations", "")),
        model_obj.get("infra_requirements", ""),
        model_obj.get("typical_users", "")
    ]
    return " | ".join([f for f in fields if f])

def main():
    db = load_db()
    texts = [build_corpus_entry(m) for m in db]

    model = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")
    embeddings = model.encode(texts, show_progress_bar=True, convert_to_numpy=True)

    dim = embeddings.shape[1]
    index = faiss.IndexFlatIP(dim)  # inner product (cosine-like after normalization)

    # Normalize for cosine similarity
    faiss.normalize_L2(embeddings)
    index.add(embeddings)

    # Save index + embeddings + metadata
    faiss.write_index(index, INDEX_PATH)
    np.save(EMB_PATH, embeddings)

    with open(META_PATH, "w", encoding="utf-8") as f:
        json.dump(db, f, ensure_ascii=False, indent=2)

    print("Index built and saved!")

if __name__ == "__main__":
    main()
