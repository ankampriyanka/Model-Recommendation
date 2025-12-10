# build/build_index.py
import json
import numpy as np
from sentence_transformers import SentenceTransformer
import faiss
import os

DB_PATH = "model_db.json"
INDEX_PATH = "model_index.faiss"
META_PATH = "model_metadata.json"
EMB_MODEL = "sentence-transformers/all-MiniLM-L6-v2"

def load_db(path=DB_PATH):
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)

def build_corpus_entry(model_obj):
    fields = [
        model_obj.get("id", ""),
        model_obj.get("family", ""),
        model_obj.get("task", ""),
        " ".join(model_obj.get("input_type", [])) if isinstance(model_obj.get("input_type", []), list) else str(model_obj.get("input_type","")),
        " ".join(model_obj.get("domain", [])) if isinstance(model_obj.get("domain", []), list) else str(model_obj.get("domain","")),
        model_obj.get("description", ""),
        " ".join(model_obj.get("best_for", [])) if isinstance(model_obj.get("best_for", []), list) else str(model_obj.get("best_for","")),
        " ".join(model_obj.get("limitations", [])) if isinstance(model_obj.get("limitations", []), list) else str(model_obj.get("limitations","")),
        model_obj.get("infra_requirements", ""),
        model_obj.get("typical_users", "")
    ]
    return " | ".join([f for f in fields if f])

def main():
    if not os.path.exists(DB_PATH):
        raise FileNotFoundError(f"{DB_PATH} not found. Run merge_catalogs first.")

    db = load_db()
    texts = [build_corpus_entry(m) for m in db]

    model = SentenceTransformer(EMB_MODEL)
    embeddings = model.encode(texts, show_progress_bar=True, convert_to_numpy=True)

    dim = embeddings.shape[1]
    index = faiss.IndexFlatIP(dim)
    faiss.normalize_L2(embeddings)
    index.add(embeddings)

    faiss.write_index(index, INDEX_PATH)
    np.save("model_embeddings.npy", embeddings)

    with open(META_PATH, "w", encoding="utf-8") as f:
        json.dump(db, f, ensure_ascii=False, indent=2)

    print("Index built and saved:", INDEX_PATH)

if __name__ == "__main__":
    main()
