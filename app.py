import json
import faiss
import numpy as np
import streamlit as st
from sentence_transformers import SentenceTransformer
from typing import List, Dict

INDEX_PATH = "model_index.faiss"
META_PATH = "model_metadata.json"

@st.cache_resource
def load_index_and_metadata():
    index = faiss.read_index(INDEX_PATH)
    with open(META_PATH, "r", encoding="utf-8") as f:
        metadata = json.load(f)
    model = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")
    return index, metadata, model

def build_query_text(use_case: str, data_type: str, task_type: str, constraints: str) -> str:
    parts = [
        f"Use case: {use_case}",
        f"Data type: {data_type}" if data_type else "",
        f"Task type: {task_type}" if task_type else "",
        f"Constraints: {constraints}" if constraints else ""
    ]
    return " | ".join([p for p in parts if p])

def search_models(
    query: str,
    index,
    metadata: List[Dict],
    emb_model,
    top_k: int = 5
):
    if not query.strip():
        return []

    query_emb = emb_model.encode([query], convert_to_numpy=True)
    faiss.normalize_L2(query_emb)

    distances, indices = index.search(query_emb, top_k)
    results = []
    for score, idx in zip(distances[0], indices[0]):
        if idx == -1:
            continue
        m = metadata[idx]
        results.append({
            "score": float(score),
            "model": m
        })
    return results

def present_model_card(item: Dict):
    model = item["model"]
    score = item["score"]

    st.markdown(f"### {model['id']}  (score: {score:.3f})")
    st.markdown(f"**Family:** {model.get('family', 'N/A')}")
    st.markdown(f"**Task:** {model.get('task', 'N/A')}")
    st.markdown(f"**Input type:** {', '.join(model.get('input_type', []))}")
    st.markdown(f"**Domain:** {', '.join(model.get('domain', []))}")

    st.markdown(f"**Description**: {model.get('description', '')}")
    if model.get("best_for"):
        st.markdown("**Best for:** " + ", ".join(model["best_for"]))
    if model.get("limitations"):
        st.markdown("**Limitations:** " + ", ".join(model["limitations"]))

    if model.get("infra_requirements"):
        st.markdown(f"**Infra:** {model['infra_requirements']}")
    if model.get("link"):
        st.markdown(f"[Model link]({model['link']})")

def main():
    st.set_page_config(
        page_title="Model Recommender",
        layout="wide"
    )
    st.title("🔍 AI Model Recommender (RAG-based)")
    st.write("Describe your use case and I'll suggest suitable models from the catalog.")

    index, metadata, emb_model = load_index_and_metadata()

    with st.form("query_form"):
        use_case = st.text_area(
            "Describe your use case",
            placeholder="e.g. I want to classify customer support emails into categories and detect sentiment."
        )
        col1, col2 = st.columns(2)
        with col1:
            data_type = st.selectbox(
                "Primary data type",
                ["", "text", "image", "tabular", "time-series", "audio"]
            )
        with col2:
            task_type = st.selectbox(
                "Likely task type",
                ["", "classification", "generation", "object-detection", "segmentation",
                 "forecasting", "clustering", "recommendation"]
            )
        constraints = st.text_input(
            "Constraints (optional)",
            placeholder="e.g. Must run on CPU only, low latency, explainable."
        )
        submitted = st.form_submit_button("Recommend Models")

    if submitted:
        query = build_query_text(use_case, data_type, task_type, constraints)
        results = search_models(query, index, metadata, emb_model, top_k=5)

        if not results:
            st.warning("No matching models found. Try adding more details to your use case.")
        else:
            st.subheader("Recommended models")
            for item in results:
                with st.expander(f"{item['model']['id']} (score: {item['score']:.3f})", expanded=True):
                    present_model_card(item)

if __name__ == "__main__":
    main()
