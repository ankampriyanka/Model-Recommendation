# app.py
import json
import faiss
import numpy as np
import streamlit as st
from sentence_transformers import SentenceTransformer
from typing import List, Dict, Any
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
import os

# --- Config
INDEX_PATH = "model_index.faiss"
META_PATH = "model_metadata.json"
EMB_MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"
TOP_K = 5

st.set_page_config(page_title="Model Recommender + FinOps", layout="wide")

@st.cache_resource
def load_index_and_metadata(index_path=INDEX_PATH, meta_path=META_PATH):
    if not os.path.exists(index_path) or not os.path.exists(meta_path):
        raise FileNotFoundError(
            f"Required files missing. Expecting {index_path} and {meta_path} in the repo."
        )
    index = faiss.read_index(index_path)
    with open(meta_path, "r", encoding="utf-8") as f:
        metadata = json.load(f)
    emb_model = SentenceTransformer(EMB_MODEL_NAME)
    return index, metadata, emb_model

def build_query_text(use_case: str, data_type: str, task_type: str, constraints: str) -> str:
    parts = [
        f"Use case: {use_case}",
        f"Data type: {data_type}" if data_type else "",
        f"Task type: {task_type}" if task_type else "",
        f"Constraints: {constraints}" if constraints else ""
    ]
    return " | ".join([p for p in parts if p])

def search_models(query: str, index, metadata: List[Dict], emb_model, top_k: int = TOP_K):
    if not query.strip():
        return []
    query_emb = emb_model.encode([query], convert_to_numpy=True)
    faiss.normalize_L2(query_emb)
    distances, indices = index.search(query_emb, top_k)
    results = []
    for score, idx in zip(distances[0], indices[0]):
        if idx == -1:
            continue
        m = metadata[idx].copy()
        results.append({"score": float(score), "model": m})
    return results

def present_model_card(item: Dict[str, Any]):
    model = item["model"]
    score = item.get("score", 0.0)
    st.markdown(f"### {model.get('id', 'unknown')}  —  score: {score:.3f}")
    cols = st.columns([2, 1])
    with cols[0]:
        st.markdown(f"**Family:** {model.get('family','N/A')}")
        st.markdown(f"**Task:** {model.get('task','N/A')}")
        st.markdown(f"**Input type:** {', '.join(model.get('input_type', []))}")
        st.markdown(f"**Description:** {model.get('description','')}")
        if model.get("best_for"):
            st.markdown("**Best for:** " + ", ".join(model.get("best_for", [])))
        if model.get("limitations"):
            st.markdown("**Limitations:** " + ", ".join(model.get("limitations", [])))
    with cols[1]:
        st.markdown(f"**Cost / 1k:** ${model.get('cost_per_1k_inferences_usd', 'N/A')}")
        st.markdown(f"**Latency:** {model.get('latency_ms', 'N/A')} ms")
        st.markdown(f"**Memory:** {model.get('memory_mb', 'N/A')} MB")
        st.markdown(f"**Accuracy:** {model.get('accuracy_score', 'N/A')}")
        st.markdown(f"**ROI Score:** {model.get('roi_score', 'N/A')}")
        if model.get("link"):
            st.markdown(f"[Model link]({model.get('link')})")

def show_cost_chart(results: List[Dict[str, Any]]):
    df = pd.DataFrame([
        {
            "Model": item["model"]["id"],
            "Cost per 1k Inferences ($)": item["model"].get("cost_per_1k_inferences_usd", 0)
        }
        for item in results
    ])
    if df.empty:
        st.info("No data for cost chart.")
        return
    fig = px.bar(df, x="Model", y="Cost per 1k Inferences ($)", title="Cost per 1,000 Inferences")
    st.plotly_chart(fig, use_container_width=True)

def show_latency_chart(results: List[Dict[str, Any]]):
    df = pd.DataFrame([
        {
            "Model": item["model"]["id"],
            "Latency (ms)": item["model"].get("latency_ms", 0)
        }
        for item in results
    ])
    if df.empty:
        st.info("No data for latency chart.")
        return
    fig = px.bar(df, x="Model", y="Latency (ms)", title="Latency (ms) — Lower is Better")
    st.plotly_chart(fig, use_container_width=True)

def show_roi_chart(results: List[Dict[str, Any]]):
    df = pd.DataFrame([
        {
            "Model": item["model"]["id"],
            "ROI Score": item["model"].get("roi_score", 0)
        }
        for item in results
    ])
    if df.empty:
        st.info("No data for ROI chart.")
        return
    fig = px.bar(df, x="Model", y="ROI Score", title="ROI Score Comparison")
    st.plotly_chart(fig, use_container_width=True)

def create_radar_chart(results: List[Dict[str, Any]]):
    metrics = ["Cost (inv)", "Latency (inv)", "Accuracy", "ROI Score", "Memory (inv)"]
    fig = go.Figure()
    for item in results:
        m = item["model"]
        cost = float(m.get("cost_per_1k_inferences_usd", 0.0))
        latency = float(m.get("latency_ms", 0.0))
        accuracy = float(m.get("accuracy_score", 0.0))
        roi = float(m.get("roi_score", 0.0))
        memory = float(m.get("memory_mb", 0.0))
        values = [
            1.0 / (cost + 1e-8),
            1.0 / (latency + 1e-8),
            accuracy,
            roi,
            1.0 / (memory + 1e-8)
        ]
        fig.add_trace(go.Scatterpolar(
            r=values,
            theta=metrics,
            fill='toself',
            name=m.get("id", "model")
        ))
    fig.update_layout(
        polar=dict(
            radialaxis=dict(visible=True)
        ),
        showlegend=True,
        height=600,
        title="ROI & Performance Radar (inverted metrics: lower cost/latency/memory => higher value)"
    )
    return fig

def main():
    st.title("🧠 Model Recommender + FinOps Dashboard")
    st.write("Describe your use case; the system will return model recommendations and FinOps metrics.")
    try:
        index, metadata, emb_model = load_index_and_metadata()
    except Exception as e:
        st.error(f"Failed to load index or metadata: {e}")
        st.stop()
    with st.form("query_form"):
        use_case = st.text_area("Describe your use case", height=120,
                                placeholder="E.g. Classify customer feedback into categories and detect sentiment in short messages.")
        col1, col2 = st.columns(2)
        with col1:
            data_type = st.selectbox("Primary data type", ["", "text", "image", "tabular", "time-series", "audio"])
        with col2:
            task_type = st.selectbox("Likely task", ["", "classification", "generation", "object-detection", "segmentation",
                                                     "forecasting", "clustering", "recommendation", "extraction"])
        constraints = st.text_input("Constraints (optional)", placeholder="e.g. CPU only, <50ms latency, explainable")
        submitted = st.form_submit_button("Recommend Models")
    if submitted:
        query = build_query_text(use_case, data_type, task_type, constraints)
        with st.spinner("Searching model catalog..."):
            results = search_models(query, index, metadata, emb_model, top_k=TOP_K)
        if not results:
            st.warning("No matching models found. Try adding more details to your use case.")
            return
        tab1, tab2, tab3 = st.tabs(["🔍 Recommendations", "💰 FinOps Dashboard", "📈 Radar Insights"])
        with tab1:
            st.subheader("🔍 Model Recommendations")
            for item in results:
                with st.expander(f"{item['model']['id']} (score: {item.get('score',0.0):.3f})", expanded=False):
                    present_model_card(item)
        with tab2:
            st.subheader("💰 FinOps Cost & Performance Metrics")
            best = results[0]["model"]
            kpi_cols = st.columns(5)
            kpi_cols[0].metric("Best Model", best.get("id", "N/A"))
            kpi_cols[1].metric("Cost per 1k", f"${best.get('cost_per_1k_inferences_usd', 'N/A')}")
            kpi_cols[2].metric("Latency (ms)", f"{best.get('latency_ms', 'N/A')}")
            kpi_cols[3].metric("Accuracy", f"{best.get('accuracy_score', 'N/A')}")
            kpi_cols[4].metric("ROI Score", f"{best.get('roi_score', 'N/A')}")
            st.markdown("---")
            show_cost_chart(results)
            show_latency_chart(results)
            show_roi_chart(results)
        with tab3:
            st.subheader("📈 ROI & Performance Radar")
            radar_fig = create_radar_chart(results)
            st.plotly_chart(radar_fig, use_container_width=True)

if __name__ == "__main__":
    main()
