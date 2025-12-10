---
title: Model Recommender
emoji: 🧠
colorFrom: blue
colorTo: green
sdk: streamlit
app_file: app.py
pinned: false
sdk_version: 1.52.1
---

# Model Recommender + FinOps Dashboard

This Space recommends ML models for a given use-case using a FAISS-based retrieval over a model catalog, and displays FinOps metrics (cost, latency, memory), ROI and visualizations.

## Files required (in the same repo)
- `app.py` (this Streamlit app)
- `model_db.json` (model catalog with FinOps fields)
- `model_index.faiss` (FAISS index built from `model_db.json`)
- `model_metadata.json` (metadata used for lookup)
- `requirements.txt` (dependencies)

## Quick Start (Hugging Face Spaces)
1. Make sure `sdk: streamlit` is set in this README (already configured).
2. Upload the files to the Space (Files → Upload).
3. Ensure `requirements.txt` contains:
