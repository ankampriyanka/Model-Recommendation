---
title: Model Recommender (Docker)
emoji: 🧠
colorFrom: blue
colorTo: green
sdk: docker
pinned: false
---

# Model Recommender + FinOps (Docker)

This repository provides a Streamlit-based Model Recommender app packaged with Docker, plus automation scripts to auto-generate a model catalog from Hugging Face and rebuild a FAISS index.

## Contents
- `app.py` - Streamlit app
- `Dockerfile` - Docker build (Ubuntu base with system deps)
- `build/` - automation scripts
- `.github/workflows/update_and_deploy.yml` - GitHub Action to auto-update catalog and push to HF Space
- `finops_overrides.json` - small file to maintain FinOps metrics
- `model_db.json` - sample merged catalog (you can regenerate)

## How to use (local / CI)
1. Populate `finops_overrides.json` with your FinOps metrics for key models.
2. Run the automation locally (or let GitHub Actions run it):
   ```bash
   python3 build/auto_catalog.py
   python3 build/merge_catalogs.py
   python3 build/build_index.py
   ```
3. Build Docker:
   ```bash
   docker build -t model-recommender:latest .
   docker run -p 7860:7860 model-recommender:latest
   ```
4. For automated runs, set `HF_TOKEN` and `HF_SPACE_REPO` secrets in GitHub and enable workflow.

**Note:** Docker builds with faiss-cpu may be heavy and take time. If you encounter build issues on certain hosts, consider using Streamlit SDK (no Docker) for Hugging Face Spaces.
