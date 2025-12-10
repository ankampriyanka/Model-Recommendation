## Flow followed:
GitHub Repo
   ↓ (run scripts)
   
auto_catalog.py → 
generates model_catalog_auto.json
merge_catalogs.py → 
generates model_db.json
build_index.py → 
generates model_index.faiss + metadata
   ↓
GitHub Actions pushes:
   model_db.json
   model_index.faiss
   model_metadata.json
→ Hugging Face Space Repo
→ Space automatically updates app

### Folder Structure:
/repo
  app.py
  build/
     auto_catalog.py
     merge_catalogs.py
     build_index.py
  finops_overrides.json
  requirements.txt
  .github/workflows/update.yaml

