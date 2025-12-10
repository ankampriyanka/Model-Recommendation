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
<img width="597" height="419" alt="image" src="https://github.com/user-attachments/assets/6cf3bdf3-813f-43fe-a430-78d569cd1bb3" />


### Folder Structure:

<img width="412" height="390" alt="image" src="https://github.com/user-attachments/assets/0878eb2d-ba9d-4c44-acf7-6d40370402c1" />

