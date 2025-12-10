# build/auto_catalog.py
from huggingface_hub import list_models
import json

def fetch_models(limit=200):
    models = list_models(limit=limit)

    catalog = []
    for m in models:
        catalog.append({
            "id": m.modelId,
            "task": m.pipeline_tag,
            "library": m.library_name,
            "downloads": getattr(m, "downloads", None),
            "likes": getattr(m, "likes", None),
            "tags": getattr(m, "tags", None),
            "description": m.cardData.get("summary", "") if getattr(m, "cardData", None) else ""
        })

    with open("model_catalog_auto.json", "w", encoding="utf-8") as f:
        json.dump(catalog, f, indent=2, ensure_ascii=False)

    print("Auto catalog generated → model_catalog_auto.json")

if __name__ == "__main__":
    fetch_models()
