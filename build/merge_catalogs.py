# build/merge_catalogs.py
import json

def merge_catalogs():
    with open("model_catalog_auto.json", "r", encoding="utf-8") as f:
        auto_catalog = json.load(f)

    try:
        with open("finops_overrides.json", "r", encoding="utf-8") as f:
            overrides = json.load(f)
    except FileNotFoundError:
        overrides = {}

    merged = []
    for entry in auto_catalog:
        model_id = entry["id"]
        if model_id in overrides:
            entry.update(overrides[model_id])
        merged.append(entry)

    with open("model_db.json", "w", encoding="utf-8") as f:
        json.dump(merged, f, indent=2, ensure_ascii=False)

    print("Merged catalog created → model_db.json")

if __name__ == "__main__":
    merge_catalogs()
