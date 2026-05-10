import io
import os
from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse
from pydantic import BaseModel
import pandas as pd
from rapidfuzz import process, fuzz

from search import PartDatabase
from inference import PartClassifier

app = FastAPI(title="HPOE Part Finder")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

BASE = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(BASE, '..', 'data')
MODEL_DIR = os.path.join(BASE, '..', 'model')
PUBLIC_DIR = os.path.join(BASE, '..', 'public')
CSV_PATH = os.path.join(DATA_DIR, 'parts.csv')

db = PartDatabase(CSV_PATH)

MODEL_PATH = os.path.join(MODEL_DIR, 'checkpoints', 'best_model.pth')
CLASS_NAMES_PATH = os.path.join(DATA_DIR, 'class_names.json')
classifier = None
if os.path.exists(MODEL_PATH) and os.path.exists(CLASS_NAMES_PATH):
    classifier = PartClassifier(MODEL_PATH, CLASS_NAMES_PATH)
    print("CNN model loaded.")
else:
    print("No trained model found — image search disabled until training is complete.")



class TextQuery(BaseModel):
    query: str


class NewPart(BaseModel):
    part_name: str
    location: str
    usage_description: str
    appearance_description: str


# ── Static files & pages ──────────────────────────────────────────────────────

app.mount("/static", StaticFiles(directory=PUBLIC_DIR), name="static")


@app.get("/")
def index():
    return FileResponse(os.path.join(PUBLIC_DIR, "index.html"))


@app.get("/db")
def db_page():
    return FileResponse(os.path.join(PUBLIC_DIR, "db.html"))


# ── Existing search routes ────────────────────────────────────────────────────

@app.get("/api/health")
def health():
    return {"status": "ok", "model_loaded": classifier is not None}


@app.post("/api/search/image")
async def search_image(file: UploadFile = File(...)):
    if classifier is None:
        raise HTTPException(
            status_code=503,
            detail="Image model not trained yet. Run model/train.py first, then restart the server.",
        )
    image_bytes = await file.read()
    predictions = classifier.predict(image_bytes, top_k=3)

    results = []
    for pred in predictions:
        matches = db.get_by_name_exact(pred['part_name'])
        for m in matches:
            m['confidence'] = pred['confidence']
            results.append(m)

    return {"results": results}


@app.post("/api/search/name")
def search_name(body: TextQuery):
    if not body.query.strip():
        raise HTTPException(status_code=400, detail="Query cannot be empty.")
    results = db.search_by_name(body.query)
    return {"results": results}


@app.post("/api/search/description")
def search_description(body: TextQuery):
    if not body.query.strip():
        raise HTTPException(status_code=400, detail="Query cannot be empty.")
    results = db.search_by_description(body.query)
    return {"results": results}


# ── Database management routes ────────────────────────────────────────────────

@app.get("/api/db/parts")
def list_parts():
    return {"parts": db.to_records()}


@app.post("/api/db/parts")
def add_part(part: NewPart):
    df = pd.read_csv(CSV_PATH)
    new_row = pd.DataFrame([{
        "Part Name": part.part_name,
        "Location": part.location,
        "Usage Description (What does it do)": part.usage_description,
        "Appearance Description": part.appearance_description,
    }])
    df = pd.concat([df, new_row], ignore_index=True)
    df.to_csv(CSV_PATH, index=False)
    db.reload(CSV_PATH)
    return {"ok": True, "total": len(db.to_records())}


@app.delete("/api/db/parts/{idx}")
def delete_part(idx: int):
    df = pd.read_csv(CSV_PATH)
    if idx < 0 or idx >= len(df):
        raise HTTPException(status_code=404, detail="Part index out of range.")
    df = df.drop(index=idx).reset_index(drop=True)
    df.to_csv(CSV_PATH, index=False)
    db.reload(CSV_PATH)
    return {"ok": True, "total": len(db.to_records())}


@app.post("/api/db/retrain")
def trigger_retrain():
    db.reload(CSV_PATH)
    return {"ok": True, "total": len(db.to_records())}


# ── Bulk upload ───────────────────────────────────────────────────────────────

# Keywords that hint at each target field (checked against the uploaded column names)
_FIELD_HINTS = {
    "part_name": [
        "part name", "part", "name", "item", "component", "title",
        "sku", "part number", "part no", "description", "product",
    ],
    "location": [
        "location", "bin", "shelf", "row", "storage", "where",
        "place", "position", "aisle", "cabinet", "slot", "spot",
    ],
    "usage_description": [
        "usage description", "usage", "use", "function", "purpose",
        "what it does", "how it works", "application", "role", "task",
    ],
    "appearance_description": [
        "appearance description", "appearance", "look", "visual",
        "color", "colour", "shape", "physical", "how it looks",
        "looks like", "exterior", "form",
    ],
}


def _map_columns(columns: list[str]) -> dict[str, str]:
    """Return {target_field: uploaded_column} for the best fuzzy match of each field."""
    mapping = {}
    used = set()
    cols_lower = {c: c.lower().strip() for c in columns}

    for field, hints in _FIELD_HINTS.items():
        best_col, best_score = None, 0
        for col, col_l in cols_lower.items():
            if col in used:
                continue
            for hint in hints:
                score = fuzz.token_set_ratio(col_l, hint)
                if score > best_score:
                    best_score, best_col = score, col
        if best_col and best_score >= 50:
            mapping[field] = best_col
            used.add(best_col)

    return mapping


def _read_csv_loose(raw: bytes) -> pd.DataFrame:
    """Try common delimiters and encodings until one works."""
    for enc in ("utf-8", "latin-1", "cp1252"):
        for sep in (",", ";", "\t", "|"):
            try:
                df = pd.read_csv(io.BytesIO(raw), sep=sep, encoding=enc, dtype=str)
                if len(df.columns) >= 1:
                    return df
            except Exception:
                pass
    raise ValueError("Could not parse the CSV with any common delimiter or encoding.")


@app.post("/api/db/bulk-upload")
async def bulk_upload(file: UploadFile = File(...)):
    raw = await file.read()
    try:
        uploaded = _read_csv_loose(raw)
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))

    uploaded.columns = [str(c).strip() for c in uploaded.columns]
    mapping = _map_columns(list(uploaded.columns))

    if "part_name" not in mapping:
        raise HTTPException(
            status_code=422,
            detail=f"Could not find a 'Part Name' column. Columns found: {list(uploaded.columns)}",
        )

    def _cell(row, col):
        if not col or col not in row:
            return ""
        v = row[col]
        return "" if pd.isna(v) else str(v).strip()

    # Build clean rows using only the mapped columns
    rows = []
    skipped = 0
    for _, row in uploaded.iterrows():
        part_name = _cell(row, mapping["part_name"])
        if not part_name:
            skipped += 1
            continue
        rows.append({
            "Part Name": part_name,
            "Location": _cell(row, mapping.get("location", "")),
            "Usage Description (What does it do)": _cell(row, mapping.get("usage_description", "")),
            "Appearance Description": _cell(row, mapping.get("appearance_description", "")),
        })

    if not rows:
        raise HTTPException(status_code=422, detail="No valid rows found after cleaning.")

    existing = pd.read_csv(CSV_PATH)
    existing_names = set(existing["Part Name"].str.strip().str.lower().dropna())

    deduped = []
    seen = set()
    duplicates = 0
    for r in rows:
        key = r["Part Name"].lower()
        if key in existing_names or key in seen:
            duplicates += 1
        else:
            seen.add(key)
            deduped.append(r)

    if deduped:
        appended = pd.concat([existing, pd.DataFrame(deduped)], ignore_index=True)
        appended.to_csv(CSV_PATH, index=False)
        db.reload(CSV_PATH)

    return {
        "ok": True,
        "added": len(deduped),
        "skipped": skipped,
        "duplicates": duplicates,
        "total": len(db.to_records()),
        "column_mapping": {k: mapping.get(k, "(not found)") for k in _FIELD_HINTS},
    }
