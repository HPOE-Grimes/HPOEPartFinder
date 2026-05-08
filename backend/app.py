import os
from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse
from pydantic import BaseModel
import pandas as pd

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
