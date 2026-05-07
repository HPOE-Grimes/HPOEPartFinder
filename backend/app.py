import os
from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

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

db = PartDatabase(os.path.join(DATA_DIR, 'parts.csv'))

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


@app.get("/health")
def health():
    return {"status": "ok", "model_loaded": classifier is not None}


@app.post("/search/image")
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


@app.post("/search/name")
def search_name(body: TextQuery):
    if not body.query.strip():
        raise HTTPException(status_code=400, detail="Query cannot be empty.")
    results = db.search_by_name(body.query)
    return {"results": results}


@app.post("/search/description")
def search_description(body: TextQuery):
    if not body.query.strip():
        raise HTTPException(status_code=400, detail="Query cannot be empty.")
    results = db.search_by_description(body.query)
    return {"results": results}
