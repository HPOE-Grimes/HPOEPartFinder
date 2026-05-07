import os
from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

from search import PartDatabase

app = FastAPI(title="HPOE Part Finder")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
CSV_PATH = os.path.join(BASE_DIR, 'data', 'parts.csv')

db = PartDatabase(CSV_PATH)


class TextQuery(BaseModel):
    query: str


@app.get("/api/health")
def health():
    return {"status": "ok", "model_loaded": False}


@app.post("/api/search/name")
def search_name(body: TextQuery):
    if not body.query.strip():
        raise HTTPException(status_code=400, detail="Query cannot be empty.")
    return {"results": db.search_by_name(body.query)}


@app.post("/api/search/description")
def search_description(body: TextQuery):
    if not body.query.strip():
        raise HTTPException(status_code=400, detail="Query cannot be empty.")
    return {"results": db.search_by_description(body.query)}


@app.post("/api/search/image")
async def search_image(file: UploadFile = File(...)):
    raise HTTPException(
        status_code=503,
        detail="Image search not available in this deployment. Use name or description search.",
    )
