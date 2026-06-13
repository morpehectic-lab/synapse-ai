# -*- coding: utf-8 -*-
"""
SYNAPSE AI — API server (FastAPI)
Ishga tushirish: uvicorn server:app --host 0.0.0.0 --port 8000
"""
import os
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

from engine import SynapseAI

DATA_DIR = os.environ.get("SYNAPSE_DATA", os.path.join(os.path.dirname(__file__), "data"))
INDEX_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "index")

app = FastAPI(title="SYNAPSE AI", version="1.0")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # production da SYNAPSE domeniga cheklang
    allow_methods=["*"],
    allow_headers=["*"],
)

print("SYNAPSE AI yuklanmoqda...")
ai = SynapseAI(data_dir=DATA_DIR, index_dir=INDEX_DIR)
print(f"Tayyor! Backend: {ai.backend} | Chunklar: {len(ai.chunks)}")


class Question(BaseModel):
    savol: str
    direction: str | None = None  # python | html | flutter | sql
    modul: int | None = None
    top_k: int = 3


@app.get("/")
def root():
    return {"service": "SYNAPSE AI", "backend": ai.backend,
            "chunks": len(ai.chunks), "status": "ok"}


@app.get("/health")
def health():
    return {"ok": True}


@app.post("/ask")
def ask(q: Question):
    return ai.ask(q.savol, direction=q.direction, modul=q.modul, top_k=q.top_k)
