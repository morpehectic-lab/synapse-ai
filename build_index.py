# -*- coding: utf-8 -*-
"""
SYNAPSE AI — Index yaratish
JSON darsliklardan embedding bazasini quradi.
Bir marta ishga tushiriladi (yoki darslik yangilanganda).
"""
import json
import os
import numpy as np
from sentence_transformers import SentenceTransformer

# Ko'p tilli kichik model (~470MB lekin RAM da ~500MB, CPU da ishlaydi)
# paraphrase-multilingual-MiniLM-L12-v2 — o'zbek/rus/ingliz tillarini tushunadi
MODEL_NAME = "paraphrase-multilingual-MiniLM-L12-v2"

DIRECTIONS = {
    "python":  ("python_65.json",  "Python Backend"),
    "html":    ("html_60.json",    "HTML/Frontend"),
    "flutter": ("flutter_55.json", "Flutter/Mobile"),
    "sql":     ("sql_45.json",     "SQL/Database"),
}

DATA_DIR = os.environ.get("SYNAPSE_DATA", "/home/claude")
OUT_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "index")
os.makedirs(OUT_DIR, exist_ok=True)


def make_chunks():
    """Har modulni qidiriladigan bo'laklarga ajratadi."""
    chunks = []
    for dir_key, (fname, label) in DIRECTIONS.items():
        path = os.path.join(DATA_DIR, fname)
        with open(path, encoding="utf-8") as f:
            modules = json.load(f)

        for m in modules:
            base = {
                "direction": dir_key,
                "direction_label": label,
                "modul": m["modul"],
                "mavzu": m["mavzu"],
            }

            # 1) Nazariya — paragraflarga bo'lib
            naz = m.get("nazariya", "")
            paras = [p.strip() for p in naz.split("\n\n") if len(p.strip()) > 80]
            if not paras and naz.strip():
                paras = [naz.strip()]
            for i, p in enumerate(paras):
                chunks.append({**base, "type": "nazariya", "part": i, "text": p})

            # 2) Quiz savollari — savol+to'g'ri javob+tushuntirish
            for qi, q in enumerate(m.get("quiz", [])):
                javob_harf = q.get("javob", "")
                javob_matn = q.get(javob_harf, "")
                t = f"Savol: {q.get('savol','')}\nJavob: {javob_matn}\nTushuntirish: {q.get('tushuntirish','')}"
                chunks.append({**base, "type": "quiz", "part": qi, "text": t})

            # 3) Topshiriq
            top = m.get("topshiriq", {})
            if top:
                t = f"Vazifa: {top.get('vazifa','')}\nYechim: {top.get('yechim','')}"
                chunks.append({**base, "type": "topshiriq", "part": 0, "text": t})

            # 4) PDF reference
            pdf = m.get("pdf_reference", "")
            if pdf and len(pdf) > 50:
                chunks.append({**base, "type": "pdf", "part": 0, "text": pdf})

    return chunks


def main():
    print("1/3 Chunklar yaratilmoqda...")
    chunks = make_chunks()
    print(f"    Jami: {len(chunks)} chunk")

    print("2/3 Model yuklanmoqda (birinchi marta internetdan yuklaydi)...")
    model = SentenceTransformer(MODEL_NAME)

    print("3/3 Embedding hisoblanmoqda (bir necha daqiqa)...")
    texts = [c["text"] for c in chunks]
    emb = model.encode(texts, batch_size=64, show_progress_bar=True,
                       normalize_embeddings=True)

    np.save(os.path.join(OUT_DIR, "embeddings.npy"), emb.astype(np.float32))
    with open(os.path.join(OUT_DIR, "chunks.json"), "w", encoding="utf-8") as f:
        json.dump(chunks, f, ensure_ascii=False)

    print(f"\nTAYYOR! index/ papkasida:")
    print(f"  embeddings.npy  ({emb.shape[0]} x {emb.shape[1]})")
    print(f"  chunks.json     ({len(chunks)} chunk)")


if __name__ == "__main__":
    main()
