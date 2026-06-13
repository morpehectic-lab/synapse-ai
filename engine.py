# -*- coding: utf-8 -*-
"""
SYNAPSE AI — yadro (engine)
Ikki backend:
  1) sentence-transformers (production, semantik qidiruv, ko'p tilli)
  2) TF-IDF (fallback — model yuklanmagan muhitda ham ishlaydi)
Backend avtomatik tanlanadi.
"""
import json
import os
import re
import time

import numpy as np

MODEL_NAME = "paraphrase-multilingual-MiniLM-L12-v2"
MIN_SCORE_ST = 0.35    # sentence-transformers uchun
MIN_SCORE_TFIDF = 0.12 # TF-IDF uchun (boshqa shkala)

# Dasturlash bilan aloqasi bo'lmagan, tez-tez so'raladigan mavzular —
# bu so'zlar topilsa va dasturlash atamasi bo'lmasa, javob berilmaydi
OFF_TOPIC_PATTERNS = [
    "ob-havo", "ob havo", "harorat bugun", "yomg'ir", "havo qanday",
    "restoran", "ovqat", "taom tayyorla", "retsept",
    "poytaxt", "futbol", "match natija", "kino", "film",
    "musiqa", "qo'shiq", "yangiliklar", "siyosat",
    "valyuta kursi", "dollar kursi", "soat necha", "bugun necha",
]
PROGRAMMING_HINTS = [
    "kod", "dastur", "funksiya", "function", "class", "metod", "method",
    "python", "html", "flutter", "sql", "dart", "widget", "css", "javascript",
    "o'zgaruvchi", "ozgaruvchi", "variable", "ma'lumot", "database", "query",
    "selector", "tag", "loop", "sikl", "massiv", "array", "list", "map",
]

DIRECTIONS = {
    "python":  ("python_65.json",  "Python Backend"),
    "html":    ("html_60.json",    "HTML/Frontend"),
    "flutter": ("flutter_55.json", "Flutter/Mobile"),
    "sql":     ("sql_45.json",     "SQL/Database"),
}


def make_chunks(data_dir: str):
    """JSON darsliklarni qidiriladigan bo'laklarga ajratadi."""
    chunks = []
    for dir_key, (fname, label) in DIRECTIONS.items():
        path = os.path.join(data_dir, fname)
        if not os.path.exists(path):
            continue
        with open(path, encoding="utf-8") as f:
            modules = json.load(f)

        for m in modules:
            base = {
                "direction": dir_key,
                "direction_label": label,
                "modul": m["modul"],
                "mavzu": m["mavzu"],
            }
            # Nazariya — paragraflar
            naz = m.get("nazariya", "")
            paras = [p.strip() for p in naz.split("\n\n") if len(p.strip()) > 80]
            if not paras and naz.strip():
                paras = [naz.strip()]
            for i, p in enumerate(paras):
                # Mavzuni matnga qo'shamiz — qidiruv aniqroq bo'ladi
                chunks.append({**base, "type": "nazariya", "part": i,
                               "text": f"{m['mavzu']}. {p}"})
            # Quiz
            for qi, q in enumerate(m.get("quiz", [])):
                harf = q.get("javob", "")
                javob = q.get(harf, "")
                t = f"Savol: {q.get('savol','')}\nJavob: {javob}\nTushuntirish: {q.get('tushuntirish','')}"
                chunks.append({**base, "type": "quiz", "part": qi, "text": t})
            # Topshiriq
            top = m.get("topshiriq", {})
            if top:
                t = f"{m['mavzu']}. Vazifa: {top.get('vazifa','')}\nYechim: {top.get('yechim','')}"
                chunks.append({**base, "type": "topshiriq", "part": 0, "text": t})
            # PDF
            pdf = m.get("pdf_reference", "")
            if pdf and len(pdf) > 50:
                chunks.append({**base, "type": "pdf", "part": 0,
                               "text": f"{m['mavzu']}. {pdf}"})
    return chunks


class SynapseAI:
    def __init__(self, data_dir: str, index_dir: str | None = None):
        self.chunks = make_chunks(data_dir)
        self.backend = None
        self.min_score = MIN_SCORE_TFIDF
        self._init_backend(index_dir)

    # ---------- backend tanlash ----------
    def _init_backend(self, index_dir):
        # 1) sentence-transformers urinish
        try:
            from sentence_transformers import SentenceTransformer
            self.model = SentenceTransformer(MODEL_NAME)
            texts = [c["text"] for c in self.chunks]
            emb_path = os.path.join(index_dir, "embeddings.npy") if index_dir else None
            if emb_path and os.path.exists(emb_path):
                self.emb = np.load(emb_path)
            else:
                self.emb = self.model.encode(
                    texts, batch_size=64, normalize_embeddings=True
                ).astype(np.float32)
                if index_dir:
                    os.makedirs(index_dir, exist_ok=True)
                    np.save(emb_path, self.emb)
            self.backend = "sentence-transformers"
            self.min_score = MIN_SCORE_ST
            return
        except Exception as e:
            print(f"[SYNAPSE AI] sentence-transformers ishlamadi: {e}")

        # 2) TF-IDF fallback
        from sklearn.feature_extraction.text import TfidfVectorizer
        texts = [c["text"] for c in self.chunks]
        uz_stop = [
            "nima", "qanday", "qachon", "qayerda", "qaysi", "nega", "necha",
            "bo'ladi", "boladi", "bo'lsa", "bolsa", "kerak", "mumkin",
            "uchun", "bilan", "haqida", "ham", "va", "yoki", "lekin",
            "bu", "shu", "u", "ular", "men", "sen", "biz", "siz",
            "deb", "degan", "edi", "ekan", "esa", "agar", "chunki",
            "kim", "ertaga", "bugun", "kecha", "hozir", "natija", "natijasi",
            "bo", "ladi", "lsa", "qilinadi", "qilish", "beradi", "berish",
            "the", "is", "are", "what", "how", "a", "an", "in", "on",
        ]
        self.vec = TfidfVectorizer(
            lowercase=True, ngram_range=(1, 2), max_features=60000,
            sublinear_tf=True, stop_words=uz_stop,
        )
        self.tfidf = self.vec.fit_transform(texts)
        self.backend = "tfidf"
        self.min_score = MIN_SCORE_TFIDF

    # ---------- qidiruv ----------
    def _scores(self, savol: str) -> np.ndarray:
        if self.backend == "sentence-transformers":
            q = self.model.encode([savol], normalize_embeddings=True)[0]
            return self.emb @ q
        q = self.vec.transform([savol])
        return (self.tfidf @ q.T).toarray().ravel()

    def ask(self, savol: str, direction: str | None = None, top_k: int = 3) -> dict:
        t0 = time.time()
        savol_lower = savol.lower()

        # Tezkor heuristika: ma'lum-mavzudan-tashqari so'zlar bor,
        # dasturlash atamasi yo'q — darhol rad etamiz
        has_off_topic = any(p in savol_lower for p in OFF_TOPIC_PATTERNS)
        has_prog_hint = any(p in savol_lower for p in PROGRAMMING_HINTS)
        if has_off_topic and not has_prog_hint:
            ms = round((time.time() - t0) * 1000)
            return {
                "javob": "Bu savolga javob bera olmayman. Men faqat SYNAPSE dasturlash "
                         "darsligi (Python, HTML/Frontend, Flutter, SQL) bo'yicha javob "
                         "beraman. Dasturlash bo'yicha savol bering!",
                "topildi": False, "score": 0.0,
                "ms": ms, "manba": None, "backend": self.backend,
            }

        raw_scores = self._scores(savol)
        scores = raw_scores

        if direction:
            mask = np.array([c["direction"] == direction for c in self.chunks])
            scores = np.where(mask, raw_scores, -1.0)

        top_idx = np.argsort(-scores)[:top_k]
        best_score = float(scores[top_idx[0]])
        global_best = float(raw_scores.max())
        ms = round((time.time() - t0) * 1000)

        # Savol shu yo'nalishga tegishli emasligini aniqlash:
        # 1) Umuman past skor (hech qayerda mos kelmaydi)
        # 2) Yoki global eng yaxshi moslik boshqa yo'nalishda ancha yuqori
        #    (savol mavzu doirasidan tashqarida)
        rejected = best_score < self.min_score
        if direction and not rejected and (global_best - best_score) > 0.08:
            rejected = True

        if rejected:
            return {
                "javob": "Bu savolga javob bera olmayman. Men faqat SYNAPSE dasturlash "
                         "darsligi (Python, HTML/Frontend, Flutter, SQL) bo'yicha javob "
                         "beraman. Dasturlash bo'yicha savol bering!",
                "topildi": False, "score": round(best_score, 3),
                "ms": ms, "manba": None, "backend": self.backend,
            }

        best = self.chunks[int(top_idx[0])]
        manbalar = []
        for i in top_idx:
            c, s = self.chunks[int(i)], float(scores[int(i)])
            if s >= self.min_score:
                manbalar.append({
                    "direction": c["direction_label"], "modul": c["modul"],
                    "mavzu": c["mavzu"], "type": c["type"], "score": round(s, 3),
                })

        text = re.sub(r"\n{3,}", "\n\n", best["text"]).strip()[:1200]
        return {
            "javob": text, "topildi": True, "score": round(best_score, 3),
            "ms": ms, "backend": self.backend,
            "manba": {"direction": best["direction_label"],
                      "modul": best["modul"], "mavzu": best["mavzu"]},
            "qoshimcha": manbalar[1:],
        }
