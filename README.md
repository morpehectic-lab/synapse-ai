# SYNAPSE AI

JSON darsliklardan javob beradigan lokal AI. Tashqi AI API ishlatilmaydi.

## Tuzilma
```
synapse_ai/
├── engine.py        # Yadro: chunk + qidiruv (2 backend)
├── server.py        # FastAPI API
├── build_index.py   # Embedding indexini oldindan qurish (ixtiyoriy)
├── data/            # JSON darsliklar (225 modul)
│   ├── python_65.json
│   ├── html_60.json
│   ├── flutter_55.json
│   └── sql_45.json
├── requirements.txt
└── Dockerfile
```

## Backend lar
1. **sentence-transformers** (asosiy) — semantik qidiruv, ko'p tilli.
   Birinchi ishga tushishda Hugging Face dan model yuklanadi (~470MB).
2. **TF-IDF** (fallback) — model yuklanmasa avtomatik ishlaydi.

## Lokal ishga tushirish
```bash
pip install -r requirements.txt
uvicorn server:app --host 0.0.0.0 --port 8000
```

## Railway ga deploy
1. Yangi Railway service → GitHub repo yoki `railway up`
2. Dockerfile avtomatik ishlatiladi
3. PORT env Railway beradi — Dockerfile dagi 8000 ni $PORT ga moslang yoki
   Start Command: `uvicorn server:app --host 0.0.0.0 --port $PORT`

## API

### POST /ask
```json
{"savol": "Flutter StatelessWidget nima?", "direction": "flutter"}
```
Javob:
```json
{
  "javob": "...",
  "topildi": true,
  "score": 0.71,
  "ms": 45,
  "manba": {"direction": "Flutter/Mobile", "modul": 4, "mavzu": "..."},
  "qoshimcha": [...]
}
```
`direction` ixtiyoriy: python | html | flutter | sql

## SYNAPSE (Node.js) bilan ulash
```js
const r = await fetch(process.env.SYNAPSE_AI_URL + '/ask', {
  method: 'POST',
  headers: {'Content-Type': 'application/json'},
  body: JSON.stringify({ savol: userQuestion, direction: dir })
});
const data = await r.json();
// data.javob — o'quvchiga ko'rsatiladi
// data.manba — "Flutter M4" deb manba ko'rsatiladi
```
