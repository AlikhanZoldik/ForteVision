from __future__ import annotations

import asyncio
import base64
import json
import os
import secrets
import sqlite3
import time
import uuid
from contextlib import asynccontextmanager
from pathlib import Path

import cv2
from fastapi import FastAPI, File, Header, HTTPException, UploadFile
from fastapi.responses import FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles

from app.model import EventModel, ROOT
from app.vlm import describe_locally


DATA = ROOT / "data"
UPLOADS = DATA / "uploads"
DB = DATA / "jobs.sqlite3"
RULE = json.loads((ROOT / "rules.json").read_text(encoding="utf-8"))["default"]
API_KEY = os.getenv("API_KEY", "").strip()
MAX_BYTES = int(float(os.getenv("MAX_UPLOAD_MB", "50")) * 1024 * 1024)
MIN_CONFIDENCE = 0.45
THRESHOLD = max(MIN_CONFIDENCE, float(os.getenv("EVENT_THRESHOLD"))) if os.getenv("EVENT_THRESHOLD", "").strip() else None
worker_slots = asyncio.Semaphore(1)


def db() -> sqlite3.Connection:
    conn = sqlite3.connect(DB)
    conn.row_factory = sqlite3.Row
    return conn


def init_db() -> None:
    DATA.mkdir(parents=True, exist_ok=True)
    UPLOADS.mkdir(parents=True, exist_ok=True)
    with db() as conn:
        conn.execute("CREATE TABLE IF NOT EXISTS jobs (id TEXT PRIMARY KEY, status TEXT, filename TEXT, created REAL, updated REAL, result TEXT, error TEXT)")
        conn.execute("UPDATE jobs SET status='failed', error='Сервис перезапущен во время анализа', updated=? WHERE status='processing'", (time.time(),))


@asynccontextmanager
async def lifespan(_: FastAPI):
    init_db()
    if not API_KEY:
        raise RuntimeError("Задайте API_KEY в .env перед запуском.")
    yield


app = FastAPI(title="Forte Vision", version="1.0.0", lifespan=lifespan)
app.mount("/static", StaticFiles(directory=ROOT / "app" / "static"), name="static")


def authorize(value: str | None) -> None:
    if not value or not secrets.compare_digest(value, API_KEY):
        raise HTTPException(status_code=401, detail="Неверный или отсутствующий X-API-Key")


@app.get("/")
async def index():
    return FileResponse(ROOT / "app" / "static" / "index.html")


@app.get("/health")
async def health():
    path = ROOT / "models" / "event_detector.joblib"
    model_info = {}
    if path.exists():
        try:
            loaded = __import__("joblib").load(path)
            model_info = {"threshold": max(MIN_CONFIDENCE, float(os.getenv("EVENT_THRESHOLD") or loaded.get("threshold", MIN_CONFIDENCE))), "validation": loaded.get("validation")}
        except Exception:
            model_info = {"warning": "Файл модели есть, но прочитать его не удалось."}
    return {"status": "ok", "model_ready": path.exists(), **model_info}


@app.post("/api/v1/analyze", status_code=202)
async def analyze(file: UploadFile = File(...), x_api_key: str | None = Header(default=None)):
    authorize(x_api_key)
    if Path(file.filename or "").suffix.lower() != ".mov":
        raise HTTPException(status_code=415, detail="Принимаются видеозаписи .mov")
    job_id = uuid.uuid4().hex[:12]
    destination = UPLOADS / f"{job_id}.mov"
    size = 0
    try:
        with destination.open("wb") as out:
            while chunk := await file.read(1024 * 1024):
                size += len(chunk)
                if size > MAX_BYTES:
                    raise HTTPException(status_code=413, detail=f"Файл превышает лимит {MAX_BYTES // 1048576} МБ")
                out.write(chunk)
    except Exception:
        destination.unlink(missing_ok=True)
        raise
    finally:
        await file.close()
    now = time.time()
    with db() as conn:
        conn.execute("INSERT INTO jobs VALUES (?, 'processing', ?, ?, ?, NULL, NULL)", (job_id, file.filename, now, now))
    asyncio.create_task(process_job(job_id, destination))
    return {"job_id": job_id, "status": "processing"}


async def process_job(job_id: str, path: Path) -> None:
    async with worker_slots:
        try:
            result = await asyncio.to_thread(run_inference, job_id, path)
            with db() as conn:
                conn.execute("UPDATE jobs SET status='done', result=?, updated=? WHERE id=?", (json.dumps(result, ensure_ascii=False), time.time(), job_id))
        except Exception as exc:
            with db() as conn:
                conn.execute("UPDATE jobs SET status='failed', error=?, updated=? WHERE id=?", (str(exc)[:1000], time.time(), job_id))
        finally:
            path.unlink(missing_ok=True)


def run_inference(job_id: str, path: Path) -> dict:
    model = EventModel()
    predictions, duration = model.analyze(path, THRESHOLD)
    violations = []
    for event in predictions:
        frame = event.pop("frame")
        desc = describe_locally(frame, event["timecode_sec"])
        if not desc:
            desc = RULE["description"]
        ok, buffer = cv2.imencode(".jpg", frame, [cv2.IMWRITE_JPEG_QUALITY, 84])
        if not ok:
            continue
        b64 = base64.b64encode(buffer.tobytes()).decode("ascii")
        violations.append({
            "type": RULE["type"],
            "timecode_sec": event["timecode_sec"],
            "description": desc,
            "confidence": event["confidence"],
            "evidence_frames": [{"timestamp_sec": event["timecode_sec"], "image_base64": b64}],
            "rule": {"title": RULE["source_title"], "section": RULE["section"], "scope_note": RULE["scope_note"], "url": RULE["source_url"]},
        })
    validation = model.bundle.get("validation", {})
    return {
        "job_id": job_id, "status": "done",
        "verdict": "violation" if violations else "no_violation",
        "duration_sec": duration, "violations": violations,
        "model_quality": {
            "threshold": max(MIN_CONFIDENCE, float(THRESHOLD if THRESHOLD is not None else model.bundle.get("threshold", MIN_CONFIDENCE))),
            "validation_video_groups": validation.get("video_groups"),
            "precision": validation.get("precision"),
            "recall": validation.get("recall"),
            "warning": "Слабая проверка на небольшом наборе; результат требует ручной проверки." if validation.get("recall", 0) < 0.8 or validation.get("precision", 0) < 0.3 else None,
        },
    }


@app.get("/api/v1/jobs/{job_id}")
async def get_job(job_id: str, x_api_key: str | None = Header(default=None)):
    authorize(x_api_key)
    with db() as conn:
        row = conn.execute("SELECT * FROM jobs WHERE id=?", (job_id,)).fetchone()
    if not row:
        raise HTTPException(status_code=404, detail="Задание не найдено")
    if row["status"] == "done":
        return JSONResponse(json.loads(row["result"]))
    response = {"job_id": job_id, "status": row["status"]}
    if row["error"]:
        response["error"] = row["error"]
    return response

