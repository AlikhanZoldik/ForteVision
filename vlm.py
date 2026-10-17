from __future__ import annotations

import base64
import json
import os
import urllib.parse
import urllib.request

import cv2


def describe_locally(frame, timestamp: float) -> str | None:
    """Optional Ollama VLM enrichment. Only loopback endpoints are allowed."""
    base_url = os.getenv("OLLAMA_BASE_URL", "").strip().rstrip("/")
    if not base_url:
        return None
    parsed = urllib.parse.urlparse(base_url)
    if parsed.hostname not in {"localhost", "127.0.0.1", "::1", "host.docker.internal", "ollama"}:
        raise RuntimeError("OLLAMA_BASE_URL должен указывать на локальный Ollama: loopback, host.docker.internal или контейнер ollama.")
    model = os.getenv("OLLAMA_VISION_MODEL", "").strip()
    if not model:
        return None
    ok, encoded = cv2.imencode(".jpg", frame, [cv2.IMWRITE_JPEG_QUALITY, 82])
    if not ok:
        return None
    body = {
        "model": model,
        "stream": False,
        "format": "json",
        "messages": [{"role": "user", "content": (
            f"Кадр из видео кассового обслуживания, время {timestamp:.1f} секунды. "
            "Проверяемый визуальный сигнал: другой человек подходит к рабочему месту кассира/входит в рабочую зону, "
            "либо сотрудник встаёт, отходит или меняется местами с другим человеком. "
            "Опиши только то, что действительно видно на этом кадре; не утверждай, что правило нарушено, если это не очевидно. "
            "Опиши только видимые действия людей и предметов на русском, без идентификации лиц "
            "и без вывода о юридической виновности. До 35 слов. Верни JSON {description: string}."
        ), "images": [base64.b64encode(encoded.tobytes()).decode("ascii")]}]
    }
    request = urllib.request.Request(
        f"{base_url}/api/chat", data=json.dumps(body).encode(),
        headers={"Content-Type": "application/json"}, method="POST"
    )
    try:
        with urllib.request.urlopen(request, timeout=25) as response:
            payload = json.loads(response.read().decode("utf-8"))
        result = json.loads(payload.get("message", {}).get("content", "{}"))
        description = str(result.get("description", "")).strip()
        return description[:500] or None
    except Exception:
        return None

