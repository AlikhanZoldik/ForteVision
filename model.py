from __future__ import annotations

import os
from pathlib import Path

import cv2
import joblib
import numpy as np


ROOT = Path(__file__).resolve().parents[1]
MODEL_PATH = Path(os.getenv("MODEL_PATH", ROOT / "models" / "event_detector.joblib"))


def frame_features(frame: np.ndarray, previous: np.ndarray | None = None) -> np.ndarray:
    """Compact appearance + motion descriptor; no face or identity features."""
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    gray = cv2.resize(gray, (96, 96), interpolation=cv2.INTER_AREA)
    hog = cv2.HOGDescriptor(
        (96, 96), (16, 16), (8, 8), (8, 8), 9
    ).compute(gray).reshape(-1)
    gray_f = gray.astype(np.float32) / 255.0
    appearance = np.array(
        [gray_f.mean(), gray_f.std(), *[gray_f[y:y + 48, x:x + 48].mean()
          for y in (0, 48) for x in (0, 48)]], dtype=np.float32
    )
    if previous is None:
        motion = np.zeros(5, dtype=np.float32)
    else:
        diff = cv2.absdiff(gray, previous).astype(np.float32) / 255.0
        motion = np.array(
            [diff.mean(), *[diff[y:y + 48, x:x + 48].mean()
              for y in (0, 48) for x in (0, 48)]], dtype=np.float32
        )
    return np.concatenate((hog.astype(np.float32), appearance, motion))


class EventModel:
    def __init__(self) -> None:
        if not MODEL_PATH.exists():
            raise RuntimeError(
                f"Модель не найдена: {MODEL_PATH}. Сначала обучите её на открытом наборе: "
                "python scripts/train.py --dataset <папка open из архива>"
            )
        self.bundle = joblib.load(MODEL_PATH)
        self.classifier = self.bundle["classifier"]
        self.sample_fps = float(self.bundle.get("sample_fps", 1.0))

    def analyze(self, video_path: Path, threshold: float | None = None) -> tuple[list[dict], float]:
        cap = cv2.VideoCapture(str(video_path))
        if not cap.isOpened():
            raise ValueError("Не удалось открыть видеозапись. Проверьте файл .mov.")
        source_fps = cap.get(cv2.CAP_PROP_FPS) or 29.0
        duration = (cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0) / source_fps
        step = max(1, round(source_fps / self.sample_fps))
        scored: list[tuple[float, float, np.ndarray]] = []
        previous = None
        index = 0
        try:
            while True:
                ok, frame = cap.read()
                if not ok:
                    break
                if index % step == 0:
                    t = index / source_fps
                    small = cv2.resize(frame, (640, 360), interpolation=cv2.INTER_AREA)
                    features = frame_features(frame, previous).reshape(1, -1)
                    score = float(self.classifier.predict_proba(features)[0, 1])
                    scored.append((t, score, small))
                    previous = cv2.resize(cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY), (96, 96), interpolation=cv2.INTER_AREA)
                index += 1
        finally:
            cap.release()

        # Product policy: scores below 45% are never emitted as violations.
        threshold = max(0.45, float(threshold if threshold is not None else self.bundle.get("threshold", 0.62)))
        peaks = [item for i, item in enumerate(scored)
                 if item[1] >= threshold
                 and (i == 0 or item[1] >= scored[i - 1][1])
                 and (i + 1 == len(scored) or item[1] >= scored[i + 1][1])]
        peaks.sort(key=lambda x: x[1], reverse=True)
        events: list[dict] = []
        for t, score, frame in peaks:
            if any(abs(t - e["timecode_sec"]) < 4.0 for e in events):
                continue
            events.append({"timecode_sec": round(t, 2), "confidence": round(score, 3), "frame": frame})
            if len(events) >= 5:
                break
        events.sort(key=lambda e: e["timecode_sec"])
        return events, round(duration, 2)

