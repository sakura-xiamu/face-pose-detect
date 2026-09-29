"""低头 / 转头检测服务。页面在 / ，接口在 /api/detect/ 。"""

from __future__ import annotations

import asyncio
import json
import threading
from contextlib import asynccontextmanager

from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.responses import FileResponse, StreamingResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

from app.engine import MODEL_LABELS, ROOT, PoseEngine, decode_image_bytes
from app.geometry import parse_turn_range

MAX_IMAGE_BYTES = 20 * 1024 * 1024
engine = PoseEngine()


class StatusHub:
    def __init__(self):
        self._condition = threading.Condition()
        self._version = 0
        self.snapshot: dict = {}

    def publish(self, snapshot: dict) -> None:
        with self._condition:
            self.snapshot = snapshot
            self._version += 1
            self._condition.notify_all()

    def _wait(self, version: int):
        with self._condition:
            if self._version == version:
                self._condition.wait(timeout=15)
            return self.snapshot, self._version

    async def stream(self):
        loop = asyncio.get_running_loop()
        version = -1
        while True:
            snapshot, version = await loop.run_in_executor(None, self._wait, version)
            payload = json.dumps({"status": snapshot}, ensure_ascii=False)
            yield f"data: {payload}\n\n"


class ServiceState:
    def __init__(self):
        self.hub = StatusHub()
        self.total_detections = 0
        self.run_state = "idle"
        self.latest_message = "请先初始化模型"
        self.latest_error = None
        self.latest_result = None

    def publish(self) -> None:
        self.hub.publish(self.snapshot())

    def snapshot(self) -> dict:
        return {
            "initialized": engine.initialized,
            "state": self.run_state,
            "provider": engine.provider_text,
            "model": engine.model_name,
            "model_label": MODEL_LABELS.get(engine.model_name or "", ""),
            "total_detections": self.total_detections,
            "latest_message": self.latest_message,
            "latest_error": self.latest_error,
            "latest_detection_result": self.latest_result,
        }


service = ServiceState()


class InitRequest(BaseModel):
    model_name: str = "mobilenetv3_small"


class Base64Request(BaseModel):
    image_data: str
    lower_head_pc: float = -15
    turnhead_pc: str = "-30,30"


def _thresholds(lower_head_pc: float, turnhead_pc: str) -> tuple[float, float, float]:
    try:
        turn_min, turn_max = parse_turn_range(turnhead_pc)
        return float(lower_head_pc), turn_min, turn_max
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc


def _run_detect(image_bytes: bytes, lower: float, turn_min: float, turn_max: float) -> dict:
    if len(image_bytes) > MAX_IMAGE_BYTES:
        raise ValueError("图片超过 20MB")
    try:
        image = decode_image_bytes(image_bytes)
    except Exception as exc:
        raise ValueError(f"无法解析图片: {exc}") from exc
    result = engine.detect(image, lower, turn_min, turn_max)
    service.total_detections += 1
    service.latest_result = result
    service.latest_error = None
    service.latest_message = result["message"]
    service.run_state = "ready"
    service.publish()
    return {"success": True, "message": result["message"], "result": result}


@asynccontextmanager
async def lifespan(_app: FastAPI):
    service.publish()
    yield
    engine.release()


app = FastAPI(lifespan=lifespan)
app.mount("/static", StaticFiles(directory=ROOT / "static"), name="static")


@app.get("/")
def index():
    return FileResponse(ROOT / "static" / "index.html")


@app.get("/api/detect/status")
def status():
    return service.snapshot()


@app.get("/api/detect/events")
async def events():
    return StreamingResponse(service.hub.stream(), media_type="text/event-stream")


@app.post("/api/detect/initialize")
async def initialize(body: InitRequest):
    service.run_state = "loading"
    service.latest_error = None
    service.latest_message = "正在初始化模型..."
    service.publish()
    try:
        message = await asyncio.to_thread(engine.initialize, body.model_name)
    except Exception as exc:
        service.run_state = "error"
        service.latest_error = str(exc)
        service.latest_message = str(exc)
        service.publish()
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    service.run_state = "ready"
    service.latest_message = message
    service.latest_error = None
    service.publish()
    return {"success": True, "message": message, "provider": engine.provider_text}


@app.post("/api/detect/release")
def release():
    engine.release()
    service.run_state = "idle"
    service.latest_message = "资源已释放"
    service.latest_error = None
    service.latest_result = None
    service.publish()
    return {"success": True, "message": "资源已释放"}


async def _detect_bytes(payload: bytes, lower_head_pc: float, turnhead_pc: str) -> dict:
    lower, turn_min, turn_max = _thresholds(lower_head_pc, turnhead_pc)
    try:
        return await asyncio.to_thread(_run_detect, payload, lower, turn_min, turn_max)
    except (RuntimeError, ValueError) as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc


@app.post("/api/detect/image")
async def detect_image(
    file: UploadFile = File(...),
    lower_head_pc: float = Form(-15),
    turnhead_pc: str = Form("-30,30"),
):
    payload = await file.read()
    return await _detect_bytes(payload, lower_head_pc, turnhead_pc)


@app.post("/api/detect/base64")
async def detect_base64(body: Base64Request):
    import base64

    encoded = body.image_data.strip()
    if "," in encoded and encoded.startswith("data:"):
        encoded = encoded.split(",", 1)[1]
    try:
        payload = base64.b64decode(encoded, validate=False)
    except Exception as exc:
        raise HTTPException(status_code=400, detail="Base64 无法解码") from exc
    return await _detect_bytes(payload, body.lower_head_pc, body.turnhead_pc)
