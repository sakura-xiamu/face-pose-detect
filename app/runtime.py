"""创建 ONNX Runtime 会话，优先 DirectML。"""

from __future__ import annotations

from pathlib import Path

import onnxruntime as ort


def provider_label(provider: str) -> str:
    if provider == "DmlExecutionProvider":
        return "DML"
    if provider == "CPUExecutionProvider":
        return "CPU"
    return provider


def open_session(model_path: Path):
    available = ort.get_available_providers()
    if "DmlExecutionProvider" in available:
        requested = ["DmlExecutionProvider", "CPUExecutionProvider"]
    else:
        requested = ["CPUExecutionProvider"]
    session = ort.InferenceSession(str(model_path), providers=requested)
    used = session.get_providers()[0]
    return session, used
