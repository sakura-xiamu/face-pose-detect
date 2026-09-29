"""MobileNet 头部姿态 ONNX 推理。"""

from __future__ import annotations

import logging
import time

import cv2
import numpy as np

from app.geometry import rotation_matrix_to_euler

logger = logging.getLogger("uvicorn.error")

IMAGENET_MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
IMAGENET_STD = np.array([0.229, 0.224, 0.225], dtype=np.float32)


class HeadPoseEstimator:
    def __init__(self, session, name: str = "MobileNet"):
        self.session = session
        self.name = name
        model_input = session.get_inputs()[0]
        self.input_name = model_input.name
        shape = model_input.shape
        height = shape[2] if isinstance(shape[2], int) and shape[2] > 0 else 224
        width = shape[3] if isinstance(shape[3], int) and shape[3] > 0 else 224
        self.input_size = (int(width), int(height))
        self.output_names = [item.name for item in session.get_outputs()]

    def estimate(self, face_bgr: np.ndarray) -> tuple[float, float, float]:
        rgb = cv2.cvtColor(face_bgr, cv2.COLOR_BGR2RGB)
        started = time.perf_counter()
        resized = cv2.resize(rgb, self.input_size, interpolation=cv2.INTER_LINEAR)
        resize_ms = (time.perf_counter() - started) * 1000.0
        started = time.perf_counter()
        tensor = resized.astype(np.float32) / 255.0
        tensor = (tensor - IMAGENET_MEAN) / IMAGENET_STD
        normalize_ms = (time.perf_counter() - started) * 1000.0
        tensor = np.transpose(tensor, (2, 0, 1))[None]
        tensor = np.ascontiguousarray(tensor, dtype=np.float32)
        started = time.perf_counter()
        outputs = self.session.run(self.output_names, {self.input_name: tensor})
        elapsed_ms = (time.perf_counter() - started) * 1000.0
        fps = 1000.0 / elapsed_ms if elapsed_ms > 0 else 0.0
        logger.info(
            "%s 缩放 %.2f ms，归一化 %.2f ms，检测耗时 %.2f ms（%.1f FPS）",
            self.name,
            resize_ms,
            normalize_ms,
            elapsed_ms,
            fps,
        )
        rotation = np.asarray(outputs[0], dtype=np.float32)
        if rotation.shape[-2:] != (3, 3):
            raise RuntimeError(f"姿态模型输出不是 3x3 旋转矩阵，实际形状 {rotation.shape}")
        pitch, yaw, roll = rotation_matrix_to_euler(rotation.reshape(-1, 3, 3))[0]
        return float(pitch), float(yaw), float(roll)
