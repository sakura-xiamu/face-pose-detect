"""加载 YuNet 和所选 MobileNet，对一帧图像做低头/转头判定。"""

from __future__ import annotations

import threading
from pathlib import Path

import cv2
import numpy as np
from PIL import Image, ImageOps

from app.face_detector import YuNetDetector
from app.geometry import classify_head_pose, expand_xywh
from app.pose_estimator import HeadPoseEstimator
from app.runtime import open_session, provider_label

ROOT = Path(__file__).resolve().parent.parent
WEIGHTS = ROOT / "weights"
POSE_FILES = {
    "mobilenetv3_small": WEIGHTS / "mobilenetv3_small.onnx",
    "mobilenetv2": WEIGHTS / "mobilenetv2.onnx",
}
FACE_FILE = WEIGHTS / "face_detection_yunet_2023mar.onnx"
MODEL_LABELS = {
    "mobilenetv3_small": "MobileNet V3-Small",
    "mobilenetv2": "MobileNet V2",
}
FACE_SCORE_THRESHOLD = 0.6
FACE_NMS_THRESHOLD = 0.3
FACE_EXPAND = 0.2


class PoseEngine:
    def __init__(self):
        self._lock = threading.Lock()
        self.detector: YuNetDetector | None = None
        self.estimator: HeadPoseEstimator | None = None
        self.model_name: str | None = None
        self.provider = ""
        self.provider_text = ""
        self.initialized = False

    def initialize(self, model_name: str) -> str:
        if model_name not in POSE_FILES:
            raise ValueError("模型只能是 mobilenetv3_small 或 mobilenetv2")
        pose_path = POSE_FILES[model_name]
        if not FACE_FILE.is_file():
            raise FileNotFoundError(f"缺少人脸模型: {FACE_FILE}")
        if not pose_path.is_file():
            raise FileNotFoundError(f"缺少姿态模型: {pose_path}")

        face_session, face_provider = open_session(FACE_FILE)
        pose_session, pose_provider = open_session(pose_path)
        detector = YuNetDetector(
            face_session,
            score_threshold=FACE_SCORE_THRESHOLD,
            nms_threshold=FACE_NMS_THRESHOLD,
        )
        estimator = HeadPoseEstimator(pose_session, name=MODEL_LABELS[model_name])
        with self._lock:
            self.detector = detector
            self.estimator = estimator
            self.model_name = model_name
            self.provider = pose_provider
            self.provider_text = provider_label(pose_provider)
            if face_provider != pose_provider:
                self.provider_text = (
                    f"{provider_label(pose_provider)} / 人脸 {provider_label(face_provider)}"
                )
            self.initialized = True
        label = MODEL_LABELS[model_name]
        if pose_provider == "DmlExecutionProvider":
            return f"已加载 {label}（DML）"
        return f"已加载 {label}（CPU，DirectML 不可用）"

    def release(self) -> None:
        with self._lock:
            self.detector = None
            self.estimator = None
            self.model_name = None
            self.provider = ""
            self.provider_text = ""
            self.initialized = False

    def detect(
        self,
        image_bgr: np.ndarray,
        lower_head_pc: float,
        turn_min: float,
        turn_max: float,
    ) -> dict:
        with self._lock:
            if self.detector is None or self.estimator is None:
                raise RuntimeError("请先初始化模型")
            return self._detect_locked(image_bgr, lower_head_pc, turn_min, turn_max)

    def _detect_locked(self, image_bgr, lower_head_pc, turn_min, turn_max) -> dict:
        detector = self.detector
        estimator = self.estimator
        height, width = image_bgr.shape[:2]
        faces = detector.detect(image_bgr)
        result = {
            "has_pitch": False,
            "has_yaw": False,
            "pitch": None,
            "yaw": None,
            "face_box": None,
            "image_width": width,
            "image_height": height,
            "message": "未检测到人脸",
        }
        if len(faces) == 0:
            return result

        areas = faces[:, 2] * faces[:, 3]
        face = faces[int(np.argmax(areas))]
        x, y, box_w, box_h = [float(value) for value in face[:4]]
        x1, y1, x2, y2 = expand_xywh(x, y, box_w, box_h, FACE_EXPAND, width, height)
        crop = image_bgr[y1:y2, x1:x2]
        if crop.size == 0 or crop.shape[0] < 2 or crop.shape[1] < 2:
            return result

        pitch, yaw, _roll = estimator.estimate(crop)
        has_pitch, has_yaw = classify_head_pose(pitch, yaw, lower_head_pc, turn_min, turn_max)
        result.update(
            {
                "has_pitch": has_pitch,
                "has_yaw": has_yaw,
                "pitch": round(pitch, 2),
                "yaw": round(yaw, 2),
                "face_box": [round(x, 1), round(y, 1), round(box_w, 1), round(box_h, 1)],
                "message": "检测完成",
            }
        )
        return result


def decode_image_bytes(payload: bytes) -> np.ndarray:
    if not payload:
        raise ValueError("图片为空")
    from io import BytesIO

    with Image.open(BytesIO(payload)) as image:
        rgb = ImageOps.exif_transpose(image).convert("RGB")
    array = np.asarray(rgb)
    if array.ndim != 3 or array.shape[2] != 3:
        raise ValueError("无法解析图片")
    return cv2.cvtColor(array, cv2.COLOR_RGB2BGR)
