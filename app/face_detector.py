"""YuNet 人脸检测。输入输出走 ONNX Runtime，以便使用 DirectML。"""

from __future__ import annotations

import numpy as np

STRIDES = (8, 16, 32)
INPUT_SIZE = 640


def prepare_input(image_bgr: np.ndarray) -> tuple[np.ndarray, float]:
    """把 BGR 图像等比放进 640x640，右下补零。检测框除以 scale 后回到原图。"""
    import cv2

    height, width = image_bgr.shape[:2]
    scale = min(INPUT_SIZE / float(width), INPUT_SIZE / float(height))
    resized_w = max(1, min(INPUT_SIZE, int(round(width * scale))))
    resized_h = max(1, min(INPUT_SIZE, int(round(height * scale))))
    resized = cv2.resize(image_bgr, (resized_w, resized_h), interpolation=cv2.INTER_LINEAR)
    canvas = np.zeros((INPUT_SIZE, INPUT_SIZE, 3), dtype=np.float32)
    canvas[:resized_h, :resized_w] = resized
    blob = np.transpose(canvas, (2, 0, 1))[None]
    return np.ascontiguousarray(blob), scale


def reshape_feature(tensor: np.ndarray, rows: int, cols: int, channels: int) -> np.ndarray:
    """把 cls/obj/bbox 统一成 (rows, cols, channels)。"""
    array = np.asarray(tensor, dtype=np.float32)
    if array.ndim >= 1 and array.shape[0] == 1:
        array = array[0]
    expected = rows * cols
    if array.ndim == 3 and array.shape == (channels, rows, cols):
        return np.transpose(array, (1, 2, 0))
    if array.ndim == 3 and array.shape == (rows, cols, channels):
        return array
    if array.ndim == 2 and array.shape == (expected, channels):
        return array.reshape(rows, cols, channels)
    if array.ndim == 2 and array.shape == (channels, expected):
        return np.transpose(array, (1, 0)).reshape(rows, cols, channels)
    if channels == 1 and array.shape == (rows, cols):
        return array[:, :, None]
    if channels == 1 and array.shape == (expected,):
        return array.reshape(rows, cols, 1)
    raise ValueError(
        f"无法解析特征图形状 {tuple(tensor.shape)}，期望 {rows}x{cols}x{channels}"
    )


def nms_xywh(boxes: np.ndarray, scores: np.ndarray, iou_threshold: float, top_k: int) -> list[int]:
    if len(boxes) == 0:
        return []
    order = scores.argsort()[::-1]
    keep: list[int] = []
    while order.size > 0 and len(keep) < top_k:
        current = int(order[0])
        keep.append(current)
        if order.size == 1:
            break
        rest = order[1:]
        keep_mask = _iou_one_to_many(boxes[current], boxes[rest]) <= iou_threshold
        order = rest[keep_mask]
    return keep


def _iou_one_to_many(box: np.ndarray, boxes: np.ndarray) -> np.ndarray:
    x1 = np.maximum(box[0], boxes[:, 0])
    y1 = np.maximum(box[1], boxes[:, 1])
    x2 = np.minimum(box[0] + box[2], boxes[:, 0] + boxes[:, 2])
    y2 = np.minimum(box[1] + box[3], boxes[:, 1] + boxes[:, 3])
    inter = np.maximum(0.0, x2 - x1) * np.maximum(0.0, y2 - y1)
    area = box[2] * box[3] + boxes[:, 2] * boxes[:, 3] - inter
    return inter / np.maximum(area, 1e-6)


def decode_yunet(
    outputs: dict[str, np.ndarray],
    pad_w: int,
    pad_h: int,
    score_threshold: float = 0.6,
    nms_threshold: float = 0.3,
    top_k: int = 5000,
) -> np.ndarray:
    """返回 N 行 [x, y, w, h, score]，坐标在补边后的网络输入上。"""
    boxes: list[list[float]] = []
    scores: list[float] = []
    for stride in STRIDES:
        rows = pad_h // stride
        cols = pad_w // stride
        if rows == 0 or cols == 0:
            continue
        cls_map = reshape_feature(outputs[f"cls_{stride}"], rows, cols, 1)
        obj_map = reshape_feature(outputs[f"obj_{stride}"], rows, cols, 1)
        bbox_map = reshape_feature(outputs[f"bbox_{stride}"], rows, cols, 4)
        cls_score = np.clip(cls_map[:, :, 0], 0.0, 1.0)
        obj_score = np.clip(obj_map[:, :, 0], 0.0, 1.0)
        score_map = np.sqrt(cls_score * obj_score)
        ys, xs = np.where(score_map >= score_threshold)
        for row, col in zip(ys.tolist(), xs.tolist()):
            raw = bbox_map[row, col]
            cx = (col + float(raw[0])) * stride
            cy = (row + float(raw[1])) * stride
            width = float(np.exp(raw[2])) * stride
            height = float(np.exp(raw[3])) * stride
            boxes.append([cx - width / 2.0, cy - height / 2.0, width, height])
            scores.append(float(score_map[row, col]))

    if not boxes:
        return np.zeros((0, 5), dtype=np.float32)

    box_array = np.asarray(boxes, dtype=np.float32)
    score_array = np.asarray(scores, dtype=np.float32)
    keep = nms_xywh(box_array, score_array, nms_threshold, top_k)
    kept = np.concatenate([box_array[keep], score_array[keep, None]], axis=1)
    return kept.astype(np.float32)


class YuNetDetector:
    def __init__(self, session, score_threshold: float = 0.6, nms_threshold: float = 0.3):
        self.session = session
        self.score_threshold = score_threshold
        self.nms_threshold = nms_threshold
        self.input_name = session.get_inputs()[0].name
        self.output_names = [item.name for item in session.get_outputs()]

    def detect(self, image_bgr: np.ndarray) -> np.ndarray:
        """返回原图像素坐标下的 [x, y, w, h, score]，可能为空。"""
        blob, scale = prepare_input(image_bgr)
        pad_h, pad_w = int(blob.shape[2]), int(blob.shape[3])
        values = self.session.run(self.output_names, {self.input_name: blob})
        outputs = dict(zip(self.output_names, values))
        faces = decode_yunet(
            outputs,
            pad_w,
            pad_h,
            score_threshold=self.score_threshold,
            nms_threshold=self.nms_threshold,
        )
        if len(faces) == 0:
            return faces
        faces = faces.copy()
        faces[:, :4] /= np.float32(scale)
        image_h, image_w = image_bgr.shape[:2]
        faces[:, 0] = np.clip(faces[:, 0], 0, image_w - 1)
        faces[:, 1] = np.clip(faces[:, 1], 0, image_h - 1)
        faces[:, 2] = np.minimum(faces[:, 2], image_w - faces[:, 0])
        faces[:, 3] = np.minimum(faces[:, 3], image_h - faces[:, 1])
        return faces[faces[:, 2] * faces[:, 3] >= 4]
