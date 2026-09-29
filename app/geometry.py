"""姿态角换算、阈值判定，以及人脸框外扩。"""

from __future__ import annotations

import numpy as np


def rotation_matrix_to_euler(rotation_matrix: np.ndarray) -> np.ndarray:
    """把 (N, 3, 3) 旋转矩阵换成 pitch、yaw、roll，单位是度。

    这套旋转里，pitch 为正时鼻尖朝画面上方（抬头），为负时朝下方（低头）。
    页面上的低头阈值直接和这个 pitch 比较。
    """
    rotation = np.asarray(rotation_matrix, dtype=np.float64)
    if rotation.ndim == 2:
        rotation = rotation.reshape(1, 3, 3)

    sy = np.sqrt(rotation[:, 0, 0] ** 2 + rotation[:, 1, 0] ** 2)
    singular = sy < 1e-6
    pitch = np.where(
        singular,
        np.arctan2(-rotation[:, 1, 2], rotation[:, 1, 1]),
        np.arctan2(rotation[:, 2, 1], rotation[:, 2, 2]),
    )
    yaw = np.arctan2(-rotation[:, 2, 0], sy)
    roll = np.where(
        singular,
        np.zeros_like(sy),
        np.arctan2(rotation[:, 1, 0], rotation[:, 0, 0]),
    )
    return np.degrees(np.stack([pitch, yaw, roll], axis=1))


def classify_head_pose(
    pitch: float,
    yaw: float,
    lower_head_pc: float,
    turn_min: float,
    turn_max: float,
) -> tuple[bool, bool]:
    """低头：pitch 小于阈值。转头：yaw 落在区间外。两者互不影响。"""
    has_pitch = pitch < lower_head_pc
    has_yaw = yaw < turn_min or yaw > turn_max
    return has_pitch, has_yaw


def parse_turn_range(text: str) -> tuple[float, float]:
    parts = [part.strip() for part in str(text).split(",")]
    if len(parts) != 2 or parts[0] == "" or parts[1] == "":
        raise ValueError("转头区间需要两个数，例如 -30,30")
    low = float(parts[0])
    high = float(parts[1])
    if low >= high:
        raise ValueError("转头区间左边必须小于右边")
    return low, high


def expand_xywh(
    x: float,
    y: float,
    width: float,
    height: float,
    factor: float,
    image_width: int,
    image_height: int,
) -> tuple[int, int, int, int]:
    """每边向外扩 factor 倍宽高，并裁到图像内。返回 x1, y1, x2, y2。"""
    x1 = max(0, int(np.floor(x - width * factor)))
    y1 = max(0, int(np.floor(y - height * factor)))
    x2 = min(image_width, int(np.ceil(x + width + width * factor)))
    y2 = min(image_height, int(np.ceil(y + height + height * factor)))
    return x1, y1, x2, y2
