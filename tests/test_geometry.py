import unittest

import numpy as np

from app.face_detector import decode_yunet, nms_xywh
from app.geometry import (
    classify_head_pose,
    expand_xywh,
    parse_turn_range,
    rotation_matrix_to_euler,
)


class GeometryTest(unittest.TestCase):
    def test_identity_rotation_is_zero(self):
        angles = rotation_matrix_to_euler(np.eye(3))[0]
        self.assertTrue(np.allclose(angles, [0, 0, 0], atol=1e-6))

    def test_pitch_roundtrip(self):
        pitch = np.deg2rad(-20.0)
        rotation = np.array(
            [
                [1, 0, 0],
                [0, np.cos(pitch), -np.sin(pitch)],
                [0, np.sin(pitch), np.cos(pitch)],
            ]
        )
        got = rotation_matrix_to_euler(rotation)[0, 0]
        self.assertAlmostEqual(got, -20.0, places=5)

    def test_lower_head_and_turn_are_independent(self):
        self.assertEqual(classify_head_pose(-20, 0, -15, -30, 30), (True, False))
        self.assertEqual(classify_head_pose(-10, 40, -15, -30, 30), (False, True))
        self.assertEqual(classify_head_pose(-20, -40, -15, -30, 30), (True, True))
        self.assertEqual(classify_head_pose(0, 0, -15, -30, 30), (False, False))

    def test_parse_turn_range(self):
        self.assertEqual(parse_turn_range("-30,30"), (-30.0, 30.0))
        with self.assertRaises(ValueError):
            parse_turn_range("30,-30")

    def test_expand_box_clips_to_image(self):
        self.assertEqual(expand_xywh(10, 10, 100, 100, 0.2, 200, 200), (0, 0, 130, 130))

    def test_nms_suppresses_overlap(self):
        boxes = np.array([[0, 0, 10, 10], [1, 1, 10, 10], [40, 40, 10, 10]], dtype=np.float32)
        scores = np.array([0.9, 0.8, 0.7], dtype=np.float32)
        self.assertEqual(nms_xywh(boxes, scores, 0.3, 10), [0, 2])

    def test_decode_yunet_one_cell(self):
        outputs = {}
        for stride, rows in ((8, 4), (16, 2), (32, 1)):
            outputs[f"cls_{stride}"] = np.zeros((1, 1, rows, rows), dtype=np.float32)
            outputs[f"obj_{stride}"] = np.zeros((1, 1, rows, rows), dtype=np.float32)
            outputs[f"bbox_{stride}"] = np.zeros((1, 4, rows, rows), dtype=np.float32)
        outputs["cls_32"][0, 0, 0, 0] = 1
        outputs["obj_32"][0, 0, 0, 0] = 1
        faces = decode_yunet(outputs, pad_w=32, pad_h=32, score_threshold=0.6)
        self.assertEqual(len(faces), 1)
        self.assertAlmostEqual(float(faces[0, 0]), -16.0, places=4)
        self.assertAlmostEqual(float(faces[0, 2]), 32.0, places=4)
        self.assertAlmostEqual(float(faces[0, 4]), 1.0, places=4)


if __name__ == "__main__":
    unittest.main()
