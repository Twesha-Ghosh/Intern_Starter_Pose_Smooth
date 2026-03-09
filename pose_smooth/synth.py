"""Synthetic pose dataset generation for demo/testing."""

from __future__ import annotations

import math
import random
from typing import List

from pose_smooth.types import FrameRecord


def generate_synthetic_frames(
    frames: int = 240,
    num_keypoints: int = 17,
    jitter_std: float = 8.0,
    dropout_prob: float = 0.08,
    teleport_prob: float = 0.03,
    seed: int = 7,
) -> List[FrameRecord]:

    random.seed(seed)

    data: List[FrameRecord] = []

    
    offsets = [
        (
            random.uniform(-40, 40),
            random.uniform(-40, 40),
        )
        for _ in range(num_keypoints)
    ]

    for i in range(frames):

        keypoints = []

       
        cx = 320 + 60 * math.sin(i / 20)
        cy = 240 + 40 * math.cos(i / 25)

        for j in range(num_keypoints):

            ox, oy = offsets[j]

            x = cx + ox
            y = cy + oy

            
            x += random.gauss(0, jitter_std)
            y += random.gauss(0, jitter_std)

            score = random.uniform(0.8, 1.0)

            
            if random.random() < dropout_prob:
                x = float("nan")
                y = float("nan")
                score = 0.0

            
            elif random.random() < teleport_prob:
                x += random.uniform(-200, 200)
                y += random.uniform(-200, 200)

            keypoints.append([x, y, score])

        frame: FrameRecord = {
            "frame_idx": i,
            "timestamp_s": i / 30.0,
            "keypoints": keypoints,
        }

        data.append(frame)

    return data