"""JSONL input/output and frame sanitization."""

from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Iterable, List

from pose_smooth.config import DEFAULT_NUM_KEYPOINTS
from pose_smooth.types import FrameRecord


def read_jsonl(path: str | Path, num_keypoints: int = DEFAULT_NUM_KEYPOINTS) -> List[FrameRecord]:

    path = Path(path)

    frames: List[FrameRecord] = []

    
    last_valid_xy = [[0.0, 0.0] for _ in range(num_keypoints)]

    with path.open("r", encoding="utf-8") as f:

        for line_num, line in enumerate(f, start=1):

            try:
                obj = json.loads(line)
            except Exception as e:
                raise ValueError(f"Invalid JSON at line {line_num}") from e

            
            if "frame_idx" not in obj:
                raise ValueError(f"Missing frame_idx at line {line_num}")

            if "keypoints" not in obj:
                raise ValueError(f"Missing keypoints at line {line_num}")

            kpts = obj["keypoints"]

            if not isinstance(kpts, list) or len(kpts) != num_keypoints:
                raise ValueError(f"Invalid keypoint count at line {line_num}")

            clean_kpts = []

            for j, kp in enumerate(kpts):

                if not isinstance(kp, list) or len(kp) != 3:
                    raise ValueError(f"Invalid keypoint format at line {line_num}")

                x, y, score = kp

                
                if not isinstance(score, (int, float)) or not math.isfinite(score):
                    score = 0.0

                score = max(0.0, min(1.0, float(score)))

                
                if not (isinstance(x, (int, float)) and math.isfinite(x) and
                        isinstance(y, (int, float)) and math.isfinite(y)):

                    
                    x, y = last_valid_xy[j]
                    score = 0.0

                else:
                    x = float(x)
                    y = float(y)

                    
                    last_valid_xy[j] = [x, y]

                clean_kpts.append([x, y, score])

            frame: FrameRecord = {
                "frame_idx": obj["frame_idx"],
                "keypoints": clean_kpts
            }

            if "timestamp_s" in obj:
                frame["timestamp_s"] = obj["timestamp_s"]

            frames.append(frame)

    return frames


def write_jsonl(path: str | Path, frames: Iterable[FrameRecord]) -> None:

    path = Path(path)

    with path.open("w", encoding="utf-8") as f:
        for frame in frames:
            line = json.dumps(frame, separators=(",", ":"))
            f.write(line + "\n")