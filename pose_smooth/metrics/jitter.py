"""Jitter and missingness metrics for pose sequences."""

from __future__ import annotations

from typing import Dict, List, Optional
import numpy as np

from pose_smooth.types import FrameRecord


def jitter(frames: List[FrameRecord], min_score: float) -> float:

    if len(frames) < 2:
        return 0.0

    per_frame_means = []

    for f1, f2 in zip(frames[:-1], frames[1:]):

        k1 = np.array(f1["keypoints"], dtype=float)
        k2 = np.array(f2["keypoints"], dtype=float)

        xy1 = k1[:, :2]
        xy2 = k2[:, :2]

        s1 = k1[:, 2]
        s2 = k2[:, 2]

        valid = (s1 >= min_score) & (s2 >= min_score)

        if not np.any(valid):
            continue

        diff = xy2 - xy1
        dist = np.sqrt((diff ** 2).sum(axis=1))

        per_frame_means.append(dist[valid].mean())

    if not per_frame_means:
        return 0.0

    return float(np.mean(per_frame_means))


def missing_rate(frames: List[FrameRecord], min_score: float) -> float:

    if not frames:
        return 0.0

    total = 0
    missing = 0

    for f in frames:

        k = np.array(f["keypoints"], dtype=float)
        scores = k[:, 2]

        total += len(scores)
        missing += np.sum(scores < min_score)

    return float(missing / total) if total > 0 else 0.0


def compute_metrics(
    raw_frames: List[FrameRecord],
    smoothed_frames: Optional[List[FrameRecord]],
    min_score: float,
) -> Dict[str, Optional[float]]:

    jitter_before = jitter(raw_frames, min_score)
    missing = missing_rate(raw_frames, min_score)

    jitter_after = None
    if smoothed_frames is not None:
        jitter_after = jitter(smoothed_frames, min_score)

    return {
        "jitter_before": jitter_before,
        "jitter_after": jitter_after,
        "missing_rate": missing,
    }