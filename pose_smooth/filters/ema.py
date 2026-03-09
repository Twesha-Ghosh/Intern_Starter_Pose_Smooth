"""Score-weighted EMA smoothing with missing-joint handling."""

from __future__ import annotations

from typing import List
import numpy as np

from pose_smooth.config import SmoothConfig
from pose_smooth.types import FrameRecord


class PoseEMASmoother:
    """Temporal pose smoother with missing data and teleport gating."""

    def __init__(self, num_keypoints: int, config: SmoothConfig):

     
        config.validate()

        self.config = config
        self.num_keypoints = num_keypoints

        
        self.xy = np.zeros((num_keypoints, 2), dtype=float)

       
        self.score = np.zeros(num_keypoints, dtype=float)


        self.seen = np.zeros(num_keypoints, dtype=bool)

    def process_frame(self, frame: FrameRecord) -> FrameRecord:

        kpts = frame["keypoints"]

        V = self.num_keypoints
        alpha = self.config.alpha
        min_score = self.config.min_score
        max_jump = self.config.max_jump_px
        score_decay = self.config.score_decay
        teleport_trust = getattr(self.config, "teleport_trust_score", 1.0)

        used_observation = np.zeros(V, dtype=bool)
        teleport_rejected = np.zeros(V, dtype=bool)
        missing = np.zeros(V, dtype=bool)

        for j in range(V):

            x, y, s = kpts[j]


            if s < min_score:
                missing[j] = True

            else:

                if self.seen[j]:

                    dx = x - self.xy[j, 0]
                    dy = y - self.xy[j, 1]
                    dist = np.sqrt(dx * dx + dy * dy)

                    if dist > max_jump and s < teleport_trust:
                        teleport_rejected[j] = True
                        missing[j] = True

            if missing[j]:

                if self.seen[j]:
                    
                    self.score[j] *= score_decay

                continue

            
            used_observation[j] = True

            if not self.seen[j]:
                
                self.xy[j, 0] = x
                self.xy[j, 1] = y
                self.score[j] = s
                self.seen[j] = True

            else:
                
                self.xy[j, 0] = alpha * x + (1 - alpha) * self.xy[j, 0]
                self.xy[j, 1] = alpha * y + (1 - alpha) * self.xy[j, 1]
                self.score[j] = alpha * s + (1 - alpha) * self.score[j]

        
        out_kpts = [
            [float(self.xy[j, 0]), float(self.xy[j, 1]), float(self.score[j])]
            for j in range(V)
        ]

        out = {
            "frame_idx": frame["frame_idx"],
            "keypoints": out_kpts,
            "meta": {
                "used_observation": used_observation.tolist(),
                "teleport_rejected": teleport_rejected.tolist(),
                "missing": missing.tolist(),
            },
        }

        if "timestamp_s" in frame:
            out["timestamp_s"] = frame["timestamp_s"]

        return out


def smooth_frames(frames: List[FrameRecord], config: SmoothConfig) -> List[FrameRecord]:

    if not frames:
        return []

    num_keypoints = len(frames[0]["keypoints"])

    smoother = PoseEMASmoother(num_keypoints, config)

    out = []
    for f in frames:
        out.append(smoother.process_frame(f))

    return out


# Bug fix:
# An early implementation allowed large coordinate jumps to pass directly
# into the EMA update, producing unrealistic pose spikes. The fix was to
# reject observations when the displacement exceeds max_jump_px and the
# detection confidence is not extremely high. This prevents noisy detections
# from destabilizing the smoothed trajectory.

# Edge Case:
# Low confidence detections often occur when joints are occluded or briefly
# lost by the pose detector. Instead of snapping the joint to a new noisy
# observation, the smoother holds the previous position and decays the
# confidence score. This preserves temporal continuity until reliable
# detections resume.
