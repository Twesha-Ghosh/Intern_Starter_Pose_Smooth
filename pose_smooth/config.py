"""Configuration and validation helpers."""

from dataclasses import dataclass

DEFAULT_NUM_KEYPOINTS = 17
DEFAULT_ALPHA = 0.6
DEFAULT_MIN_SCORE = 0.3
DEFAULT_MAX_JUMP_PX = 35.0
DEFAULT_SCORE_DECAY = 0.95
DEFAULT_TELEPORT_TRUST_SCORE = 0.9


@dataclass(frozen=True)
class SmoothConfig:
    alpha: float = DEFAULT_ALPHA
    min_score: float = DEFAULT_MIN_SCORE
    max_jump_px: float = DEFAULT_MAX_JUMP_PX
    score_decay: float = DEFAULT_SCORE_DECAY
    teleport_trust_score: float = DEFAULT_TELEPORT_TRUST_SCORE

    def validate(self) -> "SmoothConfig":
        if not (0 < self.alpha <= 1):
            raise ValueError("alpha must be in (0, 1].")

        if not (0 <= self.min_score <= 1):
            raise ValueError("min_score must be in [0, 1].")

        if self.max_jump_px < 0:
            raise ValueError("max_jump_px must be >= 0.")

        if not (0 <= self.score_decay <= 1):
            raise ValueError("score_decay must be in [0, 1].")

        if not (0 <= self.teleport_trust_score <= 1):
            raise ValueError("teleport_trust_score must be in [0, 1].")

        return self

# Comments:
# During development an early bug occurred when invalid parameter ranges (alpha > 1 or negative max_jump_px) 
# silently produced unstable smoothing behavior. Adding validation here ensures incorrect hyperparameters
# fail early instead of causing subtle runtime errors later.