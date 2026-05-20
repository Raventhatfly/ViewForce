"""
Test-time force steering utilities for ViewForce.

This module intentionally does not depend on a particular base policy.  The base
policy can live in another repository and call these classes with:

  1. the latest RGB frame,
  2. the latest gripper mask from SAM/SAM2, and
  3. the action or action chunk proposed by the frozen base policy.

The current ViewForce predictor is image-conditioned only, so the implemented
steering is feedback/scaling rather than true action-gradient guidance.  A
future action-conditioned force model can reuse the same outer API.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, Optional, Sequence

import numpy as np
from PIL import Image
import torch

from src.model.unet import build_unet


ArrayLike = Sequence[float] | np.ndarray
OUTPUT_SIZE = (256, 256)


@dataclass
class ForcePrediction:
    """Force prediction returned by the ViewForce estimator."""

    values: Dict[str, float]
    selected_key: str
    selected_force: float
    control_force: float
    force_mode: str


@dataclass
class ForceSteeringConfig:
    """
    Configuration for action scaling from an estimated force signal.

    Args:
        desired_force: Target force used for steering.  If force_mode is
            "magnitude", this is a magnitude in N.
        force_key: Force channel used for control, e.g. "Fz".
        force_mode: "magnitude" uses abs(force); "signed" uses the raw value.
        deadband: No steering is applied inside this force error band.
        slowdown_band: Force above target is mapped linearly to action scaling.
        stop_margin: If force exceeds desired_force + stop_margin, closing is
            stopped and optional opening can be commanded.
        gripper_index: Action dimension controlling gripper open/close.  If
            None, only motion_indices are scaled.
        close_positive: True if positive gripper command closes the gripper.
            False if negative gripper command closes it.
        min_close_scale: Lower bound for scaling a closing gripper command before
            the stop/open regime.
        open_command: Optional gripper command used when force is far above the
            target.  The sign is inferred from close_positive.
        close_gain: Optional proportional boost when force is below target.
            Keep this at 0.0 for conservative safety-first behavior.
        max_close_command: Optional absolute limit for boosted close commands.
        motion_indices: Optional action dimensions to scale down when force is
            above target, e.g. approach axes that can increase contact pressure.
        min_motion_scale: Lower bound for motion scaling.
    """

    desired_force: float
    force_key: str = "Fz"
    force_mode: str = "magnitude"
    deadband: float = 0.10
    slowdown_band: float = 1.0
    stop_margin: float = 0.75
    gripper_index: Optional[int] = None
    close_positive: bool = True
    min_close_scale: float = 0.0
    open_command: float = 0.0
    close_gain: float = 0.0
    max_close_command: Optional[float] = None
    motion_indices: tuple[int, ...] = ()
    min_motion_scale: float = 0.25

    def __post_init__(self) -> None:
        if self.force_mode not in {"magnitude", "signed"}:
            raise ValueError("force_mode must be 'magnitude' or 'signed'")
        if self.desired_force < 0 and self.force_mode == "magnitude":
            raise ValueError("desired_force must be non-negative in magnitude mode")
        if self.slowdown_band <= 0:
            raise ValueError("slowdown_band must be > 0")
        if self.deadband < 0:
            raise ValueError("deadband must be >= 0")
        if not 0.0 <= self.min_close_scale <= 1.0:
            raise ValueError("min_close_scale must be in [0, 1]")
        if not 0.0 <= self.min_motion_scale <= 1.0:
            raise ValueError("min_motion_scale must be in [0, 1]")


@dataclass
class ForceSteeringResult:
    """Output of the steering step."""

    action: np.ndarray
    base_action: np.ndarray
    predicted_force: ForcePrediction
    force_error: float
    close_scale: float
    motion_scale: float
    metadata: Dict[str, Any]


def _as_rgb_array(image: Image.Image | np.ndarray) -> np.ndarray:
    if isinstance(image, Image.Image):
        arr = np.array(image.convert("RGB"))
    else:
        arr = np.asarray(image)
    if arr.ndim != 3 or arr.shape[2] != 3:
        raise ValueError("frame must be an RGB array with shape (H, W, 3)")
    if arr.dtype != np.uint8:
        arr = np.clip(arr, 0, 255).astype(np.uint8)
    return arr


def _as_bool_mask(mask: Image.Image | np.ndarray) -> np.ndarray:
    if isinstance(mask, Image.Image):
        arr = np.array(mask.convert("L"))
    else:
        arr = np.asarray(mask)
        if arr.ndim == 3:
            arr = arr[..., 0]
    if arr.ndim != 2:
        raise ValueError("mask must have shape (H, W)")
    return arr > 127 if arr.dtype == np.uint8 else arr.astype(bool)


def masked_frame_to_tensor(
    frame_rgb: Image.Image | np.ndarray,
    mask: Image.Image | np.ndarray,
    output_size: tuple[int, int] = OUTPUT_SIZE,
) -> torch.Tensor:
    """Apply a binary gripper mask and return a normalized CHW tensor."""

    frame_np = _as_rgb_array(frame_rgb)
    mask_np = _as_bool_mask(mask)
    if frame_np.shape[:2] != mask_np.shape:
        mask_img = Image.fromarray(mask_np.astype(np.uint8) * 255)
        mask_img = mask_img.resize((frame_np.shape[1], frame_np.shape[0]), Image.NEAREST)
        mask_np = np.array(mask_img) > 127

    masked = frame_np.astype(np.float32) * mask_np[:, :, None]
    masked_u8 = np.clip(masked, 0, 255).astype(np.uint8)
    resized = Image.fromarray(masked_u8).resize(
        (int(output_size[1]), int(output_size[0])),
        Image.BILINEAR,
    )
    frame_t = torch.from_numpy(np.array(resized, dtype=np.float32) / 255.0)
    return frame_t.permute(2, 0, 1).float()


class ViewForceEstimator:
    """Loads a ViewForce checkpoint and predicts force from frame + SAM mask."""

    def __init__(
        self,
        checkpoint: str | Path,
        device: str | torch.device | None = None,
        force_keys: Optional[Iterable[str]] = None,
        force_dropout: float = 0.0,
    ) -> None:
        self.checkpoint_path = Path(checkpoint)
        self.device = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))

        ckpt = torch.load(self.checkpoint_path, map_location=self.device)
        self.force_keys = list(force_keys or ckpt.get("force_keys", ["Fz"]))

        self.model = build_unet(
            in_channels=3,
            encoder_channels=(32, 64, 128, 256),
            force_dim=len(self.force_keys),
            force_hidden_dim=256,
            force_dropout=force_dropout,
            encoder_only=True,
        ).to(self.device)
        self.model.load_state_dict(ckpt["model"])
        self.model.eval()

    def predict(
        self,
        frame_rgb: Image.Image | np.ndarray,
        mask: Image.Image | np.ndarray,
        force_key: str = "Fz",
        force_mode: str = "magnitude",
    ) -> ForcePrediction:
        if force_key not in self.force_keys:
            raise ValueError(f"force_key {force_key!r} not in checkpoint keys {self.force_keys}")
        if force_mode not in {"magnitude", "signed"}:
            raise ValueError("force_mode must be 'magnitude' or 'signed'")

        frame_t = masked_frame_to_tensor(frame_rgb, mask).unsqueeze(0).to(self.device)
        with torch.no_grad():
            pred = self.model(frame_t).detach().cpu().numpy()[0]

        values = {k: float(v) for k, v in zip(self.force_keys, pred)}
        selected = values[force_key]
        control = abs(selected) if force_mode == "magnitude" else selected
        return ForcePrediction(
            values=values,
            selected_key=force_key,
            selected_force=selected,
            control_force=control,
            force_mode=force_mode,
        )

    def predict_from_files(
        self,
        frame_path: str | Path,
        mask_path: str | Path,
        force_key: str = "Fz",
        force_mode: str = "magnitude",
    ) -> ForcePrediction:
        frame = Image.open(frame_path)
        mask = Image.open(mask_path)
        return self.predict(frame, mask, force_key=force_key, force_mode=force_mode)


class ForceActionSteerer:
    """
    Conservative test-time steering for a frozen base policy action.

    The base policy is treated as the task prior.  This class only modifies
    configured action dimensions according to the current estimated force.
    """

    def __init__(self, config: ForceSteeringConfig) -> None:
        self.config = config

    def steer(
        self,
        base_action: ArrayLike,
        predicted_force: ForcePrediction,
    ) -> ForceSteeringResult:
        cfg = self.config
        action = np.array(base_action, dtype=np.float32, copy=True)
        base = action.copy()

        force_error = float(cfg.desired_force - predicted_force.control_force)
        over = max(0.0, -force_error - cfg.deadband)
        under = max(0.0, force_error - cfg.deadband)

        close_scale = 1.0
        motion_scale = 1.0
        stopped_or_opened = False

        if over > 0:
            close_scale = max(cfg.min_close_scale, 1.0 - over / cfg.slowdown_band)
            motion_scale = max(cfg.min_motion_scale, 1.0 - over / cfg.slowdown_band)

        if cfg.gripper_index is not None:
            gi = cfg.gripper_index
            if gi < -action.size or gi >= action.size:
                raise IndexError(f"gripper_index {gi} out of bounds for action size {action.size}")

            gripper_cmd = float(action[gi])
            closing_sign = 1.0 if cfg.close_positive else -1.0
            closing_amount = max(0.0, closing_sign * gripper_cmd)

            if over >= cfg.stop_margin:
                if cfg.open_command > 0:
                    action[gi] = -closing_sign * abs(cfg.open_command)
                elif closing_amount > 0:
                    action[gi] = 0.0
                stopped_or_opened = True
            elif closing_amount > 0:
                action[gi] = closing_sign * closing_amount * close_scale

            if under > 0 and cfg.close_gain > 0:
                boosted = float(action[gi]) + closing_sign * cfg.close_gain * under
                if cfg.max_close_command is not None:
                    max_cmd = abs(cfg.max_close_command)
                    boosted = float(np.clip(boosted, -max_cmd, max_cmd))
                action[gi] = boosted

        for idx in cfg.motion_indices:
            if idx < -action.size or idx >= action.size:
                raise IndexError(f"motion index {idx} out of bounds for action size {action.size}")
            if over > 0:
                action[idx] = action[idx] * motion_scale

        return ForceSteeringResult(
            action=action,
            base_action=base,
            predicted_force=predicted_force,
            force_error=force_error,
            close_scale=close_scale,
            motion_scale=motion_scale,
            metadata={
                "desired_force": cfg.desired_force,
                "deadband": cfg.deadband,
                "over_target": over,
                "under_target": under,
                "stopped_or_opened": stopped_or_opened,
                "gripper_index": cfg.gripper_index,
                "motion_indices": list(cfg.motion_indices),
            },
        )

    def steer_chunk(
        self,
        base_actions: np.ndarray,
        predicted_force: ForcePrediction,
    ) -> ForceSteeringResult:
        """
        Apply the same force feedback to every action in an action chunk.

        base_actions must be shaped (T, action_dim).  The return action keeps the
        same shape; metadata and force values are shared across the chunk.
        """

        arr = np.asarray(base_actions, dtype=np.float32)
        if arr.ndim != 2:
            raise ValueError("base_actions must have shape (T, action_dim)")

        steered = []
        last_result: Optional[ForceSteeringResult] = None
        for row in arr:
            last_result = self.steer(row, predicted_force)
            steered.append(last_result.action)

        assert last_result is not None
        return ForceSteeringResult(
            action=np.stack(steered, axis=0),
            base_action=arr.copy(),
            predicted_force=predicted_force,
            force_error=last_result.force_error,
            close_scale=last_result.close_scale,
            motion_scale=last_result.motion_scale,
            metadata={**last_result.metadata, "chunk_len": int(arr.shape[0])},
        )


class ViewForceSteeringPipeline:
    """Convenience wrapper combining force prediction and action steering."""

    def __init__(
        self,
        checkpoint: str | Path,
        steering_config: ForceSteeringConfig,
        device: str | torch.device | None = None,
    ) -> None:
        self.estimator = ViewForceEstimator(checkpoint, device=device)
        self.steerer = ForceActionSteerer(steering_config)
        self.config = steering_config

    def steer_action(
        self,
        frame_rgb: Image.Image | np.ndarray,
        mask: Image.Image | np.ndarray,
        base_action: ArrayLike,
    ) -> ForceSteeringResult:
        pred = self.estimator.predict(
            frame_rgb,
            mask,
            force_key=self.config.force_key,
            force_mode=self.config.force_mode,
        )
        return self.steerer.steer(base_action, pred)

    def steer_action_chunk(
        self,
        frame_rgb: Image.Image | np.ndarray,
        mask: Image.Image | np.ndarray,
        base_actions: np.ndarray,
    ) -> ForceSteeringResult:
        pred = self.estimator.predict(
            frame_rgb,
            mask,
            force_key=self.config.force_key,
            force_mode=self.config.force_mode,
        )
        return self.steerer.steer_chunk(base_actions, pred)
