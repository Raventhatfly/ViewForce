# Test-Time Force Steering

This module lets a frozen base policy use ViewForce as an inference-time force
feedback signal.  The base policy can live outside this repository.

## Idea

At each control step:

1. Run the frozen base policy to get an action or action chunk.
2. Use the SAM/SAM2 gripper mask and the current RGB frame as ViewForce input.
3. Predict the current contact force.
4. Scale selected action dimensions toward a desired force.

The base policy parameters are never updated.  The desired force is only used by
the steering module.

```python
from src.steering import ForceSteeringConfig, ViewForceSteeringPipeline

config = ForceSteeringConfig(
    desired_force=2.0,
    force_key="Fz",
    force_mode="magnitude",
    gripper_index=6,       # set this to your policy's gripper action dim
    close_positive=True,   # set False if negative command closes the gripper
    motion_indices=(2,),   # optional: scale approach/z motion when force is high
)

pipeline = ViewForceSteeringPipeline("checkpoints/run1/best.pt", config)

base_action = base_policy(obs)
result = pipeline.steer_action(frame_rgb, gripper_mask, base_action)
action_to_execute = result.action
```

## Current Technical Detail

The current ViewForce network is an image-conditioned estimator:

```text
F_hat = g(masked_rgb_frame)
```

Because candidate actions are not inputs to `g`, this estimator cannot provide a
meaningful action gradient:

```text
d g(masked_rgb_frame) / d action = 0
```

So this first implementation performs conservative test-time **steering/scaling**
on configured action dimensions:

- below target: optionally allow/proportionally boost closing,
- near target: leave the base action unchanged,
- above target: scale down closing and optional approach-motion dimensions,
- far above target: stop closing or command a small opening action.

This is the safe first version for the coke-bottle experiment.  It tests whether
vision-based force estimation is useful to steer a frozen policy at execution
time.

## Offline Demo

```bash
python scripts/steer_action.py \
  --checkpoint checkpoints/run1/best.pt \
  --frame frame.png \
  --mask mask.png \
  --action 0.01 0.0 0.0 0.20 \
  --desired-force 2.0 \
  --gripper-index 3 \
  --close-positive
```

The script prints JSON containing the base action, steered action, predicted
force, force error, and scaling factors.

## Real Robot Policy Server

Run from the Diffusion Policy server directory:

```bash
cd /home/wfy/repos/ViewForce/third_party/forcelens_dp
```

Current coke-bottle TTS command, using the ViewForce checkpoint and SAM2 mask
generation from the wrist camera:

```bash
python policy_server.py \
  --ckpt-path outputs/2026.05.19/15.13.23_train_diffusion_unet_pick_coke_hybrid_pick_coke_hybrid_image/checkpoints/latest.ckpt \
  --tts-viewforce-ckpt /home/wfy/repos/ViewForce/ckpts/run_7479304/best.pt \
  --tts-viewforce-root /home/wfy/repos/ViewForce \
  --tts-desired-force 2.0 \
  --tts-frame-key wrist_image \
  --tts-auto-mask \
  --tts-mask-mode sam2 \
  --tts-sam2-model small \
  --tts-sam2-ckpt /home/wfy/repos/ViewForce/third_party/sam2/checkpoints/sam2.1_hiera_small.pt \
  --tts-rollout-dir /home/wfy/repos/ViewForce/tts_rollouts \
  --tts-gripper-index 7 \
  --tts-close-positive
```

SAM2 is expected as a submodule at:

```text
/home/wfy/repos/ViewForce/third_party/sam2
```

The current small checkpoint path is:

```text
/home/wfy/repos/ViewForce/third_party/sam2/checkpoints/sam2.1_hiera_small.pt
```

If the checkpoint is missing, download it from the submodule checkpoint folder:

```bash
cd /home/wfy/repos/ViewForce/third_party/sam2/checkpoints
./download_ckpts.sh
```

### Rollout Debug Videos

When TTS is enabled, the server writes rollout diagnostics under
`--tts-rollout-dir`:

```text
tts_rollout_YYYYMMDD_HHMMSS/
  first_frame.png   # first frame received by policy_server after JPEG decode
  overlay.mp4       # RGB frame with generated mask overlaid
  mask.mp4          # binary mask video
  overlay_h264.mp4  # ffmpeg/libx264 copy, preferred for VS Code playback
  mask_h264.mp4     # ffmpeg/libx264 copy, preferred for VS Code playback
  frames/           # per-frame PNG overlays and masks
```

`first_frame.png` is the actual resolution received over ZMQ by
`policy_server.py`. If it is low resolution, the image was already downsampled
before reaching the server.

As of the 2026-05-20 test, the latest observed `first_frame.png` was `84x84`.
That means ViewForce/SAM2 received an `84x84` wrist image even though the
ViewForce training pipeline uses the original episode video frames before
resizing the masked input to `256x256`.

### Useful Notes

- `--tts-frame-key` must match one of the RGB keys in the DP checkpoint. For
  the coke hybrid image checkpoint, the keys were `base_image` and
  `wrist_image`.
- `--tts-auto-mask --tts-mask-mode sam2` follows the ViewForce-style mask chain:
  HSV prompt detection, then SAM2 mask generation.
- The old `--tts-mask-key gripper_mask` path only works if the robot controller
  already sends a `gripper_mask` key in obs.
- The original DP policy server resizes images to `320x240` before feeding the
  DP policy. The current TTS path keeps the decoded source-resolution image for
  ViewForce/SAM2, then resizes only for the DP policy input.
- If the controller sends only `84x84`, server-side changes cannot recover
  detail. The controller or remote-policy image transport must send a higher
  resolution wrist image for ViewForce to match training conditions.

## Toward TouchGuide-Style Gradient Steering

To match TouchGuide more directly, train a differentiable action-conditioned
force or feasibility model:

```text
F_future = h(obs, force_history, gripper_state, candidate_action_chunk)
score = -||F_future - F_desired||^2
```

Then the base diffusion policy can remain frozen while the sampled action chunk
is refined with:

```text
a <- a + eta * grad_a score
```

That requires `h` to take candidate actions as input.  The current ViewForce
checkpoint is still useful as the immediate feedback baseline and as supervision
for collecting/training the action-conditioned version.
