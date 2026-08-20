# Inference

ViewForce adds force feedback to a frozen robot policy at inference time. The
current robot integration samples candidate action chunks in `policy_server.py`
and selects among them using the force output learned jointly by the current
berry policy. A standalone API is also available for conservatively scaling one
action.

Run robot-policy launchers from `third_party/forcelens_dp`. Root ViewForce
utilities resolve the repository root themselves and can be run from any
directory.

## Checkpoint roles

| Role | Variable or argument | Format | Purpose |
| --- | --- | --- | --- |
| Diffusion policy | `CKPT_PATH`, `--ckpt-path` | `.ckpt` | Produces candidate robot actions |
| Force critic | `CRITIC_CKPT`, `--tts-delta-force-critic-ckpt` | `.pt` | Predicts future force for an action or action chunk |
| ViewForce estimator | `--tts-viewforce-ckpt` | `.pt` | Estimates current force from the wrist image |
| SAM2 segmenter | `--tts-sam2-ckpt` | `.pt` | Produces the gripper mask used by ViewForce |

Do not mix an absolute-action checkpoint with a relative-action launcher, or a
relative-gripper checkpoint with a launcher that assumes absolute gripper
commands.

## Robot workflows

| Workflow | Policy representation | Force model | Launcher |
| --- | --- | --- | --- |
| Force-output TTS (recommended) | Absolute position, gripper, and force | Policy force output | `scripts/berry tts` |
| Absolute baseline | Absolute position and gripper | None | `scripts/berry baseline` |

The public interface intentionally exposes only the current force-output policy
and its matching baseline. Historical proxy, critic, fractional-data, and
relative-action experiments remain available through Git history, not as live
launch commands.

## Gentle berry grasp controller

`scripts/berry tts` and `scripts/berry baseline` deliberately load the same
frozen checkpoint and retain candidate 0's arm trajectory. The TTS condition
adds force-aware gripper intervention; the baseline only monitors ViewForce and
does not modify actions. This makes the comparison an inference-time ablation,
not a comparison between differently trained policies.

Do not retrain the policy between the initial TTS/baseline comparison. After
the force-feedback intervention is validated, a production policy should train
its action loss on successful demonstrations and reserve hard/failure episodes
for force or risk supervision; that is a separate experiment.

The TTS launcher now uses the following conservative controller:

1. Median-filter the absolute ViewForce estimate over three frames.
2. Accumulate a direct closing setpoint by `0.05` while force is below the
   target band. This intentionally does not let a DP open proposal suppress
   feedback closure; that policy gating caused the observed no-grasp runs.
3. Hold measured closure inside a `+/-0.5 N` band around the target and open by
   `0.05` above the band. Hysteresis prevents frame-to-frame close/open chatter.
4. The command may lead measured closure by at most `0.25`, enough to clear the
   observed actuator deadband without racing directly to the hard cap.
5. Immediately release if force reaches the target plus the `2.0 N` stop
   margin, or rises at least `20 N/s` near contact.
6. Keep an independent signed-closure cap of `0.82`, and add explicit hold/open
   candidates so candidate sampling always has a non-closing alternative.
7. Routine approach does not preempt queued DP arm actions. Entering the hold
   band or normal release preempts once; emergency releases preempt immediately.
8. Hold instead of closing when the gripper mask is invalid or SAM2 has fallen
   back to a rejected/stale propagation. A bounded policy-requested opening is
   still allowed because opening is the safe direction.

The defaults were selected from the current demonstrations: successful light
stage-2 episodes reached maximum gripper positions of `0.738--0.816`, whereas
hard failure episodes reached `0.963--0.993`. They are experimental bounds, not
hardware safety guarantees. Override them through the `TTS_*` variables listed
by `scripts/berry --help` only after reviewing replay and force-monitor logs.
The candidate and feedback target is the same absolute `8.0 N`, the hard
release margin is `2.0 N`, and the force-rate trip is `20 N/s`.

The force-aware trial is:

```bash
cd /home/wfy/repos/ViewForce/third_party/forcelens_dp
scripts/berry tts
```

The no-force comparison is:

```bash
scripts/berry baseline
```

Do not intentionally crush another berry merely to produce a baseline failure.
Use the existing failed baseline/TTS logs, or run the baseline on a
non-destructive surrogate with a physical stop. The scientific comparison must
use the same checkpoint, berry presentation, initial gripper pose, and arm
trajectory source.

Replay an existing force log before a robot trial:

```bash
scripts/berry replay \
  /home/wfy/repos/ViewForce/rollouts/berry_force_output_tts/<rollout>/force_log.csv
```

Replay is counterfactual controller evaluation only: recorded observations came
from the original executed commands, so replay cannot predict the berry's new
physical response.

For a real TTS acceptance trial, require an intact lift and verify that
`force_log.csv` shows bounded gripper increments, transition from
`close_below_band` to `hold_target_band`, no sustained release oscillation, and
no use of the `0.82` hard cap as the normal grasp strategy. If the cap is what
stops the gripper, treat the trial as a controller failure rather than a
force-aware success.

Run `scripts/berry --help` for arguments and environment overrides. Policy
training runs from `third_party/forcelens_dp` with `scripts/berry train`.

Policy runs are written under `third_party/forcelens_dp/outputs/`. Select a
compatible run without editing a launcher:

```bash
CKPT_PATH=outputs/<run>/checkpoints/latest.ckpt \
  scripts/berry baseline
```

## Standalone action scaling

`src.steering` provides a separate integration for scaling one action from a
policy outside this repository. Because the ViewForce estimator depends on the
image rather than the candidate action, it cannot provide an action gradient.
The standalone API instead slows or stops configured closing and approach
dimensions as estimated force rises.

```python
from src.steering import ForceSteeringConfig, ViewForceSteeringPipeline

config = ForceSteeringConfig(
    desired_force=2.0,
    force_key="Fz",
    force_mode="magnitude",
    gripper_index=6,
    close_positive=True,
    motion_indices=(2,),
)
pipeline = ViewForceSteeringPipeline("checkpoints/run1/best.pt", config)
result = pipeline.steer_action(frame_rgb, gripper_mask, base_policy(obs))
action_to_execute = result.action
```

The command-line equivalent is:

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

## SAM2 and rollout diagnostics

Install the SAM2 checkpoints expected by the robot launchers from the ViewForce
repository root:

```bash
(cd third_party/sam2/checkpoints && ./download_ckpts.sh)
```

When steering is enabled, `policy_server.py` creates a timestamped directory
under `ROLLOUT_DIR`:

```text
tts_rollout_YYYYMMDD_HHMMSS/
  original_h264.mp4         # wrist-camera frames
  masked_original_h264.mp4  # masked ViewForce RGB inputs
  edge_h264.mp4             # ViewForce edge inputs
  side_view_h264.mp4        # optional side-camera frames
  force_log.csv             # force estimates and selected actions
  candidate_scores.csv      # optional per-candidate scores
```

The H.264 files require `ffmpeg`. If encoding succeeds, the hidden intermediate
MP4 files are removed; otherwise they remain as the raw recordings.
`candidate_scores.csv` is written only when candidate logging is enabled.
New berry logs also include raw/filtered absolute force, force rate, target-band
edges, last-safe and commanded closure setpoints, controller mode, observed
gripper position, queue-preemption state, and whether command-lead or position
limits were applied. Legacy baseline/delta columns remain present but empty for
compatibility with older analysis scripts.

Across tasks, `baseline` always means the task's newest compatible frozen DP
checkpoint with monitor-only ViewForce. `tts` means the same checkpoint with a
force intervention. The intervention is task-specific: berry uses gripper
feedback, flip reranks manipulation chunks, and plug TTS is intentionally not
exposed until arm-force insertion control is validated. This preserves the
experimental comparison without applying berry grasp logic to an incompatible
task.

`--tts-frame-key` must name an RGB observation stored in the policy checkpoint.
Automatic SAM2 masking initializes from an HSV prompt on the first frame and
then propagates the previous accepted mask. The server retains source-resolution
frames for ViewForce and SAM2 before resizing the policy input. It cannot recover
detail discarded by the robot controller or image transport.

The berry launchers use
`checkpoints/viewforce_fz_edge_only_valmul10_trim01_v1/best.pt` for current-force
estimation and `third_party/sam2/checkpoints/sam2.1_hiera_small.pt` for
segmentation. Keep the ViewForce checkpoint provenance when comparing force
critics because their pseudo labels depend on that estimator.
