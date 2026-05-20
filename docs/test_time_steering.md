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
