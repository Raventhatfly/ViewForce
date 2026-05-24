# Force Estimation Notes

## 2026-05-22: Spatial Force Head

Motivation: real coke-grasp overlays show visible deformation in the orange
fin-ray stripes and green outer edge. The original force estimator used an
encoder-only UNet with global average pooling on the bottleneck, which can wash
out local stripe curvature and deformation cues.

Code change:

- `src/model/unet.py` now supports `force_pooling="avg"` and
  `force_pooling="spatial"`.
- `spatial` uses `AdaptiveAvgPool2d(force_spatial_size, force_spatial_size)`
  before flattening into the MLP force head.
- `scripts/train.py` defaults to `--force-pooling spatial
  --force-spatial-size 4`.
- `scripts/evaluate.py` and `src/steering.py` read pooling settings from the
  checkpoint args; old checkpoints fall back to `avg`.

Compatibility:

- `checkpoints/viewforce_fz_clean_v1/best.pt` loads with `avg` pooling.
- New checkpoints trained after this change should record `force_pooling` in
  `ckpt["args"]`.

Recommended next training command:

```bash
python scripts/train.py \
  --data-dir data/data_ball_260422 \
  --val-count 5 \
  --force-keys Fz \
  --epochs 300 \
  --batch-size 32 \
  --lr 1e-4 \
  --weight-decay 1e-4 \
  --force-pooling spatial \
  --force-spatial-size 4 \
  --output checkpoints/viewforce_fz_spatial_v1 \
  --wandb-project force_estimation
```

Observed issue with occlusion augmentation:

- `viewforce_fz_occaug_v1` should not be used for policy control.
- Real rollouts showed poor response to coke grasping; the predicted force did
  not change reliably when the object was grasped.
- The likely failure mode is that black occluders with unchanged force labels
  encouraged the model to ignore the contact/deformation region.

Current hypothesis:

- The useful visual signal is not the stripe texture by itself, but the degree
  of stripe and outer-edge deformation.
- Spatial pooling is a minimal architectural change to preserve coarse location
  information for that signal.
- A later version may add reference-frame difference or optical-flow inputs, but
  this change keeps the dataset and input format unchanged.

## 2026-05-22: Optional Edge Input

Motivation: the orange fin-ray stripes are useful only if the model reads their
deformation. A pure RGB input may let the model rely on coarse color/mask
statistics. The optional edge input adds a direct cue for stripe and local
texture boundaries.

Code change:

- `src/dataset.py` supports `input_mode="rgb"` and `input_mode="rgb_edge"`.
- `rgb_edge` appends a Sobel edge channel computed on the final 256x256 masked
  RGB input.
- The edge channel is multiplied by an eroded visible-mask interior to reduce
  the outer mask-boundary edge.
- `scripts/train.py` exposes `--input-mode`.
- `scripts/evaluate.py` and `src/steering.py` restore `input_mode` from
  checkpoint args; old checkpoints fall back to `rgb`.

Parallel experiment commands:

```bash
python scripts/train.py \
  --data-dir data/data_ball_260422 \
  --val-count 5 \
  --force-keys Fz \
  --epochs 300 \
  --batch-size 32 \
  --lr 1e-4 \
  --weight-decay 1e-4 \
  --input-mode rgb \
  --force-pooling spatial \
  --force-spatial-size 4 \
  --output checkpoints/viewforce_fz_spatial_v1 \
  --wandb-project force_estimation
```

## 2026-05-23: Validation Split by Episode Number

The earlier `--val-count 5` split held out the last five sorted episodes. In
`data_ball_260422`, those later episodes contain many near-zero Fz frames, while
the remaining training episodes are dominated by roughly `-9N` contact labels.
This made spatial-head models learn a strong `~10N` force prior and fail on real
empty-gripper rollouts.

Use `--val-multiple 10` for new force-estimator experiments:

```text
validation: EP000010, EP000020, EP000030, EP000040, EP000050, EP000060
training:   all other episodes, including EP000055-059 and EP000061
```

This keeps low-force/near-zero examples in the training set while still holding
out episodes across the recording range.

```bash
python scripts/train.py \
  --data-dir data/data_ball_260422 \
  --val-count 5 \
  --force-keys Fz \
  --epochs 300 \
  --batch-size 32 \
  --lr 1e-4 \
  --weight-decay 1e-4 \
  --input-mode rgb_edge \
  --force-pooling spatial \
  --force-spatial-size 4 \
  --output checkpoints/viewforce_fz_spatial_edge_v1 \
  --wandb-project force_estimation
```
