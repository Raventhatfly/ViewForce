# ViewForce agent harness

## Repository role

This is the GPU-side repository for ViewForce model development, dataset
processing, diffusion-policy training, policy serving, and evaluation. Robot
control and raw demonstration collection live in `robot-controller` on
`arx-nuc`; do not duplicate that repository here.

Read `README.md` first, then use `docs/training.md` for the force estimator and
`third_party/forcelens_dp/README.md` for robot policies. The working tree may
contain active experiments, so inspect `git status --short` before editing and
preserve unrelated changes.

## Machine boundary

- GPU machine: this checkout at `/home/wfy/repos/ViewForce` on
  `10.250.180.205`.
- ARX NUC: SSH alias `arx-nuc`, checkout `/home/ydu/robot-controller`, raw data
  root `/home/ydu/robot-controller/force_lens/data`.
- Pull datasets from the GPU machine with incremental `rsync`. Never move or
  delete the NUC copy, never use `--delete`, and only copy completed stages with
  `data.pkl`.
- The robot is hardware. Do not launch robot motion, inference, homing, or data
  collection without an explicit user request. Read-only status checks are
  safe; keep autonomous launch commands visible to the operator.

## Diffusion-policy workflow

Run policy commands from `third_party/forcelens_dp` with the `robodiff`
environment (`/home/wfy/miniforge3/envs/robodiff/bin/python`). Raw datasets live
under `data/<task>/`; builders create training views under a separate path,
usually by symlinking selected episodes. Generated data, outputs, logs, and
checkpoints are intentionally ignored by Git.

For a new collection:

1. Read both repositories' READMEs and identify the task/stage semantics.
2. Inspect the remote directory and confirm every stage has `data.pkl` plus the
   expected videos before transfer.
3. Add an explicit source-to-destination mapping to `README.md`.
4. Pull with the documented `rsync -avh --partial --info=progress2` command.
5. Validate pickle/video lengths and exclude empty or malformed stages.
6. Add or reuse a deterministic dataset builder with a manifest; do not train
   directly from an ambiguous raw collection.
7. Dry-run the task launcher with `PYTHON_BIN=/bin/echo`, inspect available GPUs,
   then start training with a persistent log and verify that epochs advance.

Pivoting is the reference for this flow:

```bash
cd third_party/forcelens_dp
scripts/pivot build
scripts/pivot train
```

The raw collection is `data/pivot/pivoting_cheng`; the success-only training
view is `data/pivot/pivoting_cheng_success`.

## Verification

- Shell launchers: `bash -n scripts/<task>` and a `/bin/echo` dry run.
- Python: run targeted tests with the `robodiff` interpreter, then compile new
  helpers with `python -m py_compile`.
- Training: record the PID, device, output directory, and log path; confirm the
  process survives startup and reports a first epoch or checkpoint.
- Documentation-only controller changes still require `git diff --check` on
  the NUC. Do not commit or push unless explicitly requested.
