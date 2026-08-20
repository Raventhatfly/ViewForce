# ViewForce

ViewForce estimates contact force from masked camera frames and uses that
estimate to guide a frozen robot policy at inference time.

## Documentation

- [Training](docs/training.md): train the image-based force estimator.
- [Inference](docs/inference.md): steering, compatible checkpoints, policy
  profiles, robot launchers, and rollout diagnostics.
- [Force-estimation notes](docs/force_estimation_notes.md): dated experiment and
  architecture notes.

## Machines and data flow

The robot teleoperation code is maintained in the
[`robot-controller`](https://github.com/Raventhatfly/robot-controller) repository
and runs on the ARX NUC used for data collection. See that repository's README
for hardware setup, teleoperation, and collection commands.

- Host: `arx-nuc`
- IP address: `10.250.23.89`
- User: `ydu`
- Path: `/home/ydu/robot-controller`

Connect to the machine with:

```bash
ssh arx-nuc
```

### Copy collected data to the GPU machine

Initiate dataset transfers from the ViewForce/GPU machine and pull only the
collected episodes from `arx-nuc`; the GPU machine does not need another copy of
the `robot-controller` repository. The active collection root on the NUC is:

```text
/home/ydu/robot-controller/force_lens/data
```

Use the following template after the current episode has finished saving:

```bash
rsync -avh --partial --info=progress2 \
  arx-nuc:/home/ydu/robot-controller/force_lens/data/<remote-dataset>/ \
  /home/wfy/repos/ViewForce/third_party/forcelens_dp/data/<category>/<local-dataset>/
```

The command is incremental, can be rerun, and does not delete files on either
machine. A completed episode contains `data.pkl`; do not transfer an episode
while the NUC is still writing it.

Current dataset mappings are:

| NUC collection | ViewForce dataset |
| --- | --- |
| `flip_obj_1` | `flip/flip_obj_1` |
| `flip_obj_2` | `flip/flip_obj_2` |
| `flip_obj_3` | `flip/flip_obj_3` |
| `flip_obj_4` | `flip/flip_obj_4` |
| `flip_obj_4_dynamics` | `flip/obj4_dynamics` |
| `berry_pick_staged` | `pick/berry_staged` |
| `berry_pick_staged_hard` | `pick/berry_staged_hard` |
| `berry_pick_staged_predict` | `pick/berry_staged_hard_predict` |
| `pivoting_cheng` | `pivot/pivoting_cheng` |

## Inference

The ARX robot is connected to `arx-nuc`; the policy server runs on the GPU
machine at `10.250.180.205`. The command below runs the newest absolute-action
berry policy, reranks candidates with its learned force output, and applies
stateful ViewForce feedback to the gripper:

```bash
ssh wfy@10.250.180.205
cd /home/wfy/repos/ViewForce/third_party/forcelens_dp
scripts/berry tts
```

The matching no-steering comparison uses the same checkpoint:

```bash
scripts/berry baseline
```

The flip task uses the same interface:

```bash
scripts/flip tts
scripts/flip baseline
```

Run either command with `--help` for options. See the
[inference guide](docs/inference.md) for checkpoint roles and rollout outputs.

## Training

Robot training is performed on a separate machine:

- IP address: `10.250.180.205`
- User: `wfy`

Connect to the training machine with:

```bash
ssh wfy@10.250.180.205
```

Current berry and flip policies are trained from `third_party/forcelens_dp`:

```bash
cd /home/wfy/repos/ViewForce/third_party/forcelens_dp
scripts/berry train
scripts/flip train
scripts/pivot train
```

ViewForce estimator training runs from the repository root. See the
[training guide](docs/training.md) for datasets, options, and checkpoints.
