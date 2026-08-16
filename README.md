# ViewForce

## Data collection and robot teleoperation

The robot teleoperation code is maintained in the
[`robot-controller`](https://github.com/Raventhatfly/robot-controller) repository
and runs on the ARX NUC used for data collection:

- Host: `arx-nuc`
- IP address: `10.250.23.89`
- User: `ydu`
- Path: `/home/ydu/robot-controller`

Connect to the machine with:

```bash
ssh ydu@10.250.23.89
```

## Inference

The ARX robot is connected to `arx-nuc`; the policy server runs on the GPU
machine at `10.250.180.205`.

```bash
ssh wfy@10.250.180.205
cd /home/wfy/repos/ViewForce/third_party/forcelens_dp
bash scripts/tts_berry_staged_stage2_absolute_delta_critic.sh
```

## Training

Robot training is performed on a separate machine:

- IP address: `10.250.180.205`
- User: `wfy`

Connect to the training machine with:

```bash
ssh wfy@10.250.180.205
```

The training code is located in [`third_party/forcelens_dp`](third_party/forcelens_dp).
