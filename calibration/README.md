# `calibration/` — your rig's calibration results

This directory is **not** shipped with content: every rig needs its own
calibration. The commands in [`../docs/calibration.md`](../docs/calibration.md)
write the results of the tools in [`../aura/calibration/`](../aura/calibration/)
to `$AURA_CALIB_ROOT`, which is this directory by default (`config/aura.env`),
and the pipeline scripts read them from there. Keep the commands' output paths:
without them, `Extract_intrinsic.py` writes into its `--root` recording and
`Extract_extrinsic.py` next to its input image, where nothing reads them.

Expected layout after calibration (see [`../docs/calibration.md`](../docs/calibration.md)):

```
calibration/
├── intrinsic_results/
│   ├── 1280x720/oak_<MXID>.{npz,json}      one per camera, per capture resolution
│   └── 640x400/oak_<MXID>.{npz,json}
├── extrinsic_results/oak_<MXID>.{npz,json} camera-to-world (ChArUco board frame)
└── robot_calibration_results/
    ├── robot_calib_result.{npz,json}       hand-eye: T_world_from_base etc.
    └── fingertip_calibration_results/
        └── franka_fingertip_table_calibration_simple.json
```
