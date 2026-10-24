# Third-party notices

AuRA's own code is released under the Apache License 2.0 ([LICENSE](LICENSE)). This page lists the third-party components the v1 pipeline uses, obtains or depends on, and the AuRA files that adapt code from other projects.

## Components

The third-party components below are not covered by AuRA's Apache-2.0 license:

| Component | Use in AuRA | License |
|---|---|---|
| [AnyGrasp SDK](https://github.com/graspnet/anygrasp_sdk) | not included; obtain the SDK, a machine-locked license and the checkpoint from its authors ([docs/anygrasp_license.md](docs/anygrasp_license.md)) | AnyGrasp's own terms |
| [Grounded-SAM-2](https://github.com/IDEA-Research/Grounded-SAM-2), [SAM 2](https://github.com/facebookresearch/sam2), [Grounding DINO](https://github.com/IDEA-Research/GroundingDINO) | not included; cloned and downloaded during installation | Apache-2.0; SAM 2 also contains BSD-3-Clause code (cc_torch); see each repository |
| [PyRoki](https://github.com/chungmin99/pyroki) | installed from its git repository | MIT |
| [PhysTwin](https://github.com/Jianghanxiao/PhysTwin) | not included; v1 writes cases in its data format | MIT |
| `fr3` ([third_party/fr3](third_party/fr3/README.md)) | vendored FR3 control library used by the robot scripts | to be confirmed by the authors; `JointMotionGenerator` in `third_party/fr3/src/fr3/_motion.py` is derived from [libfranka](https://github.com/frankarobotics/libfranka)'s `pylibfranka/examples/example_common.py` (Apache-2.0; see [LICENSE-libfranka](third_party/fr3/LICENSE-libfranka) and [NOTICE-libfranka](third_party/fr3/NOTICE-libfranka)) |

## Code adapted from third-party projects

`aura/perception/aura_segment_static_target_image.py` keeps the command-line interface and RGBA mask format of PhysTwin's `data_process/segment_util_image.py` (MIT), and its model settings follow Grounded-SAM-2's image demo `grounded_sam2_local_demo.py` (Apache-2.0). `aura/grasping/anygrasp_poses_from_pointcloud.py` follows the AnyGrasp SDK's `grasp_detection/demo.py` for its model options and debug view. The IK solve (`solve_jax`) in `aura/grasping/filter_grasps_table_collision_pyroki.py` follows PyRoki's `examples/pyroki_snippets/_solve_ik.py` (MIT). `aura/robot/gamepad_record_franka_3oak.py` adapts code from `fr3`. Its gamepad input filter (adaptive EMA, zero snap, input acceleration limit, velocity magnitude clamp) follows `Gamepad.update` in `third_party/fr3/src/fr3/_gamepad.py`, and its 1 kHz jerk-limited velocity smoother and pose integration follow `ControlThread._velocity_loop` in `third_party/fr3/src/fr3/_control.py`. Its stick mapping, deadzone, speed limits and gripper settings follow fr3's gamepad example `examples/07_gamepad_control.py`, which is not vendored. These parts fall under the `fr3` license, which is still to be confirmed (see the table above).

## Other dependencies

Other dependencies come under their own licenses. Most of them, such as pylibfranka, DepthAI, Open3D, PyTorch and JAX, are installed from PyPI, conda or git. AnyGrasp's MinkowskiEngine fork (MIT) is cloned and built during installation, and `bert-base-uncased` and `franka_description` (both Apache-2.0) are downloaded on first use.
