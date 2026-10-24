#!/usr/bin/env python3
r"""Solve one camera's extrinsics from a single image of a static ChArUco board.

Step 2 of the calibration chain Extract_intrinsic.py -> Extract_extrinsic.py ->
robot_world_calib_charuco.py. Run it once per camera on the same bundle of an
aura/capture/record_3oak.py recording in which one static board placement is
visible to all cameras. The board is detected with cv2.aruco.CharucoDetector,
each ChArUco corner is paired with board.getChessboardCorners()[charuco_id] (see
detect_and_estimate_pose) and the pose is solved with cv2.SOLVEPNP_IPPE.

The board frame IS the AuRA world frame: T_world_from_cam = T_board_from_cam =
phys_twin_c2w. Origin at the outer corner of the first square, +X along the
--squares-x edge, +Y along the --squares-y edge, +Z = X x Y pointing into the
board, away from the cameras: with the board lying on the table, world z points
down and points above the table have negative z. Translations are in the unit of
--square-length (metres in AuRA).

Run from the repository root in the AURA_ENV conda environment with
config/aura.env loaded (set -a; source config/aura.env; set +a):

    python aura/calibration/Extract_extrinsic.py \
        --image-path "$AURA_DATA_ROOT/calibration_captures/<capture>/bundle_000100/oak_<MXID>/rgb.jpg" \
        --intrinsic-file "$AURA_CALIB_ROOT/intrinsic_results/1280x720/oak_<MXID>.npz" \
        --camera-name oak_<MXID> \
        --squares-x 4 --squares-y 5 --square-length 0.04 --marker-length 0.03 \
        --dictionary DICT_4X4_50 \
        --save-overlay "$AURA_CALIB_ROOT/debug/extrinsic/oak_<MXID>.jpg" \
        --output-json "$AURA_CALIB_ROOT/extrinsic_results/oak_<MXID>.json" \
        --output-npz "$AURA_CALIB_ROOT/extrinsic_results/oak_<MXID>.npz"

(--bundle-image "$AURA_DATA_ROOT/calibration_captures/<capture>" oak_<MXID> bundle_000100
selects the same image.) Both outputs hold K, dist, rvec/tvec_cam_from_board,
T_cam_from_board, T_board_from_cam, T_cam_from_world, T_world_from_cam,
phys_twin_c2w, image_size, num_markers, num_charuco and reproj_rms. Keep the
.npz and the .json together: different pipeline stages read each.

Notes:
  * The board is built as cv2.aruco.CharucoBoard((squares_x, squares_y), ...)
    with no legacy-pattern option: a board printed by OpenCV < 4.6 has a
    different layout when --squares-y is even (the 4 x 5 AuRA board is the same
    in both). Check marker ids and axes on the --save-overlay image.
  * Pass the intrinsics .npz. The Extract_intrinsic.py .json report has no
    image_size key and fails with KeyError: 'image_size'.
  * The image size must equal the intrinsic image_size.
  * Without --output-json / --output-npz the results are written next to the
    input image (<stem>_extrinsic.{json,npz}), where the pipeline does not look.
  * Do not add --zero-dist when the intrinsics were fitted on frames already
    undistorted on the camera (record_3oak.py --enable-undistortion, on by
    default): their dist is the residual distortion.
  * The JSON report records image_path exactly as given.
  * calibrate_3oak_anchor108.py (same directory) imports load_intrinsics() and
    draw_overlay() from this file; keep their signatures.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Optional, Tuple

import cv2
import numpy as np


DICT_MAP = {
    name: getattr(cv2.aruco, name)
    for name in dir(cv2.aruco)
    if name.startswith("DICT_")
}


def load_intrinsics(path: Path) -> Tuple[np.ndarray, np.ndarray, Tuple[int, int]]:
    if path.suffix.lower() == ".npz":
        data = np.load(path)
        K = np.asarray(data["K"], dtype=np.float64)
        dist = np.asarray(data["dist"], dtype=np.float64).reshape(-1)
        image_size = tuple(int(x) for x in np.asarray(data["image_size"]).reshape(-1))
        return K, dist, (image_size[0], image_size[1])

    if path.suffix.lower() == ".json":
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
        K = np.asarray(data["K"], dtype=np.float64)
        dist = np.asarray(data["dist"], dtype=np.float64).reshape(-1)
        image_size = tuple(int(x) for x in data["image_size"])
        return K, dist, (image_size[0], image_size[1])

    raise ValueError(f"Unsupported intrinsic file format: {path}")


def rt_to_T(rvec: np.ndarray, tvec: np.ndarray) -> np.ndarray:
    R, _ = cv2.Rodrigues(rvec.reshape(3, 1))
    T = np.eye(4, dtype=np.float64)
    T[:3, :3] = R
    T[:3, 3] = tvec.reshape(3)
    return T


def invert_T(T: np.ndarray) -> np.ndarray:
    R = T[:3, :3]
    t = T[:3, 3]
    T_inv = np.eye(4, dtype=np.float64)
    T_inv[:3, :3] = R.T
    T_inv[:3, 3] = -R.T @ t
    return T_inv


def find_image_from_bundle(root: Path, camera_name: str, bundle_name: str) -> Path:
    candidates = [
        root / bundle_name / camera_name / "rgb.png",
        root / bundle_name / camera_name / "rgb.jpg",
        root / bundle_name / camera_name / "rgb.jpeg",
    ]
    img_path = next((p for p in candidates if p.exists()), None)
    if img_path is None:
        raise FileNotFoundError(f"No rgb image found for {camera_name} in {root / bundle_name}")
    return img_path


def compute_reproj_rms(board, charuco_corners: np.ndarray, charuco_ids: np.ndarray,
                       rvec: np.ndarray, tvec: np.ndarray,
                       K: np.ndarray, dist: np.ndarray) -> Optional[float]:
    if not hasattr(board, "getChessboardCorners"):
        return None
    all_obj = np.asarray(board.getChessboardCorners(), dtype=np.float64)
    ids = charuco_ids.reshape(-1).astype(int)
    obj_visible = all_obj[ids]
    proj, _ = cv2.projectPoints(obj_visible, rvec, tvec, K, dist)
    proj = proj.reshape(-1, 2)
    imgp = charuco_corners.reshape(-1, 2)
    return float(np.sqrt(np.mean(np.sum((proj - imgp) ** 2, axis=1))))


def detect_and_estimate_pose(img_bgr: np.ndarray, board, detector,
                             K: np.ndarray, dist: np.ndarray,
                             min_markers: int, min_charuco: int):
    charuco_corners, charuco_ids, marker_corners, marker_ids = detector.detectBoard(img_bgr)
    num_markers = 0 if marker_ids is None else len(marker_ids)
    num_charuco = 0 if charuco_ids is None else len(charuco_ids)

    if num_markers < min_markers or num_charuco < min_charuco:
        raise RuntimeError(
            f"Not enough detections: markers={num_markers}, charuco={num_charuco}; "
            f"need at least markers>={min_markers}, charuco>={min_charuco}."
        )

    # IMPORTANT:
    # Do NOT call board.matchImagePoints(charuco_corners, charuco_ids) here.
    # Board.matchImagePoints is for ArUco marker corners, while charuco_corners
    # are interpolated ChArUco chessboard corners. Passing ChArUco corners to it
    # can create wrong 3D-2D correspondences on OpenCV 4.7+.
    #
    # Correct mapping:
    #   image point  = charuco_corners[i]
    #   object point = board.getChessboardCorners()[charuco_ids[i]]
    if not hasattr(board, "getChessboardCorners"):
        raise RuntimeError("This OpenCV CharucoBoard does not expose getChessboardCorners().")

    ids = charuco_ids.reshape(-1).astype(np.int32)
    all_obj = np.asarray(board.getChessboardCorners(), dtype=np.float32)
    obj_points = all_obj[ids].reshape(-1, 3).astype(np.float32)
    img_points = charuco_corners.reshape(-1, 2).astype(np.float32)

    if len(obj_points) < min_charuco:
        raise RuntimeError(
            f"Only {len(obj_points)} ChArUco object/image points, "
            f"need at least {min_charuco}."
        )

    pnp_flag = cv2.SOLVEPNP_IPPE if hasattr(cv2, "SOLVEPNP_IPPE") else cv2.SOLVEPNP_ITERATIVE
    success, rvec, tvec = cv2.solvePnP(
        objectPoints=obj_points,
        imagePoints=img_points,
        cameraMatrix=K,
        distCoeffs=dist,
        flags=pnp_flag,
    )

    if not success or rvec is None or tvec is None:
        # Fallback to iterative PnP, just in case IPPE fails in a corner case.
        success, rvec, tvec = cv2.solvePnP(
            objectPoints=obj_points,
            imagePoints=img_points,
            cameraMatrix=K,
            distCoeffs=dist,
            flags=cv2.SOLVEPNP_ITERATIVE,
        )

    if not success or rvec is None or tvec is None:
        raise RuntimeError("Pose estimation failed even though ChArUco detection succeeded.")

    rms = compute_reproj_rms(board, charuco_corners, charuco_ids, rvec, tvec, K, dist)
    return {
        "charuco_corners": charuco_corners,
        "charuco_ids": charuco_ids,
        "marker_corners": marker_corners,
        "marker_ids": marker_ids,
        "num_markers": int(num_markers),
        "num_charuco": int(num_charuco),
        "rvec": np.asarray(rvec, dtype=np.float64).reshape(3),
        "tvec": np.asarray(tvec, dtype=np.float64).reshape(3),
        "rms": rms,
    }


def draw_overlay(img_bgr: np.ndarray, K: np.ndarray, dist: np.ndarray,
                 det: dict, axis_length: float) -> np.ndarray:
    out = img_bgr.copy()
    if det["marker_ids"] is not None and len(det["marker_ids"]) > 0:
        cv2.aruco.drawDetectedMarkers(out, det["marker_corners"], det["marker_ids"])
    if det["charuco_ids"] is not None and len(det["charuco_ids"]) > 0:
        cv2.aruco.drawDetectedCornersCharuco(out, det["charuco_corners"], det["charuco_ids"])
    cv2.drawFrameAxes(out, K, dist, det["rvec"].reshape(3, 1), det["tvec"].reshape(3, 1), axis_length)

    text_lines = [
        f"markers={det['num_markers']}",
        f"charuco={det['num_charuco']}",
        f"rms={det['rms']:.4f}" if det["rms"] is not None else "rms=n/a",
    ]
    y = 24
    for line in text_lines:
        cv2.putText(out, line, (12, y), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (0, 255, 0), 2, cv2.LINE_AA)
        y += 26
    return out


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Estimate extrinsic from a single image using a ChArUco board as world frame."
    )
    src = parser.add_mutually_exclusive_group(required=True)
    src.add_argument("--image-path", type=Path, help="Path to a single RGB image")
    src.add_argument("--bundle-image", nargs=3, metavar=("ROOT", "CAMERA_NAME", "BUNDLE_NAME"),
                     help="Convenience mode: root camera_name bundle_000123")

    parser.add_argument("--intrinsic-file", type=Path, required=True, help=".npz from Extract_intrinsic.py, e.g. $AURA_CALIB_ROOT/intrinsic_results/1280x720/oak_<MXID>.npz (a .json is read only if it has K, dist and image_size; the Extract_intrinsic.py .json report has no image_size)")
    parser.add_argument("--camera-name", type=str, default=None, help="Optional name to include in output")
    parser.add_argument("--squares-x", type=int, required=True)
    parser.add_argument("--squares-y", type=int, required=True)
    parser.add_argument("--square-length", type=float, required=True)
    parser.add_argument("--marker-length", type=float, required=True)
    parser.add_argument("--dictionary", type=str, default="DICT_4X4_50", choices=sorted(DICT_MAP.keys()))
    parser.add_argument("--min-markers", type=int, default=3)
    parser.add_argument("--min-charuco", type=int, default=5)
    parser.add_argument("--axis-length", type=float, default=None)
    parser.add_argument("--save-overlay", type=Path, default=None)
    parser.add_argument("--output-json", type=Path, default=None)
    parser.add_argument("--output-npz", type=Path, default=None)
    parser.add_argument("--zero-dist", action="store_true",
                        help="Use zero distortion for PnP/projection. Do not use it with intrinsics fitted on frames already undistorted on the camera (record_3oak.py default): their dist is the residual distortion.")
    args = parser.parse_args()

    if args.bundle_image is not None:
        root = Path(args.bundle_image[0])
        camera_name = args.bundle_image[1]
        bundle_name = args.bundle_image[2]
        image_path = find_image_from_bundle(root, camera_name, bundle_name)
        if args.camera_name is None:
            args.camera_name = camera_name
    else:
        image_path = args.image_path

    if args.camera_name is None:
        args.camera_name = image_path.parent.name if image_path.parent.name else "camera"

    K, dist, image_size = load_intrinsics(args.intrinsic_file)
    if args.zero_dist:
        dist = np.zeros(5, dtype=np.float64)

    img = cv2.imread(str(image_path))
    if img is None:
        raise RuntimeError(f"Failed to read image: {image_path}")
    h, w = img.shape[:2]
    if (w, h) != tuple(image_size):
        raise RuntimeError(
            f"Image size mismatch: image is {(w, h)} but intrinsic expects {image_size}."
        )

    aruco_dict = cv2.aruco.getPredefinedDictionary(DICT_MAP[args.dictionary])
    board = cv2.aruco.CharucoBoard(
        (args.squares_x, args.squares_y),
        args.square_length,
        args.marker_length,
        aruco_dict,
    )
    detector = cv2.aruco.CharucoDetector(board)

    axis_length = args.axis_length if args.axis_length is not None else (args.square_length * 2.0)

    det = detect_and_estimate_pose(
        img_bgr=img,
        board=board,
        detector=detector,
        K=K,
        dist=dist,
        min_markers=args.min_markers,
        min_charuco=args.min_charuco,
    )

    T_cam_from_board = rt_to_T(det["rvec"], det["tvec"])
    T_board_from_cam = invert_T(T_cam_from_board)
    # Since world = board
    T_world_from_cam = T_board_from_cam.copy()
    T_cam_from_world = T_cam_from_board.copy()

    if args.save_overlay is not None:
        args.save_overlay.parent.mkdir(parents=True, exist_ok=True)
        overlay = draw_overlay(img, K, dist, det, axis_length)
        cv2.imwrite(str(args.save_overlay), overlay)

    result = {
        "camera_name": args.camera_name,
        "image_path": str(image_path),
        "world_frame": "charuco_board",
        "board": {
            "squares_x": args.squares_x,
            "squares_y": args.squares_y,
            "square_length": args.square_length,
            "marker_length": args.marker_length,
            "dictionary": args.dictionary,
        },
        "image_size": [w, h],
        "num_markers": det["num_markers"],
        "num_charuco": det["num_charuco"],
        "reproj_rms": det["rms"],
        "K": K.tolist(),
        "dist": dist.tolist(),
        "rvec_cam_from_board": det["rvec"].tolist(),
        "tvec_cam_from_board": det["tvec"].tolist(),
        "T_cam_from_board": T_cam_from_board.tolist(),
        "T_board_from_cam": T_board_from_cam.tolist(),
        "T_cam_from_world": T_cam_from_world.tolist(),
        "T_world_from_cam": T_world_from_cam.tolist(),
        "phys_twin_c2w": T_world_from_cam.tolist(),
    }

    print("=" * 72)
    print(f"camera_name : {args.camera_name}")
    print(f"image       : {image_path}")
    print(f"image_size  : {w} x {h}")
    print(f"markers     : {det['num_markers']}")
    print(f"charuco     : {det['num_charuco']}")
    print(f"RMS error   : {det['rms']:.6f}" if det["rms"] is not None else "RMS error   : n/a")
    print("T_cam_from_board =")
    print(T_cam_from_board)
    print("T_board_from_cam =")
    print(T_board_from_cam)
    print("(PhysTwin c2w = T_world_from_cam = T_board_from_cam)")

    if args.output_json is None:
        args.output_json = image_path.with_name(image_path.stem + "_extrinsic.json")
    args.output_json.parent.mkdir(parents=True, exist_ok=True)
    with open(args.output_json, "w", encoding="utf-8") as f:
        json.dump(result, f, indent=2)
    print(f"Saved JSON: {args.output_json}")

    if args.output_npz is None:
        args.output_npz = image_path.with_name(image_path.stem + "_extrinsic.npz")
    args.output_npz.parent.mkdir(parents=True, exist_ok=True)
    np.savez(
        args.output_npz,
        K=K,
        dist=dist,
        rvec_cam_from_board=det["rvec"],
        tvec_cam_from_board=det["tvec"],
        T_cam_from_board=T_cam_from_board,
        T_board_from_cam=T_board_from_cam,
        T_cam_from_world=T_cam_from_world,
        T_world_from_cam=T_world_from_cam,
        phys_twin_c2w=T_world_from_cam,
        image_size=np.asarray([w, h], dtype=np.int32),
        num_markers=np.asarray([det["num_markers"]], dtype=np.int32),
        num_charuco=np.asarray([det["num_charuco"]], dtype=np.int32),
        reproj_rms=np.asarray([np.nan if det["rms"] is None else det["rms"]], dtype=np.float64),
    )
    print(f"Saved NPZ : {args.output_npz}")

    if args.save_overlay is not None:
        print(f"Saved overlay: {args.save_overlay}")
    print("=" * 72)


if __name__ == "__main__":
    main()
