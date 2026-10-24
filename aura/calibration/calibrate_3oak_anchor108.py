#!/usr/bin/env python3
"""Re-estimate three OAK extrinsics while preserving the existing anchor-camera world pose.

Cameras are $AURA_OAK_IDS (camera-index order); the anchor is --anchor oak_<MXID>
(default $AURA_HANDEYE_OAK_ID). Each camera's board image is read from
SESSION_ROOT/BUNDLE/oak_<MXID>/rgb.{png,jpg,jpeg} (a record_3oak.py capture).
Reads CALIB_ROOT/intrinsic_results/1280x720/oak_<MXID>.npz and the anchor's
CALIB_ROOT/extrinsic_results/oak_<MXID>.npz (CALIB_ROOT defaults to $AURA_CALIB_ROOT).
Writes all results to the empty OUTPUT_DIR; --apply then installs the non-anchor
extrinsics into CALIB_ROOT/extrinsic_results/, after backing up the previous files
to CALIB_ROOT/extrinsic_results/backups/.
Uses Extract_extrinsic.py next to this script for intrinsic loading and overlay drawing.
Does not command a robot. The board must stay fixed across the three images.
The anchor camera must not have moved if the old physical table/world frame is to persist.
Input is undistorted RGB (record_3oak.py --enable-undistortion True, the default).
Unlike Extract_extrinsic.py, which by default uses the saved dist and SOLVEPNP_IPPE, this
script by default replaces the saved dist with zeros and solves with SOLVEPNP_ITERATIVE.
Requires numpy and an OpenCV build with aruco/CharucoDetector.
The '108' in this file name is the authors' historical name for their own anchor camera;
it does not name your camera. The same name is kept unchanged in the backup prefix
before_anchor108_ and in the label world_frame='existing_world_preserved_via_oak108' of
each output JSON. The anchor actually used is --anchor, recorded there as anchor_camera.
The capture and calibration commands are in docs/calibration.md, section 5.
"""
import argparse
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import shutil
import sys
from datetime import datetime, timezone
import numpy as np

CAMERAS = tuple('oak_' + mxid for mxid in os.environ.get('AURA_OAK_IDS', '').split())
ANCHOR = os.environ.get('AURA_HANDEYE_OAK_ID') or None  # replaced by --anchor in main()


def _repo_root():
    """Repository root: $AURA_REPO_ROOT, else two levels above this file's directory."""
    return Path(os.environ.get('AURA_REPO_ROOT') or Path(__file__).resolve().parents[2])


def rigid(T):
    T = np.asarray(T, dtype=float)
    if (T.shape != (4, 4) or not np.isfinite(T).all()
        or not np.allclose(T[3], [0, 0, 0, 1], atol=1e-8)
        or not np.allclose(T[:3, :3].T @ T[:3, :3], np.eye(3), atol=1e-5)
        or not np.isclose(np.linalg.det(T[:3, :3]), 1, atol=1e-5)):
        raise ValueError('Invalid rigid transform')
    return T


def rebase(world_from_anchor, camera_from_board):
    old = rigid(world_from_anchor)
    world_from_board = old @ rigid(camera_from_board[ANCHOR])
    result = {name: rigid(world_from_board @ np.linalg.inv(rigid(T)))
              for name, T in camera_from_board.items()}
    assert np.allclose(result[ANCHOR], old, atol=1e-10)
    result[ANCHOR] = old.copy()
    return result, world_from_board


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    global ANCHOR  # rebase() and the file names below use the module-level ANCHOR
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--calib-root', type=Path,
                   default=Path(os.environ.get('AURA_CALIB_ROOT') or _repo_root() / 'calibration'),
                   help='Calibration results root (default: $AURA_CALIB_ROOT, else <repo>/calibration).')
    p.add_argument('--anchor', default=ANCHOR,
                   help='oak_<MXID> whose existing world pose is preserved '
                        '(default: $AURA_HANDEYE_OAK_ID; required when that is unset).')
    p.add_argument('--session-root', type=Path, required=True)
    p.add_argument('--bundle', default='bundle_000100')
    p.add_argument('--output-dir', type=Path, required=True)
    p.add_argument('--max-rms', type=float, default=1.0)
    p.add_argument('--raw-distorted', action='store_true',
                   help='Use the saved intrinsic dist instead of zeros, like Extract_extrinsic.py (solver '
                        'stays SOLVEPNP_ITERATIVE). Each output JSON then records rgb_undistorted=false, '
                        'although the input is still undistorted RGB.')
    p.add_argument('--apply', action='store_true',
                   help='Back up and install the two non-anchor extrinsics after validation.')
    args = p.parse_args()
    if not np.isfinite(args.max_rms) or args.max_rms <= 0:
        p.error('--max-rms must be finite and positive')
    if not CAMERAS:
        p.error('AURA_OAK_IDS is empty: set the camera MXIDs (camera-index order) in config/aura.env '
                'and load it (set -a; source config/aura.env; set +a)')
    if not args.anchor:
        p.error('no anchor camera: set AURA_HANDEYE_OAK_ID in config/aura.env and load it '
                '(set -a; source config/aura.env; set +a), or pass --anchor oak_<MXID>')
    if args.anchor not in CAMERAS:
        p.error(f'anchor {args.anchor} (from --anchor, else $AURA_HANDEYE_OAK_ID) is not one of the '
                f'$AURA_OAK_IDS cameras; use one of: {", ".join(CAMERAS)}')
    ANCHOR = args.anchor
    import cv2
    calib = args.calib_root
    source = Path(__file__).resolve().parent / 'Extract_extrinsic.py'
    spec = importlib.util.spec_from_file_location('existing_extrinsic', source)
    helper = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = helper
    spec.loader.exec_module(helper)
    active = calib / 'extrinsic_results'
    old_path = active / f'{ANCHOR}.npz'
    anchor_hash = sha(old_path)
    with np.load(old_path, allow_pickle=False) as z:
        keys = [k for k in ('T_world_from_cam', 'phys_twin_c2w') if k in z]
        if not keys:
            raise ValueError('Anchor file has no recognized camera-to-world transform')
        old = rigid(z[keys[0]])
        for key in keys[1:]:
            if not np.allclose(old, z[key], atol=1e-8):
                raise ValueError('Anchor camera-to-world fields disagree')
    robot_dir = calib / 'robot_calibration_results'
    protected = [old_path, active / f'{ANCHOR}.json']
    protected += [robot_dir / f'robot_calib_result.{ext}' for ext in ('npz', 'json')]
    protected_hashes = {path: sha(path) for path in protected if path.is_file()}
    dictionary = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_4X4_50)
    board = cv2.aruco.CharucoBoard((4, 5), .04, .03, dictionary)
    detector = cv2.aruco.CharucoDetector(board)
    records, poses, overlays = {}, {}, {}
    # All cameras must succeed before candidate outputs or active files are written.
    for name in CAMERAS:
        folder = args.session_root / args.bundle / name
        image_path = next((folder / ('rgb.' + ext) for ext in ('png', 'jpg', 'jpeg')
                           if (folder / ('rgb.' + ext)).is_file()), None)
        if image_path is None:
            raise FileNotFoundError(f'No RGB image in {folder}; choose a saved --bundle')
        image = cv2.imread(str(image_path))
        if image is None:
            raise ValueError(f'Cannot decode {image_path}')
        K, saved_dist, size = helper.load_intrinsics(
            calib / 'intrinsic_results/1280x720' / f'{name}.npz')
        if tuple(size) != (image.shape[1], image.shape[0]):
            raise ValueError(f'{name}: image/intrinsic resolution mismatch')
        dist = saved_dist if args.raw_distorted else np.zeros_like(saved_dist)
        corners, ids, markers, marker_ids = detector.detectBoard(image)
        if ids is None or len(ids) < 8 or marker_ids is None or len(marker_ids) < 4:
            raise ValueError(f'{name}: need >=8 ChArUco corners and >=4 markers')
        # Match exact corner IDs directly; avoids the old matchImagePoints issue.
        obj = np.asarray(board.getChessboardCorners(), dtype=np.float64)[ids.ravel()]
        ok, rvec, tvec = cv2.solvePnP(obj, corners.astype(np.float64), K, dist,
                                     flags=cv2.SOLVEPNP_ITERATIVE)
        if not ok:
            raise ValueError(f'{name}: PnP failed')
        R, _ = cv2.Rodrigues(rvec)
        if np.any((obj @ R.T + tvec.reshape(3))[:, 2] <= 0):
            raise ValueError(f'{name}: board behind camera')
        projected, _ = cv2.projectPoints(obj, rvec, tvec, K, dist)
        rms = float(np.sqrt(np.mean(np.sum((projected.reshape(-1, 2) -
                                          corners.reshape(-1, 2)) ** 2, axis=1))))
        if not np.isfinite(rms) or rms > args.max_rms:
            raise ValueError(f'{name}: reprojection RMS {rms:.4f}px exceeds {args.max_rms}px')
        T = np.eye(4); T[:3, :3] = R; T[:3, 3] = tvec.reshape(3)
        poses[name] = rigid(T)
        det = dict(charuco_corners=corners, charuco_ids=ids, marker_corners=markers,
                   marker_ids=marker_ids, num_markers=len(marker_ids), num_charuco=len(ids),
                   rvec=rvec.reshape(3), tvec=tvec.reshape(3), rms=rms)
        overlays[name] = helper.draw_overlay(image, K, dist, det, .08)
        records[name] = dict(camera_name=name, image_path=str(image_path.resolve()),
            image_size=list(size), K=K.tolist(), dist=dist.reshape(-1).tolist(),
            num_markers=len(marker_ids), num_charuco=len(ids), reproj_rms=rms,
            rvec_cam_from_board=rvec.reshape(3).tolist(),
            tvec_cam_from_board=tvec.reshape(3).tolist(),
            T_cam_from_board=T.tolist(), T_board_from_cam=np.linalg.inv(T).tolist())
        print(f'[CAMERA] {name}: corners={len(ids)}, RMS={rms:.4f}px')
    world, world_board = rebase(old, poses)
    output = args.output_dir.expanduser().resolve()
    if output == active.resolve():
        raise ValueError('Choose a separate output directory; --apply installs validated results')
    output.mkdir(parents=True, exist_ok=True)
    if any(output.iterdir()):
        raise FileExistsError(f'Output must be empty: {output}')
    for name, record in records.items():
        record.update(world_frame='existing_world_preserved_via_oak108', anchor_camera=ANCHOR,
            anchor_source=str(old_path), anchor_sha256=anchor_hash,
            rgb_undistorted=not args.raw_distorted,
            T_world_from_board=world_board.tolist(),
            T_world_from_cam=world[name].tolist(), phys_twin_c2w=world[name].tolist(),
            T_cam_from_world=np.linalg.inv(world[name]).tolist())
        (output / f'{name}.json').write_text(json.dumps(record, indent=2) + '\n')
        numeric = {k: np.asarray(record[k]) for k in (
            'K', 'dist', 'image_size', 'num_markers', 'num_charuco', 'reproj_rms',
            'rvec_cam_from_board', 'tvec_cam_from_board', 'T_cam_from_board',
            'T_board_from_cam', 'T_world_from_board', 'T_world_from_cam',
            'T_cam_from_world', 'phys_twin_c2w')}
        np.savez(output / f'{name}.npz', **numeric)
        if not cv2.imwrite(str(output / f'{name}_board_overlay.png'), overlays[name]):
            raise OSError('Failed to save board overlay')
    if any(sha(path) != digest for path, digest in protected_hashes.items()):
        raise RuntimeError('Calibration inputs changed while solving; active files not installed')
    if args.apply:
        stamp = datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S%fZ')
        backup = active / 'backups' / ('before_anchor108_' + stamp)
        backup.mkdir(parents=True)
        for name in [n for n in CAMERAS if n != ANCHOR]:
            for ext in ('json', 'npz'):
                dest = active / f'{name}.{ext}'
                if dest.exists():
                    shutil.copy2(dest, backup / dest.name)
        for name in [n for n in CAMERAS if n != ANCHOR]:
            for ext in ('json', 'npz'):
                dest = active / f'{name}.{ext}'
                temp = active / f'.{dest.name}.{stamp}.tmp'
                shutil.copy2(output / dest.name, temp)
                temp.replace(dest)
        print(f'[INSTALLED] two non-anchor cameras; previous files: {backup}')
    assert all(sha(path) == digest for path, digest in protected_hashes.items())
    print(f'[OUTPUT] {output}')
    print(f'[PRESERVED] {ANCHOR} world pose and robot calibration files unchanged')


if __name__ == '__main__':
    main()
