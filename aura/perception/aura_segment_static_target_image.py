#!/usr/bin/env python3
r"""AuRA-only static target segmentation for AnyGrasp observation fusion.

GroundingDINO may return multiple boxes for one text prompt. We remove highly
overlapping duplicate boxes with NMS, select the highest-confidence remaining
box, and call SAM2 with exactly that one box. The selected mask is written as
an RGBA PNG (alpha = mask, input pixels inside the mask, zeros outside) next to
a <stem>.detections.jpg overlay and a <stem>.selection.json diagnostic record.
The command-line interface and the RGBA mask format follow PhysTwin's
data_process/segment_util_image.py (https://github.com/Jianghanxiao/PhysTwin,
MIT License, Copyright (c) 2025 Hanxiao Jiang); the model settings follow
Grounded-SAM-2's grounded_sam2_local_demo.py (Apache-2.0).

Stage 02 of scripts/anygrasp_poses_one_shot.sh runs this once per camera via
aura/perception/phystwin_data_to_observation_pointcloud.py. To run it by hand
(diagnosis only: the driver's --skip-sam never picks up a hand-run result), use
the AuRA environment with config/aura.env loaded:

    python aura/perception/aura_segment_static_target_image.py \
        --img_path "$AURA_DATA_ROOT/franka_lift/<case>/anygrasp/color/0/0.png" \
        --output_path "$AURA_DATA_ROOT/debug/cam0_frame0_sam_rgba.png" \
        --TEXT_PROMPT "<object prompt>."

Model files (sam2.1_hiera_large.pt, groundingdino_swint_ogc.pth,
GroundingDINO_SwinT_OGC.py) are read from --checkpoint-dir (default
$AURA_GSAM_CHECKPOINTS, else <repo>/checkpoints/groundedSAM/; <repo> is
$AURA_REPO_ROOT, else this checkout). The Grounded-SAM-2 clone is required:
$AURA_GSAM_REPO (default <repo>/third_party/Grounded-SAM-2/; ~ is expanded, a
relative value is taken from the current directory) is put at the front of
sys.path, if that directory exists, before groundingdino/sam2 are imported,
because that fork imports grounding_dino.* relative to its clone root. Without
it the import fails with an error naming the path checked.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from contextlib import nullcontext
from pathlib import Path
from typing import Any

import cv2
import numpy as np
import torch


def _repo_root() -> Path:
    """Repository root: $AURA_REPO_ROOT, else two levels above aura/perception/."""
    return Path(os.environ.get("AURA_REPO_ROOT") or Path(__file__).resolve().parents[2])


# The Grounded-SAM-2 fork imports grounding_dino.* relative to its clone root,
# so the clone ($AURA_GSAM_REPO) must be on sys.path before groundingdino/sam2.
_GSAM_REPO = Path(
    os.environ.get("AURA_GSAM_REPO") or _repo_root() / "third_party" / "Grounded-SAM-2"
).expanduser().resolve()
if _GSAM_REPO.is_dir():
    sys.path.insert(0, str(_GSAM_REPO))

try:
    from groundingdino.util.inference import load_image, load_model, predict
    from sam2.build_sam import build_sam2
    from sam2.sam2_image_predictor import SAM2ImagePredictor
except ImportError as exc:
    # Only a missing Grounded-SAM-2 module gets the clone hint; a missing
    # dependency of groundingdino/sam2 keeps the original error.
    if (exc.name or "").split(".")[0] not in {"groundingdino", "grounding_dino", "sam2"}:
        raise
    raise ImportError(
        f"{exc} (Grounded-SAM-2 clone {_GSAM_REPO} "
        f"{'is on sys.path' if _GSAM_REPO.is_dir() else 'does not exist'}: set "
        "AURA_GSAM_REPO in config/aura.env and install groundingdino/sam2 from it)"
    ) from exc
from torchvision.ops import box_convert, nms


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Segment one static AuRA target: DINO NMS -> highest-confidence "
            "box -> single-box SAM2."
        )
    )
    parser.add_argument("--img_path", type=Path, required=True)
    parser.add_argument("--output_path", type=Path, required=True)
    parser.add_argument("--TEXT_PROMPT", required=True)
    parser.add_argument(
        "--checkpoint-dir",
        type=Path,
        default=Path(os.environ.get("AURA_GSAM_CHECKPOINTS") or _repo_root() / "checkpoints" / "groundedSAM"),
        help=(
            "Directory with sam2.1_hiera_large.pt, groundingdino_swint_ogc.pth and "
            "GroundingDINO_SwinT_OGC.py. Defaults to $AURA_GSAM_CHECKPOINTS, else "
            "<repo>/checkpoints/groundedSAM."
        ),
    )
    parser.add_argument("--box-threshold", type=float, default=0.35)
    parser.add_argument("--text-threshold", type=float, default=0.25)
    parser.add_argument("--nms-iou-threshold", type=float, default=0.50)
    parser.add_argument(
        "--fail-on-multiple",
        action="store_true",
        help=(
            "After NMS, save diagnostics and fail if multiple disjoint DINO "
            "boxes remain instead of selecting the highest-confidence one."
        ),
    )
    return parser.parse_args()


def normalized_prompt(value: str) -> str:
    value = " ".join(value.strip().lower().split())
    if not value:
        raise ValueError("--TEXT_PROMPT must not be empty")
    return value if value.endswith(".") else value + "."


def output_companion_path(output: Path, suffix: str) -> Path:
    return output.with_name(output.stem + suffix)


def json_ready(value: Any) -> Any:
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, np.ndarray):
        return value.tolist()
    return value


def draw_detections(
    image_bgr: np.ndarray,
    boxes_xyxy: np.ndarray,
    confidences: np.ndarray,
    labels: list[str],
    kept_indices: set[int],
    selected_index: int,
) -> np.ndarray:
    result = image_bgr.copy()
    height, width = result.shape[:2]
    font_scale = max(0.55, width / 1800.0)
    thickness = max(1, round(width / 1000.0))
    for index, box in enumerate(boxes_xyxy):
        x1, y1, x2, y2 = np.rint(box).astype(int)
        x1, x2 = np.clip([x1, x2], 0, width - 1)
        y1, y2 = np.clip([y1, y2], 0, height - 1)
        if index == selected_index:
            color = (0, 255, 0)
            status = "SELECTED"
        elif index in kept_indices:
            color = (0, 180, 255)
            status = "CANDIDATE"
        else:
            color = (128, 128, 128)
            status = "NMS-REMOVED"
        cv2.rectangle(result, (x1, y1), (x2, y2), color, thickness + 1)
        label = labels[index] if index < len(labels) else "unknown"
        caption = f"{index}: {label} {confidences[index]:.3f} {status}"
        (text_w, text_h), baseline = cv2.getTextSize(
            caption, cv2.FONT_HERSHEY_SIMPLEX, font_scale, thickness
        )
        top = max(0, y1 - text_h - baseline - 8)
        cv2.rectangle(
            result,
            (x1, top),
            (min(width - 1, x1 + text_w + 8), y1),
            color,
            -1,
        )
        cv2.putText(
            result,
            caption,
            (x1 + 4, max(text_h, y1 - baseline - 4)),
            cv2.FONT_HERSHEY_SIMPLEX,
            font_scale,
            (0, 0, 0),
            thickness,
            cv2.LINE_AA,
        )
    return result


def main() -> None:
    args = parse_args()
    prompt = normalized_prompt(args.TEXT_PROMPT)
    image_path = args.img_path.expanduser().resolve()
    output_path = args.output_path.expanduser().resolve()

    if not image_path.is_file():
        raise FileNotFoundError(f"input image does not exist: {image_path}")
    if not 0.0 <= args.nms_iou_threshold <= 1.0:
        raise ValueError("--nms-iou-threshold must be within [0, 1]")

    checkpoint_root = args.checkpoint_dir.expanduser().resolve()
    sam_checkpoint = checkpoint_root / "sam2.1_hiera_large.pt"
    dino_config = checkpoint_root / "GroundingDINO_SwinT_OGC.py"
    dino_checkpoint = checkpoint_root / "groundingdino_swint_ogc.pth"
    for required in (sam_checkpoint, dino_config, dino_checkpoint):
        if not required.is_file():
            raise FileNotFoundError(
                f"missing model file: {required} (set AURA_GSAM_CHECKPOINTS in "
                "config/aura.env, or pass --checkpoint-dir; for "
                "phystwin_data_to_observation_pointcloud.py: --gsam-checkpoints)"
            )

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"[AuRA static SAM] prompt={prompt!r} device={device}")

    grounding_model = load_model(
        model_config_path=str(dino_config),
        model_checkpoint_path=str(dino_checkpoint),
        device=device,
    )
    image_source, image_tensor = load_image(str(image_path))
    with torch.inference_mode():
        boxes_cxcywh, confidence_tensor, labels = predict(
            model=grounding_model,
            image=image_tensor,
            caption=prompt,
            box_threshold=args.box_threshold,
            text_threshold=args.text_threshold,
        )

    raw_count = int(boxes_cxcywh.shape[0])
    if raw_count == 0:
        raise RuntimeError(
            f"GroundingDINO found no box for prompt {prompt!r} "
            f"(box_threshold={args.box_threshold}, text_threshold={args.text_threshold})"
        )

    height, width = image_source.shape[:2]
    scale = boxes_cxcywh.new_tensor([width, height, width, height])
    boxes_xyxy_tensor = box_convert(
        boxes=boxes_cxcywh * scale,
        in_fmt="cxcywh",
        out_fmt="xyxy",
    )
    kept_tensor = nms(
        boxes_xyxy_tensor,
        confidence_tensor,
        args.nms_iou_threshold,
    )
    kept_indices = [int(index) for index in kept_tensor.detach().cpu().tolist()]
    if not kept_indices:
        raise RuntimeError("NMS unexpectedly removed all GroundingDINO boxes")
    # torchvision.ops.nms returns kept indices in descending score order.
    selected_index = kept_indices[0]

    boxes_xyxy = boxes_xyxy_tensor.detach().cpu().numpy()
    confidences = confidence_tensor.detach().cpu().numpy()
    label_strings = [str(value) for value in labels]
    raw_image_bgr = cv2.imread(str(image_path), cv2.IMREAD_COLOR)
    if raw_image_bgr is None:
        raise RuntimeError(f"OpenCV could not read input image: {image_path}")

    output_path.parent.mkdir(parents=True, exist_ok=True)
    overlay_path = output_companion_path(output_path, ".detections.jpg")
    metadata_path = output_companion_path(output_path, ".selection.json")
    detection_overlay = draw_detections(
        raw_image_bgr,
        boxes_xyxy,
        confidences,
        label_strings,
        set(kept_indices),
        selected_index,
    )
    if not cv2.imwrite(str(overlay_path), detection_overlay):
        raise RuntimeError(f"failed to write detection overlay: {overlay_path}")

    detections = []
    for index in range(raw_count):
        detections.append(
            {
                "index": index,
                "label": label_strings[index] if index < len(label_strings) else "unknown",
                "confidence": float(confidences[index]),
                "box_xyxy": [float(value) for value in boxes_xyxy[index]],
                "kept_after_nms": index in kept_indices,
                "selected": index == selected_index,
            }
        )
    metadata: dict[str, Any] = {
        "schema_version": 1,
        "pipeline": "AuRA static target segmentation",
        "input_image": str(image_path),
        "output_rgba": str(output_path),
        "detection_overlay": str(overlay_path),
        "prompt": prompt,
        "device": device,
        "box_threshold": args.box_threshold,
        "text_threshold": args.text_threshold,
        "nms_iou_threshold": args.nms_iou_threshold,
        "raw_box_count": raw_count,
        "post_nms_box_count": len(kept_indices),
        "ambiguous_after_nms": len(kept_indices) > 1,
        "selected_index": selected_index,
        "selection_policy": "highest DINO confidence after box NMS",
        "detections": detections,
    }
    metadata_path.write_text(
        json.dumps(metadata, indent=2, default=json_ready) + "\n",
        encoding="utf-8",
    )

    print(
        f"[AuRA static SAM] DINO boxes: raw={raw_count}, "
        f"after_nms={len(kept_indices)}, selected={selected_index}, "
        f"confidence={float(confidences[selected_index]):.4f}"
    )
    print(f"[AuRA static SAM] detections: {overlay_path}")
    print(f"[AuRA static SAM] metadata:   {metadata_path}")
    if args.fail_on_multiple and len(kept_indices) > 1:
        raise RuntimeError(
            f"multiple disjoint target candidates remain after NMS: {len(kept_indices)}; "
            f"inspect {overlay_path}"
        )

    sam_model = build_sam2(
        "configs/sam2.1/sam2.1_hiera_l.yaml",
        str(sam_checkpoint),
        device=device,
    )
    predictor = SAM2ImagePredictor(sam_model)
    autocast_context = (
        torch.autocast(device_type="cuda", dtype=torch.bfloat16)
        if device == "cuda"
        else nullcontext()
    )
    if device == "cuda" and torch.cuda.get_device_properties(0).major >= 8:
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True

    # A length-4 XYXY box is intentional: never give this static AuRA entry
    # point an Nx4 multi-box batch (one mask per box would come back, and
    # masks[0] below would then ignore the NMS/confidence selection above).
    selected_box = boxes_xyxy[selected_index]
    with torch.inference_mode(), autocast_context:
        predictor.set_image(image_source)
        masks, scores, _ = predictor.predict(
            point_coords=None,
            point_labels=None,
            box=selected_box,
            multimask_output=False,
        )

    masks = np.asarray(masks)
    if masks.ndim == 4 and masks.shape[1] == 1:
        masks = masks[:, 0]
    if masks.ndim != 3 or masks.shape[0] < 1:
        raise RuntimeError(f"unexpected SAM2 mask shape: {masks.shape}")
    mask = masks[0].astype(bool)
    if mask.shape != (height, width):
        raise RuntimeError(
            f"SAM2 mask resolution mismatch: {mask.shape} != {(height, width)}"
        )
    if not mask.any():
        raise RuntimeError("SAM2 returned an empty mask for the selected box")

    rgba = np.zeros((height, width, 4), dtype=np.uint8)
    rgba[mask, :3] = raw_image_bgr[mask]
    rgba[:, :, 3] = mask.astype(np.uint8) * 255
    if not cv2.imwrite(str(output_path), rgba):
        raise RuntimeError(f"failed to write RGBA mask: {output_path}")

    score_array = np.asarray(scores).reshape(-1)
    metadata["sam_mask_pixels"] = int(mask.sum())
    metadata["sam_mask_fraction"] = float(mask.mean())
    metadata["sam_score"] = float(score_array[0]) if score_array.size else None
    metadata_path.write_text(
        json.dumps(metadata, indent=2, default=json_ready) + "\n",
        encoding="utf-8",
    )
    print(
        f"[AuRA static SAM] saved mask: {output_path} "
        f"({int(mask.sum())} pixels, {100.0 * float(mask.mean()):.2f}%)"
    )


if __name__ == "__main__":
    main()
