#!/usr/bin/env python3
"""Convert serialized BEVFusion demo predictions into NuScenes-style output.

This helper is intended for quick QA when Open3D visualisation is unavailable.
It consumes ``work_dirs/bevfusion_demo_result.json`` (or equivalent) and
emits:

1. A NuScenes detection JSON suitable for downstream inspection.
2. An optional plain-text summary with per-class statistics.

Example usage::

    python projects/BEVFusion/demo/export_predictions.py \
        --result-json work_dirs/bevfusion_demo_result.json \
        --ann demo/data/nuscenes/n015-2018-07-24-11-22-45+0800.pkl \
        --out-json work_dirs/bevfusion_demo_detection.json \
        --summary-txt work_dirs/bevfusion_demo_summary.txt \
        --score-thr 0.3

The annotation pickle is only required when a genuine NuScenes sample token
is desired in the exported file. Otherwise the tool will populate the token
with ``"demo_sample"``.
"""
from __future__ import annotations

import argparse
import json
import math
import pickle
from collections import Counter, defaultdict
from pathlib import Path
from typing import Dict, Iterable, List, Sequence

NUSCENES_CLASSES: List[str] = [
    "car",
    "truck",
    "construction_vehicle",
    "bus",
    "trailer",
    "barrier",
    "motorcycle",
    "bicycle",
    "pedestrian",
    "traffic_cone",
]

# Attribute defaults aligned with the NuScenes detection schema for quick QA.
ATTRIBUTE_DEFAULTS: Dict[str, str] = {
    "car": "vehicle.moving",
    "truck": "vehicle.moving",
    "construction_vehicle": "vehicle.moving",
    "bus": "vehicle.moving",
    "trailer": "vehicle.moving",
    "barrier": "other",
    "motorcycle": "cycle.with_rider",
    "bicycle": "cycle.with_rider",
    "pedestrian": "pedestrian.moving",
    "traffic_cone": "other",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--result-json",
        type=Path,
        default=Path("work_dirs/bevfusion_demo_result.json"),
        help="Path to the serialized Det3DDataSample JSON.",
    )
    parser.add_argument(
        "--ann",
        type=Path,
        default=None,
        help="Optional annotation pickle to recover the NuScenes sample token.",
    )
    parser.add_argument(
        "--out-json",
        type=Path,
        default=Path("work_dirs/bevfusion_demo_detection.json"),
        help="Destination file for NuScenes-style detection results.",
    )
    parser.add_argument(
        "--summary-txt",
        type=Path,
        default=None,
        help="Optional text file for per-class counts and score statistics.",
    )
    parser.add_argument(
        "--score-thr",
        type=float,
        default=0.0,
        help="Drop detections below this confidence threshold.",
    )
    return parser.parse_args()


def load_sample_token(ann_path: Path | None) -> str:
    if ann_path is None:
        return "demo_sample"
    with ann_path.open("rb") as fh:
        ann_data = pickle.load(fh)
    data_list: Sequence[dict] = ann_data.get("data_list", [])
    if not data_list:
        return "demo_sample"
    entry = data_list[0]
    return entry.get("token", "demo_sample")


def yaw_to_quaternion(yaw: float) -> List[float]:
    half = yaw / 2.0
    return [math.cos(half), 0.0, 0.0, math.sin(half)]


def convert_predictions(
    detections: dict,
    sample_token: str,
    score_thr: float,
) -> Dict[str, List[dict]]:
    scores: Iterable[float] = detections.get("scores_3d", [])
    labels: Iterable[int] = detections.get("labels_3d", [])
    boxes: Iterable[Sequence[float]] = detections.get("bboxes_3d", [])

    output: Dict[str, List[dict]] = defaultdict(list)
    for score, label, box in zip(scores, labels, boxes):
        if score < score_thr:
            continue
        if not (0 <= label < len(NUSCENES_CLASSES)):
            continue
        class_name = NUSCENES_CLASSES[label]
        x, y, z, dx, dy, dz, yaw, *rest = box
        vx, vy = (rest + [0.0, 0.0])[:2]
        detection = {
            "sample_token": sample_token,
            "translation": [float(x), float(y), float(z)],
            # NuScenes expects [width, length, height].
            "size": [float(dy), float(dx), float(dz)],
            "rotation": yaw_to_quaternion(float(yaw)),
            "velocity": [float(vx), float(vy)],
            "detection_name": class_name,
            "detection_score": float(score),
            "attribute_name": ATTRIBUTE_DEFAULTS.get(class_name, "other"),
        }
        output[sample_token].append(detection)
    return output


def summarise(detections: Dict[str, List[dict]]) -> str:
    counter: Counter[str] = Counter()
    score_totals: Dict[str, float] = defaultdict(float)
    for det_list in detections.values():
        for det in det_list:
            cls = det["detection_name"]
            counter[cls] += 1
            score_totals[cls] += det["detection_score"]
    lines = ["class,count,mean_score"]
    for cls in NUSCENES_CLASSES:
        count = counter.get(cls, 0)
        mean_score = score_totals.get(cls, 0.0) / count if count else 0.0
        lines.append(f"{cls},{count},{mean_score:.4f}")
    return "\n".join(lines)


def main() -> None:
    args = parse_args()
    with args.result_json.open() as fh:
        data = json.load(fh)
    sample_token = load_sample_token(args.ann)
    converted = convert_predictions(
        data.get("pred_instances_3d", {}), sample_token, args.score_thr
    )

    args.out_json.parent.mkdir(parents=True, exist_ok=True)
    with args.out_json.open("w") as fh:
        json.dump({"results": converted, "meta": {"score_threshold": args.score_thr}}, fh, indent=2)

    if args.summary_txt is not None:
        summary_text = summarise(converted)
        args.summary_txt.parent.mkdir(parents=True, exist_ok=True)
        args.summary_txt.write_text(summary_text + "\n")

    print(f"Exported {sum(len(v) for v in converted.values())} detections to {args.out_json}")
    if args.summary_txt is not None:
        print(f"Wrote summary to {args.summary_txt}")


if __name__ == "__main__":
    main()
