#!/usr/bin/env python3
"""Batch convert BEVFusion demo results into a single NuScenes-style JSON."""
from __future__ import annotations

import argparse
import importlib.util
import json
from collections import defaultdict
from pathlib import Path
from typing import DefaultDict, List

MODULE_PATH = Path(__file__).resolve().parent / "export_predictions.py"

_spec = importlib.util.spec_from_file_location("bevfusion_export_single", MODULE_PATH)
_module = importlib.util.module_from_spec(_spec)
assert _spec and _spec.loader
_spec.loader.exec_module(_module)

convert_predictions = _module.convert_predictions
load_sample_token = _module.load_sample_token
summarise = _module.summarise


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--result-glob",
        type=str,
        default="work_dirs/*_result.json",
        help="Glob pattern that matches per-sample prediction JSON files.",
    )
    parser.add_argument(
        "--ann-root",
        type=Path,
        default=None,
        help="Optional directory containing matching annotation pickles.",
    )
    parser.add_argument(
        "--out-json",
        type=Path,
        default=Path("work_dirs/bevfusion_batch_detection.json"),
        help="Combined NuScenes-style detection output.",
    )
    parser.add_argument(
        "--summary-txt",
        type=Path,
        default=Path("work_dirs/bevfusion_batch_summary.txt"),
        help="Aggregate per-class summary file.",
    )
    parser.add_argument(
        "--score-thr",
        type=float,
        default=0.0,
        help="Discard detections below this score threshold.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    result_paths = sorted(Path().glob(args.result_glob))
    if not result_paths:
        raise SystemExit(f"No files matched pattern: {args.result_glob}")

    aggregated: DefaultDict[str, List[dict]] = defaultdict(list)

    for result_path in result_paths:
        with result_path.open() as fh:
            data = json.load(fh)
        sample_token = None
        ann_path: Path | None = None
        if args.ann_root is not None:
            candidate = args.ann_root / f"{result_path.stem}.pkl"
            if candidate.is_file():
                ann_path = candidate
        if ann_path is not None:
            sample_token = load_sample_token(ann_path)
        if not sample_token:
            sample_token = result_path.stem

        converted = convert_predictions(
            data.get("pred_instances_3d", {}), sample_token, args.score_thr
        )
        for token, dets in converted.items():
            aggregated[token].extend(dets)

    args.out_json.parent.mkdir(parents=True, exist_ok=True)
    with args.out_json.open("w") as fh:
        json.dump({"results": dict(aggregated), "meta": {"score_threshold": args.score_thr}}, fh, indent=2)

    if args.summary_txt is not None:
        summary = summarise(dict(aggregated))
        args.summary_txt.parent.mkdir(parents=True, exist_ok=True)
        args.summary_txt.write_text(summary + "\n")

    total = sum(len(v) for v in aggregated.values())
    print(f"Processed {len(result_paths)} files -> {total} detections saved to {args.out_json}")
    if args.summary_txt is not None:
        print(f"Summary written to {args.summary_txt}")


if __name__ == "__main__":
    main()
