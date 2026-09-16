"""Measure uniform sampling alternatives on actual transcoded clip bytes."""

import argparse
import io
import json
import random
import time
from pathlib import Path

import av
import cv2
import numpy as np

from nemo_curator.stages.video.analysis.frame_sampling import ClipFrameSamplingStage
from nemo_curator.utils.operation_utils import make_pipeline_named_temporary_file


def main(args: argparse.Namespace) -> None:
    paths = sorted(args.root.rglob("*.mp4"))
    random.Random(20260916).shuffle(paths)  # noqa: S311 - reproducible benchmark ordering
    stage = ClipFrameSamplingStage(frame_count=3)
    records = []
    for path in paths[: args.count]:
        contents = path.read_bytes()
        start = time.perf_counter()
        with make_pipeline_named_temporary_file(sub_dir="sampling_probe", suffix=".mp4") as tmp:
            tmp.write_bytes(contents)
            reference = stage._sample_uniform(str(tmp))
        baseline = time.perf_counter() - start
        capture = cv2.VideoCapture(str(path))
        total = int(capture.get(cv2.CAP_PROP_FRAME_COUNT))
        capture.release()
        count = min(3, total)
        indices = [total // 2] if count == 1 else np.linspace(0, total - 1, count, dtype=int).tolist()
        start = time.perf_counter()
        frames = []
        with av.open(io.BytesIO(contents)) as container:
            for index, frame in enumerate(container.decode(video=0)):
                if index in indices:
                    frames.append(frame.to_ndarray(format="rgb24"))
                if index >= indices[-1]:
                    break
        in_memory = time.perf_counter() - start
        candidate = np.stack(frames)
        record = {
            "path": str(path),
            "frames": total,
            "baseline_seconds": baseline,
            "memory_sequential_seconds": in_memory,
            "same_pixels": np.array_equal(reference, candidate),
        }
        records.append(record)
    args.result.write_text(json.dumps(records, indent=2))
    print(
        json.dumps(
            {
                "clips": len(records),
                "same_pixels": all(r["same_pixels"] for r in records),
                "baseline_seconds": sum(r["baseline_seconds"] for r in records),
                "memory_sequential_seconds": sum(r["memory_sequential_seconds"] for r in records),
            }
        ),
        flush=True,
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--result", type=Path, required=True)
    parser.add_argument("--count", type=int, default=24)
    main(parser.parse_args())
