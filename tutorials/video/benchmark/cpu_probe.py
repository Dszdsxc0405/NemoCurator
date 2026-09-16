"""Compare isolated scene detection paths without changing production stages."""

import argparse
import json
import multiprocessing as mp
import queue
import time
from pathlib import Path
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from nemo_curator.stages.video.analysis.scene_split import _SceneDetectionConfig


def detect(path: str, config: "_SceneDetectionConfig") -> list[tuple[float, float, int, int]]:
    from nemo_curator.stages.video.analysis.scene_split import _detect_scenes_worker

    result = queue.Queue()
    _detect_scenes_worker(result, path, config)
    status, payload = result.get()
    if status != "ok":
        raise RuntimeError(payload)
    return payload


def main(args: argparse.Namespace) -> None:
    from nemo_curator.stages.video.analysis.scene_split import VideoSceneSplitStage, _SceneDetectionConfig
    from nemo_curator.utils.operation_utils import make_pipeline_named_temporary_file

    rows = [json.loads(line) for line in args.manifest.read_text().splitlines()]
    paths = [row["videos"][0] for row in rows[: args.count]]
    config = _SceneDetectionConfig("ContentDetector", 27.0, 10, {}, False)
    stage = VideoSceneSplitStage()
    records = []
    with mp.get_context("spawn").Pool(1) as pool:
        # Separate one-time import/setup from per-video steady-state work.
        pool.apply(detect, (paths[0], config))
        for path in paths:
            start = time.perf_counter()
            contents = Path(path).read_bytes()
            read_seconds = time.perf_counter() - start
            with make_pipeline_named_temporary_file(sub_dir="cpu_probe", suffix=".mp4") as tmp:
                start = time.perf_counter()
                tmp.write_bytes(contents)
                write_seconds = time.perf_counter() - start
                start = time.perf_counter()
                original, error = stage._detect(str(tmp))
                spawn_seconds = time.perf_counter() - start
                if error:
                    raise RuntimeError(error)
                start = time.perf_counter()
                reused = pool.apply_async(detect, (str(tmp), config)).get(timeout=60)
                reused_seconds = time.perf_counter() - start
            start = time.perf_counter()
            direct = pool.apply_async(detect, (path, config)).get(timeout=60)
            direct_seconds = time.perf_counter() - start
            record = {
                "path": path,
                "bytes": len(contents),
                "read_seconds": read_seconds,
                "write_seconds": write_seconds,
                "spawn_seconds": spawn_seconds,
                "reused_seconds": reused_seconds,
                "direct_seconds": direct_seconds,
                "same_scenes": original == reused == direct,
                "scene_count": len(original),
            }
            records.append(record)
            args.result.write_text(json.dumps(records, indent=2))
            print(json.dumps(record), flush=True)
    print(
        json.dumps(
            {
                key: sum(r[key] for r in records)
                for key in ["read_seconds", "write_seconds", "spawn_seconds", "reused_seconds", "direct_seconds"]
            }
        ),
        flush=True,
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--result", type=Path, required=True)
    parser.add_argument("--count", type=int, default=12)
    main(parser.parse_args())
