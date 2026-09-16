"""Compare clip metadata from ffprobe and in-process container inspection."""

import argparse
import io
import json
import random
import time
from pathlib import Path

import av

from nemo_curator.utils.decoder_utils import extract_video_metadata


def main(args: argparse.Namespace) -> None:
    paths = sorted(args.root.rglob("*.mp4"))
    random.Random(20260916).shuffle(paths)  # noqa: S311 - reproducible benchmark ordering
    results = []
    for path in paths[: args.count]:
        contents = path.read_bytes()
        start = time.perf_counter()
        reference = extract_video_metadata(contents)
        ffprobe_seconds = time.perf_counter() - start
        start = time.perf_counter()
        with av.open(io.BytesIO(contents)) as container:
            stream = container.streams.video[0]
            fps = float(stream.average_rate)
            duration = float(f"{float(stream.duration * stream.time_base):.6f}")
            candidate = {
                "width": stream.width,
                "height": stream.height,
                "framerate": fps,
                "num_frames": int(duration * fps),
                "video_codec": stream.codec_context.name,
            }
        pyav_seconds = time.perf_counter() - start
        expected = {
            "width": reference.width,
            "height": reference.height,
            "framerate": reference.fps,
            "num_frames": reference.num_frames,
            "video_codec": reference.video_codec,
        }
        results.append(
            {
                "path": str(path),
                "ffprobe_seconds": ffprobe_seconds,
                "pyav_seconds": pyav_seconds,
                "equal": candidate == expected,
                "expected": expected,
                "candidate": candidate,
            }
        )
    args.result.write_text(json.dumps(results, indent=2))
    print(
        json.dumps(
            {
                "clips": len(results),
                "all_equal": all(r["equal"] for r in results),
                "ffprobe_seconds": sum(r["ffprobe_seconds"] for r in results),
                "pyav_seconds": sum(r["pyav_seconds"] for r in results),
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
