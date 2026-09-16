#!/usr/bin/env python3
"""Measure video-task batch sizes using cached, real scene-split videos."""

import argparse
import concurrent.futures
import copy
import json
import os
import pickle
import random
import subprocess
import sys
import time
from pathlib import Path

import yaml

STAGES = {
    "aesthetics": "video_aesthetics_filter",
    "caption": "video_captioning_from_frames",
    "camera": "video_captioning_camera_motion",
    "ocr": "video_ocr_area_ratio_filter",
    "flow": "video_optical_flow_filter",
}


def prepare(args):
    from nemo_curator.stages.video.analysis import ClipFrameSamplingStage, VideoSceneSplitStage
    from nemo_curator.stages.video.clipping.clip_extraction_stages import ClipTranscodingStage
    from nemo_curator.tasks.video import Video, VideoTask

    cfg = yaml.safe_load(Path(args.config).read_text())["stages"]
    rows = [json.loads(line) for line in Path(args.manifest).open() if line.strip()]
    paths = [v for row in rows for v in ([row["videos"]] if isinstance(row["videos"], str) else row["videos"])]
    present = [v for v in paths if Path(v).is_file()]
    missing = [v for v in paths if not Path(v).is_file()]
    args.root.mkdir(parents=True, exist_ok=True)
    (args.root / "missing_paths.json").write_text(json.dumps(missing, indent=2))
    random.Random(20260915).shuffle(present)
    selected = present[:args.count]
    previous_path = args.root / 'corpus.json'
    previous = {r['index']: r for r in json.loads(previous_path.read_text())} if previous_path.exists() else {}

    def one(pair):
        index, path = pair
        dest = args.root / f"{index:04d}.pickle"
        if dest.exists():
            return previous[index]
        video = Video(input_video=Path(path), source_bytes=Path(path).read_bytes())
        video.populate_metadata()
        task = VideoTask(dataset_name="batch_sweep", data=video)
        VideoSceneSplitStage(**cfg["video_scene_split"]["params"]).process(task)
        if not video.clips:
            raise RuntimeError(str(video.errors))
        parts = ClipTranscodingStage(**cfg["clip_transcoding"]["params"]).process(task)
        if not isinstance(parts, list):
            parts = [parts]
        for part in parts:
            for name, key in [("uniform_3_frame_sampling", "video_analysis:uniform:3"),
                              ("all_keyframe_sampling", "video_analysis:all_keyframes"),
                              ("uniform_16_frame_sampling", "video_analysis:uniform:16")]:
                ClipFrameSamplingStage(frame_key=key, **cfg[name]["params"]).process(part)
        task.data.clips = [clip for part in parts for clip in part.data.clips]
        for clip in task.data.clips:
            clip.buffer = None
        task.data.filtered_clips = []
        record = {"index": index, "path": path, "clips": len(task.data.clips),
                  "keyframes": sum(len(c.extracted_frames.get("video_analysis:all_keyframes", [])) for c in task.data.clips),
                  "height": video.metadata.height, "width": video.metadata.width}
        if not task.data.clips:
            raise RuntimeError("No clips survived frame extraction")
        with dest.open("wb") as stream:
            pickle.dump(task, stream, protocol=pickle.HIGHEST_PROTOCOL)
        return record

    records = []
    with concurrent.futures.ThreadPoolExecutor(max_workers=8) as pool:
        futures = {pool.submit(one, pair): pair for pair in enumerate(selected)}
        for future in concurrent.futures.as_completed(futures):
            try:
                records.append(future.result())
            except Exception as exc:
                records.append({"index": futures[future][0], "path": futures[future][1], "error": str(exc)})
            (args.root / "corpus.json").write_text(json.dumps(records, indent=2))
            if len(records) % 16 == 0:
                print(f"Prepared {len(records)}/{len(selected)}", flush=True)
    print(json.dumps({"input": len(paths), "missing": len(missing), "prepared": len(list(args.root.glob('*.pickle')))}), flush=True)


def measure(args):
    import torch
    from nemo_curator.stages.video.analysis import (
        VideoAestheticsFilterStage, VideoFrameCaptionStage, VideoCameraMotionStage,
        VideoOcrAreaRatioFilterStage, VideoOpticalFlowFilterStage,
    )

    classes = dict(zip(STAGES, [VideoAestheticsFilterStage, VideoFrameCaptionStage, VideoCameraMotionStage,
                               VideoOcrAreaRatioFilterStage, VideoOpticalFlowFilterStage]))
    cfg = yaml.safe_load(Path(args.config).read_text())["stages"][STAGES[args.stage]]
    stage = classes[args.stage](**cfg["params"])
    stage.batch_size = args.batch
    stage.setup()
    records = json.loads((args.root / "corpus.json").read_text())
    valid = [r for r in records if "error" not in r]
    valid.sort(key=lambda r: r.get("keyframes", 0) if args.stage == "caption" else r.get("clips", 0), reverse=True)
    if args.timing_videos:
        valid.sort(key=lambda r: r['index'])
        random.Random(42).shuffle(valid)
    if len(valid) < args.batch:
        raise ValueError(f"Only {len(valid)} unique cached video tasks available")

    def load(offset, count):
        tasks = []
        for i in range(count):
            index = valid[(offset + i) % len(valid)]["index"]
            with (args.root / f"{index:04d}.pickle").open("rb") as stream:
                task = pickle.load(stream)
            for clip in task.data.clips:
                clip.extracted_frames = {stage.frame_key: clip.extracted_frames[stage.frame_key]}
            tasks.append(task)
        return tasks

    stage.process_batch(load(0, min(2, args.batch)))
    torch.cuda.synchronize()
    torch.cuda.empty_cache()
    timings = []
    failures = []
    workloads = []
    peak = 0
    reserved = 0
    for repeat in range(args.repeats):
        count = min(args.timing_videos, len(valid)) if args.timing_videos else args.batch
        tasks = load(0 if args.timing_videos else repeat * args.batch, count)
        workloads.append({"videos": len(tasks), "clips": sum(len(t.data.clips) for t in tasks),
                          "frames": sum(len(c.extracted_frames[stage.frame_key]) for t in tasks for c in t.data.clips)})
        torch.manual_seed(42 + repeat)
        torch.cuda.reset_peak_memory_stats()
        torch.cuda.synchronize()
        start = time.perf_counter()
        stage.process_batch(tasks)
        torch.cuda.synchronize()
        timings.append(time.perf_counter() - start)
        peak = max(peak, torch.cuda.max_memory_allocated())
        reserved = max(reserved, torch.cuda.max_memory_reserved())
        for task in tasks:
            for clip in task.data.clips + task.data.filtered_clips:
                for key, detail in clip.errors.items():
                    if key == stage.name and 'outside [' not in detail and detail != 'insufficient_frames':
                        failures.append(detail)
        del tasks
        if failures:
            break
    result = {"stage": args.stage, "batch_videos": args.batch, "ok": not failures,
              "seconds": timings, "videos_per_second": count * len(timings) / sum(timings),
              "timing_videos": args.timing_videos,
              "peak_allocated_gib": peak / 1024**3, "workloads": workloads,
              "peak_reserved_gib": reserved / 1024**3,
              "errors": list(dict.fromkeys(failures)), "gpu": os.environ.get('CUDA_VISIBLE_DEVICES')}
    print(json.dumps(result), flush=True)
    Path(args.result).write_text(json.dumps(result, indent=2))


def sweep(args):
    args.output.mkdir(parents=True, exist_ok=True)
    results = []
    for batch in args.batches:
        result = args.output / f"{args.stage}_{batch}.json"
        log = args.output / f"{args.stage}_{batch}.log"
        command = [sys.executable, __file__, "measure", "--config", args.config, "--root", str(args.root),
                   "--stage", args.stage, "--batch", str(batch), "--result", str(result), "--repeats", str(args.repeats),
                   "--timing-videos", str(args.timing_videos)]
        print('Starting', args.stage, batch, flush=True)
        with log.open('w') as stream:
            try:
                proc = subprocess.run(command, stdout=stream, stderr=subprocess.STDOUT, timeout=args.timeout)
                code = proc.returncode
            except subprocess.TimeoutExpired:
                code = 'timeout'
        if result.exists():
            record = json.loads(result.read_text())
        else:
            record = {'stage': args.stage, 'batch_videos': batch, 'ok': False, 'exit_code': code, 'log': str(log)}
        results.append(record)
        (args.output / f"{args.stage}_summary.json").write_text(json.dumps(results, indent=2))
        print(json.dumps(record), flush=True)
        if not record['ok']:
            break


def joint(args):
    from nemo_curator.stages.video.analysis import (
        VideoAestheticsFilterStage, VideoFrameCaptionStage, VideoCameraMotionStage,
        VideoOcrAreaRatioFilterStage, VideoOpticalFlowFilterStage,
    )
    from nemo_curator.stages.resources import Resources
    from nemo_curator.core.client import RayClient
    from nemo_curator.pipeline import Pipeline
    from nemo_curator.backends.xenna import XennaExecutor
    classes = {'aesthetics': VideoAestheticsFilterStage, 'flow': VideoOpticalFlowFilterStage,
               'ocr': VideoOcrAreaRatioFilterStage, 'caption': VideoFrameCaptionStage,
               'camera': VideoCameraMotionStage}
    cfg = yaml.safe_load(Path(args.config).read_text())
    pipeline = Pipeline('batch_capacity_joint')
    for name, cls in classes.items():
        settings = cfg['stages'][STAGES[name]]
        params = dict(settings['params'])
        if name in ['aesthetics', 'flow']:
            params.update(min_score=0, max_score=1e10)
        elif name == 'ocr':
            params.update(min_area_ratio=0, max_area_ratio=1e10)
        pipeline.add_stage(cls(**params).with_(resources=Resources(cpus=settings['cpus'], gpus=settings['gpus']),
                                              batch_size=settings['batch_size'], num_workers=settings['num_workers']))
    records = [r for r in json.loads((args.root/'corpus.json').read_text()) if 'error' not in r]
    records.sort(key=lambda r: r.get('keyframes', 0), reverse=True)
    tasks = []
    for record in records[:args.count]:
        with (args.root/f"{record['index']:04d}.pickle").open('rb') as stream:
            tasks.append(pickle.load(stream))
    start = time.perf_counter()
    with RayClient(num_cpus=96, num_gpus=4, include_dashboard=False, ray_temp_dir='/tmp/curator_batch_tuning'):
        results = pipeline.run(XennaExecutor(), initial_tasks=tasks)
    errors = {}
    clips = 0
    for task in results:
        clips += len(task.data.clips)
        for clip in task.data.clips + task.data.filtered_clips:
            for key, detail in clip.errors.items():
                if detail != 'insufficient_frames':
                    errors.setdefault(key, {}).setdefault(detail, 0)
                    errors[key][detail] += 1
    report = {'config': args.config, 'input_tasks': len(tasks), 'output_tasks': len(results),
              'remaining_clips': clips, 'errors': errors, 'seconds': time.perf_counter()-start,
              'threshold_filters_disabled': True}
    Path(args.result).write_text(json.dumps(report, indent=2))
    print(json.dumps(report), flush=True)


def ocr_check(args):
    import numpy as np
    from nemo_curator.stages.video.analysis import VideoOcrAreaRatioFilterStage
    cfg = yaml.safe_load(Path(args.config).read_text())['stages']['video_ocr_area_ratio_filter']
    stage = VideoOcrAreaRatioFilterStage(**cfg['params'])
    stage.batch_size = 16
    stage.setup()
    tasks = []
    for path in sorted(args.root.glob('*.pickle'))[:args.count]:
        with path.open('rb') as stream:
            tasks.append(pickle.load(stream))
    expected = {str(clip.uuid): float(np.mean([stage._frame_ratio(frame) for frame in clip.extracted_frames[stage.frame_key]]))
                for task in tasks for clip in task.data.clips}
    stage.process_batch(tasks)
    actual = {str(clip.uuid): clip.ocr_area_ratio for task in tasks for clip in task.data.clips + task.data.filtered_clips}
    differences = {key: abs(expected[key]-actual[key]) for key in expected}
    decisions = {key: (stage.min_area_ratio <= expected[key] <= stage.max_area_ratio,
                       stage.min_area_ratio <= actual[key] <= stage.max_area_ratio) for key in expected}
    changed = {key: value for key, value in decisions.items() if value[0] != value[1]}
    report = {'clips': len(expected), 'maximum_difference': max(differences.values()), 'differences': differences,
              'changed_filter_decisions': changed, 'expected': expected, 'actual': actual}
    Path(args.result).write_text(json.dumps(report, indent=2))
    print(json.dumps(report), flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('mode', choices=['prepare', 'measure', 'sweep', 'joint', 'ocr_check'])
    parser.add_argument('--config', default='tutorials/video/run-op-6.yaml')
    parser.add_argument('--manifest', default='tutorials/video/dataset/T2V-5k-filter.jsonl')
    parser.add_argument('--root', type=Path, required=True)
    parser.add_argument('--count', type=int, default=512)
    parser.add_argument('--stage', choices=list(STAGES))
    parser.add_argument('--batch', type=int)
    parser.add_argument('--batches', nargs='+', type=int)
    parser.add_argument('--result')
    parser.add_argument('--output', type=Path)
    parser.add_argument('--repeats', type=int, default=3)
    parser.add_argument('--timing-videos', type=int, default=0)
    parser.add_argument('--timeout', type=int, default=240)
    parsed = parser.parse_args()
    globals()[parsed.mode](parsed)
