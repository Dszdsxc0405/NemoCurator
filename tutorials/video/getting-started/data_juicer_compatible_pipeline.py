# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.

"""Run migrated Data-Juicer video operators with Xenna."""

import argparse
import json
import os
import socket
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml

from nemo_curator.backends.xenna import XennaExecutor
from nemo_curator.core.client import RayClient
from nemo_curator.pipeline import Pipeline
from nemo_curator.stages.resources import Resources
from nemo_curator.stages.video.analysis import (
    ClipFrameSamplingStage,
    VideoAestheticsFilterStage,
    VideoCameraMotionStage,
    VideoFrameCaptionStage,
    VideoOcrAreaRatioFilterStage,
    VideoOpticalFlowFilterStage,
    VideoSceneSplitStage,
    WholeVideoClipStage,
)
from nemo_curator.stages.video.analysis._utils import ALL_KEYFRAMES, UNIFORM_3_FRAMES, UNIFORM_16_FRAMES
from nemo_curator.stages.video.clipping.clip_extraction_stages import ClipTranscodingStage
from nemo_curator.stages.video.io.clip_writer import ClipWriterStage
from nemo_curator.stages.video.io.video_manifest_reader import VideoManifestReaderStage
from nemo_curator.stages.video.io.video_reader import VideoReader, VideoReaderStage

_DATA_JUICER_ROOT = Path("/home/xyq/VSCode/2025_10_TeleDataJuicer/my_dj")
_DEFAULT_MANIFEST = _DATA_JUICER_ROOT / "demos/process_video_on_ray/data/openvid-64.jsonl"
_DEFAULT_RAY_TEMP_DIR = Path("/dev/shm/nemo_curator_data_juicer_ray")  # noqa: S108
_DEFAULT_PIPELINE_CONFIG = Path(__file__).with_name("curator_pipeline_config.yaml")


@dataclass(frozen=True)
class ExperimentConfig:
    num_cpus: int
    num_gpus: int
    input_manifest: str | None
    video_dir: str | None
    operator_names: tuple[str, ...]
    stage_names: tuple[str, ...]
    stage_configs: dict[str, dict[str, Any]]


def _bypass_proxy_for_local_ray() -> None:
    local_hosts = {"localhost", "127.0.0.1", socket.gethostbyname(socket.gethostname())}
    configured_hosts = {
        host.strip()
        for key in ("NO_PROXY", "no_proxy")
        for host in os.environ.get(key, "").split(",")
        if host.strip()
    }
    value = ",".join(sorted(configured_hosts | local_hosts))
    os.environ["NO_PROXY"] = value
    os.environ["no_proxy"] = value


def _load_experiment_config(config_path: str, experiment: str | None) -> ExperimentConfig:
    path = Path(config_path).expanduser().resolve()
    with path.open(encoding="utf-8") as stream:
        raw_config = json.load(stream) if path.suffix.lower() == ".json" else yaml.safe_load(stream)
    if not isinstance(raw_config, dict):
        message = f"Pipeline config must contain an object: {path}"
        raise ValueError(message)

    profiles = raw_config.get("experiments", {})
    if experiment is None:
        if len(profiles) == 1:
            experiment = next(iter(profiles))
        elif "6op" in profiles:
            experiment = "6op"
        else:
            available = ", ".join(sorted(profiles)) or "none"
            message = f"Experiment must be specified. Available experiments: {available}"
            raise ValueError(message)
    if experiment not in profiles:
        available = ", ".join(sorted(profiles)) or "none"
        message = f"Unknown experiment '{experiment}'. Available experiments: {available}"
        raise ValueError(message)
    profile = profiles[experiment]
    operator_names = profile.get("operators", [])
    operator_stages = raw_config.get("operators", {})
    active_stages = set(raw_config.get("always_stages", []))
    active_stages.update(profile.get("extra_stages", []))
    for operator_name in operator_names:
        if operator_name not in operator_stages:
            message = f"Experiment '{experiment}' references unknown operator '{operator_name}'"
            raise ValueError(message)
        active_stages.update(operator_stages[operator_name])

    stage_order = raw_config.get("stage_order", [])
    missing_from_order = active_stages - set(stage_order)
    if missing_from_order:
        message = f"Active stages missing from stage_order: {sorted(missing_from_order)}"
        raise ValueError(message)
    stage_names = tuple(stage_name for stage_name in stage_order if stage_name in active_stages)

    base_stage_configs = raw_config.get("stages", {})
    stage_overrides = profile.get("stage_overrides", {})
    missing_stage_configs = active_stages - set(base_stage_configs)
    if missing_stage_configs:
        message = f"Missing resource config for stages: {sorted(missing_stage_configs)}"
        raise ValueError(message)
    stage_configs = {}
    for stage_name, base_stage_config in base_stage_configs.items():
        stage_override = stage_overrides.get(stage_name, {})
        stage_configs[stage_name] = {
            **base_stage_config,
            **stage_override,
            "params": {
                **base_stage_config.get("params", {}),
                **stage_override.get("params", {}),
            },
        }

    ray_config = raw_config.get("ray", {})
    dataset_config = raw_config.get("dataset", {})
    if not isinstance(dataset_config, dict):
        message = "dataset must contain an object"
        raise ValueError(message)
    input_manifest = dataset_config.get("input_manifest")
    video_dir = dataset_config.get("video_dir")
    if input_manifest is not None and not isinstance(input_manifest, str):
        message = "dataset.input_manifest must be a path string"
        raise ValueError(message)
    if video_dir is not None and not isinstance(video_dir, str):
        message = "dataset.video_dir must be a path string"
        raise ValueError(message)
    if input_manifest and video_dir:
        message = "dataset must specify only one of input_manifest or video_dir"
        raise ValueError(message)

    def resolve_dataset_path(value: str | None) -> str | None:
        if not value:
            return None
        dataset_path = Path(value).expanduser()
        if not dataset_path.is_absolute():
            dataset_path = path.parent / dataset_path
        return dataset_path.resolve().as_posix()

    return ExperimentConfig(
        num_cpus=int(ray_config.get("num_cpus", 96)),
        num_gpus=int(ray_config.get("num_gpus", 4)),
        input_manifest=resolve_dataset_path(input_manifest),
        video_dir=resolve_dataset_path(video_dir),
        operator_names=tuple(operator_names),
        stage_names=stage_names,
        stage_configs=stage_configs,
    )


def _stage_overrides(config: ExperimentConfig, stage_name: str) -> dict[str, Any]:
    stage_config = config.stage_configs[stage_name]
    supported_keys = {"cpus", "gpus", "batch_size", "num_workers", "params"}
    unexpected_keys = set(stage_config) - supported_keys
    if unexpected_keys:
        message = f"Unsupported resource keys for stage '{stage_name}': {sorted(unexpected_keys)}"
        raise ValueError(message)
    cpus = float(stage_config.get("cpus", 1.0))
    gpus = float(stage_config.get("gpus", 0.0))
    batch_size = int(stage_config.get("batch_size", 1))
    num_workers = int(stage_config.get("num_workers", 1))
    if cpus <= 0 or gpus < 0 or batch_size <= 0 or num_workers <= 0:
        message = f"Invalid resource config for stage '{stage_name}': {stage_config}"
        raise ValueError(message)
    return {
        "resources": Resources(cpus=cpus, gpus=gpus),
        "batch_size": batch_size,
        "num_workers": num_workers,
    }


def _stage_params(config: ExperimentConfig, stage_name: str) -> dict[str, Any]:
    params = config.stage_configs[stage_name].get("params", {})
    if not isinstance(params, dict):
        message = f"Stage params must be an object for stage '{stage_name}'"
        raise ValueError(message)
    return dict(params)


def _manifest_input_root(manifest_path: str, video_key: str = "videos") -> str:
    manifest = Path(manifest_path).expanduser().resolve()
    paths = []
    with manifest.open(encoding="utf-8") as stream:
        for line in stream:
            row = json.loads(line)
            values = row.get(video_key, [])
            values = [values] if isinstance(values, str) else values
            for value in values:
                path = Path(value).expanduser()
                paths.append((manifest.parent / path).resolve() if not path.is_absolute() else path)
    if not paths:
        message = f"No video paths found in manifest: {manifest}"
        raise ValueError(message)
    common_path = Path(os.path.commonpath(paths))
    return (common_path if common_path.is_dir() else common_path.parent).as_posix()


def create_pipeline(args: argparse.Namespace, config: ExperimentConfig) -> Pipeline:
    pipeline = Pipeline(
        name="data_juicer_compatible_video",
        description=f"{len(config.operator_names)} Data-Juicer operators implemented as native Curator stages",
    )
    stage_names = set(config.stage_names)
    if not {"input_partition", "video_reader", "clip_writer"}.issubset(stage_names):
        message = "always_stages must include input_partition, video_reader, and clip_writer"
        raise ValueError(message)

    if getattr(args, "input_manifest", None):
        input_root = _manifest_input_root(args.input_manifest)
        pipeline.add_stage(
            VideoManifestReaderStage(input_manifest=args.input_manifest, limit=args.video_limit).with_(
                **_stage_overrides(config, "input_partition")
            )
        )
        pipeline.add_stage(
            VideoReaderStage(input_path=input_root).with_(**_stage_overrides(config, "video_reader"))
        )
    else:
        input_root = args.video_dir
        pipeline.add_stage(
            VideoReader(input_video_path=input_root, video_limit=args.video_limit).with_(
                {
                    "file_partitioning": _stage_overrides(config, "input_partition"),
                    "video_reader": _stage_overrides(config, "video_reader"),
                }
            )
        )

    stage_factories = {
        "whole_video_clip": WholeVideoClipStage,
        "video_scene_split": lambda: VideoSceneSplitStage(
            **_stage_params(config, "video_scene_split")
        ),
        "clip_transcoding": lambda: ClipTranscodingStage(
            **_stage_params(config, "clip_transcoding")
        ),
        "uniform_3_frame_sampling": lambda: ClipFrameSamplingStage(
            frame_key=UNIFORM_3_FRAMES,
            name="uniform_3_frame_sampling",
            **_stage_params(config, "uniform_3_frame_sampling"),
        ),
        "video_aesthetics_filter": lambda: VideoAestheticsFilterStage(
            **_stage_params(config, "video_aesthetics_filter")
        ),
        "video_optical_flow_filter": lambda: VideoOpticalFlowFilterStage(
            **_stage_params(config, "video_optical_flow_filter")
        ),
        "video_ocr_area_ratio_filter": lambda: VideoOcrAreaRatioFilterStage(
            **_stage_params(config, "video_ocr_area_ratio_filter")
        ),
        "all_keyframe_sampling": lambda: ClipFrameSamplingStage(
            frame_key=ALL_KEYFRAMES,
            name="all_keyframe_sampling",
            **_stage_params(config, "all_keyframe_sampling"),
        ),
        "video_captioning_from_frames": lambda: VideoFrameCaptionStage(
            **_stage_params(config, "video_captioning_from_frames")
        ),
        "uniform_16_frame_sampling": lambda: ClipFrameSamplingStage(
            frame_key=UNIFORM_16_FRAMES,
            name="uniform_16_frame_sampling",
            **_stage_params(config, "uniform_16_frame_sampling"),
        ),
        "video_captioning_camera_motion": lambda: VideoCameraMotionStage(
            **_stage_params(config, "video_captioning_camera_motion")
        ),
        "clip_writer": lambda: ClipWriterStage(
            output_path=args.output_path,
            input_path=input_root,
            upload_clips=not args.no_upload_clips,
            dry_run=args.dry_run,
            **_stage_params(config, "clip_writer"),
        ),
    }
    unsupported_stages = stage_names - {"input_partition", "video_reader"} - set(stage_factories)
    if unsupported_stages:
        message = f"No Curator implementation registered for stages: {sorted(unsupported_stages)}"
        raise ValueError(message)
    for stage_name in config.stage_names:
        if stage_name in stage_factories:
            pipeline.add_stage(stage_factories[stage_name]().with_(**_stage_overrides(config, stage_name)))
    return pipeline


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    inputs = parser.add_mutually_exclusive_group()
    inputs.add_argument("--video-dir")
    inputs.add_argument("--input-manifest")
    parser.add_argument("--output-path", required=True)
    parser.add_argument("--video-limit", type=int)
    parser.add_argument("--pipeline-config", default=_DEFAULT_PIPELINE_CONFIG.as_posix())
    parser.add_argument("--experiment")
    parser.add_argument("--num-cpus", type=int)
    parser.add_argument("--num-gpus", type=int)
    parser.add_argument("--ray-temp-dir", default=_DEFAULT_RAY_TEMP_DIR.as_posix())
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--no-upload-clips", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config = _load_experiment_config(args.pipeline_config, args.experiment)
    if args.video_dir is None and args.input_manifest is None:
        args.video_dir = config.video_dir
        args.input_manifest = config.input_manifest
    if args.video_dir is None and args.input_manifest is None:
        args.input_manifest = _DEFAULT_MANIFEST.as_posix()
    pipeline = create_pipeline(args, config)
    print(pipeline.describe())
    _bypass_proxy_for_local_ray()
    with RayClient(
        include_dashboard=False,
        ray_temp_dir=args.ray_temp_dir,
        num_cpus=args.num_cpus if args.num_cpus is not None else config.num_cpus,
        num_gpus=args.num_gpus if args.num_gpus is not None else config.num_gpus,
    ):
        pipeline.run(
            XennaExecutor(
                config={
                    "execution_mode": "streaming",
                    "logging_interval": 60,
                    "autoscale_interval_s": 180,
                }
            )
        )


if __name__ == "__main__":
    main()
