# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.

"""Run the migrated Data-Juicer video operators with Xenna autoscaling."""

import argparse
import json
import os
import socket
from pathlib import Path

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
)
from nemo_curator.stages.video.analysis._utils import ALL_KEYFRAMES, UNIFORM_3_FRAMES, UNIFORM_16_FRAMES
from nemo_curator.stages.video.clipping.clip_extraction_stages import ClipTranscodingStage
from nemo_curator.stages.video.io.clip_writer import ClipWriterStage
from nemo_curator.stages.video.io.video_manifest_reader import VideoManifestReaderStage
from nemo_curator.stages.video.io.video_reader import VideoReader, VideoReaderStage

_DATA_JUICER_ROOT = Path("/home/xyq/VSCode/2025_10_TeleDataJuicer/my_dj")
_DEFAULT_MANIFEST = _DATA_JUICER_ROOT / "demos/process_video_on_ray/data/openvid-64.jsonl"
_DEFAULT_MODEL_ROOT = Path("/home/xyq/.cache/data_juicer/models")
_DEFAULT_RAY_TEMP_DIR = Path("/dev/shm/nemo_curator_data_juicer_ray")  # noqa: S108


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


def create_pipeline(args: argparse.Namespace) -> Pipeline:
    pipeline = Pipeline(
        name="data_juicer_compatible_video",
        description="Six Data-Juicer video operators implemented as native Curator stages",
    )
    if getattr(args, "input_manifest", None):
        input_root = _manifest_input_root(args.input_manifest)
        pipeline.add_stage(VideoManifestReaderStage(input_manifest=args.input_manifest, limit=args.video_limit))
        pipeline.add_stage(VideoReaderStage(input_path=input_root))
    else:
        input_root = args.video_dir
        pipeline.add_stage(VideoReader(input_video_path=input_root, video_limit=args.video_limit))
    pipeline.add_stage(
        VideoSceneSplitStage(
            detector="ContentDetector",
            threshold=27.0,
            min_scene_len=10,
            show_progress=False,
            max_scene_num=3,
            meta_time_out=60,
        )
    )
    pipeline.add_stage(ClipTranscodingStage())
    pipeline.add_stage(
        ClipFrameSamplingStage(
            frame_key=UNIFORM_3_FRAMES,
            sampling_method="uniform",
            frame_count=3,
            name="uniform_3_frame_sampling",
        )
    )
    pipeline.add_stage(
        VideoAestheticsFilterStage(
            hf_scorer_model=args.aesthetic_model,
            min_score=0.4,
            max_score=1.0,
            frame_sampling_method="uniform",
            frame_num=3,
            reduce_mode="avg",
            any_or_all="any",
        )
    )
    pipeline.add_stage(
        VideoOpticalFlowFilterStage(
            optflow_model_path=args.unimatch_weights,
            min_score=0.15,
            max_score=20.0,
            frame_sampling_method="uniform",
            frame_num=3,
            reduce_mode="avg",
            any_or_all="any",
            stride=2,
            scale_by_fps=True,
        )
    )
    pipeline.add_stage(
        VideoOcrAreaRatioFilterStage(
            model_storage_dir=args.easyocr_model_dir,
            min_area_ratio=0.001,
            max_area_ratio=1.0,
            frame_sample_num=3,
            languages_to_detect=["ch_sim", "en"],
            any_or_all="all",
        )
    )
    pipeline.add_stage(
        ClipFrameSamplingStage(
            frame_key=ALL_KEYFRAMES,
            sampling_method="all_keyframes",
            frame_count=3,
            name="all_keyframe_sampling",
        )
    )
    pipeline.add_stage(
        VideoFrameCaptionStage(
            hf_img2seq=args.blip2_model,
            caption_num=1,
            keep_candidate_mode="random_any",
            keep_original_sample=False,
            prompt=None,
            prompt_key=None,
            frame_sampling_method="all_keyframes",
            frame_num=3,
            horizontal_flip=False,
            vertical_flip=False,
        )
    )
    pipeline.add_stage(
        ClipFrameSamplingStage(
            frame_key=UNIFORM_16_FRAMES,
            sampling_method="uniform",
            frame_count=16,
            name="uniform_16_frame_sampling",
        )
    )
    pipeline.add_stage(
        VideoCameraMotionStage(
            hf_model_path=args.camera_motion_model,
            num_frames=16,
            max_duration=8,
            torch_dtype="bfloat16",
        )
    )
    pipeline.add_stage(
        ClipWriterStage(
            output_path=args.output_path,
            input_path=input_root,
            upload_clips=not args.no_upload_clips,
            dry_run=args.dry_run,
            generate_embeddings=False,
            generate_previews=False,
            generate_captions=True,
            caption_models=[],
            enhanced_caption_models=[],
        ).with_(resources=Resources(cpus=1.0, gpu_memory_gb=2.0), batch_size=4)
    )
    return pipeline


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    inputs = parser.add_mutually_exclusive_group()
    inputs.add_argument("--video-dir")
    inputs.add_argument("--input-manifest")
    parser.add_argument("--output-path", required=True)
    parser.add_argument(
        "--aesthetic-model",
        default=(_DEFAULT_MODEL_ROOT / "shunk031/aesthetics-predictor-v2-sac-logos-ava1-l14-linearMSE").as_posix(),
    )
    parser.add_argument(
        "--unimatch-weights",
        default=(
            _DEFAULT_MODEL_ROOT / "Unimatch/gmflow-scale2-regrefine6-mixdata-train320x576-4e7b215d.pth"
        ).as_posix(),
    )
    parser.add_argument(
        "--easyocr-model-dir",
        default=(_DEFAULT_MODEL_ROOT / "Ceceliachenen/easyocr").as_posix(),
    )
    parser.add_argument(
        "--blip2-model",
        default=(_DEFAULT_MODEL_ROOT / "Salesforce/blip2-opt-2.7b").as_posix(),
    )
    parser.add_argument(
        "--camera-motion-model",
        default=(_DEFAULT_MODEL_ROOT / "MCG-NJU/videomae-large").as_posix(),
    )
    parser.add_argument("--video-limit", type=int)
    parser.add_argument("--ray-temp-dir", default=_DEFAULT_RAY_TEMP_DIR.as_posix())
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--no-upload-clips", action="store_true")
    args = parser.parse_args()
    if args.video_dir is None and args.input_manifest is None:
        args.input_manifest = _DEFAULT_MANIFEST.as_posix()
    return args


def main() -> None:
    args = parse_args()
    pipeline = create_pipeline(args)
    print(pipeline.describe())
    _bypass_proxy_for_local_ray()
    with RayClient(include_dashboard=False, ray_temp_dir=args.ray_temp_dir):
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
