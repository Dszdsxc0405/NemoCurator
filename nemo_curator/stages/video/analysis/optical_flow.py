# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.

from dataclasses import dataclass, field
from pathlib import Path
from typing import Literal

import numpy as np

from nemo_curator.backends.base import WorkerMetadata
from nemo_curator.stages.base import ProcessingStage
from nemo_curator.stages.resources import Resources
from nemo_curator.tasks.video import Clip, VideoTask

from ._utils import UNIFORM_3_FRAMES, active_clips, chunks, filter_failed_clips

_DEFAULT_FPS = 24.0
_MAX_VALID_FPS = 240.0


@dataclass
class VideoOpticalFlowFilterStage(ProcessingStage[VideoTask, VideoTask]):
    """Data-Juicer-compatible UniMatch optical-flow filter."""

    optflow_model_path: str = ""
    min_score: float = 0.15
    max_score: float = 20.0
    frame_sampling_method: Literal["uniform", "all_keyframes"] = "uniform"
    frame_num: int = 3
    stride: int = 2
    scale_by_fps: bool = True
    any_or_all: Literal["any", "all"] = "any"
    reduce_mode: Literal["avg", "max", "min"] = "avg"
    frame_key: str = UNIFORM_3_FRAMES
    batch_size: int = 4
    resources: Resources = field(default_factory=lambda: Resources(cpus=1.0, gpu_memory_gb=12.0))
    name: str = "video_optical_flow_filter"

    def __post_init__(self) -> None:
        if not self.optflow_model_path:
            message = "optflow_model_path must be provided"
            raise ValueError(message)
        if self.stride <= 0:
            message = "stride must be positive"
            raise ValueError(message)
        if self.reduce_mode not in ("avg", "max", "min"):
            message = f"Unsupported reduce_mode: {self.reduce_mode}"
            raise ValueError(message)

    def inputs(self) -> tuple[list[str], list[str]]:
        return ["data"], ["clips"]

    def outputs(self) -> tuple[list[str], list[str]]:
        return ["data"], ["clips"]

    def setup(self, worker_metadata: WorkerMetadata | None = None) -> None:  # noqa: ARG002
        import torch

        from nemo_curator.models.unimatch.unimatch import UniMatch

        self._torch = torch
        self._model = UniMatch(
            feature_channels=128,
            num_scales=2,
            upsample_factor=4,
            num_head=1,
            ffn_dim_expansion=4,
            num_transformer_layers=6,
            reg_refine=True,
            task="flow",
        )
        checkpoint = torch.load(Path(self.optflow_model_path), map_location="cpu", weights_only=False)
        self._model.load_state_dict(checkpoint["model"])
        self._model.to("cuda").eval()

    def process(self, task: VideoTask) -> VideoTask:
        return self.process_batch([task])[0]

    def process_batch(self, tasks: list[VideoTask]) -> list[VideoTask]:
        for batch in chunks(tasks, self.batch_size):
            self._process_video_batch(batch)
        return tasks

    def _process_video_batch(self, tasks: list[VideoTask]) -> list[VideoTask]:  # noqa: C901
        entries = active_clips(tasks)
        failed: set[int] = set()
        errors: dict[int, str] = {}
        pairs: list[tuple[VideoTask, Clip, np.ndarray, np.ndarray]] = []
        scores: dict[int, list[float]] = {id(clip): [] for _, clip in entries}
        for task, clip in entries:
            frames = clip.extracted_frames.get(self.frame_key)
            if frames is None or len(frames) <= 1:
                failed.add(id(clip))
                errors[id(clip)] = f"not enough frames: {self.frame_key}"
                continue
            pairs.extend(
                (task, clip, frames[index], frames[index + 1]) for index in range(0, len(frames) - 1, self.stride)
            )
        if pairs:
            batch = pairs
            try:
                batch_scores = self._infer_pairs([(first, second) for _, _, first, second in batch])
                for (task, clip, _, _), score in zip(batch, batch_scores, strict=True):
                    fps = task.data.metadata.framerate or _DEFAULT_FPS
                    scaled_score = score
                    if self.scale_by_fps and 1 <= fps <= _MAX_VALID_FPS:
                        scaled_score *= fps / _DEFAULT_FPS
                    scores[id(clip)].append(scaled_score)
            except Exception as exc:  # noqa: BLE001
                message = f"{type(exc).__name__}: {exc}"
                for _, clip, _, _ in batch:
                    failed.add(id(clip))
                    errors[id(clip)] = message
        reducer = {"avg": np.mean, "max": np.max, "min": np.min}[self.reduce_mode]
        for _, clip in entries:
            if id(clip) in failed:
                continue
            clip.optical_flow_score = float(reducer(scores[id(clip)])) if scores[id(clip)] else 0.0
            if not self.min_score <= clip.optical_flow_score <= self.max_score:
                failed.add(id(clip))
                errors[id(clip)] = f"score {clip.optical_flow_score} outside [{self.min_score}, {self.max_score}]"
        filter_failed_clips(
            tasks,
            failed,
            self.name,
            lambda clip: errors[id(clip)],
            "num_filtered_by_optical_flow",
        )
        return tasks

    def _infer_pairs(self, pairs: list[tuple[np.ndarray, np.ndarray]]) -> list[float]:
        from torch.nn import functional

        def prepare(frame: np.ndarray):  # noqa: ANN202
            tensor = self._torch.from_numpy(np.array(frame, copy=True)).permute(2, 0, 1).cuda()
            if tensor.shape[-2] > tensor.shape[-1]:
                tensor = tensor.permute(0, 2, 1)
            return functional.interpolate(
                tensor.unsqueeze(0).float(), size=(320, 576), mode="bilinear", align_corners=True
            )[0].to(self._torch.uint8)

        first = self._torch.stack([prepare(a) for a, _ in pairs])
        second = self._torch.stack([prepare(b) for _, b in pairs])
        with self._torch.inference_mode():
            output = self._model(
                first,
                second,
                attn_type="swin",
                attn_splits_list=[2, 8],
                corr_radius_list=[-1, 4],
                prop_radius_list=[-1, 1],
                num_reg_refine=6,
                task="flow",
                pred_bidir_flow=False,
            )
        return output["flow_preds"][-1].abs().float().mean(dim=(1, 2, 3)).cpu().tolist()
