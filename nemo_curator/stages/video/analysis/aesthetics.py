# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.

from dataclasses import dataclass, field
from typing import Literal

import numpy as np

from nemo_curator.backends.base import WorkerMetadata
from nemo_curator.stages.base import ProcessingStage
from nemo_curator.stages.resources import Resources
from nemo_curator.tasks.video import Clip, VideoTask

from ._utils import UNIFORM_3_FRAMES, active_clips, chunks, filter_failed_clips


@dataclass
class VideoAestheticsFilterStage(ProcessingStage[VideoTask, VideoTask]):
    """Data-Juicer-compatible simple aesthetics filter for Curator clips."""

    hf_scorer_model: str = ""
    min_score: float = 0.4
    max_score: float = 1.0
    frame_sampling_method: Literal["uniform", "all_keyframes"] = "uniform"
    frame_num: int = 3
    reduce_mode: Literal["avg", "max", "min"] = "avg"
    any_or_all: Literal["any", "all"] = "any"
    trust_remote_code: bool = False
    frame_key: str = UNIFORM_3_FRAMES
    batch_size: int = 4
    resources: Resources = field(default_factory=lambda: Resources(cpus=1.0, gpu_memory_gb=4.0))
    name: str = "video_aesthetics_filter"

    def __post_init__(self) -> None:
        if not self.hf_scorer_model:
            message = "hf_scorer_model must be provided"
            raise ValueError(message)
        if self.reduce_mode not in ("avg", "max", "min"):
            message = f"Unsupported reduce_mode: {self.reduce_mode}"
            raise ValueError(message)
        if self.any_or_all not in ("any", "all"):
            message = f"Unsupported any_or_all: {self.any_or_all}"
            raise ValueError(message)
        self._normalize = "shunk031/aesthetics-predictor" in self.hf_scorer_model

    def inputs(self) -> tuple[list[str], list[str]]:
        return ["data"], ["clips"]

    def outputs(self) -> tuple[list[str], list[str]]:
        return ["data"], ["clips"]

    def setup(self, worker_metadata: WorkerMetadata | None = None) -> None:  # noqa: ARG002
        import torch
        from aesthetics_predictor import AestheticsPredictorV1, AestheticsPredictorV2Linear, AestheticsPredictorV2ReLU
        from transformers import CLIPProcessor

        self._torch = torch
        self._processor = CLIPProcessor.from_pretrained(self.hf_scorer_model, trust_remote_code=self.trust_remote_code)
        if "v1" in self.hf_scorer_model:
            model_class = AestheticsPredictorV1
        elif "v2" in self.hf_scorer_model and "linear" in self.hf_scorer_model:
            model_class = AestheticsPredictorV2Linear
        elif "v2" in self.hf_scorer_model and "relu" in self.hf_scorer_model:
            model_class = AestheticsPredictorV2ReLU
        else:
            message = f"Unsupported aesthetics predictor: {self.hf_scorer_model}"
            raise ValueError(message)
        self._model = model_class.from_pretrained(self.hf_scorer_model, trust_remote_code=self.trust_remote_code).to(
            "cuda"
        )
        self._model.eval()

    def process(self, task: VideoTask) -> VideoTask:
        return self.process_batch([task])[0]

    def process_batch(self, tasks: list[VideoTask]) -> list[VideoTask]:  # noqa: C901
        entries = active_clips(tasks)
        failed: set[int] = set()
        errors: dict[int, str] = {}
        frame_scores: dict[int, list[float]] = {id(clip): [] for _, clip in entries}
        flat: list[tuple[Clip, np.ndarray]] = []
        for _, clip in entries:
            frames = clip.extracted_frames.get(self.frame_key)
            if frames is None or len(frames) == 0:
                failed.add(id(clip))
                errors[id(clip)] = f"missing frames: {self.frame_key}"
            else:
                flat.extend((clip, frame) for frame in frames)
        from PIL import Image

        for batch in chunks(flat, self.batch_size):
            try:
                images = [Image.fromarray(frame) for _, frame in batch]
                inputs = self._processor(images=images, return_tensors="pt").to(self._model.device)
                with self._torch.inference_mode():
                    logits = self._model(**inputs).logits.detach().float().cpu().reshape(-1).numpy()
                if self._normalize:
                    logits = logits / 10.0
                for (clip, _), score in zip(batch, logits, strict=True):
                    frame_scores[id(clip)].append(float(score))
            except Exception as exc:  # noqa: BLE001
                message = f"{type(exc).__name__}: {exc}"
                for clip, _ in batch:
                    failed.add(id(clip))
                    errors[id(clip)] = message
        reducer = {"avg": np.mean, "max": np.max, "min": np.min}[self.reduce_mode]
        for _, clip in entries:
            if id(clip) in failed:
                continue
            clip.aesthetic_score = float(reducer(frame_scores[id(clip)]))
            if not self.min_score <= clip.aesthetic_score <= self.max_score:
                failed.add(id(clip))
                errors[id(clip)] = f"score {clip.aesthetic_score} outside [{self.min_score}, {self.max_score}]"
        filter_failed_clips(
            tasks,
            failed,
            self.name,
            lambda clip: errors[id(clip)],
            "num_filtered_by_aesthetic",
        )
        return tasks
