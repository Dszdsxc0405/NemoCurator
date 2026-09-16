# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.

from dataclasses import dataclass, field

import numpy as np

from nemo_curator.backends.base import WorkerMetadata
from nemo_curator.stages.base import ProcessingStage
from nemo_curator.stages.resources import Resources
from nemo_curator.tasks.video import VideoTask

from ._utils import UNIFORM_16_FRAMES, active_clips, chunks, filter_failed_clips

_LABEL_THRESHOLD = 0.5


@dataclass
class VideoCameraMotionStage(ProcessingStage[VideoTask, VideoTask]):
    """Classify one globally sampled 16-frame sequence per clip."""

    hf_model_path: str = ""
    trust_remote_code: bool = False
    num_frames: int = 16
    max_duration: int = 8
    torch_dtype: str = "bfloat16"
    frame_key: str = UNIFORM_16_FRAMES
    batch_size: int = 4
    resources: Resources = field(default_factory=lambda: Resources(cpus=1.0, gpu_memory_gb=2.0))
    name: str = "video_captioning_camera_motion"

    def __post_init__(self) -> None:
        if not self.hf_model_path:
            message = "hf_model_path must be provided"
            raise ValueError(message)
        if self.num_frames <= 0:
            message = "num_frames must be positive"
            raise ValueError(message)

    def inputs(self) -> tuple[list[str], list[str]]:
        return ["data"], ["clips"]

    def outputs(self) -> tuple[list[str], list[str]]:
        return ["data"], ["clips"]

    def setup(self, worker_metadata: WorkerMetadata | None = None) -> None:  # noqa: ARG002
        import torch
        from transformers import VideoMAEForVideoClassification, VideoMAEImageProcessor

        dtype = {
            "fp32": torch.float32,
            "float32": torch.float32,
            "fp16": torch.float16,
            "float16": torch.float16,
            "bf16": torch.bfloat16,
            "bfloat16": torch.bfloat16,
        }.get(self.torch_dtype, torch.bfloat16)
        self._torch = torch
        self._processor = VideoMAEImageProcessor.from_pretrained(
            self.hf_model_path, trust_remote_code=self.trust_remote_code
        )
        self._model = VideoMAEForVideoClassification.from_pretrained(
            self.hf_model_path,
            trust_remote_code=self.trust_remote_code,
            attn_implementation="sdpa",
            dtype=dtype,
        ).to("cuda")
        self._model.eval()

    def process(self, task: VideoTask) -> VideoTask:
        return self.process_batch([task])[0]

    def process_batch(self, tasks: list[VideoTask]) -> list[VideoTask]:
        for batch in chunks(tasks, self.batch_size):
            self._process_video_batch(batch)
        return tasks

    def _process_video_batch(self, tasks: list[VideoTask]) -> list[VideoTask]:
        entries = active_clips(tasks)
        valid = []
        failed: set[int] = set()
        errors: dict[int, str] = {}
        for _, clip in entries:
            frames = clip.extracted_frames.pop(self.frame_key, None)
            if frames is None or len(frames) < self.num_frames:
                clip.camera_motion_labels = ["static"]
                clip.errors[self.name] = "insufficient_frames"
                continue
            valid.append((clip, frames[: self.num_frames]))
        if valid:
            batch = valid
            try:
                videos = [list(frames) for _, frames in batch]
                inputs = self._processor(videos, return_tensors="pt")
                pixel_values = inputs["pixel_values"].to(self._model.device, dtype=self._model.dtype)
                with self._torch.inference_mode():
                    logits = self._model(pixel_values).logits.float()
                predictions = (self._torch.sigmoid(logits) > _LABEL_THRESHOLD).cpu().numpy()
                for (clip, _), prediction in zip(batch, predictions, strict=True):
                    labels = [self._model.config.id2label[index] for index in np.flatnonzero(prediction)]
                    clip.camera_motion_labels = labels or ["static"]
            except Exception as exc:  # noqa: BLE001
                message = f"{type(exc).__name__}: {exc}"
                for clip, _ in batch:
                    failed.add(id(clip))
                    errors[id(clip)] = message
        filter_failed_clips(tasks, failed, self.name, lambda clip: errors[id(clip)])
        return tasks
