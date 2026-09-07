# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.

from dataclasses import dataclass, field
from typing import Literal

import numpy as np
from loguru import logger

from nemo_curator.stages.base import ProcessingStage
from nemo_curator.stages.resources import Resources
from nemo_curator.tasks.video import Clip, VideoTask
from nemo_curator.utils.operation_utils import make_pipeline_named_temporary_file

from ._utils import filter_failed_clips


@dataclass
class ClipFrameSamplingStage(ProcessingStage[VideoTask, VideoTask]):
    """Decode an exact number of uniform frames or every encoded keyframe."""

    frame_key: str = "video_analysis:uniform:3"
    sampling_method: Literal["uniform", "all_keyframes"] = "uniform"
    frame_count: int = 3
    batch_size: int = 4
    resources: Resources = field(default_factory=lambda: Resources(cpus=1.0))
    name: str = "clip_frame_sampling"

    def __post_init__(self) -> None:
        if self.sampling_method not in ("uniform", "all_keyframes"):
            message = f"Unsupported sampling method: {self.sampling_method}"
            raise ValueError(message)
        if self.frame_count <= 0:
            message = "frame_count must be positive"
            raise ValueError(message)

    def inputs(self) -> tuple[list[str], list[str]]:
        return ["data"], ["clips"]

    def outputs(self) -> tuple[list[str], list[str]]:
        return ["data"], ["clips"]

    def process(self, task: VideoTask) -> VideoTask:
        failed: set[int] = set()
        errors: dict[int, str] = {}
        for clip in task.data.clips:
            try:
                frames = self._sample_checked(clip)
                clip.extracted_frames[self.frame_key] = frames
            except Exception as exc:  # noqa: BLE001
                logger.warning(f"Failed to sample frames for clip {clip.uuid}: {exc}")
                failed.add(id(clip))
                errors[id(clip)] = f"{type(exc).__name__}: {exc}"
        filter_failed_clips([task], failed, f"frames:{self.frame_key}", lambda clip: errors[id(clip)])
        return task

    def _sample_checked(self, clip: Clip) -> np.ndarray:
        if not clip.buffer:
            message = "clip buffer is empty"
            raise ValueError(message)
        frames = self._sample(clip)
        if len(frames) == 0:
            message = "decoder returned no frames"
            raise ValueError(message)
        return frames

    def _sample(self, clip: Clip) -> np.ndarray:
        with make_pipeline_named_temporary_file(sub_dir="video_analysis", suffix=".mp4") as path:
            path.write_bytes(clip.buffer or b"")
            if self.sampling_method == "uniform":
                return self._sample_uniform(path.as_posix())
            return self._sample_keyframes(path.as_posix())

    def _sample_uniform(self, path: str) -> np.ndarray:
        import cv2

        capture = cv2.VideoCapture(path)
        try:
            total = int(capture.get(cv2.CAP_PROP_FRAME_COUNT))
            if total <= 0:
                return np.empty((0, 0, 0, 3), dtype=np.uint8)
            count = min(self.frame_count, total)
            indices = [total // 2] if count == 1 else np.linspace(0, total - 1, count, dtype=int).tolist()
            frames = []
            for index in indices:
                capture.set(cv2.CAP_PROP_POS_FRAMES, index)
                ok, frame = capture.read()
                if not ok:
                    message = f"failed to decode frame {index}"
                    raise RuntimeError(message)
                frames.append(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
            return np.stack(frames)
        finally:
            capture.release()

    @staticmethod
    def _sample_keyframes(path: str) -> np.ndarray:
        import av

        frames = []
        with av.open(path) as container:
            stream = container.streams.video[0]
            stream.codec_context.skip_frame = "NONKEY"
            frames.extend(frame.to_ndarray(format="rgb24") for frame in container.decode(stream))
        if frames:
            return np.stack(frames)
        with av.open(path) as container:
            stream = container.streams.video[0]
            first = next(container.decode(stream), None)
            if first is not None:
                return np.stack([first.to_ndarray(format="rgb24")])
        return np.empty((0, 0, 0, 3), dtype=np.uint8)
