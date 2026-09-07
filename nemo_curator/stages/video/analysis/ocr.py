# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.

from dataclasses import dataclass, field

import numpy as np

from nemo_curator.backends.base import WorkerMetadata
from nemo_curator.stages.base import ProcessingStage
from nemo_curator.stages.resources import Resources
from nemo_curator.tasks.video import Clip, VideoTask

from ._utils import UNIFORM_3_FRAMES, active_clips, filter_failed_clips


def triangle_area(p1: list[float], p2: list[float], p3: list[float]) -> float:
    x1, y1 = p1
    x2, y2 = p2
    x3, y3 = p3
    return float(0.5 * abs(x1 * y2 + x2 * y3 + x3 * y1 - x2 * y1 - x3 * y2 - x1 * y3))


@dataclass
class VideoOcrAreaRatioFilterStage(ProcessingStage[VideoTask, VideoTask]):
    """Data-Juicer-compatible EasyOCR text-area filter."""

    model_storage_dir: str = ""
    min_area_ratio: float = 0.001
    max_area_ratio: float = 1.0
    frame_sample_num: int = 3
    languages_to_detect: list[str] = field(default_factory=lambda: ["ch_sim", "en"])
    any_or_all: str = "all"
    frame_key: str = UNIFORM_3_FRAMES
    batch_size: int = 4
    resources: Resources = field(default_factory=lambda: Resources(cpus=1.0, gpu_memory_gb=4.0))
    name: str = "video_ocr_area_ratio_filter"

    def __post_init__(self) -> None:
        if self.any_or_all not in ("any", "all"):
            message = f"Unsupported any_or_all: {self.any_or_all}"
            raise ValueError(message)

    def inputs(self) -> tuple[list[str], list[str]]:
        return ["data"], ["clips"]

    def outputs(self) -> tuple[list[str], list[str]]:
        return ["data"], ["clips"]

    def setup(self, worker_metadata: WorkerMetadata | None = None) -> None:  # noqa: ARG002
        import easyocr

        self._reader = easyocr.Reader(
            self.languages_to_detect,
            gpu=False,
            recognizer=False,
            verbose=False,
            model_storage_directory=self.model_storage_dir or None,
        )
        self._reader.device = "cuda"
        self._reader.detector = self._reader.detector.to("cuda")

    def process(self, task: VideoTask) -> VideoTask:
        return self.process_batch([task])[0]

    def process_batch(self, tasks: list[VideoTask]) -> list[VideoTask]:
        failed: set[int] = set()
        errors: dict[int, str] = {}
        for _, clip in active_clips(tasks):
            frames = clip.extracted_frames.get(self.frame_key)
            try:
                self._process_clip(clip, frames)
            except Exception as exc:  # noqa: BLE001
                failed.add(id(clip))
                errors[id(clip)] = f"{type(exc).__name__}: {exc}"
            finally:
                clip.extracted_frames.pop(self.frame_key, None)
        filter_failed_clips(tasks, failed, self.name, lambda clip: errors[id(clip)], "num_filtered_by_ocr")
        return tasks

    def _process_clip(self, clip: Clip, frames: np.ndarray | None) -> None:
        if frames is None or len(frames) == 0:
            message = f"missing frames: {self.frame_key}"
            raise ValueError(message)
        ratios = [self._frame_ratio(frame) for frame in frames]
        clip.ocr_area_ratio = float(np.mean(ratios))
        if not self.min_area_ratio <= clip.ocr_area_ratio <= self.max_area_ratio:
            message = f"ratio {clip.ocr_area_ratio} outside [{self.min_area_ratio}, {self.max_area_ratio}]"
            raise ValueError(message)

    def _frame_ratio(self, frame: np.ndarray) -> float:
        horizontal_lists, free_lists = self._reader.detect(frame)
        height, width = frame.shape[:2]
        rectangle_area = sum(
            (xmax - xmin) * (ymax - ymin)
            for xmin, xmax, ymin, ymax in horizontal_lists[0]
            if xmax >= xmin and ymax >= ymin
        )
        quadrilateral_area = 0.0
        for points in free_lists[0]:
            quadrilateral_area += triangle_area(*points[:3])
            quadrilateral_area += triangle_area(*[*points[2:], points[0]])
        return float((rectangle_area + quadrilateral_area) / (width * height))
