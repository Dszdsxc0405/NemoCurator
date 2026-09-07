# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.

import multiprocessing as mp
import uuid
from dataclasses import dataclass, field
from multiprocessing.queues import Queue
from queue import Empty
from typing import Any

from nemo_curator.stages.base import ProcessingStage
from nemo_curator.stages.resources import Resources
from nemo_curator.tasks.video import Clip, VideoTask
from nemo_curator.utils.operation_utils import make_pipeline_named_temporary_file


@dataclass
class _SceneDetectionConfig:
    detector_name: str
    threshold: float
    min_scene_len: int
    detector_kwargs: dict[str, Any]
    show_progress: bool


def _detect_scenes_worker(
    queue: Queue,
    path: str,
    config: _SceneDetectionConfig,
) -> None:
    try:
        import scenedetect

        detector_class = getattr(scenedetect.detectors, config.detector_name)
        detector = detector_class(config.threshold, config.min_scene_len, **config.detector_kwargs)
        scenes = scenedetect.detect(path, detector, show_progress=config.show_progress, start_in_scene=True) or []
        queue.put(
            (
                "ok",
                [
                    (start.get_seconds(), end.get_seconds(), start.get_frames(), end.get_frames())
                    for start, end in scenes
                ],
            )
        )
    except BaseException as exc:  # noqa: BLE001
        queue.put(("error", f"{type(exc).__name__}: {exc}"))


@dataclass
class VideoSceneSplitStage(ProcessingStage[VideoTask, VideoTask]):
    """Detect scene spans while leaving clip transcoding to Curator."""

    detector: str = "ContentDetector"
    threshold: float = 27.0
    min_scene_len: int = 10
    show_progress: bool = False
    max_scene_num: int = 3
    meta_time_out: int = 60
    detector_kwargs: dict[str, Any] = field(default_factory=dict)
    batch_size: int = 4
    resources: Resources = field(default_factory=lambda: Resources(cpus=1.0))
    name: str = "video_scene_split"

    def __post_init__(self) -> None:
        if self.detector not in ("ContentDetector", "ThresholdDetector", "AdaptiveDetector"):
            message = f"Unsupported scene detector: {self.detector}"
            raise ValueError(message)
        if self.max_scene_num <= 0:
            message = "max_scene_num must be positive"
            raise ValueError(message)

    def inputs(self) -> tuple[list[str], list[str]]:
        return ["data"], ["source_bytes"]

    def outputs(self) -> tuple[list[str], list[str]]:
        return ["data"], ["clips"]

    def process(self, task: VideoTask) -> VideoTask:
        video = task.data
        video.clips = []
        if not video.source_bytes:
            video.errors[self.name] = "source_bytes is empty"
            return task
        try:
            with make_pipeline_named_temporary_file(sub_dir="scene_split", suffix=".mp4") as path:
                path.write_bytes(video.source_bytes)
                scenes, error = self._detect(path.as_posix())
        except Exception as exc:  # noqa: BLE001
            video.errors[self.name] = f"{type(exc).__name__}: {exc}"
            return task
        if error is not None:
            video.errors[self.name] = error
            return task

        if len(scenes) <= 1:
            if video.metadata.duration is None or video.metadata.num_frames is None:
                video.errors[self.name] = "video duration or frame count is missing"
                return task
            scenes = [(0.0, video.metadata.duration, 0, video.metadata.num_frames)]
        for start_s, end_s, start_frame, end_frame in scenes[: self.max_scene_num]:
            clip_id = uuid.uuid5(uuid.NAMESPACE_URL, f"{video.input_path}_{start_frame}_{end_frame}")
            video.clips.append(Clip(uuid=clip_id, source_video=video.input_path, span=(start_s, end_s)))
        video.num_total_clips = len(video.clips)
        return task

    def _detect(self, path: str) -> tuple[list[tuple[float, float, int, int]], str | None]:
        ctx = mp.get_context("spawn")
        queue = ctx.Queue(maxsize=1)
        process = ctx.Process(
            target=_detect_scenes_worker,
            args=(
                queue,
                path,
                _SceneDetectionConfig(
                    detector_name=self.detector,
                    threshold=self.threshold,
                    min_scene_len=self.min_scene_len,
                    detector_kwargs=self.detector_kwargs,
                    show_progress=self.show_progress,
                ),
            ),
        )
        process.start()
        process.join(self.meta_time_out)
        if process.is_alive():
            process.terminate()
            process.join(1)
            if process.is_alive():
                process.kill()
                process.join()
            return [], "timeout"
        try:
            status, payload = queue.get(timeout=0.2)
        except Empty:
            return [], "scene detector returned no result"
        return (payload, None) if status == "ok" else ([], payload)
