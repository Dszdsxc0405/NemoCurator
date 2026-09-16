# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.

import uuid
from dataclasses import dataclass, field

from nemo_curator.stages.base import ProcessingStage
from nemo_curator.stages.resources import Resources
from nemo_curator.tasks.video import Clip, VideoTask


@dataclass
class WholeVideoClipStage(ProcessingStage[VideoTask, VideoTask]):
    """Expose an unsplit source video as one clip for clip-oriented stages."""

    batch_size: int = 1
    resources: Resources = field(default_factory=lambda: Resources(cpus=0.25))
    name: str = "whole_video_clip"

    def inputs(self) -> tuple[list[str], list[str]]:
        return ["data"], ["source_bytes", "metadata"]

    def outputs(self) -> tuple[list[str], list[str]]:
        return ["data"], ["clips"]

    def process(self, task: VideoTask) -> VideoTask:
        video = task.data
        if video.clips:
            return task
        if not video.source_bytes:
            video.errors[self.name] = "source_bytes is empty"
            return task
        if video.metadata.duration is None:
            video.errors[self.name] = "video duration is missing"
            return task

        end_marker = video.metadata.num_frames if video.metadata.num_frames is not None else video.metadata.duration
        clip = Clip(
            uuid=uuid.uuid5(uuid.NAMESPACE_URL, f"{video.input_path}_0_{end_marker}"),
            source_video=video.input_path,
            span=(0.0, video.metadata.duration),
            buffer=video.source_bytes,
        )
        video.clips = [clip]
        video.num_total_clips = 1
        video.source_bytes = None
        return task
