# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.

from collections.abc import Callable, Iterable
from typing import TypeVar

from nemo_curator.tasks.video import Clip, VideoTask

UNIFORM_3_FRAMES = "video_analysis:uniform:3"
ALL_KEYFRAMES = "video_analysis:all_keyframes"
UNIFORM_16_FRAMES = "video_analysis:uniform:16"

T = TypeVar("T")


def chunks(values: list[T], size: int) -> Iterable[list[T]]:
    for index in range(0, len(values), size):
        yield values[index : index + size]


def active_clips(tasks: list[VideoTask]) -> list[tuple[VideoTask, Clip]]:
    return [(task, clip) for task in tasks for clip in task.data.clips]


def filter_failed_clips(
    tasks: list[VideoTask],
    failed: set[int],
    error_key: str,
    error: str | Callable[[Clip], str],
    counter: str | None = None,
) -> None:
    for task in tasks:
        passed = []
        for clip in task.data.clips:
            if id(clip) not in failed:
                passed.append(clip)
                continue
            clip.errors[error_key] = error(clip) if callable(error) else error
            task.data.filtered_clips.append(clip)
            if counter is not None:
                setattr(task.data.clip_stats, counter, getattr(task.data.clip_stats, counter) + 1)
        task.data.clips = passed
