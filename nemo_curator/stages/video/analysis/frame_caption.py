# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.

from __future__ import annotations

import random
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Literal

if TYPE_CHECKING:
    import numpy as np

    from nemo_curator.backends.base import WorkerMetadata

from nemo_curator.stages.base import ProcessingStage
from nemo_curator.stages.resources import Resources
from nemo_curator.tasks.video import Clip, VideoTask

from ._utils import ALL_KEYFRAMES, active_clips, chunks, filter_failed_clips


@dataclass
class VideoFrameCaptionStage(ProcessingStage[VideoTask, VideoTask]):
    """Generate and concatenate BLIP-2 captions for sampled clip frames."""

    hf_img2seq: str = ""
    caption_num: int = 1
    keep_candidate_mode: Literal["random_any", "all"] = "random_any"
    keep_original_sample: bool = False
    prompt: str | None = None
    prompt_key: str | None = None
    frame_sampling_method: Literal["uniform", "all_keyframes"] = "all_keyframes"
    frame_num: int = 3
    horizontal_flip: bool = False
    vertical_flip: bool = False
    trust_remote_code: bool = False
    frame_key: str = ALL_KEYFRAMES
    max_new_tokens: int = 128
    batch_size: int = 4
    resources: Resources = field(default_factory=lambda: Resources(cpus=1.0, gpu_memory_gb=20.0))
    name: str = "video_captioning_from_frames"

    def __post_init__(self) -> None:
        if not self.hf_img2seq:
            message = "hf_img2seq must be provided"
            raise ValueError(message)
        if self.caption_num <= 0:
            message = "caption_num must be positive"
            raise ValueError(message)
        if self.keep_candidate_mode not in ("random_any", "all"):
            message = f"Unsupported keep_candidate_mode: {self.keep_candidate_mode}"
            raise ValueError(message)
        if self.prompt_key is not None:
            message = "prompt_key is not available on native VideoTask; pass prompt instead"
            raise ValueError(message)

    def inputs(self) -> tuple[list[str], list[str]]:
        return ["data"], ["clips"]

    def outputs(self) -> tuple[list[str], list[str]]:
        return ["data"], ["clips"]

    def setup(self, worker_metadata: WorkerMetadata | None = None) -> None:  # noqa: ARG002
        import torch
        from transformers import AutoModelForImageTextToText, AutoProcessor

        self._torch = torch
        self._processor = AutoProcessor.from_pretrained(
            self.hf_img2seq,
            trust_remote_code=self.trust_remote_code,
            use_fast=False,
        )
        self._model = AutoModelForImageTextToText.from_pretrained(
            self.hf_img2seq,
            trust_remote_code=self.trust_remote_code,
            dtype=torch.bfloat16,
        ).to("cuda")
        self._model.eval()

    def process(self, task: VideoTask) -> VideoTask:
        return self.process_batch([task])[0]

    def process_batch(self, tasks: list[VideoTask]) -> list[VideoTask]:  # noqa: C901
        entries = active_clips(tasks)
        flat: list[tuple[Clip, int, np.ndarray]] = []
        failed: set[int] = set()
        errors: dict[int, str] = {}
        per_clip: dict[int, list[list[str]]] = {}
        for _, clip in entries:
            frames = clip.extracted_frames.get(self.frame_key)
            if frames is None or len(frames) == 0:
                failed.add(id(clip))
                errors[id(clip)] = f"missing frames: {self.frame_key}"
                continue
            per_clip[id(clip)] = [[] for _ in range(self.caption_num)]
            flat.extend((clip, index, frame) for index, frame in enumerate(frames))
        from PIL import Image, ImageOps

        for batch in chunks(flat, self.batch_size):
            try:
                images = [Image.fromarray(frame) for _, _, frame in batch]
                if self.horizontal_flip:
                    images = [ImageOps.mirror(image) for image in images]
                if self.vertical_flip:
                    images = [ImageOps.flip(image) for image in images]
                prompts = [self.prompt] * len(images) if self.prompt else None
                inputs = self._processor(text=prompts, images=images, return_tensors="pt").to(self._model.device)
                with self._torch.inference_mode():
                    generated = self._model.generate(
                        **inputs,
                        max_new_tokens=self.max_new_tokens,
                        do_sample=True,
                        num_return_sequences=self.caption_num,
                    )
                decoded = self._processor.batch_decode(generated, skip_special_tokens=True)
                for frame_index, (clip, _, _) in enumerate(batch):
                    start = frame_index * self.caption_num
                    for candidate_index in range(self.caption_num):
                        per_clip[id(clip)][candidate_index].append(decoded[start + candidate_index].strip())
            except Exception as exc:  # noqa: BLE001
                message = f"{type(exc).__name__}: {exc}"
                for clip, _, _ in batch:
                    failed.add(id(clip))
                    errors[id(clip)] = message
        for _, clip in entries:
            clip.extracted_frames.pop(self.frame_key, None)
            if id(clip) in failed:
                continue
            candidates = [". ".join(frame_captions) for frame_captions in per_clip[id(clip)]]
            clip.frame_caption_candidates = candidates
            clip.frame_caption = (
                random.choice(candidates)  # noqa: S311 - matches Data-Juicer's non-security selection.
                if self.keep_candidate_mode == "random_any"
                else candidates[0]
            )
        filter_failed_clips(tasks, failed, self.name, lambda clip: errors[id(clip)])
        return tasks
