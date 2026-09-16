# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.

import numpy as np
import torch

from nemo_curator.stages.video.analysis._utils import ALL_KEYFRAMES
from nemo_curator.stages.video.analysis.frame_caption import VideoFrameCaptionStage


class _Inputs(dict):
    def to(self, _device):
        return self


class _Processor:
    def __call__(self, **kwargs):
        self.count = len(kwargs["images"])
        return _Inputs()

    def batch_decode(self, _generated, **_kwargs):
        return [f"frame {index}" for index in range(self.count)]


class _Model:
    device = "cpu"

    def generate(self, **_kwargs):
        return torch.arange(2)


def test_caption_concatenates_all_keyframes(video_task):
    clip = video_task.data.clips[0]
    clip.extracted_frames[ALL_KEYFRAMES] = np.zeros((2, 2, 2, 3), dtype=np.uint8)
    stage = VideoFrameCaptionStage(hf_img2seq="model", caption_num=1, batch_size=1)
    stage._torch = torch
    stage._processor = _Processor()
    stage._model = _Model()

    stage.process(video_task)

    assert clip.frame_caption_candidates == ["frame 0. frame 1"]
    assert clip.frame_caption == "frame 0. frame 1"
    assert ALL_KEYFRAMES not in clip.extracted_frames
