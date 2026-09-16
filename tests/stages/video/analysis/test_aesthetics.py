# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.

from types import SimpleNamespace

import numpy as np
import pytest
import torch

from nemo_curator.stages.video.analysis._utils import UNIFORM_3_FRAMES
from nemo_curator.stages.video.analysis.aesthetics import VideoAestheticsFilterStage


class _Inputs(dict):
    def to(self, _device):
        return self


class _Model:
    device = "cpu"

    def __call__(self, **_kwargs):
        return SimpleNamespace(logits=torch.tensor([0.3, 0.6, 0.9]))


def test_aesthetics_averages_frame_scores(video_task):
    clip = video_task.data.clips[0]
    clip.extracted_frames[UNIFORM_3_FRAMES] = np.zeros((3, 2, 2, 3), dtype=np.uint8)
    stage = VideoAestheticsFilterStage(hf_scorer_model="predictor-v2-linear", batch_size=1)
    stage._torch = torch
    stage._model = _Model()
    stage._processor = lambda **_kwargs: _Inputs()

    stage.process(video_task)

    assert clip.aesthetic_score == pytest.approx(0.6)
    assert video_task.data.clips == [clip]
