# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.

from copy import deepcopy

import numpy as np

from nemo_curator.stages.video.analysis._utils import UNIFORM_3_FRAMES
from nemo_curator.stages.video.analysis.optical_flow import VideoOpticalFlowFilterStage


def test_optical_flow_preserves_stride_and_fps_scaling(video_task, monkeypatch):
    clip = video_task.data.clips[0]
    clip.extracted_frames[UNIFORM_3_FRAMES] = np.zeros((3, 2, 2, 3), dtype=np.uint8)
    video_task.data.metadata.framerate = 48.0
    stage = VideoOpticalFlowFilterStage(optflow_model_path="weights.pth", stride=2)
    calls = []

    def infer(pairs):
        calls.append(pairs)
        return [0.25]

    monkeypatch.setattr(stage, "_infer_pairs", infer)
    stage.process(video_task)

    assert len(calls) == 1
    assert len(calls[0]) == 1
    assert clip.optical_flow_score == 0.5


def test_batch_size_counts_videos_not_frame_pairs(video_task, monkeypatch):
    clip = video_task.data.clips[0]
    clip.extracted_frames[UNIFORM_3_FRAMES] = np.zeros((5, 2, 2, 3), dtype=np.uint8)
    other = deepcopy(video_task)
    stage = VideoOpticalFlowFilterStage(optflow_model_path="weights.pth", batch_size=1)
    sizes = []

    def infer(pairs):
        sizes.append(len(pairs))
        return [0.25] * len(pairs)

    monkeypatch.setattr(stage, "_infer_pairs", infer)
    stage.process_batch([video_task, other])
    assert sizes == [2, 2]
    assert video_task.data.clips[0].optical_flow_score == 0.25
