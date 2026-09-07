from types import SimpleNamespace

import numpy as np
import torch

from nemo_curator.stages.video.analysis._utils import UNIFORM_16_FRAMES
from nemo_curator.stages.video.analysis.camera_motion import VideoCameraMotionStage


class _Processor:
    def __call__(self, videos, **_kwargs):
        assert len(videos) == 1
        assert len(videos[0]) == 16
        return {"pixel_values": torch.zeros((1, 16, 3, 2, 2))}


class _Model:
    device = "cpu"
    dtype = torch.float32
    config = SimpleNamespace(id2label={0: "pan", 1: "zoom"})

    def __call__(self, _pixel_values):
        return SimpleNamespace(logits=torch.tensor([[1.0, -1.0]]))


def test_camera_motion_runs_one_global_sequence(video_task):
    clip = video_task.data.clips[0]
    clip.extracted_frames[UNIFORM_16_FRAMES] = np.zeros((16, 2, 2, 3), dtype=np.uint8)
    stage = VideoCameraMotionStage(hf_model_path="model", num_frames=16, max_duration=8)
    stage._torch = torch
    stage._processor = _Processor()
    stage._model = _Model()

    stage.process(video_task)

    assert clip.camera_motion_labels == ["pan"]
    assert UNIFORM_16_FRAMES not in clip.extracted_frames
