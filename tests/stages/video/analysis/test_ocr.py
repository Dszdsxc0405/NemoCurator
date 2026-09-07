import numpy as np
import pytest

from nemo_curator.stages.video.analysis._utils import UNIFORM_3_FRAMES
from nemo_curator.stages.video.analysis.ocr import VideoOcrAreaRatioFilterStage


class _Reader:
    def detect(self, _frame):
        return [[(0, 5, 0, 2)]], [[]]


def test_ocr_computes_mean_area_and_consumes_shared_frames(video_task):
    clip = video_task.data.clips[0]
    clip.extracted_frames[UNIFORM_3_FRAMES] = np.zeros((3, 10, 10, 3), dtype=np.uint8)
    stage = VideoOcrAreaRatioFilterStage()
    stage._reader = _Reader()

    stage.process(video_task)

    assert clip.ocr_area_ratio == pytest.approx(0.1)
    assert UNIFORM_3_FRAMES not in clip.extracted_frames
