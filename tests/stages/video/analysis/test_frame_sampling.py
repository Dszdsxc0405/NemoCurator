import numpy as np

from nemo_curator.stages.video.analysis.frame_sampling import ClipFrameSamplingStage


def test_sampling_stores_exact_frame_array(video_task, monkeypatch):
    frames = np.zeros((3, 4, 5, 3), dtype=np.uint8)
    stage = ClipFrameSamplingStage(frame_key="sample", frame_count=3)
    monkeypatch.setattr(stage, "_sample", lambda _clip: frames)

    result = stage.process(video_task)

    assert result.data.clips[0].extracted_frames["sample"] is frames


def test_sampling_failure_filters_clip(video_task, monkeypatch):
    stage = ClipFrameSamplingStage(frame_key="sample", frame_count=3)

    def fail(_clip):
        message = "decode failed"
        raise RuntimeError(message)

    monkeypatch.setattr(stage, "_sample", fail)
    stage.process(video_task)

    assert video_task.data.clips == []
    assert "decode failed" in video_task.data.filtered_clips[0].errors["frames:sample"]
