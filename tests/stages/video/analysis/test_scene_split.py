import os
import signal
import uuid

import pytest

from nemo_curator.stages.video.analysis.scene_split import VideoSceneSplitStage


def test_scene_split_creates_curator_clip_spans(video_task, monkeypatch):
    stage = VideoSceneSplitStage(max_scene_num=3)
    scenes = [(0.0, 1.0, 0, 30), (1.0, 2.0, 30, 60)]
    monkeypatch.setattr(stage, "_detect", lambda _path: (scenes, None))

    stage.process(video_task)

    assert [clip.span for clip in video_task.data.clips] == [(0.0, 1.0), (1.0, 2.0)]
    expected = uuid.uuid5(uuid.NAMESPACE_URL, f"{video_task.data.input_path}_0_30")
    assert video_task.data.clips[0].uuid == expected
    assert video_task.data.num_total_clips == 2


def test_single_scene_preserves_full_video(video_task, monkeypatch):
    stage = VideoSceneSplitStage()
    monkeypatch.setattr(stage, "_detect", lambda _path: ([(0.1, 1.9, 3, 57)], None))

    stage.process(video_task)

    assert video_task.data.clips[0].span == (0.0, 2.0)


@pytest.fixture
def real_scene_video(tmp_path):
    cv2 = pytest.importorskip("cv2")
    np = pytest.importorskip("numpy")
    pytest.importorskip("scenedetect")
    path = tmp_path / "scenes.mp4"
    writer = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*"mp4v"), 24, (64, 64))
    assert writer.isOpened()
    for value in (0, 255, 0):
        for _ in range(24):
            writer.write(np.full((64, 64, 3), value, dtype=np.uint8))
    writer.release()
    return str(path)


def test_reused_detector_matches_fresh_process_and_recovers_after_bad_input(real_scene_video, tmp_path):
    expected, error = VideoSceneSplitStage()._detect(real_scene_video)
    assert error is None
    assert len(expected) == 3
    stage = VideoSceneSplitStage(reuse_detector_process=True)
    try:
        assert stage._detect(real_scene_video) == (expected, None)
        pid = stage._detector_process.pid
        assert stage._detect(str(tmp_path / "missing.mp4"))[1] is not None
        assert stage._detect(real_scene_video) == (expected, None)
        assert stage._detector_process.pid == pid
    finally:
        stage.teardown()
    assert stage._detector_process is None


@pytest.mark.skipif(not hasattr(signal, "SIGSTOP"), reason="requires POSIX signals")
def test_reused_detector_timeout_kills_child_and_next_video_restarts(real_scene_video):
    stage = VideoSceneSplitStage(reuse_detector_process=True)
    try:
        expected = stage._detect(real_scene_video)
        pid = stage._detector_process.pid
        os.kill(pid, signal.SIGSTOP)
        stage.meta_time_out = 0.1
        assert stage._detect(real_scene_video) == ([], "timeout")
        assert stage._detector_process is None
        with pytest.raises(ProcessLookupError):
            os.kill(pid, 0)
        stage.meta_time_out = 60
        assert stage._detect(real_scene_video) == expected
        assert stage._detector_process.pid != pid
    finally:
        stage.teardown()
