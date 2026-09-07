import uuid

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
