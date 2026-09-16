import uuid

from nemo_curator.stages.video.analysis.whole_video import WholeVideoClipStage


def test_whole_video_becomes_single_clip(video_task):
    video = video_task.data
    video.clips = []

    WholeVideoClipStage().process(video_task)

    assert len(video.clips) == 1
    assert video.clips[0].span == (0.0, 2.0)
    assert video.clips[0].buffer == b"video"
    assert video.clips[0].uuid == uuid.uuid5(uuid.NAMESPACE_URL, f"{video.input_path}_0_60")
    assert video.num_total_clips == 1
    assert video.source_bytes is None


def test_whole_video_records_missing_source(video_task):
    video_task.data.clips = []
    video_task.data.source_bytes = None

    WholeVideoClipStage().process(video_task)

    assert video_task.data.clips == []
    assert video_task.data.errors["whole_video_clip"] == "source_bytes is empty"
