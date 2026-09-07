from nemo_curator.stages.video.analysis._utils import filter_failed_clips


def test_filter_failed_clips_moves_clip_and_increments_counter(video_task):
    clip = video_task.data.clips[0]

    filter_failed_clips(
        [video_task],
        {id(clip)},
        "test",
        "failed",
        "num_filtered_by_ocr",
    )

    assert video_task.data.clips == []
    assert video_task.data.filtered_clips == [clip]
    assert clip.errors["test"] == "failed"
    assert video_task.data.clip_stats.num_filtered_by_ocr == 1
