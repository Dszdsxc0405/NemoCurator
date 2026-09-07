import pathlib
import uuid

import pytest

from nemo_curator.tasks.video import Clip, Video, VideoMetadata, VideoTask


@pytest.fixture
def video_task(tmp_path: pathlib.Path) -> VideoTask:
    source = tmp_path / "source.mp4"
    source.write_bytes(b"video")
    clip = Clip(
        uuid=uuid.uuid4(),
        source_video=source.as_posix(),
        span=(0.0, 2.0),
        buffer=b"clip",
    )
    video = Video(
        input_video=source,
        source_bytes=b"video",
        metadata=VideoMetadata(duration=2.0, num_frames=60, framerate=24.0),
        clips=[clip],
    )
    return VideoTask(dataset_name="test", data=video)
