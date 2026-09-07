import json
from pathlib import Path

from nemo_curator.stages.video.io.video_manifest_reader import VideoManifestReaderStage
from nemo_curator.tasks import EmptyTask


def test_manifest_reader_explodes_video_paths_and_preserves_row(tmp_path: Path) -> None:
    first = tmp_path / "first.mp4"
    second = tmp_path / "second.mp4"
    manifest = tmp_path / "input.jsonl"
    manifest.write_text(
        json.dumps({"videos": [first.name, second.as_posix()], "text": "caption"}) + "\n",
        encoding="utf-8",
    )
    stage = VideoManifestReaderStage(input_manifest=manifest.as_posix())

    tasks = stage.process(EmptyTask())

    assert [task.data for task in tasks] == [[first.as_posix()], [second.as_posix()]]
    assert tasks[0]._metadata["manifest_row"]["text"] == "caption"
    assert stage.num_workers() == 1


def test_manifest_reader_honors_video_limit(tmp_path: Path) -> None:
    manifest = tmp_path / "input.jsonl"
    manifest.write_text(json.dumps({"videos": ["one.mp4", "two.mp4"]}) + "\n", encoding="utf-8")

    tasks = VideoManifestReaderStage(input_manifest=manifest.as_posix(), limit=1).process(EmptyTask())

    assert len(tasks) == 1
