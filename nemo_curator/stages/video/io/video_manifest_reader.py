# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from loguru import logger

from nemo_curator.stages.base import ProcessingStage
from nemo_curator.stages.resources import Resources
from nemo_curator.tasks import EmptyTask
from nemo_curator.tasks.file_group import FileGroupTask


@dataclass
class VideoManifestReaderStage(ProcessingStage[EmptyTask, FileGroupTask]):
    """Read Data-Juicer JSONL video paths into native Curator file tasks."""

    input_manifest: str
    video_key: str = "videos"
    limit: int | None = None
    resources: Resources = field(default_factory=lambda: Resources(cpus=1.0))
    name: str = "video_manifest_reader"

    def inputs(self) -> tuple[list[str], list[str]]:
        return [], []

    def outputs(self) -> tuple[list[str], list[str]]:
        return [], []

    def num_workers(self) -> int | None:
        return 1

    def process(self, _: EmptyTask) -> list[FileGroupTask]:
        manifest = Path(self.input_manifest).expanduser().resolve()
        if not manifest.is_file():
            message = f"Video manifest does not exist: {manifest}"
            raise FileNotFoundError(message)

        tasks = []
        with manifest.open(encoding="utf-8") as stream:
            for line_number, raw_line in enumerate(stream, 1):
                if self.limit is not None and len(tasks) >= self.limit:
                    break
                line = raw_line.strip()
                if not line:
                    continue
                try:
                    row = json.loads(line)
                except json.JSONDecodeError as exc:
                    logger.warning(f"Skipping invalid JSON at {manifest}:{line_number}: {exc}")
                    continue
                paths = self._video_paths(row, manifest.parent)
                for path in paths:
                    if self.limit is not None and len(tasks) >= self.limit:
                        break
                    tasks.append(
                        FileGroupTask(
                            dataset_name=manifest.stem,
                            data=[path],
                            reader_config={},
                            _metadata={
                                "manifest_line": line_number,
                                "manifest_row": row,
                                "source_files": [path],
                            },
                        )
                    )
        logger.info(f"Loaded {len(tasks)} videos from {manifest}")
        return tasks

    def _video_paths(self, row: dict[str, Any], manifest_dir: Path) -> list[str]:
        values = row.get(self.video_key, [])
        if isinstance(values, str):
            values = [values]
        if not isinstance(values, list):
            return []
        paths = []
        for value in values:
            if not isinstance(value, str) or not value:
                continue
            path = Path(value).expanduser()
            paths.append((manifest_dir / path).resolve().as_posix() if not path.is_absolute() else path.as_posix())
        return paths
