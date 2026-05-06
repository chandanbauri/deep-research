from __future__ import annotations

import pytest
from pathlib import Path

from deep_research.tools.file_writer import FileWriterTool
from deep_research.tools.file_reader import FileReaderTool
from deep_research import config


@pytest.mark.asyncio
async def test_file_write_and_read(tmp_path, monkeypatch):
    monkeypatch.setattr(config, "OUTPUT_DIR", tmp_path)

    writer = FileWriterTool()
    reader = FileReaderTool()

    write_result = await writer.run(path="test/hello.txt", content="Hello, world!")
    assert write_result.success is True
    assert (tmp_path / "test" / "hello.txt").exists()

    read_result = await reader.run(path="test/hello.txt")
    assert read_result.success is True
    assert "Hello, world!" in read_result.data["content"]


@pytest.mark.asyncio
async def test_file_writer_blocks_path_traversal(tmp_path, monkeypatch):
    monkeypatch.setattr(config, "OUTPUT_DIR", tmp_path)
    writer = FileWriterTool()
    result = await writer.run(path="../../etc/passwd", content="bad")
    assert result.success is False
    assert "traversal" in result.error.lower()


@pytest.mark.asyncio
async def test_file_reader_missing_file(tmp_path, monkeypatch):
    monkeypatch.setattr(config, "OUTPUT_DIR", tmp_path)
    reader = FileReaderTool()
    result = await reader.run(path="nonexistent.txt")
    assert result.success is False
    assert "not found" in result.error.lower()
