"""CLI tests for RAG commands."""

from __future__ import annotations

from unittest.mock import MagicMock, patch

from typer.testing import CliRunner

from aiforge.cli.commands_rag import qa_app
from aiforge.modules.qa.schemas import IngestResponse

runner = CliRunner()


def _ingest_response() -> IngestResponse:
    return IngestResponse(
        chunks_indexed=1,
        files_processed=1,
        languages={"py": 1},
        skipped_files=[],
        collection="test",
    )


def test_qa_ingest_defaults_to_full_mode() -> None:
    with patch("aiforge.DocQA") as MockQA:
        qa = MagicMock()
        qa.ingest.return_value = _ingest_response()
        qa.ingest_incremental.return_value = _ingest_response()
        MockQA.return_value = qa

        result = runner.invoke(qa_app, ["ingest", ".", "--collection", "test"])

    assert result.exit_code == 0
    qa.ingest.assert_called_once()
    qa.ingest_incremental.assert_not_called()


def test_qa_ingest_incremental_mode() -> None:
    with patch("aiforge.DocQA") as MockQA:
        qa = MagicMock()
        qa.ingest.return_value = _ingest_response()
        qa.ingest_incremental.return_value = _ingest_response()
        MockQA.return_value = qa

        result = runner.invoke(
            qa_app,
            ["ingest", ".", "--collection", "test", "--mode", "incremental"],
        )

    assert result.exit_code == 0
    qa.ingest_incremental.assert_called_once()
    qa.ingest.assert_not_called()
