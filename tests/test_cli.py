"""Tests for the public command-line behavior."""

from __future__ import annotations

import json
import tempfile
import unittest
from contextlib import redirect_stderr, redirect_stdout
from io import StringIO
from pathlib import Path

from src.cli import run


class CliTests(unittest.TestCase):
    """Exercise successful and failed CLI validation flows."""

    def _write_json(self, directory: Path, name: str, value: object) -> Path:
        """Write a JSON fixture and return its path."""
        path = directory / name
        path.write_text(json.dumps(value), encoding="utf-8")
        return path

    def test_run_accepts_valid_inputs(self) -> None:
        """Valid documents produce a clear success summary."""
        functions = [
            {
                "name": "fn_ping",
                "description": "Ping.",
                "parameters": {},
                "returns": {"type": "string"},
            }
        ]
        prompts = [{"prompt": "Ping the server"}]
        with tempfile.TemporaryDirectory() as temporary:
            directory = Path(temporary)
            functions_path = self._write_json(directory, "functions.json", functions)
            prompts_path = self._write_json(directory, "prompts.json", prompts)
            output = StringIO()
            with redirect_stdout(output):
                status = run(
                    [
                        "--functions_definition",
                        str(functions_path),
                        "--input",
                        str(prompts_path),
                    ]
                )
        self.assertEqual(status, 0)
        self.assertIn("Inputs validated: 1 function(s), 1 prompt(s).", output.getvalue())

    def test_run_reports_missing_file(self) -> None:
        """A missing input returns a controlled error instead of crashing."""
        error = StringIO()
        with redirect_stderr(error):
            status = run(["--functions_definition", "missing.json"])
        self.assertEqual(status, 2)
        self.assertIn("file not found", error.getvalue())
