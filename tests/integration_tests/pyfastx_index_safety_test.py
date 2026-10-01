"""Isolate a native custom-index-path defect so it cannot abort the test runner."""

from __future__ import annotations

import subprocess
import sys

import pytest


@pytest.mark.xfail(
    strict=True, reason="Pyfastx 2.3.1 custom index path writes beyond its allocation"
)
def test_custom_index_path_lengths_do_not_corrupt_the_heap():
    code = """
import gc
import os
import tempfile
from pathlib import Path
import pyfastx
with tempfile.TemporaryDirectory(prefix="binette-index-repro-") as root:
    os.chdir(root)
    Path("input.fa").write_text(">contig\\nACGTACGT\\n")
    for length in (24, 40, 56, 72, 88, 104):
        result = pyfastx.Fasta("input.fa", index_file="x" * (length - 4) + ".fxi")
        del result
        gc.collect()
"""
    result = subprocess.run(
        [sys.executable, "-c", code], capture_output=True, text=True, timeout=30
    )
    assert result.returncode == 0, result.stderr
