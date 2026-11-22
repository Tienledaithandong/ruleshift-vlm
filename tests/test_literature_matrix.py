"""Structural checks on the source-grounded literature matrix."""

from __future__ import annotations

import csv
from pathlib import Path

MATRIX = Path(__file__).resolve().parents[1] / "research" / "literature_matrix.csv"

REQUIRED_COLUMNS = {
    "paper",
    "short_name",
    "year",
    "venue",
    "url",
    "code_url",
    "setting",
    "input_modality",
    "hidden_state",
    "budget",
    "planning_role",
    "oracle_privileges",
    "metrics",
    "strongest_baseline",
    "remaining_gap",
    "contrast_with_ruleshift",
    "threat_to_novelty",
    "read_status",
    "verification_source",
    "verified_on",
}

REQUIRED_PAPERS = {"DIVA", "ZendoWorld", "EVOLvE"}

CANONICAL_URLS = {
    "https://arxiv.org/abs/2610.32181",
    "https://arxiv.org/abs/2607.08233",
    "https://proceedings.mlr.press/v267/nie25b.html",
    "https://proceedings.neurips.cc/paper_files/paper/2024/hash/"
    "5f1b350fc0c2affd56f465faa36be343-Abstract-Conference.html",
    "https://proceedings.iclr.cc/paper_files/paper/2026/hash/"
    "a24d08f2eecc6c96f4a0ad37a74ddeca-Abstract-Conference.html",
}


def load_rows() -> list[dict[str, str]]:
    with MATRIX.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def test_matrix_has_required_columns() -> None:
    rows = load_rows()
    assert rows, "literature matrix is empty"
    assert REQUIRED_COLUMNS <= set(rows[0])


def test_matrix_covers_required_papers_and_urls() -> None:
    rows = load_rows()
    names = {row["short_name"] for row in rows}
    urls = {row["url"] for row in rows}
    assert REQUIRED_PAPERS <= names
    assert CANONICAL_URLS <= urls
    assert len(rows) >= 5


def test_every_row_is_flagged_as_not_fully_read() -> None:
    rows = load_rows()
    for row in rows:
        assert "FULLTEXT_NOT_READ" in row["read_status"]
        assert row["url"].startswith("https://")
        assert row["verification_source"], "missing verification source"
        assert row["year"].isdigit()
