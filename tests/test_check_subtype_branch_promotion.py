from __future__ import annotations

from pathlib import Path

import pytest

from src.common.io import write_json
from scripts.check_subtype_branch_promotion import check_subtype_branch_promotion


def test_check_subtype_branch_promotion_fails_when_json_missing(tmp_path: Path) -> None:
    with pytest.raises(SystemExit, match="promotion json not found"):
        check_subtype_branch_promotion(
            promotion_json=tmp_path / "missing_promotion.json"
        )


def test_check_subtype_branch_promotion_fails_when_branch_not_promoted(tmp_path: Path) -> None:
    payload_path = tmp_path / "promotion.json"
    write_json(payload_path, {"promote": False})

    with pytest.raises(SystemExit, match="did not qualify for stage3 promotion"):
        check_subtype_branch_promotion(promotion_json=payload_path)


def test_check_subtype_branch_promotion_passes_when_branch_is_promoted(tmp_path: Path) -> None:
    payload_path = tmp_path / "promotion.json"
    write_json(payload_path, {"promote": True})

    payload = check_subtype_branch_promotion(promotion_json=payload_path)

    assert payload["promote"] is True
    assert payload["override_used"] is False


def test_check_subtype_branch_promotion_allow_override_passes(tmp_path: Path) -> None:
    payload_path = tmp_path / "promotion.json"
    write_json(payload_path, {"promote": False})

    payload = check_subtype_branch_promotion(
        promotion_json=payload_path,
        allow_unpromoted=True,
    )

    assert payload["promote"] is False
    assert payload["override_used"] is True
    assert payload["promotion_json_found"] is True
