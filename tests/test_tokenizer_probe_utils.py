from __future__ import annotations

from pathlib import Path

from src.competition.tokenizer_probe_utils import TOKENIZER_CACHE_DIR


REPO_ROOT = Path(__file__).resolve().parents[1]


def test_probe_and_recheck_share_tokenizer_probe_utils() -> None:
    probe_text = (REPO_ROOT / "scripts" / "probe_chat_template.py").read_text(
        encoding="utf-8"
    )
    recheck_text = (REPO_ROOT / "scripts" / "recheck_chat_template_sha16.py").read_text(
        encoding="utf-8"
    )

    assert "from src.competition.tokenizer_probe_utils import" in probe_text
    assert "from src.competition.tokenizer_probe_utils import" in recheck_text
    assert "from scripts.probe_chat_template import" not in recheck_text


def test_tokenizer_probe_utils_default_cache_dir_matches_repo_contract() -> None:
    assert TOKENIZER_CACHE_DIR == Path("artifacts/_tokenizer_cache")
