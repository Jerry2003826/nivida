from __future__ import annotations

import json
from pathlib import Path

import pytest


class _FakeOutput:
    def __init__(self, text: str, token_ids: list[int]) -> None:
        self.text = text
        self.token_ids = token_ids


class _FakeRequest:
    def __init__(self, outputs) -> None:
        self.outputs = outputs


class FakeLLM:
    def __init__(self, responses: list[tuple[str, list[int]]] | None = None, *args, **kwargs) -> None:
        self._kwargs = kwargs
        self._responses = responses or [(r"Reasoning... \boxed{42}", list(range(50)))]

    def get_tokenizer(self):
        class _Tokenizer:
            def apply_chat_template(self, messages, **kwargs):
                return messages[0]["content"]

        return _Tokenizer()

    def generate(self, prompts, sampling_params=None, lora_request=None):
        return [
            _FakeRequest([_FakeOutput(text, token_ids)])
            for text, token_ids in self._responses[: len(prompts)]
        ]


def _write_input_rows(tmp_path: Path, rows: list[dict[str, str]]) -> Path:
    inp = tmp_path / "in.jsonl"
    inp.write_text("\n".join(json.dumps(row) for row in rows), encoding="utf-8")
    return inp


def _run_eval(
    tmp_path: Path,
    monkeypatch,
    *,
    responses,
    rows,
    write_raw_predictions: bool = False,
) -> tuple[dict, list[dict]]:
    from scripts import eval_official_vllm_proxy as mod

    monkeypatch.setattr(
        mod,
        "_instantiate_vllm",
        lambda *args, **kwargs: FakeLLM(responses=responses),
    )
    monkeypatch.setattr(mod, "_build_lora_request", lambda *args, **kwargs: None)
    monkeypatch.setattr(mod, "_build_sampling_params", lambda: object())

    inp = _write_input_rows(tmp_path, rows)
    out = tmp_path / "eval.json"
    raw_dir = tmp_path / "raw"
    argv = [
        "--adapter-dir",
        "does-not-exist-ok-mock",
        "--input",
        str(inp),
        "--output",
        str(out),
        "--config",
        "configs/train_stage2_selected_trace.yaml",
        "--num-repeats",
        "1",
        "--no-load-base-model",
    ]
    if write_raw_predictions:
        argv.extend(
            [
                "--write-raw-predictions",
                "--raw-predictions-dir",
                str(raw_dir),
            ]
        )
    mod.main(
        argv
    )
    payload = json.loads(out.read_text(encoding="utf-8"))
    raw_rows: list[dict] = []
    raw_path = payload["repeats"][0].get("raw_predictions_path")
    if raw_path:
        raw_rows = [
            json.loads(line)
            for line in Path(raw_path).read_text(encoding="utf-8").splitlines()
            if line.strip()
        ]
    return payload, raw_rows


def test_end_to_end_with_fake_llm(tmp_path, monkeypatch) -> None:
    data, _raw_rows = _run_eval(
        tmp_path,
        monkeypatch,
        responses=[
            (r"Reasoning... \boxed{42}", list(range(50))),
            (r"Reasoning... \boxed{42}", list(range(50))),
        ],
        rows=[
            {"id": "a", "family": "bit", "prompt": "q1", "target_answer": "42"},
            {"id": "b", "family": "cipher", "prompt": "q2", "target_answer": "43"},
        ],
    )
    assert data["num_examples"] == 2
    assert data["num_repeats"] == 1
    assert data["repeats"][0]["num_correct"] == 1
    assert 0.49 < data["mean_competition_correct_rate"] < 0.51
    assert "verify_sha256" in data["contract_fingerprint"]


def test_hit_max_tokens_boundary_counts_as_truncation(tmp_path, monkeypatch) -> None:
    data, raw_rows = _run_eval(
        tmp_path,
        monkeypatch,
        responses=[(r"Reasoning... \boxed{42}", list(range(3584)))],
        rows=[{"id": "a", "family": "bit", "prompt": "q1", "target_answer": "42"}],
        write_raw_predictions=True,
    )
    repeat = data["repeats"][0]
    assert repeat["num_hit_max_tokens"] == 1
    assert repeat["truncate_rate"] == 1.0
    assert repeat["generation_length_p50"] == 3584
    assert raw_rows[0]["hit_max_tokens"] is True
    assert data["repeats"][0]["num_correct"] == 1


def test_low_boxed_rate_marks_fallback_rows(tmp_path, monkeypatch) -> None:
    data, raw_rows = _run_eval(
        tmp_path,
        monkeypatch,
        responses=[
            (r"Reasoning... \boxed{42}", list(range(20))),
            ("Reasoning... final 43", list(range(12))),
        ],
        rows=[
            {"id": "a", "family": "bit", "prompt": "q1", "target_answer": "42"},
            {"id": "b", "family": "cipher", "prompt": "q2", "target_answer": "43"},
        ],
        write_raw_predictions=True,
    )
    repeat = data["repeats"][0]
    assert repeat["boxed_rate"] == 0.5
    assert repeat["num_fallback_used"] == 1
    assert raw_rows[1]["fallback_used"] is True


def test_cli_rejects_sampling_override_flags() -> None:
    from scripts import eval_official_vllm_proxy as mod

    with pytest.raises(SystemExit):
        mod.main(
            [
                "--adapter-dir",
                "adapter",
                "--input",
                "in.jsonl",
                "--output",
                "out.json",
                "--config",
                "configs/train_stage2_selected_trace.yaml",
                "--temperature",
                "0.5",
            ]
        )
