from __future__ import annotations

import json


class _FakeOutput:
    def __init__(self, text: str, token_ids: list[int]) -> None:
        self.text = text
        self.token_ids = token_ids


class _FakeRequest:
    def __init__(self, outputs) -> None:
        self.outputs = outputs


class FakeLLM:
    def __init__(self, *args, **kwargs) -> None:
        self._kwargs = kwargs

    def get_tokenizer(self):
        class _Tokenizer:
            def apply_chat_template(self, messages, **kwargs):
                return messages[0]["content"]

        return _Tokenizer()

    def generate(self, prompts, sampling_params=None, lora_request=None):
        return [
            _FakeRequest([_FakeOutput(r"Reasoning... \boxed{42}", list(range(50)))])
            for _ in prompts
        ]


def test_end_to_end_with_fake_llm(tmp_path, monkeypatch) -> None:
    from scripts import eval_official_vllm_proxy as mod

    monkeypatch.setattr(mod, "_instantiate_vllm", lambda *args, **kwargs: FakeLLM())
    monkeypatch.setattr(mod, "_build_lora_request", lambda *args, **kwargs: None)
    monkeypatch.setattr(mod, "_build_sampling_params", lambda: object())

    inp = tmp_path / "in.jsonl"
    rows = [
        {"id": "a", "family": "bit", "prompt": "q1", "target_answer": "42"},
        {"id": "b", "family": "cipher", "prompt": "q2", "target_answer": "43"},
    ]
    inp.write_text("\n".join(json.dumps(row) for row in rows), encoding="utf-8")

    out = tmp_path / "eval.json"
    mod.main(
        [
            "--adapter-dir",
            "does-not-exist-ok-mock",
            "--input",
            str(inp),
            "--output",
            str(out),
            "--config",
            "configs/train_stage2_selected_trace.yaml",
            "--num-repeats",
            "2",
            "--no-load-base-model",
        ]
    )
    data = json.loads(out.read_text(encoding="utf-8"))
    assert data["num_examples"] == 2
    assert data["num_repeats"] == 2
    assert data["repeats"][0]["num_correct"] == 1
    assert 0.49 < data["mean_competition_correct_rate"] < 0.51
    assert "verify_sha256" in data["contract_fingerprint"]
