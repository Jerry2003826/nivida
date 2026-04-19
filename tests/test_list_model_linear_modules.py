from __future__ import annotations

import json
from pathlib import Path

class FakeLinear:
    def __init__(self) -> None:
        self.weight = object()
        self.in_features = 128
        self.out_features = 128


class FakeAttention:
    def __init__(self) -> None:
        self.q_proj = FakeLinear()
        self.k_proj = FakeLinear()
        self.v_proj = FakeLinear()
        self.o_proj = FakeLinear()


class FakeMamba:
    def __init__(self) -> None:
        self.in_proj = FakeLinear()
        self.out_proj = FakeLinear()


class FakeMLP:
    def __init__(self) -> None:
        self.up_proj = FakeLinear()
        self.down_proj = FakeLinear()


class FakeLayer:
    def __init__(self, kind: str) -> None:
        self.mixer = FakeMamba() if kind == "mamba" else FakeAttention()
        self.mlp = FakeMLP()


class FakeModel:
    def __init__(self) -> None:
        self.layers = [FakeLayer("mamba"), FakeLayer("attn"), FakeLayer("mamba")]

    def named_modules(self):
        yield "", self
        yield "layers", self.layers
        for index, layer in enumerate(self.layers):
            layer_name = f"layers.{index}"
            yield layer_name, layer
            yield f"{layer_name}.mixer", layer.mixer
            if isinstance(layer.mixer, FakeMamba):
                yield f"{layer_name}.mixer.in_proj", layer.mixer.in_proj
                yield f"{layer_name}.mixer.out_proj", layer.mixer.out_proj
            else:
                yield f"{layer_name}.mixer.q_proj", layer.mixer.q_proj
                yield f"{layer_name}.mixer.k_proj", layer.mixer.k_proj
                yield f"{layer_name}.mixer.v_proj", layer.mixer.v_proj
                yield f"{layer_name}.mixer.o_proj", layer.mixer.o_proj
            yield f"{layer_name}.mlp", layer.mlp
            yield f"{layer_name}.mlp.up_proj", layer.mlp.up_proj
            yield f"{layer_name}.mlp.down_proj", layer.mlp.down_proj


def test_enumeration_and_regex_match(tmp_path: Path, monkeypatch) -> None:
    cfg = tmp_path / "fake_cfg.yaml"
    cfg.write_text(
        "model:\n"
        "  name: fake/Nemotron-test\n"
        "  target_modules: '.*\\\\.(in_proj|out_proj|up_proj|down_proj)$'\n",
        encoding="utf-8",
    )
    out = tmp_path / "audit.json"

    from scripts import list_model_linear_modules as mod

    monkeypatch.setattr(mod, "_load_model_for_audit", lambda **_kwargs: FakeModel())

    mod.main(
        [
            "--config",
            str(cfg),
            "--output",
            str(out),
            "--no-load-weights",
        ]
    )
    data = json.loads(out.read_text(encoding="utf-8"))
    counts = data["linear_suffix_counts"]
    assert counts["q_proj"] == 1
    assert counts["in_proj"] == 2
    assert counts["up_proj"] == 3
    assert set(data["uncovered_candidate_suffixes"]) == {
        "q_proj",
        "k_proj",
        "v_proj",
        "o_proj",
    }
    assert data["wide_branch_recommendation"]["eligible"] in (True, False)
