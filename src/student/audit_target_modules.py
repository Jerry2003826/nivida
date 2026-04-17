from __future__ import annotations

import argparse
import re
from pathlib import Path

from src.common.io import read_yaml, write_json
from src.student.lora_train import _import_or_raise, normalise_target_modules, resolve_model_path


def audit_modules(config: dict[str, object]) -> dict[str, object]:
    transformers = _import_or_raise("transformers")
    model_path = resolve_model_path(config)
    model = transformers.AutoModelForCausalLM.from_pretrained(
        model_path,
        trust_remote_code=bool(config.get("trust_remote_code", True)),
        device_map="cpu",
    )
    target_modules = normalise_target_modules(dict(config.get("lora", {})).get("target_modules"))
    module_names = [name for name, _ in model.named_modules() if name]
    if isinstance(target_modules, list):
        matches = [name for name in module_names if any(name.endswith(token) for token in target_modules)]
    else:
        pattern = re.compile(target_modules)
        matches = [name for name in module_names if pattern.search(name)]
    return {
        "model_path": model_path,
        "target_modules": target_modules,
        "num_modules": len(module_names),
        "num_matches": len(matches),
        "matches": matches,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Audit model module names for LoRA targeting.")
    parser.add_argument("--config", default="configs/train_stage2_selected_trace.yaml")
    parser.add_argument("--output", default="artifacts/target_module_audit.json")
    args = parser.parse_args()

    config = read_yaml(args.config)
    payload = audit_modules(config)
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    write_json(output_path, payload)


if __name__ == "__main__":
    main()
