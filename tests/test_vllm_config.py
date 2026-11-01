import pytest

from pydantic import ValidationError

from tests.conftest import load_module_file

schemas = load_module_file("vllm", "schemas")

# Equivalent of configs/inference_template.yaml of leaderboard-backend after envsubst
LEADERBOARD_CONFIG = {
    "run_name": None,
    "reasoning_content_length": 1024,
    "wandb": {"launch": False, "entity": "entity", "project": "project"},
    "apply_chat_template": True,
    "tokenize_kwargs": {"add_special_tokens": False, "enable_thinking": True},
    "model": {
        "model": "llm-jp/llm-jp-3-3.7b-instruct",
        "dtype": "bfloat16",
        "revision": "main",
        "reasoning_parser": "",
        "tensor_parallel_size": 8,
        "max_model_len": 4096,
        "gpu_memory_utilization": 0.7,
        "enable_prefix_caching": False,
        "trust_remote_code": False,
    },
    "tokenizer": {
        "pretrained_model_name_or_path": "llm-jp/llm-jp-3-3.7b-instruct",
        "revision": "abc123",
        "trust_remote_code": False,
        "use_fast": True,
        "padding_side": "left",
    },
    "generation_config": {"temperature": 0.0, "top_p": 1.0, "max_tokens": None},
    "meta": {"base_model_name": None, "quantization": None},
}


def test_leaderboard_config_is_accepted_and_typos_are_rejected():
    cfg = schemas.InferenceConfig.model_validate(LEADERBOARD_CONFIG)
    assert cfg.tokenizer.model_dump()["revision"] == "abc123"

    with pytest.raises(ValidationError):
        schemas.InferenceConfig.model_validate({**LEADERBOARD_CONFIG, "apply_chat_tempalte": True})
    with pytest.raises(ValidationError):
        schemas.InferenceConfig.model_validate(
            {**LEADERBOARD_CONFIG, "tokenizer": {**LEADERBOARD_CONFIG["tokenizer"], "revison": "abc123"}}
        )
