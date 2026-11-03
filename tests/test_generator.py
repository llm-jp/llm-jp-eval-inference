import json

import pytest

from transformers import AutoTokenizer

from llm_jp_eval_inference.generator import GeneratorBase
from llm_jp_eval_inference.schemas import BaseInferenceConfig

TOKENIZER = "Qwen/Qwen3-0.6B"
TOOLS = [{"type": "function", "function": {"name": "Search_hotel", "parameters": {"type": "object"}}}]
CHAT_MESSAGES = [
    {"role": "user", "content": "札幌のホテルを探して"},
    {
        "role": "assistant",
        "content": None,
        "tool_calls": [
            {"id": "c0", "type": "function", "function": {"name": "Search_hotel", "arguments": '{"city": "札幌"}'}}
        ],
    },
    {"role": "tool", "tool_call_id": "c0", "content": "{}"},
]


class _Generator(GeneratorBase):
    def __init__(self, apply_chat_template: bool, supports_tools: bool = False):
        cfg = BaseInferenceConfig(
            generation_config={}, apply_chat_template=apply_chat_template, meta={"lib_type": "test"}
        )
        super().__init__(cfg, "test", None)
        self.tokenizer = AutoTokenizer.from_pretrained(TOKENIZER)
        self._supports_tools = supports_tools

    def supports_tool_calls(self) -> bool:
        return self._supports_tools


def _text_dataset(system_prompt: str | None = None) -> dict:
    prompt = "### 入力:\nこんにちは\n### 応答:\n"
    system = [{"role": "system", "content": system_prompt}] if system_prompt else []
    return {
        "tools": None,
        "samples": [{"prompt": prompt, "messages": [*system, {"role": "user", "content": prompt}]}],
    }


def _chat_dataset() -> dict:
    return {"tools": TOOLS, "samples": [{"prompt": None, "messages": CHAT_MESSAGES}]}


def test_tokenize_text_dataset_matches_previous_tokens():
    generator = _Generator(apply_chat_template=False)
    tokenized = generator.tokenize_dataset({"text": _text_dataset()})["text"]
    assert tokenized.prompt_token_ids == [generator.tokenizer("### 入力:\nこんにちは\n### 応答:\n")["input_ids"]]

    generator = _Generator(apply_chat_template=True)
    tokenized = generator.tokenize_dataset({"text": _text_dataset("system")})["text"]
    previous = generator.tokenizer.apply_chat_template(
        [{"role": "system", "content": "system"}, {"role": "user", "content": "### 入力:\nこんにちは\n### 応答:\n"}],
        add_generation_prompt=True,
        return_dict=False,
        add_special_tokens=True,
    )
    assert tokenized.prompt_token_ids == [previous]


def test_tokenize_chat_dataset_with_tools():
    generator = _Generator(apply_chat_template=True, supports_tools=True)

    tokenized = generator.tokenize_dataset({"chat": _chat_dataset()})["chat"]

    prompt = generator.tokenizer.decode(tokenized.prompt_token_ids[0])
    assert json.dumps(TOOLS[0], ensure_ascii=False) in prompt
    # Arguments are rendered as an object, not as an escaped JSON string
    assert '{"name": "Search_hotel", "arguments": {"city": "札幌"}}' in prompt


def test_tokenize_chat_dataset_requires_chat_template_and_tool_parser():
    with pytest.raises(ValueError, match="chat template"):
        _Generator(apply_chat_template=False, supports_tools=True).tokenize_dataset({"chat": _chat_dataset()})
    with pytest.raises(ValueError, match="tool_call_parser"):
        _Generator(apply_chat_template=True).tokenize_dataset({"chat": _chat_dataset()})
