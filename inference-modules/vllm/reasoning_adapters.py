"""Bridge vLLM ReasoningParser implementations to offline ``LLM.generate()``.

vLLM's parsers are wired for the OpenAI HTTP server path; their
``adjust_request`` hook and any non-text parsing are unreachable when
calling ``LLM.generate()`` directly. Adapters restore those pieces.

Adapter selection reuses the user-facing ``reasoning_parser`` name. An
adapter may rewrite ``model_cfg.reasoning_parser`` in its ``__init__``
to redirect vLLM to a different parser (e.g. llm-jp-4 reuses
``openai_gptoss`` because vLLM only preserves Harmony special tokens in
the output token IDs when a Harmony-aware parser is configured).
"""

import logging
import re

from abc import ABC, abstractmethod
from typing import TypeVar

import vllm

from schemas import ModelConfig
from transformers import PreTrainedTokenizerBase
from vllm.entrypoints.openai.chat_completion.protocol import ChatCompletionRequest
from vllm.entrypoints.openai.parser.harmony_utils import get_encoding, parse_chat_output
from vllm.reasoning.abs_reasoning_parsers import ReasoningParserManager

logger = logging.getLogger(__name__)


class ReasoningAdapter(ABC):
    parser_name: str | None = None

    def __init__(self, model_cfg: ModelConfig, tokenizer: PreTrainedTokenizerBase) -> None:
        self.tokenizer = tokenizer
        # Subclasses may rewrite model_cfg.reasoning_parser before this point
        # if they need vLLM started with a different parser.
        parser_class = ReasoningParserManager.get_reasoning_parser(model_cfg.reasoning_parser)
        self.parser = parser_class(tokenizer=tokenizer)

    def adjust_sampling_params(self, sampling_params: vllm.sampling_params.SamplingParams) -> None:  # noqa: B027
        """No-op by default; override to mirror the parser's adjust_request."""

    @abstractmethod
    def parse_output(self, output_token_ids: list[int], output_text: str) -> tuple[str | None, str]:
        """Return (reasoning_content, final_content). final_content is always str."""


class DefaultReasoningAdapter(ReasoningAdapter):
    """Delegate parsing to a vLLM ``ReasoningParser.extract_reasoning(text)``."""

    def parse_output(self, output_token_ids: list[int], output_text: str) -> tuple[str | None, str]:
        reasoning, content = self.parser.extract_reasoning(output_text, request=ChatCompletionRequest(messages=[]))
        # On cut-off, surface partial reasoning as final so it is not lost.
        if content is None:
            return (None, reasoning) if reasoning else (None, "")
        return reasoning, content


class Gemma4ReasoningAdapter(DefaultReasoningAdapter):
    parser_name = "gemma4"

    def adjust_sampling_params(self, sampling_params: vllm.sampling_params.SamplingParams) -> None:
        # Gemma4 boundary tokens must survive in the output text.
        if sampling_params.skip_special_tokens:
            sampling_params.skip_special_tokens = False


class GptOssReasoningAdapter(ReasoningAdapter):
    parser_name = "openai_gptoss"

    def parse_output(self, output_token_ids: list[int], output_text: str) -> tuple[str | None, str]:
        reasoning, final, _ = parse_chat_output(output_token_ids)
        if final is None:
            return (None, reasoning) if reasoning else (None, "")
        return reasoning, final


class Llmjp4ReasoningAdapter(ReasoningAdapter):
    # llm-jp-4 emits Harmony channels but its tokenizer is not the
    # openai-harmony vocabulary, so the model's token IDs cannot be fed to
    # parse_chat_output directly. Decode with special tokens preserved,
    # collapse the trailing spaces some tokenizers introduce after special
    # tokens, then re-encode through openai-harmony before parsing.
    parser_name = "llmjp4"
    _SPECIAL_TOKEN_TRAILING_SPACE = re.compile(r"(<\|[^|]+\|>)\s+")

    def __init__(self, model_cfg: ModelConfig, tokenizer: PreTrainedTokenizerBase) -> None:
        # vLLM has no "llmjp4"; borrow openai_gptoss so Harmony special tokens
        # stay in the output token IDs.
        model_cfg.reasoning_parser = "openai_gptoss"
        super().__init__(model_cfg, tokenizer)

    def parse_output(self, output_token_ids: list[int], output_text: str) -> tuple[str | None, str]:
        text = self.tokenizer.decode(output_token_ids, skip_special_tokens=False)
        normalized = self._SPECIAL_TOKEN_TRAILING_SPACE.sub(r"\1", text)
        try:
            harmony_token_ids = get_encoding().encode(normalized, allowed_special="all")
            reasoning, final, _ = parse_chat_output(harmony_token_ids)
        except Exception as e:
            # Truncated / malformed Harmony output: surface raw text instead
            # of failing the whole batch.
            logger.warning("Harmony parse failed (likely truncated): %s", e)
            return None, output_text
        if final is None:
            return (None, reasoning) if reasoning else (None, "")
        return reasoning, final


_T = TypeVar("_T")


def _walk_subclasses(cls: type[_T]) -> list[type[_T]]:
    direct: list[type[_T]] = cls.__subclasses__()
    return direct + [s for sub in direct for s in _walk_subclasses(sub)]


def find_adapter_class(name: str) -> type[ReasoningAdapter]:
    """Return the adapter class bound to ``name``, or DefaultReasoningAdapter."""
    for cls in _walk_subclasses(ReasoningAdapter):  # type: ignore[type-abstract]
        if cls.parser_name == name:
            return cls
    return DefaultReasoningAdapter
