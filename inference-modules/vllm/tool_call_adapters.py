"""Bridge vLLM ToolParser implementations to offline ``LLM.generate()``.

As with reasoning parsers, vLLM wires tool parsers only into the OpenAI HTTP server, so the request
adjustment (e.g. keeping special tokens such as ``<tool_call>``) and the extraction are applied here.
"""

import dataclasses

from typing import Any

import vllm

from output_parsing import ParsedOutput
from transformers import PreTrainedTokenizerBase
from vllm.entrypoints.openai.chat_completion.protocol import ChatCompletionRequest
from vllm.tool_parsers import ToolParserManager


def _request(tools: list[dict[str, Any]]) -> ChatCompletionRequest:
    return ChatCompletionRequest(messages=[], tools=tools, tool_choice="auto")


class ToolCallAdapter:
    def __init__(self, parser_name: str, plugin_path: str | None, tokenizer: PreTrainedTokenizerBase) -> None:
        if plugin_path:
            ToolParserManager.import_tool_parser(plugin_path)
        self.parser = ToolParserManager.get_tool_parser(parser_name)(tokenizer)

    def adjust_sampling_params(
        self, sampling_params: vllm.sampling_params.SamplingParams, tools: list[dict[str, Any]]
    ) -> None:
        request = self.parser.adjust_request(_request(tools))
        # Only tighten the settings, so that those requested by the reasoning parser are kept
        if not request.skip_special_tokens:
            sampling_params.skip_special_tokens = False
        if request.stop_token_ids:
            sampling_params.stop_token_ids = sorted(
                set(sampling_params.stop_token_ids or []) | set(request.stop_token_ids)
            )

    def __call__(
        self, parsed: ParsedOutput, output_token_ids: list[int], tools: list[dict[str, Any]] | None
    ) -> ParsedOutput:
        if not tools:
            return parsed
        info = self.parser.extract_tool_calls(parsed.content, _request(tools))
        tool_calls = [
            {
                "id": call.id,
                "type": "function",
                "function": {"name": call.function.name, "arguments": call.function.arguments},
            }
            for call in (info.tool_calls if info.tools_called else [])
        ]
        return dataclasses.replace(parsed, content=info.content or "", tool_calls=tool_calls)
