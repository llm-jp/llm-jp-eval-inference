"""Bridge vLLM's Parser to offline ``LLM.generate()``.

The OpenAI-compatible server adjusts each request through the parser (e.g. keeping the special tokens that mark
reasoning or tool calls) and extracts reasoning and tool calls from the output with it. ``LLM.generate()`` does
neither, so both are applied here with the same Parser the server builds.
"""

from typing import Any

import vllm.sampling_params

from output_parsing import ParsedOutput, to_parsed_output
from transformers import PreTrainedTokenizerBase
from vllm.entrypoints.chat_utils import make_tool_call_id
from vllm.entrypoints.openai.chat_completion.protocol import ChatCompletionRequest
from vllm.parser import Parser, ParserManager


def _request(tools: list[dict[str, Any]] | None) -> ChatCompletionRequest:
    return ChatCompletionRequest(messages=[], tools=tools or None, tool_choice="auto" if tools else "none")


class ParserAdapter:
    def __init__(
        self,
        tokenizer: PreTrainedTokenizerBase,
        reasoning_parser: str | None,
        tool_call_parser: str | None,
        is_harmony: bool,
        chat_template_kwargs: dict[str, Any],
    ) -> None:
        """Build the parser as the server does.

        Args:
            tokenizer: Tokenizer of the model.
            reasoning_parser: Name of a registered reasoning parser, or None.
            tool_call_parser: Name of a registered tool parser, or None.
            is_harmony: Whether the model uses the Harmony format, which vLLM parses from token IDs.
            chat_template_kwargs: Template variables, read by parsers whose output depends on them.

        Raises:
            ValueError: If neither parser is given.
        """
        parser_cls = ParserManager.get_parser(
            tool_parser_name=tool_call_parser,
            reasoning_parser_name=reasoning_parser,
            enable_auto_tools=True,
            is_harmony=is_harmony,
        )
        if parser_cls is None:
            raise ValueError("Either reasoning_parser or tool_call_parser must be set")
        self.parser_cls = parser_cls
        self.tokenizer = tokenizer
        self.chat_template_kwargs = chat_template_kwargs

    def _parser(self, request: ChatCompletionRequest) -> Parser:
        # Parsers keep per-request state, so a new one is made for each request as on the server
        return self.parser_cls(self.tokenizer, request.tools, chat_template_kwargs=self.chat_template_kwargs)

    def adjust_sampling_params(
        self, sampling_params: vllm.sampling_params.SamplingParams, tools: list[dict[str, Any]] | None
    ) -> None:
        """Apply the request adjustment of the parser to the sampling parameters.

        Args:
            sampling_params: Sampling parameters, updated in place.
            tools: Tools given to the dataset, or None.
        """
        # As on the server, a tool parser alone adjusts only requests with tools
        if self.parser_cls.reasoning_parser_cls is None and not tools:
            return
        request = _request(tools)
        request = self._parser(request).adjust_request(request)
        # Only tighten the settings, so that those of the generation config are kept
        if not request.skip_special_tokens:
            sampling_params.skip_special_tokens = False
        if request.stop_token_ids:
            sampling_params.stop_token_ids = sorted(
                set(sampling_params.stop_token_ids or []) | set(request.stop_token_ids)
            )

    def parse(self, output_text: str, output_token_ids: list[int], tools: list[dict[str, Any]] | None) -> ParsedOutput:
        """Extract reasoning and tool calls from a generated output.

        Args:
            output_text: Generated text.
            output_token_ids: Generated token IDs.
            tools: Tools given to the sample, or None.

        Returns:
            The parsed output.
        """
        request = _request(tools)
        reasoning, content, calls = self._parser(request).parse(
            output_text, request, enable_auto_tools=True, model_output_token_ids=output_token_ids
        )
        tool_calls = [
            {
                "id": call.id or make_tool_call_id(),
                "type": "function",
                "function": {"name": call.name, "arguments": call.arguments},
            }
            for call in calls or []
        ]
        return to_parsed_output(reasoning, content, tool_calls, with_tools=bool(tools))
