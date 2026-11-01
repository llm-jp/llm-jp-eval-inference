"""Post-processing of generated text into the fields written to the results.

Kept free of vLLM imports so that the order of the steps can be tested without a GPU.
"""

from dataclasses import dataclass
from typing import Any, Protocol


@dataclass(frozen=True)
class ParsedOutput:
    content: str
    reasoning_content: str | None = None
    # None when the sample has no tools; otherwise a list, empty when no tool is called
    tool_calls: list[dict[str, Any]] | None = None

    def to_result_fields(self, with_reasoning: bool, with_tools: bool) -> dict[str, Any]:
        """Fields written to a sample of the results; optional ones only when the corresponding step ran."""
        fields: dict[str, Any] = {"generated": self.content}
        if with_reasoning:
            fields["reasoning_content"] = self.reasoning_content
        if with_tools:
            fields["tool_calls"] = self.tool_calls
        return fields


class OutputParser(Protocol):
    def __call__(
        self, parsed: ParsedOutput, output_token_ids: list[int], tools: list[dict[str, Any]] | None
    ) -> ParsedOutput: ...


def parse_output(
    output_text: str,
    output_token_ids: list[int],
    tools: list[dict[str, Any]] | None,
    parsers: list[OutputParser],
) -> ParsedOutput:
    """Apply the parsers in order, each refining the result of the previous one.

    Args:
        output_text: Generated text.
        output_token_ids: Generated token IDs.
        tools: Tools given to the sample, or None.
        parsers: Steps such as reasoning extraction followed by tool call extraction.

    Returns:
        The parsed output.
    """
    parsed = ParsedOutput(content=output_text)
    for parser in parsers:
        parsed = parser(parsed, output_token_ids, tools)
    return parsed
