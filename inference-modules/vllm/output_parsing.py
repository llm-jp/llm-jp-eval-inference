"""Post-processing of generated text into the fields written to the results.

Kept free of vLLM imports so that it can be tested without a GPU.
"""

from dataclasses import dataclass
from typing import Any


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


def to_parsed_output(
    reasoning: str | None,
    content: str | None,
    tool_calls: list[dict[str, Any]] | None,
    with_tools: bool,
) -> ParsedOutput:
    """Convert the result of vLLM's parser into the output written to the results.

    Args:
        reasoning: Extracted reasoning, or None.
        content: Final answer, or None when the parser found none.
        tool_calls: Extracted tool calls, or None.
        with_tools: Whether the sample has tools.

    Returns:
        The parsed output.
    """
    # On cut-off, surface partial reasoning as the answer so that it is not lost
    if content is None and not tool_calls:
        reasoning, content = None, reasoning
    return ParsedOutput(
        content=content.lstrip() if content else "",
        reasoning_content=reasoning.lstrip() if reasoning else None,
        tool_calls=(tool_calls or []) if with_tools else None,
    )
