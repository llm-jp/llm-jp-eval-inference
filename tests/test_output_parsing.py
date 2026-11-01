import dataclasses

from tests.conftest import load_module_file

output_parsing = load_module_file("vllm", "output_parsing")


def _reasoning(parsed, output_token_ids, tools):
    reasoning, content = parsed.content.split("</think>")
    return dataclasses.replace(parsed, content=content, reasoning_content=reasoning.removeprefix("<think>"))


def _tool_calls(parsed, output_token_ids, tools):
    if not tools:
        return parsed
    assert "</think>" not in parsed.content, "tool calls are parsed after reasoning is removed"
    calls = [{"id": "c0", "type": "function", "function": {"name": "f", "arguments": "{}"}}] if "<call>" in parsed.content else []
    return dataclasses.replace(parsed, content=parsed.content.replace("<call>", ""), tool_calls=calls)


def test_parse_output_applies_reasoning_then_tool_calls():
    tools = [{"type": "function", "function": {"name": "f"}}]

    parsed = output_parsing.parse_output("<think>r</think>text<call>", [], tools, [_reasoning, _tool_calls])

    assert parsed.to_result_fields(with_reasoning=True, with_tools=True) == {
        "generated": "text",
        "reasoning_content": "r",
        "tool_calls": [{"id": "c0", "type": "function", "function": {"name": "f", "arguments": "{}"}}],
    }
    no_call = output_parsing.parse_output("<think>r</think>text", [], tools, [_reasoning, _tool_calls])
    assert no_call.tool_calls == []
    assert output_parsing.parse_output("text", [], None, []).to_result_fields(False, False) == {"generated": "text"}
