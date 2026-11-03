from tests.conftest import load_module_file

output_parsing = load_module_file("vllm", "output_parsing")

CALL = {"id": "c0", "type": "function", "function": {"name": "f", "arguments": "{}"}}


def test_to_parsed_output_keeps_reasoning_apart_from_answer_and_tool_calls():
    parsed = output_parsing.to_parsed_output(" r", " text", [], with_tools=True)
    assert parsed.to_result_fields(with_reasoning=True, with_tools=True) == {
        "generated": "text",
        "reasoning_content": "r",
        "tool_calls": [],
    }

    called = output_parsing.to_parsed_output("r", None, [CALL], with_tools=True)
    assert (called.content, called.reasoning_content, called.tool_calls) == ("", "r", [CALL])

    assert output_parsing.to_parsed_output(None, "text", None, with_tools=False).tool_calls is None


def test_to_parsed_output_surfaces_cut_off_reasoning_as_answer():
    parsed = output_parsing.to_parsed_output("partial", None, None, with_tools=False)
    assert (parsed.content, parsed.reasoning_content) == ("partial", None)
