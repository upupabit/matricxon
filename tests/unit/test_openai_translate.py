import json

import pytest

from app.openai_compat.schemas import OpenAIChatRequest
from app.openai_compat.translate import OpenAITranslator
from app.server.errors import UnsupportedOptionError


def _translate(**fields: object):
    body = {"model": "m", "messages": [{"role": "user", "content": "hi"}], **fields}
    return OpenAITranslator(OpenAIChatRequest(**body)).chat_request()


def test_options_map_onto_matricxon_names() -> None:
    request = _translate(
        temperature=0.2, max_completion_tokens=7, stop="END", presence_penalty=0.5, num_ctx=1024
    )
    assert request.options.temperature == 0.2
    assert request.options.num_predict == 7
    assert request.options.stop == ["END"]
    assert request.options.presence_penalty == 0.5
    assert request.options.num_ctx == 1024


def test_omitted_options_keep_matricxon_defaults() -> None:
    options = _translate().options
    assert options.temperature == 0.8 and options.num_predict == -1


def test_messages_roles_images_and_tool_calls() -> None:
    call = {"id": "c1", "type": "function", "function": {"name": "f", "arguments": '{"x": 1}'}}
    image = {"type": "image_url", "image_url": {"url": "data:image/png;base64,QUJD"}}
    request = _translate(
        messages=[
            {"role": "developer", "content": "be brief"},
            {"role": "user", "content": [{"type": "text", "text": "look"}, image]},
            {"role": "assistant", "content": None, "tool_calls": [call]},
            {"role": "tool", "tool_call_id": "c1", "content": "42"},
        ]
    )
    system, user, assistant, tool = request.messages
    assert system.role == "system"
    assert user.content == "look" and user.images == ["QUJD"]
    assert assistant.tool_calls[0].function.arguments == {"x": 1}
    assert tool.tool_call_id == "c1"


def test_tool_choice_none_drops_the_tools() -> None:
    tools = [{"type": "function", "function": {"name": "f", "parameters": {}}}]
    assert _translate(tools=tools, tool_choice="none").tools is None
    assert _translate(tools=tools).tools == tools


def test_response_formats() -> None:
    assert _translate(response_format={"type": "json_object"}).format == "json"
    schema = {"type": "object"}
    fmt = {"type": "json_schema", "json_schema": {"name": "x", "schema": schema}}
    assert _translate(response_format=fmt).format == schema
    with pytest.raises(UnsupportedOptionError):
        _translate(response_format={"type": "grammar"})


def test_reply_tool_calls_get_ids_and_json_arguments() -> None:
    calls = OpenAITranslator.tool_calls([{"function": {"name": "f", "arguments": {"x": 1}}}])
    assert calls[0]["type"] == "function" and calls[0]["id"].startswith("call_")
    assert json.loads(calls[0]["function"]["arguments"]) == {"x": 1}


def test_finish_reasons() -> None:
    assert OpenAITranslator.finish_reason({"done_reason": "length"}, False) == "length"
    assert OpenAITranslator.finish_reason({"done_reason": "stop"}, True) == "tool_calls"
    assert OpenAITranslator.finish_reason({}, False) == "stop"
