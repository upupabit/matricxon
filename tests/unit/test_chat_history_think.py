"""Tests for app/runtime/chat_history_think.py: earlier assistant turns get back the empty think
block the reply was generated after, so a chat's next prompt extends the previous one (prompt cache
reuse on hybrid models)."""

from jinja2 import TemplateError

from app.runtime.chat_history_think import HistoryThinkBlock
from app.runtime.chat_template import ChatTemplatePromptBuilder
from app.schemas.chat import ChatMessage, ToolCall, ToolFunctionCall

# Qwen 3.5's shape: history replies without the think block, the generation prompt with it.
_QWEN35_LIKE = (
    "{% for m in messages %}"
    "<|im_start|>{{ m.role }}\n{{ m.content }}"
    "{% if m.tool_calls %}<tool_call>{{ m.tool_calls[0].function.name }}</tool_call>{% endif %}"
    "<|im_end|>\n"
    "{% endfor %}"
    "{% if add_generation_prompt %}<|im_start|>assistant\n"
    "{% if enable_thinking is defined and enable_thinking %}<think>\n"
    "{% else %}<think>\n\n</think>\n\n{% endif %}"
    "{% endif %}"
)
# Keeps its generation prompt in history: nothing to restore.
_PLAIN = (
    "{% for m in messages %}<|im_start|>{{ m.role }}\n{{ m.content }}<|im_end|>\n{% endfor %}"
    "{% if add_generation_prompt %}<|im_start|>assistant\n{% endif %}"
)
# Drops a generation-prompt prefix that is not a think block: left as the template wrote it.
_NOT_THINK = (
    "{% for m in messages %}<|im_start|>{{ m.role }}\n{{ m.content }}<|im_end|>\n{% endfor %}"
    "{% if add_generation_prompt %}<|im_start|>assistant\nSure: {% endif %}"
)
_BLOCK = "<think>\n\n</think>\n\n"


def _builder(template: str) -> ChatTemplatePromptBuilder:
    return ChatTemplatePromptBuilder(template, "", "<|im_end|>")


def _chat(*turns: tuple[str, str]) -> list[ChatMessage]:
    return [ChatMessage(role=role, content=content) for role, content in turns]


def test_earlier_assistant_turns_get_the_block_back() -> None:
    messages = _chat(("user", "Hi"), ("assistant", "Hello!"), ("user", "Ok"))

    prompt = _builder(_QWEN35_LIKE).build(messages)

    assert prompt == (
        "<|im_start|>user\nHi<|im_end|>\n"
        f"<|im_start|>assistant\n{_BLOCK}Hello!<|im_end|>\n"
        "<|im_start|>user\nOk<|im_end|>\n"
        f"<|im_start|>assistant\n{_BLOCK}"
    )


def test_next_prompt_extends_the_previous_prompt_and_reply() -> None:
    builder = _builder(_QWEN35_LIKE)
    first = builder.build(_chat(("user", "Hi")))
    second = builder.build(_chat(("user", "Hi"), ("assistant", "Hello!"), ("user", "Ok")))
    third = builder.build(
        _chat(("user", "Hi"), ("assistant", "Hello!"), ("user", "Ok"), ("assistant", "Fine."))
        + _chat(("user", "Bye"))
    )

    assert second.startswith(first + "Hello!<|im_end|>")
    assert third.startswith(second + "Fine.<|im_end|>")


def test_tool_call_turns_get_the_block_too() -> None:
    call = ToolCall(function=ToolFunctionCall(name="search"))
    messages = [
        ChatMessage(role="user", content="Find it"),
        ChatMessage(role="assistant", content="", tool_calls=[call]),
        ChatMessage(role="tool", content="found"),
    ]

    prompt = _builder(_QWEN35_LIKE).build(messages)

    assert f"<|im_start|>assistant\n{_BLOCK}<tool_call>search</tool_call>" in prompt


def test_generation_prompt_block_is_not_doubled() -> None:
    prompt = _builder(_QWEN35_LIKE).build(_chat(("user", "Hi")))

    assert prompt == f"<|im_start|>user\nHi<|im_end|>\n<|im_start|>assistant\n{_BLOCK}"


def test_template_keeping_its_generation_prompt_is_unchanged() -> None:
    messages = _chat(("user", "Hi"), ("assistant", "Hello!"), ("user", "Ok"))
    builder = _builder(_PLAIN)

    assert builder.build(messages) == builder._render(messages)


def test_dropped_text_that_is_not_a_think_block_is_unchanged() -> None:
    messages = _chat(("user", "Hi"), ("assistant", "Hello!"), ("user", "Ok"))
    builder = _builder(_NOT_THINK)

    assert builder.build(messages) == builder._render(messages)


def test_template_rejecting_the_probe_restores_nothing() -> None:
    def render(messages: list[ChatMessage], generation: bool) -> str:
        raise TemplateError("needs a system message")

    history = HistoryThinkBlock(render)

    assert history.block == ""
    assert history.restore("<|im_start|>assistant\nHello!") == "<|im_start|>assistant\nHello!"


def test_probe_runs_once() -> None:
    calls: list[bool] = []

    def render(messages: list[ChatMessage], generation: bool) -> str:
        calls.append(generation)
        return _builder(_QWEN35_LIKE)._render(messages, None, generation)

    history = HistoryThinkBlock(render)
    history.restore("a")
    history.restore("b")

    assert len(calls) == 3
