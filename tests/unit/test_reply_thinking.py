"""`think` on /api/chat: the template's `enable_thinking`, reasoning returned apart from the answer
(ReplyTextStream), and the history fix (HistoryThinkBlock) leaving the generation prompt and turns
that kept their reasoning as the template rendered them."""

from app.gguf.metadata import GGUFMetadata
from app.models.tool_support import _switch_changes_history
from app.runtime.chat_template import ChatTemplatePromptBuilder
from app.runtime.reply_text_stream import ReplyTextStream
from app.schemas.chat import ChatMessage

VOCAB = ["<think>", "\n", "Two plus two.", "</think>", "\n\n", "It is ", "4."]

# Qwen 3.5's shape: thinking only with enable_thinking, history replies without the block.
TEMPLATE = (
    "{% for m in messages %}<|im_start|>{{ m.role }}\n{{ m.content }}<|im_end|>\n{% endfor %}"
    "{% if add_generation_prompt %}<|im_start|>assistant\n"
    "{% if enable_thinking is defined and enable_thinking %}<think>\n"
    "{% else %}<think>\n\n</think>\n\n{% endif %}{% endif %}"
)
BLOCK = "<think>\n\n</think>\n\n"


class _Tokens:
    def token_to_bytes(self, token_id: int) -> bytes:
        return VOCAB[token_id].encode()


def _stream(ids: list[int], opened: bool) -> tuple[str, str]:
    reply = ReplyTextStream(_Tokens(), thinking_until="</think>" if opened else None)
    thinking, answer = "", ""
    for token_id in ids:
        t, a = reply.push(token_id)
        thinking, answer = thinking + t, answer + a
    t, a, calls = reply.finish()
    assert calls == []
    return thinking + t, answer + a


def test_reasoning_after_an_opened_block_comes_apart_from_the_answer() -> None:
    assert _stream([2, 1, 3, 4, 5, 6], opened=True) == ("Two plus two.\n", "It is 4.")


def test_a_reply_that_opens_the_block_itself_is_split_too() -> None:
    assert _stream([0, 1, 2, 3, 4, 5, 6], opened=False) == ("\nTwo plus two.", "It is 4.")


def test_a_plain_reply_is_unchanged() -> None:
    assert _stream([5, 6], opened=False) == ("", "It is 4.")


def _chat(*turns: tuple[str, str]) -> list[ChatMessage]:
    return [ChatMessage(role=role, content=content) for role, content in turns]


def test_think_reaches_the_template_and_the_generation_prompt_is_left_alone() -> None:
    builder = ChatTemplatePromptBuilder(TEMPLATE, "", "<|im_end|>")
    messages = _chat(("user", "Hi"), ("assistant", "Hello!"), ("user", "2+2?"))

    on = builder.build(messages, think=True)
    off = builder.build(messages, think=False)

    assert on.endswith("<|im_start|>assistant\n<think>\n")
    assert f"<|im_start|>assistant\n{BLOCK}Hello!" in on
    assert off.endswith(f"<|im_start|>assistant\n{BLOCK}") and off.count(BLOCK) == 2
    assert builder.build(messages) == off  # None: the template's default (off here)


def test_a_turn_that_kept_its_reasoning_gets_no_empty_block() -> None:
    builder = ChatTemplatePromptBuilder(TEMPLATE, "", "<|im_end|>")
    thought = ChatMessage(role="assistant", content="It is 4.", thinking="Two plus two.")
    plain = ChatMessage(role="assistant", content="Bye.")
    messages = [*_chat(("user", "2+2?")), thought, *_chat(("user", "ok")), plain]

    prompt = builder.build([*messages, *_chat(("user", "again"))])

    assert "<|im_start|>assistant\nIt is 4." in prompt
    assert f"<|im_start|>assistant\n{BLOCK}Bye." in prompt


def test_a_switch_in_the_system_turn_means_a_live_chat_is_read_again() -> None:
    gemma_like = (
        "{% if enable_thinking is defined and enable_thinking %}<|think|>\n{% endif %}"
        "{% for m in messages %}<|turn>{{ m.role }}\n{{ m.content }}<turn|>\n{% endfor %}"
        "{% if add_generation_prompt %}<|turn>model\n{% endif %}"
    )

    def metadata(template: str) -> GGUFMetadata:
        return GGUFMetadata({"tokenizer.chat_template": template, "tokenizer.ggml.tokens": ["a"]})

    assert _switch_changes_history(metadata(gemma_like))
    assert not _switch_changes_history(metadata(TEMPLATE))
