"""Tests for app/runtime/thinking_splitter.py: reasoning and answer separated from a streamed reply,
whatever the chunk boundaries."""

import pytest

from app.runtime.thinking_splitter import ThinkingSplitter


def _run(chunks: list[str], starts_inside: bool = False) -> tuple[str, str]:
    splitter = ThinkingSplitter("</think>" if starts_inside else None)
    thinking, answer = "", ""
    for chunk in chunks:
        t, a = splitter.push(chunk)
        thinking, answer = thinking + t, answer + a
    t, a = splitter.finish()
    return thinking + t, answer + a


def _every_split(text: str) -> list[list[str]]:
    return [[text[:i], text[i:]] for i in range(len(text) + 1)] + [list(text)]


@pytest.mark.parametrize("chunks", _every_split("Let me add 2 and 2.\n</think>\n\nIt is 4."))
def test_a_reply_inside_a_prefilled_block_splits_at_any_boundary(chunks: list[str]) -> None:
    assert _run(chunks, starts_inside=True) == ("Let me add 2 and 2.\n", "It is 4.")


@pytest.mark.parametrize("chunks", _every_split("\n<think>\nhmm\n</think>\n\nHi!"))
def test_a_model_that_opens_the_block_itself(chunks: list[str]) -> None:
    assert _run(chunks) == ("\nhmm\n", "Hi!")


def test_a_plain_reply_is_all_answer_including_a_later_tag() -> None:
    assert _run(["Hello", " <think> is a tag"]) == ("", "Hello <think> is a tag")


def test_unfinished_reasoning_stays_reasoning() -> None:
    assert _run(["still thinking </thi"], starts_inside=True) == ("still thinking </thi", "")


@pytest.mark.parametrize("chunks", _every_split("<|channel>thought\nhmm\n<channel|>Hi!"))
def test_gemmas_thought_channel(chunks: list[str]) -> None:
    assert _run(chunks) == ("\nhmm\n", "Hi!")


def test_a_prompt_that_opened_gemmas_channel() -> None:
    splitter = ThinkingSplitter("<channel|>")
    assert splitter.push("hmm<channel|>Hi") == ("hmm", "Hi")
