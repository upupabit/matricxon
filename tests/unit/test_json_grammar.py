import pytest

from app.runtime.json_grammar import JsonGrammar, JsonTokenConstraint

SCHEMA = {
    "type": "object",
    "properties": {
        "name": {"type": "string"},
        "age": {"type": "integer"},
        "tag": {"enum": ["x", "yy"]},
        "items": {"type": "array", "items": {"type": "number"}},
    },
    "required": ["name", "age"],
}


def _check(text: str, schema: dict | None = None) -> tuple[bool, bool]:
    grammar = JsonGrammar(schema)
    return grammar.feed(text.encode()), grammar.complete


class TestPlainJson:
    @pytest.mark.parametrize(
        "text",
        [
            "{}",
            '{"a": 1}',
            '{"a": [true, false, null, "x\\n\\u00e9", -2.5e3, 0]}',
            '{"a": {"b": {}}}',
        ],
    )
    def test_complete_objects(self, text: str) -> None:
        assert _check(text) == (True, True)

    def test_a_prefix_is_valid_but_not_complete(self) -> None:
        assert _check('{"a": [1, 2') == (True, False)

    @pytest.mark.parametrize("text", ['{"a" 1}', "[1]", '{"a": 01}', "{,}", '{"a": tru}x', " {}"])
    def test_invalid(self, text: str) -> None:
        assert _check(text)[0] is False

    def test_long_whitespace_runs_are_refused(self) -> None:
        assert _check("{" + " " * 17 + "}")[0] is False
        assert _check("{" + " " * 16 + "}") == (True, True)


class TestSchema:
    def test_matching_reply(self) -> None:
        assert _check('{"name": "Dana", "age": 30, "tag": "yy", "items": [1, 2.5]}', SCHEMA) == (
            True,
            True,
        )

    @pytest.mark.parametrize(
        "text",
        [
            '{"name": "Dana"}',  # required age missing
            '{"nickname": 1}',  # not a listed property
            '{"name": "D", "age": 3.5}',  # integer expected
            '{"name": "D", "age": 1, "tag": "z"}',  # not in the enum
            '{"name": "a", "name": "b"}',  # same key twice
            '{"name": "D", "age": 1, "items": ["x"]}',  # numbers expected
        ],
    )
    def test_breaking_replies(self, text: str) -> None:
        assert _check(text, SCHEMA) != (True, True)

    def test_a_root_of_another_type(self) -> None:
        assert _check("42", {"type": "integer"}) == (True, True)

    def test_free_properties_when_allowed(self) -> None:
        schema = {"type": "object", "properties": {"a": {}}, "additionalProperties": True}
        assert _check('{"zzz": 1}', schema) == (True, True)


class FakeTokenizer:
    def __init__(self, pieces: dict[int, bytes]) -> None:
        self.pieces = pieces

    def token_to_bytes(self, token_id: int) -> bytes:
        return self.pieces[token_id]


class TestTokenConstraint:
    def test_allows_only_tokens_that_keep_valid_json(self) -> None:
        tokenizer = FakeTokenizer({1: b"{", 2: b"}", 3: b"hello", 4: b'"a"', 5: b"<|im_end|>"})
        constraint = JsonTokenConstraint(tokenizer, {0}, None)

        assert constraint.allows(1) and not constraint.allows(3) and not constraint.allows(0)
        constraint.accept(1)
        assert constraint.allows(2) and constraint.allows(4)
        assert not constraint.allows(5)  # special tokens never belong in JSON
        constraint.accept(2)
        assert constraint.allows(0)  # complete: only the end-of-reply token now
        assert not constraint.allows(1)
