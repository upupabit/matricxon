"""The states of the structured-output JSON parser (app.runtime.json_grammar), one per open value.

Each frame reads one byte at a time and answers with an action: the byte was consumed, the frame
ended (and the byte belongs to the parent, or was its last byte), a child value starts, or the byte
is not allowed here. A JSON Schema subset narrows what is allowed: `type`, `properties`, `required`,
`additionalProperties`, `items`, `enum` and `const`. Other keywords are ignored (allowed freely).
"""

import json
from typing import Any

WHITESPACE = b" \t\n\r"
MAX_WHITESPACE = 16  # in a row: stops a model from padding forever

CONSUMED, POP, POP_CONSUMED, ERROR = "consumed", "pop", "pop_consumed", "error"


def types_of(schema: dict[str, Any]) -> set[str]:
    """The JSON types a schema allows (all of them when it doesn't say)."""
    given = schema.get("type")
    if given is None:
        return {"object", "array", "string", "number", "integer", "boolean", "null"}
    return {given} if isinstance(given, str) else set(given)


class ValueFrame:
    """Waits for a value to start, then hands over to the frame for that kind of value."""

    def __init__(self, schema: dict[str, Any]) -> None:
        self.schema = schema

    def feed(self, byte: int) -> tuple[str, object]:
        child = self._start(byte)
        return (CONSUMED, child) if child is not None else (ERROR, None)

    def _start(self, byte: int) -> object:
        schema = self.schema
        options = schema.get("enum", [schema["const"]] if "const" in schema else None)
        if options is not None:
            return LiteralFrame([json.dumps(o).encode() for o in options]).first(byte)
        allowed = types_of(schema)
        char = chr(byte)
        if char == "{" and "object" in allowed:
            return ObjectFrame(schema)
        if char == "[" and "array" in allowed:
            return ArrayFrame(schema.get("items") or {})
        if char == '"' and "string" in allowed:
            return StringFrame()
        if (char == "-" or char.isdigit()) and allowed & {"number", "integer"}:
            return NumberFrame("number" not in allowed).first(byte)
        words = ([b"true", b"false"] if "boolean" in allowed else []) + (
            [b"null"] if "null" in allowed else []
        )
        return LiteralFrame(words).first(byte) if words else None


class LiteralFrame:
    """One of a few exact texts: true/false/null, or an enum's values written as JSON."""

    def __init__(self, options: list[bytes]) -> None:
        self.options = options
        self.text = b""

    def first(self, byte: int) -> "LiteralFrame | None":
        return self if self.feed(byte)[0] != ERROR else None

    def feed(self, byte: int) -> tuple[str, object]:
        longer = self.text + bytes([byte])
        if any(o.startswith(longer) for o in self.options):
            self.text = longer
            return CONSUMED, None
        # Not a continuation: fine only if the text so far is a whole option (the byte is the
        # parent's).
        return (POP, None) if self.text in self.options else (ERROR, None)

    def done(self) -> bool:
        return self.text in self.options


class NumberFrame:
    """-?(0|[1-9][0-9]*)(.[0-9]+)?([eE][+-]?[0-9]+)?, with no fraction or exponent for integers."""

    def __init__(self, integer: bool) -> None:
        self.integer = integer
        self.state = "start"

    def first(self, byte: int) -> "NumberFrame | None":
        return self if self.feed(byte)[0] == CONSUMED else None

    def feed(self, byte: int) -> tuple[str, object]:
        char, state = chr(byte), self.state
        moves = {
            ("start", "-"): "sign", ("start", "0"): "zero", ("sign", "0"): "zero",
            ("int", "."): "dot", ("zero", "."): "dot", ("int", "e"): "e", ("zero", "e"): "e",
            ("frac", "e"): "e", ("int", "E"): "e", ("zero", "E"): "e", ("frac", "E"): "e",
            ("e", "+"): "esign", ("e", "-"): "esign",
        }  # fmt: skip
        nxt = moves.get((state, char))
        if nxt is None and char.isdigit():
            nxt = {"start": "int", "sign": "int", "int": "int", "dot": "frac", "frac": "frac",
                   "e": "exp", "esign": "exp", "exp": "exp"}.get(state)  # fmt: skip
        if nxt in ("dot", "e") and self.integer:
            nxt = None
        if nxt is not None:
            self.state = nxt
            return CONSUMED, None
        return (POP, None) if self.done() else (ERROR, None)

    def done(self) -> bool:
        return self.state in ("int", "zero", "frac", "exp")


class StringFrame:
    """A string's content after the opening quote. `names` limits it to one of these texts (an
    object's remaining property names)."""

    def __init__(self, names: list[str] | None = None) -> None:
        self.names = [n.encode() for n in names] if names is not None else None
        self.text = b""
        self.escape = 0  # bytes of an escape still to come (-1: just after the backslash)

    def feed(self, byte: int) -> tuple[str, object]:
        if self.names is not None:
            if byte == ord('"'):
                return (
                    (POP_CONSUMED, self.text.decode()) if self.text in self.names else (ERROR, None)
                )
            longer = self.text + bytes([byte])
            if byte == ord("\\") or not any(n.startswith(longer) for n in self.names):
                return ERROR, None
            self.text = longer
            return CONSUMED, None
        if self.escape == -1:
            if chr(byte) in '"\\/bfnrt':
                self.escape = 0
            elif byte == ord("u"):
                self.escape = 4
            else:
                return ERROR, None
            return CONSUMED, None
        if self.escape > 0:
            if chr(byte) not in "0123456789abcdefABCDEF":
                return ERROR, None
            self.escape -= 1
            return CONSUMED, None
        if byte == ord('"'):
            return POP_CONSUMED, self.text.decode("utf-8", "replace")
        if byte == ord("\\"):
            self.escape = -1
            return CONSUMED, None
        if byte < 0x20:
            return ERROR, None
        self.text += bytes([byte])
        return CONSUMED, None


class ObjectFrame:
    """An object after `{`: keys (limited to the schema's properties when it lists them), colons,
    values, commas; `}` only once every required key is there."""

    def __init__(self, schema: dict[str, Any]) -> None:
        self.properties: dict[str, Any] = schema.get("properties") or {}
        self.required = set(schema.get("required") or [])
        self.free = not self.properties or schema.get("additionalProperties") not in (None, False)
        self.used: frozenset[str] = frozenset()
        self.state = "start"  # start, key, colon, value, next
        self.key = ""

    def feed(self, byte: int) -> tuple[str, object]:
        char = chr(byte)
        if self.state in ("start", "key") and char == '"':
            left = [n for n in self.properties if n not in self.used]
            if not self.free and not left:
                return ERROR, None
            self.state = "colon"
            return CONSUMED, StringFrame(None if self.free else left)
        if self.state == "start" and char == "}" or self.state == "next" and char == "}":
            return (POP_CONSUMED, None) if self.required <= self.used else (ERROR, None)
        if self.state == "colon" and char == ":":
            self.state = "next"
            return CONSUMED, ValueFrame(self.properties.get(self.key, {}))
        if self.state == "next" and char == ",":
            self.state = "key"
            return CONSUMED, None
        return ERROR, None

    def child_done(self, result: object) -> None:
        if self.state == "colon" and isinstance(result, str):  # the key just read
            self.key = result
            self.used = self.used | {result}


class ArrayFrame:
    """An array after `[`: values that each match `items`, separated by commas."""

    def __init__(self, items: dict[str, Any]) -> None:
        self.items = items
        self.state = "start"  # start, value, next

    def feed(self, byte: int) -> tuple[str, object]:
        char = chr(byte)
        if self.state in ("start", "next") and char == "]":
            return POP_CONSUMED, None
        if self.state == "next" and char == ",":
            self.state = "value"
            return CONSUMED, None
        if self.state in ("start", "value"):
            self.state = "next"
            action, child = ValueFrame(self.items).feed(byte)
            return (CONSUMED, child) if action == CONSUMED and child is not None else (ERROR, None)
        return ERROR, None

    def child_done(self, result: object) -> None:
        pass
