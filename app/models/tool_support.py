"""What a model's chat template shows it can do: tool calling (a call format we can parse back), a
thinking switch (`enable_thinking`, ChatRequest.think), and whether flipping that switch changes the
history part of the prompt (Gemma 4 marks the system turn), so a live chat is read again once."""

from pathlib import Path

from app.gguf.metadata import GGUFMetadata
from app.gguf.reader import GGUFReader
from app.runtime.chat_template import ChatTemplatePromptBuilder
from app.runtime.tool_call_formats import ToolCallFormats
from app.schemas.chat import ChatMessage
from app.server.errors import MatricxonError

# Keyed by (path, mtime, size): read on every GET /api/tags, and a file's header never changes
# without the file (like has_confirmed_chat_format). Value: (tools, thinking, thinking rereads).
_cache: dict[tuple[str, int, int], tuple[bool, bool, bool]] = {}

_PROBE = [
    ChatMessage(role="system", content="Be brief."),
    ChatMessage(role="user", content="Hi"),
    ChatMessage(role="assistant", content="Hello!"),
    ChatMessage(role="user", content="Bye"),
]


def _switch_changes_history(metadata: GGUFMetadata) -> bool:
    """True when the history renders differently with thinking on and off (the generation prompt
    aside): a live chat's cache then stops matching near its start."""
    builder = ChatTemplatePromptBuilder.from_metadata(metadata)
    if builder is None:
        return False
    try:
        on = builder._render(_PROBE, None, False, True)
        off = builder._render(_PROBE, None, False, False)
    except MatricxonError:
        return False
    return on != off


def _template_features(gguf_path: str | Path) -> tuple[bool, bool, bool]:
    path = Path(gguf_path)
    stat = path.stat()
    key = (str(path), stat.st_mtime_ns, stat.st_size)
    if key not in _cache:
        metadata = GGUFReader(path).read().metadata
        template = metadata.get("tokenizer.chat_template")
        tools = ToolCallFormats.for_template(template, metadata.architecture) is not None
        thinks = isinstance(template, str) and "enable_thinking" in template
        _cache[key] = (tools, thinks, thinks and _switch_changes_history(metadata))
    return _cache[key]


def supports_tools(gguf_path: str | Path) -> bool:
    return _template_features(gguf_path)[0]


def supports_thinking(gguf_path: str | Path) -> bool:
    return _template_features(gguf_path)[1]


def thinking_rereads(gguf_path: str | Path) -> bool:
    return _template_features(gguf_path)[2]
