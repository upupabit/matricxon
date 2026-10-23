from collections.abc import Iterable
from typing import Protocol


class _StoredEntry(Protocol):
    model_id: str
    length: int
    blocks: list[str]
    saved_at: float
    tag: str


class SupersededCopies:
    """Older saved copies of a chat that a newly saved one fully covers.

    Each time a chat leaves RAM a new file is written, and an older one stayed until it expired
    (one chat showed as several "saved chats"). Every older copy of the same chat and model is
    superseded, in either precision: the newest state is the one worth restoring. That includes
    another branch (a message since deleted or edited, about 160 MB each for a 5000-token chat),
    which the chat never goes back to.
    """

    @staticmethod
    def find(
        new_name: str, new: _StoredEntry, stored: Iterable[tuple[str, _StoredEntry]]
    ) -> list[str]:
        if not new.tag:
            return []
        return [
            name
            for name, old in stored
            if name != new_name
            and old.tag == new.tag
            and old.model_id == new.model_id
            and old.saved_at <= new.saved_at
        ]

    @staticmethod
    def chat_count(stored: Iterable[_StoredEntry]) -> int:
        """Distinct chats with a saved copy (files without a chat tag count one each)."""
        tags: set[str] = set()
        untagged = 0
        for entry in stored:
            if entry.tag:
                tags.add(entry.tag)
            else:
                untagged += 1
        return len(tags) + untagged
