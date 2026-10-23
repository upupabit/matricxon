import logging
import os
import struct
import tempfile
import time
import uuid

from app.runtime.cache_files import MAGIC, SUFFIX, CacheFilesBase, StoredEntry
from app.runtime.cache_superseded import SupersededCopies
from app.runtime.prompt_slot import CacheSlot
from app.runtime.slot_codec import EncodedSlot, SlotCodec, SlotState

logger = logging.getLogger("app.runtime.cache_store")


class CacheWriter(CacheFilesBase):
    """The writing half of `PersistentCacheStore`: chats leaving RAM (`spill_async`) and copies of
    chats still in RAM (`checkpoint_async`), encoded, sealed and written by one background
    thread."""

    def spill_async(self, model_id: str, slot: CacheSlot) -> None:
        """Queues `slot` for the background writer to encode and write; the caller must not touch
        the slot afterwards (the pool has dropped it)."""
        if self.policy.enabled and self.policy.budget_bytes > 0:
            with self._lock:
                self._spills_pending += 1
            self._writer.submit(self._spill, model_id, slot)

    def checkpoint_async(self, model_id: str, state: SlotState, tag: str) -> None:
        """Writes a copy of a chat still in RAM. A newer copy of the same chat queued before this
        one is written replaces it, so a slow disk never falls behind by more than one copy."""
        if not (self.policy.enabled and self.policy.budget_bytes > 0):
            return
        with self._lock:
            queued = (model_id, tag) in self._checkpoints
            self._checkpoints[(model_id, tag)] = state
        if not queued:
            self._writer.submit(self._write_checkpoint, model_id, tag)

    def _write_checkpoint(self, model_id: str, tag: str) -> None:
        with self._lock:
            state = self._checkpoints.pop((model_id, tag))
        try:
            if tag not in self._forgotten:
                self._write(model_id, SlotCodec.serialize(state), tag)
        except Exception:  # noqa: BLE001 - a failed copy only costs a later re-prefill
            logger.exception("saving a prompt cache copy failed")

    def holds_copy(self, model_id: str, tag: str, length: int) -> bool:
        """Is a file of this chat with exactly `length` tokens on disk (or queued)?"""
        with self._lock:
            return (model_id, tag) in self._checkpoints or any(
                e.model_id == model_id and e.tag == tag and e.length == length
                for e in self._index.values()
            )

    def flush(self) -> None:
        """Blocks until every queued spill is on disk (shutdown)."""
        started = time.monotonic()
        self._writer.submit(lambda: None).result()
        if time.monotonic() - started > 1.0:
            logger.info("prompt caches written to disk in %.1fs", time.monotonic() - started)

    def _spill(self, model_id: str, slot: CacheSlot) -> None:
        try:
            encoded = SlotCodec.encode(slot)
            if encoded is not None and slot.tag not in self._forgotten:
                self._write(model_id, encoded, slot.tag)
        except Exception:  # noqa: BLE001 - a failed spill only costs a later re-prefill
            logger.exception("persisting a prompt cache failed")
        finally:
            with self._lock:
                self._spills_pending -= 1

    def _write(self, model_id: str, encoded: EncodedSlot, tag: str = "") -> None:
        entry = StoredEntry(
            model_id,
            encoded.num_ctx,
            encoded.dtype_name,
            len(encoded.token_ids),
            self._cipher.prefix_fingerprints(encoded.token_ids, self._block_tokens()),
            time.time(),
            tag,
        )
        meta = self._cipher.seal(entry.meta_json(), MAGIC)
        body = self._cipher.seal(encoded.payload, MAGIC + meta)
        blob = MAGIC + struct.pack(">I", len(meta)) + meta + body
        if len(blob) > self.policy.budget_bytes:
            return
        fd, tmp = tempfile.mkstemp(dir=self._dir, prefix=f"{self._node_id}-", suffix=".tmp")
        with os.fdopen(fd, "wb") as handle:
            handle.write(blob)
        name = f"{uuid.uuid4().hex}{SUFFIX}"
        os.replace(tmp, self._dir / name)
        entry.size = len(blob)
        with self._lock:
            self._index[name] = entry
            gone = tag in self._forgotten
            superseded = SupersededCopies.find(name, entry, list(self._index.items()))
        if gone:
            self._remove(name)
        for old in superseded:
            self._remove(old)
        self.enforce_limits()
