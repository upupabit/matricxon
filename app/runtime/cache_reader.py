import logging
import os
import struct
import time

import torch

from app.runtime.cache_crypto import CacheDecryptError
from app.runtime.cache_files import MAGIC, CacheFilesBase, StoredEntry

logger = logging.getLogger("app.runtime.cache_store")


class CacheReader(CacheFilesBase):
    """The restoring half of `PersistentCacheStore`: finds the stored chat sharing the longest
    prefix with a prompt and consumes its file."""

    def take(
        self,
        model_id: str,
        prompt_ids: list[int],
        dtype: torch.dtype,
        min_tokens: int,
        tag: str = "",
    ) -> tuple[bytes, str] | None:
        """The serialized slot sharing the longest prefix (more than `min_tokens`) with
        `prompt_ids`, removed from disk, with the chat tag it was saved under, or None. Matching
        uses the sealed per-block hashes, so a file that wouldn't help is never read, let alone
        consumed."""
        if not self.policy.enabled:
            return None
        # A save still being written (e.g. this model's own unload a moment ago) would be missed.
        # Copies of chats still in RAM (`checkpoint_async`) aren't waited for.
        if self._spills_pending:
            self._writer.submit(lambda: None).result()
        self._rescan_if_due()
        wanted = self._cipher.prefix_fingerprints(prompt_ids, self._block_tokens())
        best = self._best_match(model_id, dtype, wanted, min_tokens, tag)
        if best is None and time.monotonic() - self._last_scan >= self.MISS_RESCAN_SECONDS:
            self._refresh()
            best = self._best_match(model_id, dtype, wanted, min_tokens, tag)
        if best is None:
            return None
        body = self._consume(best[1])
        return None if body is None else (body, best[2])

    def _best_match(
        self, model_id: str, dtype: torch.dtype, wanted: list[str], min_tokens: int, tag: str = ""
    ) -> tuple[int, str, str] | None:
        """(shared blocks, file name, chat tag) of the stored file sharing the longest prefix. Any
        precision: a restore converts it (`KVCache.load`). On a tie (a trimmed chat shares only its
        system prompt with every copy) the asking chat's own copy wins, then the same precision,
        then the copy holding more tokens: a chat's old branch must not beat its newest state."""
        best: tuple[int, str, str] | None = None
        best_rank: tuple[int, bool, bool, int] | None = None
        block = self._block_tokens()
        with self._lock:
            for name, entry in self._index.items():
                if entry.model_id != model_id:
                    continue
                shared = 0
                for mine, theirs in zip(wanted, entry.blocks, strict=False):
                    if mine != theirs:
                        break
                    shared += 1
                rank = (
                    shared,
                    bool(tag) and entry.tag == tag,
                    entry.dtype_name == str(dtype),
                    entry.length,
                )
                if shared * block > min_tokens and (best_rank is None or rank > best_rank):
                    best, best_rank = (shared, name, entry.tag), rank
        return best

    def inspect(self, prompt_ids: list[int]) -> tuple[list[StoredEntry], list[str]]:
        """The files' headers plus `prompt_ids`' block hashes, to explain a miss."""
        with self._lock:
            entries = list(self._index.values())
        return entries, self._cipher.prefix_fingerprints(prompt_ids, self._block_tokens())

    def _consume(self, name: str) -> bytes | None:
        """Claims the file by renaming it (atomic: of several nodes, one wins), reads, deletes."""
        path = self._dir / f"{name}{self._claim_suffix}"
        try:
            os.replace(self._dir / name, path)
        except FileNotFoundError:
            with self._lock:
                self._index.pop(name, None)
            return None
        try:
            with path.open("rb") as handle:
                handle.seek(len(MAGIC))
                (meta_len,) = struct.unpack(">I", handle.read(4))
                meta = handle.read(meta_len)
                body = self._cipher.open(handle.read(), MAGIC + meta)
        except (CacheDecryptError, OSError, struct.error):
            logger.warning("prompt-cache file %s unreadable, dropping it", name)
            self._remove(name, path)
            return None
        self._remove(name, path)
        return body
