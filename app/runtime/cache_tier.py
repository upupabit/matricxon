import logging
import time

import torch

from app.architectures.base import ModelArchitecture
from app.runtime.cache_miss_diagnosis import CacheMissDiagnosis
from app.runtime.cache_store import BLOCK_TOKENS, PersistentCacheStore
from app.runtime.prompt_slot import CacheSlot
from app.runtime.slot_codec import SlotCodec

logger = logging.getLogger(__name__)


class ModelCacheTier:
    """One loaded model's view of the on-disk store: its `PromptCache` spills evicted slots here and
    asks here for a conversation it no longer holds in RAM. `model_id` ties files to exact weights
    (tag + file size), so a replaced model never picks up another's cache."""

    def __init__(self, store: PersistentCacheStore, model_id: str) -> None:
        self._store = store
        self._model_id = model_id

    def spill(self, slot: CacheSlot) -> None:
        """Leaving RAM: written unless this exact state is already on disk (`checkpoint`)."""
        if slot.saved_digest == slot.digest() and self._store.holds_copy(
            self._model_id, slot.tag, len(slot.token_ids)
        ):
            return
        self._store.spill_async(self._model_id, slot)

    def checkpoint(self, slot: CacheSlot) -> None:
        """After a reply, at most every `save_every_seconds`: a copy of the chat goes to disk in
        the background, so a crash or a forced stop loses at most the last few replies, not the
        whole chat (it used to be written only on eviction, unload and shutdown). The copy is
        taken here, under the pool's lock; serializing and writing run on the store's writer."""
        policy = self._store.policy
        if not slot.tag or not policy.enabled or policy.save_every_seconds <= 0:
            return
        if time.monotonic() - slot.saved_at < policy.save_every_seconds:
            return  # the copy on disk is a prefix of this chat: restoring it still reuses most
        digest = slot.digest()
        if digest == slot.saved_digest:
            return
        state = SlotCodec.capture(slot)
        if state is not None:
            slot.saved_digest, slot.saved_at = digest, time.monotonic()
            self._store.checkpoint_async(self._model_id, state, slot.tag)

    def restore(
        self,
        architecture: ModelArchitecture,
        prompt_ids: list[int],
        num_ctx: int,
        dtype: torch.dtype,
        min_tokens: int,
        tag: str = "",
    ) -> CacheSlot | None:
        """A slot rebuilt from disk sharing more than `min_tokens` leading tokens with `prompt_ids`,
        or None (nothing stored, or the file no longer fits this model)."""
        taken = self._store.take(self._model_id, prompt_ids, dtype, min_tokens, tag)
        if taken is None:
            return None
        payload, tag = taken
        try:
            slot = SlotCodec.decode(payload, architecture, num_ctx, dtype)
        except (ValueError, KeyError, RuntimeError):
            logger.warning("a stored prompt cache no longer fits its model, ignoring it")
            return None
        slot.tag = tag
        logger.info("restored a %d-token prompt cache from disk", len(slot.token_ids))
        return slot

    def explain(self, prompt_ids: list[int], dtype: torch.dtype, tag: str) -> str:
        """Why the disk tier had nothing for this prompt (for the log)."""
        entries, wanted = self._store.inspect(prompt_ids)
        return CacheMissDiagnosis.disk(
            entries,
            wanted,
            BLOCK_TOKENS,
            self._model_id,
            dtype,
            tag,
            self._store.policy.enabled,
        )
