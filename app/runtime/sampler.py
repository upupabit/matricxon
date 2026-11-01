import logging
import time

import torch

from app.native.fused_ops import FusedOps
from app.runtime.generation_request import SamplingConfig, TokenConstraint

logger = logging.getLogger(__name__)


class RepetitionPenaltyFilter:
    """Divides (or, for negative logits, multiplies) already-generated tokens' scores by `penalty`,
    per Ollama/llama.cpp's `repeat_penalty`: suppresses repeats without fully forbidding a token."""

    def __init__(self, penalty: float) -> None:
        self._penalty = penalty

    def apply(self, logits: torch.Tensor, generated_ids: list[int]) -> torch.Tensor:
        if self._penalty == 1.0 or not generated_ids:
            return logits
        logits = logits.clone()
        indices = torch.tensor(sorted(set(generated_ids)), dtype=torch.long)
        scores = logits[indices]
        logits[indices] = torch.where(scores > 0, scores / self._penalty, scores * self._penalty)
        return logits


class TemperatureScaler:
    """`temperature <= 0` short-circuits to greedy argmax (Ollama's convention); the rest of the
    pipeline never runs."""

    def __init__(self, temperature: float) -> None:
        self._temperature = temperature

    @property
    def is_greedy(self) -> bool:
        return self._temperature <= 0.0

    def apply(self, logits: torch.Tensor) -> torch.Tensor:
        return logits / self._temperature


class TopKFilter:
    def __init__(self, k: int) -> None:
        self._k = k

    def apply(self, logits: torch.Tensor) -> torch.Tensor:
        if self._k <= 0 or self._k >= logits.shape[-1]:
            return logits
        threshold = torch.topk(logits, self._k).values[-1]
        return logits.masked_fill(logits < threshold, float("-inf"))


class TopPFilter:
    """Nucleus sampling: keeps the smallest prefix of sorted probabilities whose cumulative mass
    reaches `p`, always keeping at least the top token."""

    def __init__(self, p: float) -> None:
        self._p = p

    def apply(self, logits: torch.Tensor) -> torch.Tensor:
        if self._p >= 1.0:
            return logits
        sorted_logits, sorted_indices = torch.sort(logits, descending=True)
        cumulative_probs = torch.cumsum(torch.softmax(sorted_logits, dim=-1), dim=-1)

        remove = cumulative_probs - torch.softmax(sorted_logits, dim=-1) > self._p
        sorted_logits = sorted_logits.masked_fill(remove, float("-inf"))

        result = torch.full_like(logits, float("-inf"))
        result.scatter_(-1, sorted_indices, sorted_logits)
        return result


class PresenceFrequencyPenalty:
    """OpenAI's penalties: a token already in the reply loses `presence` once and `frequency` for
    each time it appeared, so repeats get rarer without being forbidden."""

    def __init__(self, presence: float, frequency: float) -> None:
        self._presence = presence
        self._frequency = frequency

    def apply(self, logits: torch.Tensor, generated_ids: list[int]) -> torch.Tensor:
        if (self._presence == 0.0 and self._frequency == 0.0) or not generated_ids:
            return logits
        counts = torch.bincount(
            torch.tensor(generated_ids, dtype=torch.long), minlength=len(logits)
        )
        counts = counts[: len(logits)].to(logits.dtype)
        return logits - (counts > 0).to(logits.dtype) * self._presence - counts * self._frequency


class MinPFilter:
    """Keeps tokens whose probability is at least `p` times the most likely token's."""

    def __init__(self, p: float) -> None:
        self._p = p

    def apply(self, logits: torch.Tensor) -> torch.Tensor:
        if self._p <= 0.0:
            return logits
        probabilities = torch.softmax(logits, dim=-1)
        return logits.masked_fill(probabilities < self._p * probabilities.max(), float("-inf"))


class Sampler:
    """Ollama-option-shaped sampling pipeline: repeat-penalty -> temperature -> top-k -> top-p ->
    multinomial, run once per generated token on that step's vocab-sized logits."""

    # Draws tried before a constrained pick falls back to the best allowed token.
    CONSTRAINED_TRIES = 32

    def __init__(self, sampling: SamplingConfig, constraint: TokenConstraint | None = None) -> None:
        self._sampling = sampling
        self._constraint = constraint
        self._repetition_penalty = RepetitionPenaltyFilter(sampling.repeat_penalty)
        self._penalties = PresenceFrequencyPenalty(
            sampling.presence_penalty, sampling.frequency_penalty
        )
        self._temperature = TemperatureScaler(sampling.temperature)
        self._top_k = TopKFilter(sampling.top_k)
        self._min_p = MinPFilter(sampling.min_p)
        self._top_p = TopPFilter(sampling.top_p)
        # The native/Numba sampler covers the original four options only; the rest run here.
        self._fused_ok = (
            constraint is None
            and sampling.min_p <= 0.0
            and not (sampling.presence_penalty or sampling.frequency_penalty)
        )

        self._generator = torch.Generator()
        if sampling.seed is not None:
            self._generator.manual_seed(sampling.seed)
        else:
            self._generator.seed()

    def sample(self, logits: torch.Tensor, generated_ids: list[int]) -> int:
        stage_started = time.monotonic()
        logits = logits.to(torch.float32)
        ops = FusedOps.active()
        if self._fused_ok and ops is not None and ops.can_sample(logits, self._sampling):
            u = float(torch.rand((), generator=self._generator))
            token_id = ops.sample(logits, generated_ids, self._sampling, u)
            logger.debug(
                "picking next word: fused sample -> token %d in %.2fms",
                token_id,
                (time.monotonic() - stage_started) * 1000,
            )
            return token_id
        logits = self._repetition_penalty.apply(logits, generated_ids)
        logits = self._penalties.apply(logits, generated_ids)
        if self._constraint is not None:
            return self._constrained(logits, stage_started)

        if self._temperature.is_greedy:
            token_id = int(torch.argmax(logits).item())
            logger.debug(
                "picking next word: greedy argmax -> token %d in %.2fms",
                token_id,
                (time.monotonic() - stage_started) * 1000,
            )
            return token_id

        probabilities = torch.softmax(self._filtered(logits), dim=-1)
        sampled = torch.multinomial(probabilities, num_samples=1, generator=self._generator)
        token_id = int(sampled.item())
        logger.debug(
            "picking next word: temperature/top-k/top-p sample -> token %d in %.2fms",
            token_id,
            (time.monotonic() - stage_started) * 1000,
        )
        return token_id

    def _filtered(self, logits: torch.Tensor) -> torch.Tensor:
        logits = self._temperature.apply(logits)
        logits = self._top_k.apply(logits)
        logits = self._min_p.apply(logits)
        return self._top_p.apply(logits)

    def _constrained(self, logits: torch.Tensor, stage_started: float) -> int:
        """Draws as usual but only keeps a token the constraint allows: a refused draw is removed
        and drawn again. When the filters left no allowed token (or after CONSTRAINED_TRIES), the
        most likely allowed token of the whole vocabulary wins."""
        constraint = self._constraint
        assert constraint is not None
        token_id = -1
        if not self._temperature.is_greedy:
            probabilities = torch.softmax(self._filtered(logits), dim=-1)
            for _ in range(self.CONSTRAINED_TRIES):
                if not bool(torch.isfinite(probabilities).all()) or probabilities.sum() <= 0:
                    break
                drawn = int(torch.multinomial(probabilities, 1, generator=self._generator).item())
                if constraint.allows(drawn):
                    token_id = drawn
                    break
                probabilities[drawn] = 0.0
        if token_id < 0:
            token_id = next(
                (
                    int(i)
                    for i in torch.argsort(logits, descending=True)
                    if constraint.allows(int(i))
                ),
                int(torch.argmax(logits).item()),
            )
        constraint.accept(token_id)
        logger.debug(
            "picking next word: constrained -> token %d in %.2fms",
            token_id,
            (time.monotonic() - stage_started) * 1000,
        )
        return token_id
