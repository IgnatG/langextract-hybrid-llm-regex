"""Hybrid rule + LLM provider implementation.

Dispatches each prompt to a rule engine first.  If a rule
matches with sufficient confidence, the deterministic result
is returned immediately (no LLM call).  On rule miss or low
confidence, the prompt falls back to the wrapped LLM provider.
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING, Any

import langextract as lx
from langextract.core.base_model import BaseLanguageModel
from langextract.core.types import ScoredOutput

from langextract_hybrid.rules import RuleConfig, RuleResult

if TYPE_CHECKING:
    from collections.abc import Iterator, Sequence

logger = logging.getLogger(__name__)


@lx.providers.registry.register(r"^hybrid", priority=5)
class HybridLanguageModel(BaseLanguageModel):
    """Provider combining deterministic rules with LLM fallback.

    For each prompt, rules are evaluated in order.  The first
    matching rule's output is returned as a ``ScoredOutput`` with
    the rule's confidence as the score.  If no rule matches (or
    the match confidence is too low), the prompt is forwarded to
    the inner LLM provider.

    This can save 50-80% of LLM costs on well-structured
    documents where many entities (dates, amounts, reference
    numbers) follow predictable patterns.

    Parameters:
        model_id: The model identifier (typically prefixed with
            ``hybrid/``).
        inner: The fallback ``BaseLanguageModel`` for prompts that
            rules cannot handle.
        rule_config: A ``RuleConfig`` containing the ordered list
            of extraction rules and confidence settings.
        **kwargs: Additional keyword arguments forwarded to the
            base class.
    """

    def __init__(
        self,
        model_id: str,
        *,
        inner: BaseLanguageModel,
        rule_config: RuleConfig,
        **kwargs: Any,
    ) -> None:
        super().__init__(**kwargs)
        self.model_id = model_id
        self._inner = inner
        self._rule_config = rule_config

        # Counters for observability
        self._rule_hits: int = 0
        self._llm_fallbacks: int = 0

    # -- Public accessors --

    @property
    def inner(self) -> BaseLanguageModel:
        """Return the fallback LLM provider.

        Returns:
            The inner ``BaseLanguageModel`` instance.
        """
        return self._inner

    @property
    def rule_config(self) -> RuleConfig:
        """Return the rule configuration.

        Returns:
            The ``RuleConfig`` instance.
        """
        return self._rule_config

    @property
    def rule_hits(self) -> int:
        """Return the number of prompts resolved by rules.

        Returns:
            Count of rule hits since instantiation.
        """
        return self._rule_hits

    @property
    def llm_fallbacks(self) -> int:
        """Return the number of prompts that fell back to the LLM.

        Returns:
            Count of LLM fallbacks since instantiation.
        """
        return self._llm_fallbacks

    def reset_counters(self) -> None:
        """Reset the hit/fallback counters to zero."""
        self._rule_hits = 0
        self._llm_fallbacks = 0

    # -- Private helpers --

    def _try_rules(self, prompt: str) -> RuleResult:
        """Evaluate rules against a prompt, return first hit.

        Parameters:
            prompt: The prompt text.

        Returns:
            A ``RuleResult`` from the first matching rule, or a
            miss result if no rule matched.
        """
        for rule in self._rule_config.rules:
            result = rule.evaluate(prompt)
            if result.hit:
                # Check confidence threshold
                if (
                    self._rule_config.fallback_on_low_confidence
                    and result.confidence < self._rule_config.min_confidence
                ):
                    logger.debug(
                        "Rule hit with low confidence %.2f "
                        "(threshold %.2f) — falling back to LLM",
                        result.confidence,
                        self._rule_config.min_confidence,
                    )
                    continue
                return result
        return RuleResult(hit=False)

    # -- BaseLanguageModel interface --

    def infer(
        self,
        batch_prompts: Sequence[str],
        **kwargs: Any,
    ) -> Iterator[Sequence[ScoredOutput]]:
        """Inference with rule pre-check and LLM fallback.

        Each prompt is first checked against the configured rules.
        On a rule hit, the deterministic result is yielded
        immediately.  On a miss, the prompt is forwarded to the
        inner LLM provider.

        Parameters:
            batch_prompts: A sequence of prompt strings.
            **kwargs: Additional keyword arguments forwarded to the
                inner provider on fallback.

        Yields:
            Sequences of ``ScoredOutput`` per prompt.
        """
        for prompt in batch_prompts:
            rule_result = self._try_rules(prompt)

            if rule_result.hit:
                self._rule_hits += 1
                logger.debug("Rule hit for prompt — skipping LLM")
                yield [
                    ScoredOutput(
                        score=rule_result.confidence,
                        output=rule_result.output,
                    )
                ]
            else:
                self._llm_fallbacks += 1
                logger.debug("Rule miss — falling back to LLM")
                # Delegate to the inner provider for this single
                # prompt
                results = list(self._inner.infer([prompt], **kwargs))
                if results:
                    yield results[0]
                else:
                    yield [
                        ScoredOutput(
                            score=0.0,
                            output="",
                        )
                    ]

    async def async_infer(
        self,
        batch_prompts: Sequence[str],
        **kwargs: Any,
    ) -> list[Sequence[ScoredOutput]]:
        """Async inference with rule pre-check and LLM fallback.

        Rule evaluation is synchronous (it's fast).  Only prompts
        that miss all rules are batched together and sent to the
        inner provider's ``async_infer`` for concurrent processing.

        Parameters:
            batch_prompts: A sequence of prompt strings.
            **kwargs: Additional keyword arguments forwarded to the
                inner provider on fallback.

        Returns:
            A list of ``ScoredOutput`` sequences per prompt.
        """
        # Phase 1 — evaluate rules synchronously
        resolved: dict[int, list[ScoredOutput]] = {}
        fallback_indices: list[int] = []
        fallback_prompts: list[str] = []

        for idx, prompt in enumerate(batch_prompts):
            rule_result = self._try_rules(prompt)
            if rule_result.hit:
                self._rule_hits += 1
                resolved[idx] = [
                    ScoredOutput(
                        score=rule_result.confidence,
                        output=rule_result.output,
                    )
                ]
            else:
                self._llm_fallbacks += 1
                fallback_indices.append(idx)
                fallback_prompts.append(prompt)

        # Phase 2 — send misses to the LLM in one batch
        if fallback_prompts:
            llm_results = await self._inner.async_infer(fallback_prompts, **kwargs)
            for i, llm_idx in enumerate(fallback_indices):
                if i < len(llm_results):
                    resolved[llm_idx] = list(llm_results[i])
                else:
                    resolved[llm_idx] = [ScoredOutput(score=0.0, output="")]

        # Phase 3 — reassemble in original order
        return [
            resolved.get(i, [ScoredOutput(score=0.0, output="")])
            for i in range(len(batch_prompts))
        ]
