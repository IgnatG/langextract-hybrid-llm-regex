"""Rule definitions for the hybrid provider.

Rules are tried before the LLM fallback.  Each rule receives
a prompt string and either returns an extraction result (hit)
or signals that it cannot handle the prompt (miss), causing
fallback to the LLM.
"""

from __future__ import annotations

import abc
import json
import logging
import re
from dataclasses import dataclass, field
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from collections.abc import Callable

__all__ = [
    "CallableRule",
    "ExtractionRule",
    "RegexRule",
    "RuleConfig",
    "RuleResult",
]

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class RuleResult:
    """Result of a rule evaluation.

    Attributes:
        hit: ``True`` if the rule matched and produced output.
        output: The extracted text when ``hit`` is ``True``.
            ``None`` on a miss.
        confidence: Confidence score for the extraction.
            Defaults to ``1.0`` for deterministic rules.
    """

    hit: bool
    output: str | None = None
    confidence: float = 1.0


class ExtractionRule(abc.ABC):
    """Abstract base class for extraction rules."""

    @abc.abstractmethod
    def evaluate(self, prompt: str) -> RuleResult:
        """Attempt to extract data from the prompt using rules.

        Parameters:
            prompt: The full prompt text that would be sent to the
                LLM.

        Returns:
            A ``RuleResult`` indicating hit or miss.
        """


class RegexRule(ExtractionRule):
    """Rule that extracts data by matching regex patterns.

    When the pattern matches, the rule constructs a JSON output
    from named capture groups.  This is ideal for extracting
    dates, amounts, reference numbers, and other structured
    entities from well-formatted text.

    Parameters:
        pattern: A regex pattern with named groups.
        output_template: Optional callable that transforms the
            match dict into the desired output string.  Defaults
            to ``json.dumps(match.groupdict())``.
        description: Human-readable description of the rule.
        confidence: Confidence score to assign on match.
    """

    def __init__(
        self,
        pattern: str,
        output_template: Callable[[dict[str, str]], str] | None = None,
        description: str = "regex rule",
        confidence: float = 1.0,
    ) -> None:
        self._pattern = re.compile(pattern, re.DOTALL)
        self._output_template = output_template
        self._description = description
        self._confidence = confidence

    @property
    def description(self) -> str:
        """Return the human-readable rule description.

        Returns:
            The rule description string.
        """
        return self._description

    def evaluate(self, prompt: str) -> RuleResult:
        """Evaluate the regex pattern against the prompt.

        Parameters:
            prompt: The full prompt text.

        Returns:
            A ``RuleResult`` with the extraction on match, or a
            miss if the pattern does not match.
        """
        match = self._pattern.search(prompt)
        if match is None:
            return RuleResult(hit=False)

        groups = match.groupdict()
        if self._output_template is not None:
            output = self._output_template(groups)
        else:
            output = json.dumps(groups, ensure_ascii=False)

        return RuleResult(
            hit=True,
            output=output,
            confidence=self._confidence,
        )


class CallableRule(ExtractionRule):
    """Rule backed by an arbitrary callable.

    The callable receives the prompt text and returns either a
    string (hit) or ``None`` (miss).

    Parameters:
        func: A callable ``(str) -> str | None``.
        description: Human-readable description of the rule.
        confidence: Confidence score to assign on hit.
    """

    def __init__(
        self,
        func: Callable[[str], str | None],
        description: str = "callable rule",
        confidence: float = 1.0,
    ) -> None:
        self._func = func
        self._description = description
        self._confidence = confidence

    @property
    def description(self) -> str:
        """Return the human-readable rule description.

        Returns:
            The rule description string.
        """
        return self._description

    def evaluate(self, prompt: str) -> RuleResult:
        """Evaluate the callable against the prompt.

        Parameters:
            prompt: The full prompt text.

        Returns:
            A ``RuleResult`` based on the callable's return value.
        """
        try:
            result = self._func(prompt)
        except Exception:
            logger.exception(
                "Callable rule '%s' raised an exception",
                self._description,
            )
            return RuleResult(hit=False)

        if result is None:
            return RuleResult(hit=False)

        return RuleResult(
            hit=True,
            output=result,
            confidence=self._confidence,
        )


@dataclass
class RuleConfig:
    """Configuration bundle for a set of extraction rules.

    Parameters:
        rules: Ordered list of ``ExtractionRule`` instances.
            Rules are evaluated in order; the first hit wins.
        fallback_on_low_confidence: If ``True``, results with
            ``confidence`` below ``min_confidence`` trigger LLM
            fallback instead.
        min_confidence: Minimum confidence threshold for rule
            results. Only used when ``fallback_on_low_confidence``
            is ``True``.
        text_extractor: Optional callable that extracts the
            document text portion from a full prompt string.
            When set, rules receive only the extracted text
            instead of the entire prompt (which may include
            instructions and examples that could cause false
            matches).  ``None`` (default) passes the full
            prompt to rules unchanged.
        output_formatter: Optional callable that post-processes
            rule output strings before returning them as
            ``ScoredOutput``.  Useful for conforming rule
            output to the format expected by downstream
            pipeline stages.  ``None`` (default) returns
            rule output as-is.
    """

    rules: list[ExtractionRule] = field(default_factory=list)
    fallback_on_low_confidence: bool = False
    min_confidence: float = 0.8
    text_extractor: Callable[[str], str] | None = None
    output_formatter: Callable[[str], str] | None = None
