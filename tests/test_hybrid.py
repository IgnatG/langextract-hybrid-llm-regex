"""Tests for HybridLanguageModel provider and extraction rules."""

from __future__ import annotations

import json
from typing import TYPE_CHECKING
from unittest import mock

import pytest
from langcore.core.base_model import BaseLanguageModel
from langcore.core.types import ScoredOutput

from langcore_hybrid import (
    CallableRule,
    HybridLanguageModel,
    RegexRule,
    RuleConfig,
)

if TYPE_CHECKING:
    from collections.abc import Iterator, Sequence


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


class _StubLLM(BaseLanguageModel):
    """Minimal LLM stub that returns a fixed response."""

    def __init__(
        self,
        response: str = '{"llm": true}',
    ) -> None:
        super().__init__()
        self._response = response
        self.call_count = 0
        self.prompts_received: list[str] = []

    def infer(
        self,
        batch_prompts: Sequence[str],
        **kwargs,
    ) -> Iterator[Sequence[ScoredOutput]]:
        for prompt in batch_prompts:
            self.call_count += 1
            self.prompts_received.append(prompt)
            yield [ScoredOutput(score=1.0, output=self._response)]

    async def async_infer(
        self,
        batch_prompts: Sequence[str],
        **kwargs,
    ) -> list[Sequence[ScoredOutput]]:
        results: list[Sequence[ScoredOutput]] = []
        for prompt in batch_prompts:
            self.call_count += 1
            self.prompts_received.append(prompt)
            results.append([ScoredOutput(score=1.0, output=self._response)])
        return results


# ---------------------------------------------------------------------------
# Rule tests
# ---------------------------------------------------------------------------


class TestRegexRule:
    """Tests for RegexRule."""

    def test_match_returns_json_groups(self) -> None:
        rule = RegexRule(
            r"Date:\s*(?P<date>\d{4}-\d{2}-\d{2})",
            description="date extraction",
        )
        result = rule.evaluate("The contract Date: 2026-01-15 is effective.")
        assert result.hit
        parsed = json.loads(result.output or "")
        assert parsed["date"] == "2026-01-15"

    def test_no_match_returns_miss(self) -> None:
        rule = RegexRule(
            r"Date:\s*(?P<date>\d{4}-\d{2}-\d{2})",
        )
        result = rule.evaluate("No date here.")
        assert not result.hit
        assert result.output is None

    def test_custom_output_template(self) -> None:
        rule = RegexRule(
            r"Amount:\s*\$(?P<amount>[\d,.]+)",
            output_template=lambda g: json.dumps({"amount_usd": g["amount"]}),
        )
        result = rule.evaluate("Amount: $1,234.56")
        assert result.hit
        parsed = json.loads(result.output or "")
        assert parsed["amount_usd"] == "1,234.56"

    def test_confidence_score(self) -> None:
        rule = RegexRule(
            r"(?P<val>\d+)",
            confidence=0.95,
        )
        result = rule.evaluate("42")
        assert result.hit
        assert result.confidence == 0.95


class TestCallableRule:
    """Tests for CallableRule."""

    def test_callable_hit(self) -> None:
        def extract_amount(prompt: str) -> str | None:
            if "$" in prompt:
                return '{"found_dollar": true}'
            return None

        rule = CallableRule(
            func=extract_amount,
            description="dollar check",
        )
        result = rule.evaluate("Pay $100")
        assert result.hit
        assert result.output == '{"found_dollar": true}'

    def test_callable_miss(self) -> None:
        rule = CallableRule(func=lambda p: None)
        result = rule.evaluate("no match")
        assert not result.hit

    def test_callable_exception_returns_miss(self) -> None:
        def bad_func(prompt: str) -> str | None:
            raise ValueError("boom")

        rule = CallableRule(func=bad_func)
        result = rule.evaluate("anything")
        assert not result.hit


class TestRuleConfig:
    """Tests for RuleConfig."""

    def test_default_empty_rules(self) -> None:
        config = RuleConfig()
        assert config.rules == []
        assert config.fallback_on_low_confidence is False
        assert config.min_confidence == 0.8


# ---------------------------------------------------------------------------
# Provider tests — sync
# ---------------------------------------------------------------------------


class TestHybridProviderSync:
    """Tests for HybridLanguageModel sync inference."""

    def test_rule_hit_skips_llm(self) -> None:
        llm = _StubLLM()
        rule = RegexRule(
            r"Date:\s*(?P<date>\d{4}-\d{2}-\d{2})",
        )
        hybrid = HybridLanguageModel(
            model_id="hybrid/test",
            inner=llm,
            rule_config=RuleConfig(rules=[rule]),
        )
        results = list(hybrid.infer(["Extract: Date: 2026-03-01"]))
        assert len(results) == 1
        parsed = json.loads(results[0][0].output or "")
        assert parsed["date"] == "2026-03-01"
        # LLM should NOT have been called
        assert llm.call_count == 0
        assert hybrid.rule_hits == 1
        assert hybrid.llm_fallbacks == 0

    def test_rule_miss_falls_back_to_llm(self) -> None:
        llm = _StubLLM(response='{"llm_result": true}')
        rule = RegexRule(r"WONTMATCH")
        hybrid = HybridLanguageModel(
            model_id="hybrid/test",
            inner=llm,
            rule_config=RuleConfig(rules=[rule]),
        )
        results = list(hybrid.infer(["Some prompt"]))
        assert results[0][0].output == '{"llm_result": true}'
        assert llm.call_count == 1
        assert hybrid.rule_hits == 0
        assert hybrid.llm_fallbacks == 1

    def test_mixed_batch(self) -> None:
        llm = _StubLLM(response='{"fallback": true}')
        rule = RegexRule(
            r"Ref:\s*(?P<ref>[A-Z]+-\d+)",
        )
        hybrid = HybridLanguageModel(
            model_id="hybrid/test",
            inner=llm,
            rule_config=RuleConfig(rules=[rule]),
        )
        prompts = [
            "Contract Ref: ABC-123",  # rule hit
            "No structured data here",  # rule miss
            "Another Ref: XYZ-789",  # rule hit
        ]
        results = list(hybrid.infer(prompts))
        assert len(results) == 3

        # First — rule hit
        p0 = json.loads(results[0][0].output or "")
        assert p0["ref"] == "ABC-123"

        # Second — LLM fallback
        assert results[1][0].output == '{"fallback": true}'

        # Third — rule hit
        p2 = json.loads(results[2][0].output or "")
        assert p2["ref"] == "XYZ-789"

        assert hybrid.rule_hits == 2
        assert hybrid.llm_fallbacks == 1
        assert llm.call_count == 1

    def test_first_rule_wins(self) -> None:
        rule_a = RegexRule(
            r"(?P<val>\d+)",
            output_template=lambda g: f'{{"rule": "a", "val": "{g["val"]}"}}',
            description="rule A",
        )
        rule_b = RegexRule(
            r"(?P<val>\d+)",
            output_template=lambda g: f'{{"rule": "b", "val": "{g["val"]}"}}',
            description="rule B",
        )
        llm = _StubLLM()
        hybrid = HybridLanguageModel(
            model_id="hybrid/test",
            inner=llm,
            rule_config=RuleConfig(rules=[rule_a, rule_b]),
        )
        results = list(hybrid.infer(["test 42"]))
        parsed = json.loads(results[0][0].output or "")
        assert parsed["rule"] == "a"

    def test_low_confidence_triggers_fallback(self) -> None:
        rule = RegexRule(
            r"(?P<val>\d+)",
            confidence=0.5,
        )
        llm = _StubLLM(response='{"from_llm": true}')
        hybrid = HybridLanguageModel(
            model_id="hybrid/test",
            inner=llm,
            rule_config=RuleConfig(
                rules=[rule],
                fallback_on_low_confidence=True,
                min_confidence=0.8,
            ),
        )
        results = list(hybrid.infer(["test 42"]))
        # Low confidence rule should be skipped
        assert results[0][0].output == '{"from_llm": true}'
        assert llm.call_count == 1

    def test_kwargs_forwarded_to_inner(self) -> None:
        llm = _StubLLM()
        hybrid = HybridLanguageModel(
            model_id="hybrid/test",
            inner=llm,
            rule_config=RuleConfig(rules=[]),
        )
        with mock.patch.object(llm, "infer", wraps=llm.infer) as m:
            list(hybrid.infer(["prompt"], pass_num=2))
        m.assert_called_with(["prompt"], pass_num=2)

    def test_reset_counters(self) -> None:
        llm = _StubLLM()
        hybrid = HybridLanguageModel(
            model_id="hybrid/test",
            inner=llm,
            rule_config=RuleConfig(rules=[]),
        )
        list(hybrid.infer(["p1", "p2"]))
        assert hybrid.llm_fallbacks == 2
        hybrid.reset_counters()
        assert hybrid.rule_hits == 0
        assert hybrid.llm_fallbacks == 0

    def test_get_counters_returns_snapshot(self) -> None:
        """get_counters() must return a frozen snapshot that is not
        affected by subsequent counter updates."""
        llm = _StubLLM()
        rule = RegexRule(r"MATCH:\s*(?P<val>\w+)")
        hybrid = HybridLanguageModel(
            model_id="hybrid/test",
            inner=llm,
            rule_config=RuleConfig(rules=[rule]),
        )
        list(hybrid.infer(["MATCH: hello", "no match"]))
        snapshot = hybrid.get_counters()
        assert snapshot == {"rule_hits": 1, "llm_fallbacks": 1}

        # Subsequent calls should not mutate the snapshot
        list(hybrid.infer(["MATCH: world"]))
        assert snapshot == {"rule_hits": 1, "llm_fallbacks": 1}
        # But live counters advanced
        assert hybrid.rule_hits == 2

    def test_counters_thread_safe(self) -> None:
        """Rule-hit and LLM-fallback counters must be accurate when
        the provider is used from multiple threads simultaneously."""
        import threading

        rule = RegexRule(r"MATCH:\s*(?P<val>\w+)")
        llm = _StubLLM()
        hybrid = HybridLanguageModel(
            model_id="hybrid/test",
            inner=llm,
            rule_config=RuleConfig(rules=[rule]),
        )

        n_threads = 20
        errors: list[Exception] = []

        def run_infer(match: bool) -> None:
            try:
                prompt = "MATCH: hello" if match else "no match here"
                list(hybrid.infer([prompt]))
            except (
                Exception
            ) as exc:  # broad catch intentional for thread error collection
                errors.append(exc)

        threads = [
            threading.Thread(target=run_infer, args=(i % 2 == 0,))
            for i in range(n_threads)
        ]
        for t in threads:
            t.start()
        for t in threads:
            t.join()

        assert not errors, f"Thread errors: {errors}"
        assert hybrid.rule_hits + hybrid.llm_fallbacks == n_threads
        assert hybrid.rule_hits == n_threads // 2
        assert hybrid.llm_fallbacks == n_threads // 2

    def test_empty_rules_always_falls_back(self) -> None:
        llm = _StubLLM()
        hybrid = HybridLanguageModel(
            model_id="hybrid/test",
            inner=llm,
            rule_config=RuleConfig(rules=[]),
        )
        results = list(hybrid.infer(["p1"]))
        assert results[0][0].score == 1.0
        assert llm.call_count == 1


# ---------------------------------------------------------------------------
# Provider tests — async
# ---------------------------------------------------------------------------


class TestHybridProviderAsync:
    """Tests for HybridLanguageModel async inference."""

    @pytest.mark.asyncio
    async def test_async_rule_hit(self) -> None:
        llm = _StubLLM()
        rule = RegexRule(r"Date:\s*(?P<date>\d{4}-\d{2}-\d{2})")
        hybrid = HybridLanguageModel(
            model_id="hybrid/test",
            inner=llm,
            rule_config=RuleConfig(rules=[rule]),
        )
        results = await hybrid.async_infer(["Extract: Date: 2026-01-01"])
        parsed = json.loads(results[0][0].output or "")
        assert parsed["date"] == "2026-01-01"
        assert llm.call_count == 0

    @pytest.mark.asyncio
    async def test_async_fallback(self) -> None:
        llm = _StubLLM(response='{"async_llm": true}')
        hybrid = HybridLanguageModel(
            model_id="hybrid/test",
            inner=llm,
            rule_config=RuleConfig(rules=[]),
        )
        results = await hybrid.async_infer(["prompt"])
        assert results[0][0].output == '{"async_llm": true}'
        assert llm.call_count == 1

    @pytest.mark.asyncio
    async def test_async_mixed_batch(self) -> None:
        llm = _StubLLM(response='{"fallback": true}')
        rule = RegexRule(r"Ref:\s*(?P<ref>[A-Z]+-\d+)")
        hybrid = HybridLanguageModel(
            model_id="hybrid/test",
            inner=llm,
            rule_config=RuleConfig(rules=[rule]),
        )
        results = await hybrid.async_infer(
            [
                "Ref: ABC-123",
                "no match",
                "Ref: DEF-456",
            ]
        )
        assert len(results) == 3

        # Rule hits
        assert json.loads(results[0][0].output or "")["ref"] == "ABC-123"
        assert json.loads(results[2][0].output or "")["ref"] == "DEF-456"

        # LLM fallback
        assert results[1][0].output == '{"fallback": true}'

        # Only 1 LLM call — the miss was batched
        assert llm.call_count == 1

    @pytest.mark.asyncio
    async def test_async_preserves_order(self) -> None:
        llm = _StubLLM(response="llm")
        rule = RegexRule(
            r"HIT",
            output_template=lambda _: "rule",
        )
        hybrid = HybridLanguageModel(
            model_id="hybrid/test",
            inner=llm,
            rule_config=RuleConfig(rules=[rule]),
        )
        results = await hybrid.async_infer(
            [
                "miss1",
                "HIT",
                "miss2",
                "HIT",
                "miss3",
            ]
        )
        outputs = [r[0].output for r in results]
        assert outputs == ["llm", "rule", "llm", "rule", "llm"]


# ---------------------------------------------------------------------------
# text_extractor and output_formatter tests
# ---------------------------------------------------------------------------


class TestTextExtractor:
    """Tests for text_extractor in RuleConfig."""

    def test_rules_receive_extracted_text(self) -> None:
        """Rules should only see the extracted text, not the full prompt."""
        llm = _StubLLM()
        rule = RegexRule(
            r"Date:\s*(?P<date>\d{4}-\d{2}-\d{2})",
        )

        def extract_doc(prompt: str) -> str:
            # Simulate extracting only the document section
            marker = "---DOCUMENT---\n"
            idx = prompt.find(marker)
            if idx >= 0:
                return prompt[idx + len(marker) :]
            return prompt

        hybrid = HybridLanguageModel(
            model_id="hybrid/test",
            inner=llm,
            rule_config=RuleConfig(
                rules=[rule],
                text_extractor=extract_doc,
            ),
        )
        # The rule pattern exists in the instructions, but
        # text_extractor should strip instructions
        prompt = (
            "Instructions: Extract dates like Date: 2000-01-01\n"
            "---DOCUMENT---\n"
            "Contract Date: 2026-06-15"
        )
        results = list(hybrid.infer([prompt]))
        parsed = json.loads(results[0][0].output or "")
        # Should match the document date, not the instruction
        assert parsed["date"] == "2026-06-15"
        assert llm.call_count == 0

    def test_without_extractor_matches_full_prompt(self) -> None:
        """Without text_extractor, rules match the full prompt."""
        llm = _StubLLM()
        rule = RegexRule(
            r"Example:\s*(?P<val>\w+)",
        )
        hybrid = HybridLanguageModel(
            model_id="hybrid/test",
            inner=llm,
            rule_config=RuleConfig(rules=[rule]),
        )
        # This matches the instruction example, not document
        prompt = "Example: WRONG\nActual data here"
        results = list(hybrid.infer([prompt]))
        parsed = json.loads(results[0][0].output or "")
        assert parsed["val"] == "WRONG"


class TestOutputFormatter:
    """Tests for output_formatter in RuleConfig."""

    def test_formatter_transforms_output(self) -> None:
        llm = _StubLLM()
        rule = RegexRule(
            r"(?P<amount>\d+)",
        )

        def format_output(raw: str) -> str:
            data = json.loads(raw)
            return json.dumps({"extracted_amount": int(data["amount"])})

        hybrid = HybridLanguageModel(
            model_id="hybrid/test",
            inner=llm,
            rule_config=RuleConfig(
                rules=[rule],
                output_formatter=format_output,
            ),
        )
        results = list(hybrid.infer(["Pay 500"]))
        parsed = json.loads(results[0][0].output or "")
        assert parsed["extracted_amount"] == 500


# ---------------------------------------------------------------------------
# Plugin registration test
# ---------------------------------------------------------------------------


class TestPluginRegistration:
    """Tests for entry-point discovery."""

    def test_hybrid_prefix_resolves(self) -> None:
        import langcore as lx
        from langcore.providers import registry

        lx.providers.load_plugins_once()
        cls = registry.resolve("hybrid/my-model")
        assert cls.__name__ == "HybridLanguageModel"
