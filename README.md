# LangCore Hybrid Provider

A provider plugin for [LangCore](https://github.com/google/langcore) that combines deterministic rule-based extraction (regex, callable functions) with LLM fallback. Saves 50–80% of LLM costs on well-structured documents.

> **Note**: This is a third-party provider plugin for LangCore. For the main LangCore library, visit [google/langcore](https://github.com/google/langcore).

## Installation

Install from source:

```bash
git clone <repo-url>
cd langcore-hybrid
pip install -e .
```

For optional spaCy NER support:

```bash
pip install -e ".[spacy]"
```

## Features

- **Rules first, LLM fallback** — deterministic extraction for patterns that don't need an LLM
- **Regex rules** — extract dates, amounts, reference numbers via named capture groups
- **Callable rules** — plug in any Python function for custom extraction logic
- **Confidence thresholds** — optionally fall back to LLM when rule confidence is low
- **Batch-aware async** — only prompts that miss all rules are batched for LLM inference
- **Observability** — built-in counters for rule hits vs LLM fallbacks
- **Thread-safe counters** — rule-hit and LLM-fallback counters are protected by a lock for safe concurrent use
- **Text extractor hook** — optional `text_extractor` callable in `RuleConfig` isolates document text from prompt instructions before rule evaluation
- **Output formatter hook** — optional `output_formatter` callable in `RuleConfig` normalises rule outputs (e.g. wrap in JSON)
- **Zero overhead on hits** — rule evaluation is pure Python, no network calls

## Usage

### Regex Rules for Contract Extraction

```python
import langcore as lx
from langcore_hybrid import (
    HybridLanguageModel,
    RegexRule,
    RuleConfig,
)

# Create the fallback LLM provider
inner_config = lx.factory.ModelConfig(
    model_id="litellm/azure/gpt-4o",
    provider="LiteLLMLanguageModel",
)
inner_model = lx.factory.create_model(inner_config)

# Define rules for common patterns
rules = RuleConfig(rules=[
    RegexRule(
        r"Date:\s*(?P<date>\d{4}-\d{2}-\d{2})",
        description="ISO date extraction",
    ),
    RegexRule(
        r"Amount:\s*\$(?P<amount>[\d,.]+)",
        description="USD amount extraction",
    ),
    RegexRule(
        r"Ref(?:erence)?[:\s]+(?P<ref>[A-Z]+-\d+)",
        description="Reference number extraction",
    ),
])

# Create hybrid provider
hybrid_model = HybridLanguageModel(
    model_id="hybrid/gpt-4o",
    inner=inner_model,
    rule_config=rules,
)

# Use as normal
result = lx.extract(
    text_or_documents="Contract Ref: ABC-123, Date: 2026-01-15...",
    model=hybrid_model,
    prompt_description="Extract contract metadata.",
)

# Check cost savings
print(f"Rule hits: {hybrid_model.rule_hits}")
print(f"LLM fallbacks: {hybrid_model.llm_fallbacks}")
```

### Callable Rules

```python
import json
from langcore_hybrid import CallableRule, RuleConfig

def extract_email(prompt: str) -> str | None:
    """Extract email addresses deterministically."""
    import re
    emails = re.findall(r'\b[\w.+-]+@[\w-]+\.[\w.]+\b', prompt)
    if emails:
        return json.dumps({"emails": emails})
    return None  # Signal miss — fall back to LLM

rules = RuleConfig(rules=[
    CallableRule(
        func=extract_email,
        description="email extraction",
    ),
])
```

### Custom Output Templates

```python
import json
from langcore_hybrid import RegexRule

rule = RegexRule(
    r"Amount:\s*\$(?P<amount>[\d,.]+)",
    output_template=lambda groups: json.dumps({
        "amount": float(groups["amount"].replace(",", "")),
        "currency": "USD",
    }),
)
```

### Confidence Thresholds

When `fallback_on_low_confidence` is enabled, rule results with confidence below
`min_confidence` trigger LLM fallback instead:

```python
from langcore_hybrid import RegexRule, RuleConfig

rules = RuleConfig(
    rules=[
        RegexRule(r"(?P<date>\d{1,2}/\d{1,2}/\d{2,4})", confidence=0.6),
        RegexRule(r"(?P<date>\d{4}-\d{2}-\d{2})", confidence=1.0),
    ],
    fallback_on_low_confidence=True,
    min_confidence=0.8,
)
# Ambiguous dates (MM/DD/YY) fall back to LLM; ISO dates are trusted
```

### Rule Evaluation Order

Rules are evaluated **in list order**. The first rule that hits *and* meets the
confidence threshold wins — later rules are not evaluated.

When `fallback_on_low_confidence=True`, a rule that hits below `min_confidence`
is **skipped** and evaluation continues to the next rule. If no subsequent rule
produces a confident hit, the prompt falls through to the LLM.

This means rule ordering matters:

1. Place **high-confidence, specific** rules first (e.g. ISO dates).
2. Follow with **lower-confidence, broader** rules (e.g. ambiguous date formats).
3. The LLM acts as the final catch-all.

```python
rules = RuleConfig(
    rules=[
        RegexRule(r"(?P<date>\d{4}-\d{2}-\d{2})", confidence=1.0),   # ① specific
        RegexRule(r"(?P<date>\d{1,2}/\d{1,2}/\d{2,4})", confidence=0.6),  # ② broad
    ],
    fallback_on_low_confidence=True,
    min_confidence=0.8,
)
# "2026-01-15" → rule ① (confidence 1.0 ≥ 0.8) → instant result
# "1/15/26"   → rule ① miss, rule ② hit (0.6 < 0.8) → LLM fallback
```

### Async Usage

The async path is batch-optimised — only prompts that miss all rules are sent to the LLM in a single batch:

```python
results = await hybrid_model.async_infer([
    "Date: 2026-01-15",       # Rule hit — instant
    "Complex clause text...",  # Rule miss — LLM
    "Ref: ABC-123",           # Rule hit — instant
])
# Only 1 LLM call for the 1 miss, not 3
```

## Observability

Counters are **per-instance** and **cumulative** — they track totals since
the provider was created (or last reset).  In long-running applications
where a single `HybridLanguageModel` handles unrelated jobs, call
`reset_counters()` between jobs or use `get_counters()` to take
point-in-time snapshots for differential reporting.

```python
print(f"Rule hits: {hybrid_model.rule_hits}")
print(f"LLM fallbacks: {hybrid_model.llm_fallbacks}")

# Atomic snapshot (thread-safe dict copy)
snapshot = hybrid_model.get_counters()
# {"rule_hits": 42, "llm_fallbacks": 7}

# Reset counters (also thread-safe)
hybrid_model.reset_counters()
```

### Text Extractor

When prompts contain instructions followed by document text, rules may match
instruction fragments by mistake.  Use `text_extractor` to isolate the document:

```python
def extract_after_marker(prompt: str) -> str:
    """Return text after '---DOCUMENT---' marker."""
    marker = "---DOCUMENT---"
    idx = prompt.find(marker)
    return prompt[idx + len(marker) :].strip() if idx >= 0 else prompt

rules = RuleConfig(
    rules=[RegexRule(r"Date:\s*(?P<date>\d{4}-\d{2}-\d{2})")],
    text_extractor=extract_after_marker,
)
```

### Output Formatter

Normalise rule outputs for downstream consumers:

```python
import json

rules = RuleConfig(
    rules=[RegexRule(r"Ref:\s*(?P<ref>[A-Z]+-\d+)")],
    output_formatter=lambda raw: json.dumps({"result": json.loads(raw)}),
)
```

## Custom Rules

Implement the `ExtractionRule` interface:

```python
from langcore_hybrid.rules import ExtractionRule, RuleResult

class SpacyNERRule(ExtractionRule):
    def __init__(self, nlp_model: str = "en_core_web_sm") -> None:
        import spacy
        self._nlp = spacy.load(nlp_model)

    def evaluate(self, prompt: str) -> RuleResult:
        doc = self._nlp(prompt)
        entities = [
            {"text": ent.text, "label": ent.label_}
            for ent in doc.ents
        ]
        if entities:
            import json
            return RuleResult(
                hit=True,
                output=json.dumps(entities),
                confidence=0.85,
            )
        return RuleResult(hit=False)
```

## Development

```bash
pip install -e ".[dev]"
pytest
```

## License

Apache 2.0
