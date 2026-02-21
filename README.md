# LangExtract Hybrid Provider

A provider plugin for [LangExtract](https://github.com/google/langextract) that combines deterministic rule-based extraction (regex, callable functions) with LLM fallback. Saves 50–80% of LLM costs on well-structured documents.

> **Note**: This is a third-party provider plugin for LangExtract. For the main LangExtract library, visit [google/langextract](https://github.com/google/langextract).

## Installation

Install from source:

```bash
git clone <repo-url>
cd langextract-hybrid
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
- **Zero overhead on hits** — rule evaluation is pure Python, no network calls

## Usage

### Regex Rules for Contract Extraction

```python
import langextract as lx
from langextract_hybrid import (
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
from langextract_hybrid import CallableRule, RuleConfig

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
from langextract_hybrid import RegexRule

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
from langextract_hybrid import RegexRule, RuleConfig

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

```python
print(f"Rule hits: {hybrid_model.rule_hits}")
print(f"LLM fallbacks: {hybrid_model.llm_fallbacks}")

# Reset counters
hybrid_model.reset_counters()
```

## Custom Rules

Implement the `ExtractionRule` interface:

```python
from langextract_hybrid.rules import ExtractionRule, RuleResult

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
