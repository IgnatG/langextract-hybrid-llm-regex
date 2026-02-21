"""LangExtract hybrid rule + LLM provider plugin.

Combines deterministic rule-based extraction (regex, callable
functions, or spaCy NER) with LLM fallback for prompts that
rules cannot handle.  Saves 50-80% of LLM costs on
well-structured documents.
"""

from langextract_hybrid.provider import HybridLanguageModel
from langextract_hybrid.rules import (
    CallableRule,
    ExtractionRule,
    RegexRule,
    RuleConfig,
    RuleResult,
)

__all__ = [
    "CallableRule",
    "ExtractionRule",
    "HybridLanguageModel",
    "RegexRule",
    "RuleConfig",
    "RuleResult",
]
__version__ = "1.0.1"
