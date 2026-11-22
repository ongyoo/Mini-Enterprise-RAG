import re
from collections.abc import Iterable, Mapping
from dataclasses import dataclass

from app.domain import ReasonCode

_THAI_DIGITS = str.maketrans("๐๑๒๓๔๕๖๗๘๙", "0123456789")
_NUMBER = re.compile(r"\d+(?:[.,:]\d+)*")
# "1. step" / "2) step" at the start of a line is list numbering, not a claim.
_LIST_MARKER = re.compile(r"^\s*\d+[.)]\s+", re.MULTILINE)
_LEAK_MARKERS = re.compile(r"<\s*/?\s*source\b|cited_source_ids|cannot_answer", re.IGNORECASE)


@dataclass(frozen=True, slots=True)
class VerificationResult:
    ok: bool
    reason_code: ReasonCode | None = None
    cited_chunk_ids: tuple[str, ...] = ()
    detail: str | None = None


def extract_numbers(text: str) -> set[str]:
    normalized = _LIST_MARKER.sub("", text.translate(_THAI_DIGITS))
    return {m.group().replace(",", "") for m in _NUMBER.finditer(normalized)}


def verify_answer(
    *,
    answer: str,
    cited_chunk_ids: Iterable[str],
    allowed_chunks: Mapping[str, str],
    query: str,
    max_answer_chars: int,
) -> VerificationResult:
    """Citation-ID validity plus a deterministic numeric-grounding check.

    Passing this does not prove factual correctness; it only rejects answers whose
    citations are fabricated or whose numbers cannot be found in the cited text.
    """
    text = answer.strip()
    if not text:
        return VerificationResult(False, ReasonCode.MODEL_CANNOT_ANSWER, detail="empty answer")
    if len(text) > max_answer_chars:
        return VerificationResult(False, ReasonCode.UNSUPPORTED_CLAIM, detail="answer too long")
    if _LEAK_MARKERS.search(text):
        return VerificationResult(False, ReasonCode.UNSUPPORTED_CLAIM, detail="prompt leakage")

    cited = tuple(dict.fromkeys(cited_chunk_ids))
    if not cited:
        return VerificationResult(False, ReasonCode.INVALID_CITATION, detail="no citations")
    unknown = [c for c in cited if c not in allowed_chunks]
    if unknown:
        return VerificationResult(
            False, ReasonCode.INVALID_CITATION, detail=f"{len(unknown)} unknown citation(s)"
        )

    evidence = " ".join(allowed_chunks[c] for c in cited) + " " + query
    unsupported = extract_numbers(text) - extract_numbers(evidence)
    if unsupported:
        return VerificationResult(
            False,
            ReasonCode.UNSUPPORTED_CLAIM,
            detail=f"numbers not in cited sources: {sorted(unsupported)[:5]}",
        )
    return VerificationResult(True, cited_chunk_ids=cited)
