from enum import StrEnum
from typing import Literal

Status = Literal["pending", "answered", "fallback", "refused", "error"]


class ReasonCode(StrEnum):
    INVALID_INPUT = "INVALID_INPUT"
    PROMPT_INJECTION = "PROMPT_INJECTION"
    OUT_OF_SCOPE = "OUT_OF_SCOPE"
    NO_RESULTS = "NO_RESULTS"
    LOW_EVIDENCE = "LOW_EVIDENCE"
    NO_AUTHORITATIVE_SOURCE = "NO_AUTHORITATIVE_SOURCE"
    NEEDS_CLARIFICATION = "NEEDS_CLARIFICATION"
    UNCLEAR_QUERY = "UNCLEAR_QUERY"
    MODEL_CANNOT_ANSWER = "MODEL_CANNOT_ANSWER"
    INVALID_CITATION = "INVALID_CITATION"
    UNSUPPORTED_CLAIM = "UNSUPPORTED_CLAIM"
    GENERATION_PARSE_ERROR = "GENERATION_PARSE_ERROR"
    PROVIDER_UNAVAILABLE = "PROVIDER_UNAVAILABLE"
    RETRIEVAL_UNAVAILABLE = "RETRIEVAL_UNAVAILABLE"
    INTERNAL_ERROR = "INTERNAL_ERROR"


_INJECTION_MSG = "ขออภัย คำขอนี้เปลี่ยนกฎหรือขอข้อมูลที่ระบบไม่อนุญาต จึงดำเนินการให้ไม่ได้"
_OUT_OF_SCOPE_MSG = "ขออภัย คำถามนี้อยู่นอกขอบเขตของคู่มือพนักงาน"
_NO_DATA_MSG = "ขออภัย ไม่มีข้อมูลส่วนนี้ในคู่มือพนักงาน"
_UNVERIFIED_MSG = "ขออภัย ยังยืนยันคำตอบจากเอกสารได้ไม่ครบ จึงขอไม่ตอบเพื่อป้องกันข้อมูลผิด"
_UNCLEAR_MSG = (
    'ขออภัย ยังไม่เข้าใจคำถาม ช่วยพิมพ์ใหม่อีกครั้งได้ไหม เช่น "ลาป่วยต้องทำยังไง" หรือ "เบิกค่าเดินทางได้เท่าไหร่"'
)
_UNAVAILABLE_MSG = "ขออภัย ตอนนี้ระบบขัดข้องชั่วคราว ลองถามใหม่อีกครั้งในอีกสักครู่"
_INVALID_INPUT_MSG = "ขออภัย ระบบอ่านคำถามนี้ไม่ได้ ลองพิมพ์ใหม่เป็นข้อความสั้นๆ อีกครั้ง"
_ASK_ELSEWHERE = "เรื่องอื่นสอบถามฝ่าย HR หรือฝ่ายการเงินได้โดยตรง"

_NO_REFERRAL = frozenset(
    {ReasonCode.PROMPT_INJECTION, ReasonCode.INVALID_INPUT, ReasonCode.UNCLEAR_QUERY}
)

CATEGORY_LABELS = {
    "leave": "การลา",
    "finance": "การเบิกค่าใช้จ่าย",
    "Human Resources": "งานบุคคลและการลา",
    "Finance": "การเงินและการเบิกจ่าย",
}

REFUSAL_MESSAGES: dict[ReasonCode, str] = {
    ReasonCode.INVALID_INPUT: _INVALID_INPUT_MSG,
    ReasonCode.PROMPT_INJECTION: _INJECTION_MSG,
    ReasonCode.OUT_OF_SCOPE: _OUT_OF_SCOPE_MSG,
}

ERROR_REASONS: frozenset[ReasonCode] = frozenset(
    {
        ReasonCode.PROVIDER_UNAVAILABLE,
        ReasonCode.RETRIEVAL_UNAVAILABLE,
        ReasonCode.INTERNAL_ERROR,
    }
)


def format_topics(topics: dict[str, list[str]]) -> str:
    """Turn {category: [titles]} into a short "what you can ask" list."""
    if not topics:
        return ""
    lines = [
        f"• {CATEGORY_LABELS.get(category, category)}: {', '.join(titles)}"
        for category, titles in topics.items()
    ]
    return "ตอนนี้มีข้อมูลให้สอบถาม:\n" + "\n".join(lines)


def fallback_message(reason: ReasonCode, topics: dict[str, list[str]] | None = None) -> str:
    if reason in ERROR_REASONS:
        return _UNAVAILABLE_MSG
    if reason in (ReasonCode.INVALID_CITATION, ReasonCode.UNSUPPORTED_CLAIM):
        head = _UNVERIFIED_MSG
    elif reason is ReasonCode.UNCLEAR_QUERY:
        head = _UNCLEAR_MSG
    else:
        head = REFUSAL_MESSAGES.get(reason, _NO_DATA_MSG)

    parts = [head, format_topics(topics or {})]
    if reason not in _NO_REFERRAL:
        parts.append(_ASK_ELSEWHERE)
    return "\n\n".join(p for p in parts if p)


class ProviderUnavailableError(RuntimeError):
    pass


class RetrievalUnavailableError(RuntimeError):
    pass


class GenerationParseError(RuntimeError):
    pass


class ConfigurationError(RuntimeError):
    pass
