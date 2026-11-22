import re
import unicodedata
from dataclasses import dataclass

from app.domain import ReasonCode
from app.guardrails.redaction import redact

_INVISIBLE = re.compile(r"[​-‏‪-‮⁠-⁤﻿]")
_CONTROL = re.compile(r"[\x00-\x08\x0b-\x1f\x7f]")
_WS = re.compile(r"\s+")

_INJECTION_RULES: tuple[tuple[str, re.Pattern[str]], ...] = tuple(
    (name, re.compile(p, re.IGNORECASE))
    for name, p in (
        (
            "ignore_instructions",
            r"\b(ignore|disregard|forget)\b.{0,30}\b(previous|prior|above|earlier|all|system)\b"
            r".{0,20}\b(instructions?|rules?|prompts?|guidelines?)",
        ),
        (
            "reveal_prompt",
            r"\b(reveal|show|print|leak|repeat|output)\b.{0,30}"
            r"(system prompt|hidden instructions?|your instructions?|developer message)",
        ),
        ("role_override", r"\byou are now\b|\b(developer|god|dan|jailbreak)\s+mode\b"),
        ("role_tag", r"<\s*/?\s*(system|assistant|source|document|instructions?)\b"),
        ("system_prefix", r"(^|\n|\s)(system|assistant)\s*:"),
        (
            "privilege_request",
            r"\b(grant|give)\b.{0,20}\b(admin|root|superuser)\b.{0,10}\b(access|rights?|privileges?)",
        ),
        (
            "credential_request",
            r"(password|api[ _-]?key|secret|credential|token|รหัสผ่าน|พาสเวิร์ด)"
            r".{0,15}(of|for|ของ|admin|แอดมิน|ผู้ดูแล|คนอื่น|ผู้อื่น)",
        ),
        ("sql_payload", r";\s*(drop|delete|truncate|update|insert)\s+|\bunion\s+select\b"),
        (
            "th_ignore_instructions",
            r"(ลืม|เพิกเฉย|ไม่ต้องสนใจ|ข้าม|ยกเลิก|ละทิ้ง).{0,10}(คำสั่ง|กฎ|ข้อกำหนด|ข้อจำกัด)",
        ),
        (
            "th_reveal_prompt",
            r"(แสดง|เปิดเผย|บอก|พิมพ์).{0,15}(system prompt|พรอมต์|พร้อมต์|คำสั่งระบบ|คำสั่งลับ)",
        ),
        ("th_role_override", r"(คุณ|เธอ|นาย|แก)\s*(คือ|เป็น)\s*(แอดมิน|ผู้ดูแลระบบ|admin|root)"),
        ("th_privilege_request", r"(ขอ|ให้|เพิ่ม).{0,10}สิทธิ์.{0,10}(แอดมิน|ผู้ดูแล|admin|root|สูงสุด)"),
        (
            "th_bulk_personal_data",
            r"(เงินเดือน|ข้อมูลส่วนตัว|เลขบัตรประชาชน|ที่อยู่).{0,15}(ของ)?(พนักงาน)?\s*"
            r"(ทุกคน|ทั้งหมด|คนอื่น)",
        ),
    )
)

_OUT_OF_SCOPE_RULES: tuple[tuple[str, re.Pattern[str]], ...] = tuple(
    (name, re.compile(p, re.IGNORECASE))
    for name, p in (
        ("fortune", r"ดูดวง|horoscope|ทำนายดวง"),
        ("lottery", r"หวย|ลอตเตอรี่|lottery"),
        ("investment", r"ราคาหุ้น|หุ้นตัวไหน|stock price|คริปโต|บิตคอยน์|bitcoin|crypto"),
        ("coding", r"(เขียน|write|generate).{0,15}(โค้ด|code|script|สคริปต์|โปรแกรม)"),
        ("weather", r"สภาพอากาศ|พยากรณ์อากาศ|ฝนจะตก|\bweather\b"),
        ("cooking", r"สูตรอาหาร|วิธีทำอาหาร|\brecipe\b"),
        ("sports", r"ผลบอล|ผลฟุตบอล|football score"),
    )
)


@dataclass(frozen=True, slots=True)
class InputDecision:
    allowed: bool
    sanitized_query: str
    reason_code: ReasonCode | None = None
    matched_rule: str | None = None
    pii_types: tuple[str, ...] = ()


def normalize(text: str) -> str:
    text = unicodedata.normalize("NFC", text)
    text = _INVISIBLE.sub("", text)
    text = _CONTROL.sub(" ", text)
    return _WS.sub(" ", text).strip()


def evaluate_input(query: str, max_chars: int) -> InputDecision:
    text = normalize(query)
    if not text or len(text) > max_chars:
        return InputDecision(False, "", ReasonCode.INVALID_INPUT, "length")

    # NFKC folds full-width Latin tricks but also splits Thai "ำ", so check both forms.
    variants = (text, unicodedata.normalize("NFKC", text))
    for name, pattern in _INJECTION_RULES:
        if any(pattern.search(v) for v in variants):
            return InputDecision(False, "", ReasonCode.PROMPT_INJECTION, name)
    for name, pattern in _OUT_OF_SCOPE_RULES:
        if any(pattern.search(v) for v in variants):
            return InputDecision(False, "", ReasonCode.OUT_OF_SCOPE, name)

    redacted = redact(text)
    return InputDecision(True, redacted.text, pii_types=redacted.pii_types)
