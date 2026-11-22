import hashlib
import hmac
import re
from dataclasses import dataclass

_PATTERNS: tuple[tuple[str, re.Pattern[str]], ...] = (
    ("EMAIL", re.compile(r"[\w.+-]+@[\w-]+\.[\w.-]+")),
    ("THAI_NATIONAL_ID", re.compile(r"(?<!\d)\d[- ]?\d{4}[- ]?\d{5}[- ]?\d{2}[- ]?\d(?!\d)")),
    ("BANK_ACCOUNT", re.compile(r"(?<!\d)\d{3}-\d-\d{5}-\d(?!\d)")),
    ("PHONE", re.compile(r"(?<!\d)(?:\+66|0)\d{1,2}[- ]?\d{3}[- ]?\d{4}(?!\d)")),
    ("EMPLOYEE_ID", re.compile(r"\bEMP-?\d{4,}\b", re.IGNORECASE)),
)


@dataclass(frozen=True, slots=True)
class RedactionResult:
    text: str
    pii_types: tuple[str, ...]


def redact(text: str) -> RedactionResult:
    found: list[str] = []
    for label, pattern in _PATTERNS:
        text, n = pattern.subn(f"[{label}]", text)
        if n:
            found.append(label)
    return RedactionResult(text=text, pii_types=tuple(found))


def hash_identifier(value: str, salt: str) -> str:
    return hmac.new(salt.encode(), value.encode(), hashlib.sha256).hexdigest()
