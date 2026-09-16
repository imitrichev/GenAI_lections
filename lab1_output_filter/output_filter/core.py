# output_filter/core.py

import re
from dataclasses import dataclass
from typing import List


@dataclass(frozen=True)
class Finding:
    """Один найденный фрагмент конфиденциальной информации."""

    category: str
    value: str
    start: int
    end: int


class OutputFilter:
    """
    Проверяет текст (например, ответ LLM) на наличие конфиденциальной
    информации: email-адресов, паролей и API-ключей, и умеет заменять
    найденные фрагменты маской перед показом пользователю.
    """

    name = "output_filter"
    description = (
        "Ищет в тексте email-адреса, пароли и API-ключи с помощью "
        "регулярных выражений и заменяет их маской."
    )

    _PATTERNS = {
        "email": re.compile(
            r"[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}"
        ),
        "api_key": re.compile(
            r"sk-[A-Za-z0-9]{20,}"
            r"|AKIA[0-9A-Z]{16}"
            r"|ghp_[A-Za-z0-9]{30,}"
            r"|(?i:(?:api[_-]?key|token|secret)\s*[:=]\s*['\"]?[A-Za-z0-9_\-]{12,}['\"]?)"
        ),
        "password": re.compile(
            r"(?i:(?:password|passwd|pwd|пароль)\s*[:=]\s*\S+)"
        ),
    }

    def __init__(self, mask_template: str = "[REDACTED:{category}]"):
        self.mask_template = mask_template

    def scan(self, text: str) -> List[Finding]:
        """Возвращает список найденных конфиденциальных фрагментов по порядку."""
        findings = []
        for category, pattern in self._PATTERNS.items():
            for match in pattern.finditer(text):
                findings.append(
                    Finding(category, match.group(0), match.start(), match.end())
                )
        findings.sort(key=lambda f: f.start)
        return self._drop_overlaps(findings)

    @staticmethod
    def _drop_overlaps(findings: List[Finding]) -> List[Finding]:
        result = []
        last_end = -1
        for finding in findings:
            if finding.start >= last_end:
                result.append(finding)
                last_end = finding.end
        return result

    def contains_sensitive(self, text: str) -> bool:
        """True, если в тексте найден хотя бы один чувствительный фрагмент."""
        return len(self.scan(text)) > 0

    def redact(self, text: str) -> str:
        """Возвращает текст с замаскированными конфиденциальными фрагментами."""
        findings = self.scan(text)
        if not findings:
            return text

        parts = []
        cursor = 0
        for finding in findings:
            parts.append(text[cursor:finding.start])
            parts.append(self.mask_template.format(category=finding.category))
            cursor = finding.end
        parts.append(text[cursor:])
        return "".join(parts)

    def use(self, text: str) -> str:
        """Точка входа в стиле инструментов агента: вернуть очищенный текст."""
        return self.redact(text)
