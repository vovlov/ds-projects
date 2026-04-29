"""
PII Detection for ML Datasets / Обнаружение персональных данных в датасетах.

Scans DataFrame columns for Personally Identifiable Information (PII) using
regex patterns. Supports GDPR Article 4 and CCPA definitions of PII.

EU AI Act Article 10 requires data governance measures including PII audits
for high-risk AI systems.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from enum import StrEnum
from typing import Any


class PIIType(StrEnum):
    EMAIL = "email"
    PHONE = "phone"
    SSN = "ssn"  # US Social Security Number
    CREDIT_CARD = "credit_card"
    IP_ADDRESS = "ip_address"
    PASSPORT = "passport"
    IBAN = "iban"
    DATE_OF_BIRTH = "date_of_birth"
    FULL_NAME = "full_name"  # heuristic: "FirstName LastName"


# Compiled regex patterns for each PII type
_PATTERNS: dict[PIIType, re.Pattern[str]] = {
    PIIType.EMAIL: re.compile(r"\b[A-Za-z0-9._%+\-]+@[A-Za-z0-9.\-]+\.[A-Za-z]{2,}\b"),
    PIIType.PHONE: re.compile(r"(?<!\d)(\+?1[\s\-.]?)?\(?\d{3}\)?[\s\-.]?\d{3}[\s\-.]?\d{4}(?!\d)"),
    PIIType.SSN: re.compile(r"\b(?!000|666|9\d{2})\d{3}[-\s]?(?!00)\d{2}[-\s]?(?!0000)\d{4}\b"),
    PIIType.CREDIT_CARD: re.compile(
        r"\b(?:4\d{3}|5[1-5]\d{2}|3[47]\d{2}|6(?:011|5\d{2}))"
        r"[\s\-]?\d{4}[\s\-]?\d{4}[\s\-]?\d{4}\b"
    ),
    PIIType.IP_ADDRESS: re.compile(
        r"\b(?:(?:25[0-5]|2[0-4]\d|[01]?\d\d?)\.){3}"
        r"(?:25[0-5]|2[0-4]\d|[01]?\d\d?)\b"
    ),
    PIIType.PASSPORT: re.compile(r"\b[A-Z]{1,2}\d{6,9}\b"),
    PIIType.IBAN: re.compile(r"\b[A-Z]{2}\d{2}[A-Z0-9]{4}\d{7}(?:[A-Z0-9]{0,16})\b"),
    PIIType.DATE_OF_BIRTH: re.compile(
        r"\b(?:19|20)\d{2}[-/](0[1-9]|1[0-2])[-/](0[1-9]|[12]\d|3[01])\b"
        r"|\b(0[1-9]|[12]\d|3[01])[-/](0[1-9]|1[0-2])[-/](?:19|20)\d{2}\b"
    ),
    PIIType.FULL_NAME: re.compile(
        # Two consecutive capitalised words that look like first/last name
        r"\b[A-Z][a-z]{2,}\s[A-Z][a-z]{2,}\b"
    ),
}

_SEVERITY: dict[PIIType, str] = {
    PIIType.SSN: "critical",
    PIIType.CREDIT_CARD: "critical",
    PIIType.PASSPORT: "high",
    PIIType.IBAN: "high",
    PIIType.EMAIL: "high",
    PIIType.DATE_OF_BIRTH: "medium",
    PIIType.PHONE: "medium",
    PIIType.IP_ADDRESS: "low",
    PIIType.FULL_NAME: "low",
}


def _mask(value: str, pii_type: PIIType) -> str:
    """
    Маскировать PII для безопасного логирования.
    Return a masked version safe for audit logs.
    """
    if pii_type == PIIType.EMAIL:
        parts = value.split("@")
        return parts[0][:2] + "***@" + parts[1] if len(parts) == 2 else "***"
    if pii_type in (PIIType.CREDIT_CARD, PIIType.SSN, PIIType.IBAN):
        digits = re.sub(r"\D", "", value)
        return "*" * (len(digits) - 4) + digits[-4:]
    if pii_type == PIIType.PHONE:
        digits = re.sub(r"\D", "", value)
        return "***-***-" + digits[-4:] if len(digits) >= 4 else "***"
    return value[:2] + "***" + value[-2:] if len(value) > 4 else "***"


@dataclass
class PIIFinding:
    """PII найденная в конкретном столбце / PII detected in a column."""

    column: str
    pii_type: PIIType
    severity: str
    match_count: int
    sample_fraction: float  # fraction of non-null values that matched
    masked_examples: list[str] = field(default_factory=list)


@dataclass
class PIIReport:
    """Полный отчёт об обнаруженных PII / Full PII detection report."""

    findings: list[PIIFinding]
    affected_columns: list[str]
    critical_columns: list[str]
    total_rows_scanned: int
    gdpr_compliant: bool  # True if no critical/high PII found

    def to_dict(self) -> dict[str, Any]:
        return {
            "gdpr_compliant": self.gdpr_compliant,
            "total_rows_scanned": self.total_rows_scanned,
            "affected_columns": self.affected_columns,
            "critical_columns": self.critical_columns,
            "findings": [
                {
                    "column": f.column,
                    "pii_type": f.pii_type.value,
                    "severity": f.severity,
                    "match_count": f.match_count,
                    "sample_fraction": round(f.sample_fraction, 4),
                    "masked_examples": f.masked_examples,
                }
                for f in self.findings
            ],
        }


def detect_pii(
    columns: dict[str, list[Any]],
    max_examples: int = 3,
) -> PIIReport:
    """
    Просканировать столбцы датасета на наличие PII.

    Scan dataset columns for Personally Identifiable Information.

    Args:
        columns: mapping of column_name → list of values (strings or mixed).
        max_examples: max masked examples to include per finding.

    Returns:
        PIIReport with all findings, affected columns, and GDPR compliance flag.
    """
    findings: list[PIIFinding] = []
    affected: set[str] = set()
    critical: set[str] = set()

    total_rows = max((len(v) for v in columns.values()), default=0)

    for col_name, values in columns.items():
        str_values = [str(v) for v in values if v is not None and str(v).strip()]
        if not str_values:
            continue

        for pii_type, pattern in _PATTERNS.items():
            matched_values: list[str] = []
            for val in str_values:
                if pattern.search(val):
                    matched_values.append(val)

            if not matched_values:
                continue

            examples = []
            for val in matched_values[:max_examples]:
                m = pattern.search(val)
                if m:
                    examples.append(_mask(m.group(), pii_type))

            finding = PIIFinding(
                column=col_name,
                pii_type=pii_type,
                severity=_SEVERITY[pii_type],
                match_count=len(matched_values),
                sample_fraction=len(matched_values) / len(str_values),
                masked_examples=examples,
            )
            findings.append(finding)
            affected.add(col_name)
            if _SEVERITY[pii_type] in ("critical", "high"):
                critical.add(col_name)

    gdpr_compliant = len(critical) == 0

    return PIIReport(
        findings=findings,
        affected_columns=sorted(affected),
        critical_columns=sorted(critical),
        total_rows_scanned=total_rows,
        gdpr_compliant=gdpr_compliant,
    )
