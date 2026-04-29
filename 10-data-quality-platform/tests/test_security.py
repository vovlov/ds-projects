"""
Tests for OWASP ML Security Audit and PII Detection modules.
Тесты для модулей аудита безопасности OWASP ML и обнаружения PII.
"""

from __future__ import annotations

from fastapi.testclient import TestClient
from quality.api.app import app
from quality.security.owasp import (
    OWASPMLAudit,
    OWASPRisk,
    RiskSeverity,
    _iqr_outlier_ratio,
    _label_entropy,
)
from quality.security.pii_detector import PIIType, detect_pii

client = TestClient(app)


# ---------------------------------------------------------------------------
# Unit tests: helper functions / вспомогательные функции
# ---------------------------------------------------------------------------


class TestHelperFunctions:
    def test_iqr_outlier_ratio_no_outliers(self) -> None:
        values = list(range(100))  # uniform, no outliers
        ratio = _iqr_outlier_ratio(values)
        assert ratio < 0.15

    def test_iqr_outlier_ratio_with_extreme_outliers(self) -> None:
        values = list(range(50)) + [10_000, -10_000]  # two extreme outliers
        ratio = _iqr_outlier_ratio(values)
        assert ratio > 0.0

    def test_iqr_outlier_ratio_too_few_values(self) -> None:
        assert _iqr_outlier_ratio([1.0, 2.0]) == 0.0

    def test_iqr_outlier_ratio_zero_iqr(self) -> None:
        # All same value → IQR = 0 → no outliers
        assert _iqr_outlier_ratio([5.0] * 20) == 0.0

    def test_label_entropy_balanced(self) -> None:
        labels = [0, 1] * 50
        entropy = _label_entropy(labels)
        assert abs(entropy - 1.0) < 0.01  # perfectly balanced → entropy = 1

    def test_label_entropy_imbalanced(self) -> None:
        labels = [0] * 95 + [1] * 5
        entropy = _label_entropy(labels)
        assert entropy < 0.5

    def test_label_entropy_single_class(self) -> None:
        assert _label_entropy([0] * 100) == 0.0

    def test_label_entropy_empty(self) -> None:
        assert _label_entropy([]) == 1.0


# ---------------------------------------------------------------------------
# Unit tests: OWASP individual checks
# ---------------------------------------------------------------------------


class TestOWASPChecks:
    def setup_method(self) -> None:
        self.auditor = OWASPMLAudit()

    def test_ml01_no_outliers(self) -> None:
        cols = {"age": list(range(20, 60)), "income": list(range(30_000, 70_000, 1000))}
        finding = self.auditor.check_ml01_input_manipulation(cols)
        assert finding is None

    def test_ml01_with_adversarial_outliers(self) -> None:
        # Normal values + many extreme outliers (>15%)
        normal = list(range(80))
        outliers = [999_999] * 20
        cols = {"feature_x": normal + outliers}
        finding = self.auditor.check_ml01_input_manipulation(cols)
        assert finding is not None
        assert finding.risk_id == OWASPRisk.ML01
        assert finding.severity == RiskSeverity.HIGH

    def test_ml02_balanced_labels(self) -> None:
        labels = [0, 1] * 50
        finding = self.auditor.check_ml02_data_poisoning(labels)
        assert finding is None

    def test_ml02_severely_imbalanced_labels(self) -> None:
        labels = [0] * 98 + [1] * 2
        finding = self.auditor.check_ml02_data_poisoning(labels)
        assert finding is not None
        assert finding.risk_id == OWASPRisk.ML02

    def test_ml02_no_labels(self) -> None:
        assert self.auditor.check_ml02_data_poisoning(None) is None
        assert self.auditor.check_ml02_data_poisoning([]) is None

    def test_ml03_no_risky_fields(self) -> None:
        finding = self.auditor.check_ml03_model_inversion(["prediction", "confidence"])
        assert finding is None

    def test_ml03_exposes_logits(self) -> None:
        finding = self.auditor.check_ml03_model_inversion(
            ["prediction", "raw_logits", "embedding_vector"]
        )
        assert finding is not None
        assert finding.risk_id == OWASPRisk.ML03
        assert finding.severity == RiskSeverity.MEDIUM

    def test_ml04_low_cardinality(self) -> None:
        cols = {"gender": ["M", "F"] * 50, "age_group": ["young", "old"] * 50}
        finding = self.auditor.check_ml04_membership_inference(cols)
        assert finding is None

    def test_ml04_high_cardinality_ids(self) -> None:
        ids = [f"user_{i:06d}" for i in range(1000)]
        finding = self.auditor.check_ml04_membership_inference({"user_id": ids})
        assert finding is not None
        assert finding.risk_id == OWASPRisk.ML04

    def test_ml05_with_rate_limiting(self) -> None:
        finding = self.auditor.check_ml05_model_theft(
            has_rate_limiting=True, exposes_probabilities=True
        )
        assert finding is None

    def test_ml05_no_rate_limiting_probs(self) -> None:
        finding = self.auditor.check_ml05_model_theft(
            has_rate_limiting=False, exposes_probabilities=True
        )
        assert finding is not None
        assert finding.risk_id == OWASPRisk.ML05
        assert finding.severity == RiskSeverity.HIGH

    def test_ml05_no_rate_limiting_no_probs(self) -> None:
        finding = self.auditor.check_ml05_model_theft(
            has_rate_limiting=False, exposes_probabilities=False
        )
        assert finding is not None
        assert finding.severity == RiskSeverity.MEDIUM

    def test_ml08_low_missing(self) -> None:
        cols = {"age": [25, 30, None, 35], "score": [0.5, 0.7, 0.8, 0.9]}
        finding = self.auditor.check_ml08_model_skewing(cols)
        assert finding is None

    def test_ml08_high_missing(self) -> None:
        cols = {"critical_feature": [None, None, None, 1.0, None, None, 1.0, None, None, None]}
        finding = self.auditor.check_ml08_model_skewing(cols)
        assert finding is not None
        assert finding.risk_id == OWASPRisk.ML08

    def test_ml09_no_signature(self) -> None:
        finding = self.auditor.check_ml09_output_integrity(
            ["prediction", "confidence", "model_version"]
        )
        assert finding is not None
        assert finding.risk_id == OWASPRisk.ML09

    def test_ml09_with_signature(self) -> None:
        finding = self.auditor.check_ml09_output_integrity(
            ["prediction", "confidence", "signature"]
        )
        assert finding is None


# ---------------------------------------------------------------------------
# Unit tests: Full audit report
# ---------------------------------------------------------------------------


class TestAuditReport:
    def setup_method(self) -> None:
        self.auditor = OWASPMLAudit()

    def test_clean_dataset_high_score(self) -> None:
        report = self.auditor.run_audit(
            numeric_columns={"age": list(range(20, 70))},
            all_columns={"gender": ["M", "F"] * 25},
            label_column=[0, 1] * 25,
            output_fields=["prediction", "hmac_signature"],
            has_rate_limiting=True,
            exposes_probabilities=False,
        )
        assert report.score >= 60
        assert report.total_checks == 7

    def test_vulnerable_dataset_low_score(self) -> None:
        # Many extreme outliers + severely imbalanced labels + no rate limiting
        normal = list(range(80))
        extreme = [999_999] * 30
        labels = [0] * 98 + [1] * 2
        ids = [f"user_{i}" for i in range(100)]

        report = self.auditor.run_audit(
            numeric_columns={"value": normal + extreme},
            all_columns={"user_id": ids, "value": [float(v) for v in normal + extreme]},
            label_column=labels,
            output_fields=["prediction", "raw_logit"],
            has_rate_limiting=False,
            exposes_probabilities=True,
        )
        assert report.score < 60
        assert len(report.findings) >= 3
        assert report.high_risk_count >= 2

    def test_report_to_dict_structure(self) -> None:
        report = self.auditor.run_audit()
        d = report.to_dict()
        assert "score" in d
        assert "passed" in d
        assert "findings" in d
        assert "risk_summary" in d

    def test_passed_property(self) -> None:
        # No inputs → only ML05 and ML09 fire (medium/low)
        report = self.auditor.run_audit()
        # ML05 fires as HIGH (no rate limiting, exposes_probabilities=True by default)
        # so passed should be False
        assert report.passed is False

    def test_empty_audit_runs_all_checks(self) -> None:
        report = self.auditor.run_audit()
        assert report.total_checks == 7


# ---------------------------------------------------------------------------
# Unit tests: PII Detector
# ---------------------------------------------------------------------------


class TestPIIDetector:
    def test_detect_email(self) -> None:
        cols = {"contact": ["hello@example.com", "no-email-here", "user@corp.org"]}
        report = detect_pii(cols)
        emails = [f for f in report.findings if f.pii_type == PIIType.EMAIL]
        assert len(emails) == 1
        assert emails[0].column == "contact"
        assert emails[0].match_count == 2

    def test_detect_phone(self) -> None:
        cols = {"phone": ["555-123-4567", "not-a-phone", "+1 (800) 555-0100"]}
        report = detect_pii(cols)
        phones = [f for f in report.findings if f.pii_type == PIIType.PHONE]
        assert len(phones) == 1
        assert phones[0].match_count >= 1

    def test_detect_credit_card(self) -> None:
        cols = {"payment": ["4111111111111111", "nothing", "5500 0000 0000 0004"]}
        report = detect_pii(cols)
        cards = [f for f in report.findings if f.pii_type == PIIType.CREDIT_CARD]
        assert len(cards) == 1
        assert cards[0].severity == "critical"

    def test_detect_ip_address(self) -> None:
        cols = {"log": ["192.168.1.100", "10.0.0.1", "not-an-ip"]}
        report = detect_pii(cols)
        ips = [f for f in report.findings if f.pii_type == PIIType.IP_ADDRESS]
        assert len(ips) == 1

    def test_no_pii_clean_dataset(self) -> None:
        cols = {"score": ["0.9", "0.7", "0.85"], "label": ["fraud", "normal", "fraud"]}
        report = detect_pii(cols)
        assert len(report.findings) == 0
        assert report.gdpr_compliant is True

    def test_gdpr_not_compliant_with_email(self) -> None:
        cols = {"email": ["user@example.com"]}
        report = detect_pii(cols)
        assert report.gdpr_compliant is False
        assert "email" in report.critical_columns

    def test_affected_columns(self) -> None:
        cols = {
            "email_col": ["test@example.com"],
            "clean_col": ["no pii here"],
        }
        report = detect_pii(cols)
        assert "email_col" in report.affected_columns
        assert "clean_col" not in report.affected_columns

    def test_masked_examples_not_empty(self) -> None:
        cols = {"card": ["4111111111111111"]}
        report = detect_pii(cols)
        assert len(report.findings) > 0
        for f in report.findings:
            if f.pii_type == PIIType.CREDIT_CARD:
                assert len(f.masked_examples) > 0
                assert "***" in f.masked_examples[0] or f.masked_examples[0][-4:].isdigit()

    def test_empty_columns(self) -> None:
        report = detect_pii({})
        assert report.findings == []
        assert report.gdpr_compliant is True

    def test_to_dict_structure(self) -> None:
        cols = {"email": ["a@b.com"]}
        d = detect_pii(cols).to_dict()
        assert "gdpr_compliant" in d
        assert "findings" in d
        assert "affected_columns" in d


# ---------------------------------------------------------------------------
# API endpoint tests
# ---------------------------------------------------------------------------


class TestSecurityAuditAPIEndpoint:
    def test_audit_clean_returns_200(self) -> None:
        payload = {
            "numeric_columns": {"age": [25, 30, 35, 40]},
            "all_columns": {"gender": ["M", "F", "M", "F"]},
            "label_column": [0, 1, 0, 1],
            "output_fields": ["prediction", "hmac_signature"],
            "has_rate_limiting": True,
            "exposes_probabilities": False,
        }
        resp = client.post("/security/audit", json=payload)
        assert resp.status_code == 200
        data = resp.json()
        assert "score" in data
        assert "findings" in data
        assert "risk_summary" in data

    def test_audit_vulnerable_has_findings(self) -> None:
        payload = {
            "has_rate_limiting": False,
            "exposes_probabilities": True,
            "output_fields": ["prediction"],
        }
        resp = client.post("/security/audit", json=payload)
        assert resp.status_code == 200
        data = resp.json()
        assert data["score"] < 100
        assert len(data["findings"]) > 0

    def test_audit_score_range(self) -> None:
        resp = client.post("/security/audit", json={})
        assert resp.status_code == 200
        score = resp.json()["score"]
        assert 0 <= score <= 100


class TestPIIScanAPIEndpoint:
    def test_pii_scan_with_email(self) -> None:
        payload = {"columns": {"email": ["user@example.com", "other@corp.org"]}}
        resp = client.post("/security/pii", json=payload)
        assert resp.status_code == 200
        data = resp.json()
        assert data["gdpr_compliant"] is False
        assert len(data["findings"]) > 0

    def test_pii_scan_clean_data(self) -> None:
        payload = {"columns": {"score": ["0.9", "0.7"], "label": ["ok", "fraud"]}}
        resp = client.post("/security/pii", json=payload)
        assert resp.status_code == 200
        data = resp.json()
        assert data["gdpr_compliant"] is True
        assert data["findings"] == []

    def test_pii_scan_returns_affected_columns(self) -> None:
        payload = {
            "columns": {
                "emails": ["a@b.com"],
                "names": ["John Doe"],
                "safe": ["some text"],
            }
        }
        resp = client.post("/security/pii", json=payload)
        assert resp.status_code == 200
        data = resp.json()
        assert "emails" in data["affected_columns"]


class TestSecurityChecklistEndpoint:
    def test_checklist_returns_10_items(self) -> None:
        resp = client.get("/security/checklist")
        assert resp.status_code == 200
        data = resp.json()
        assert len(data["owasp_ml_top_10"]) == 10

    def test_checklist_has_required_fields(self) -> None:
        resp = client.get("/security/checklist")
        items = resp.json()["owasp_ml_top_10"]
        for item in items:
            assert "id" in item
            assert "title" in item
            assert "mitigation" in item

    def test_checklist_ids_sequential(self) -> None:
        resp = client.get("/security/checklist")
        ids = [item["id"] for item in resp.json()["owasp_ml_top_10"]]
        assert ids == [f"ML{i:02d}" for i in range(1, 11)]
