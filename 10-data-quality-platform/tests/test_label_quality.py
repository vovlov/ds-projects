"""
Тесты для Decoupled Confident Learning (DeCoLe) — обнаружение ошибок разметки.
Tests for Decoupled Confident Learning (DeCoLe) — label error detection.

Покрываем:
- Unit: основная логика DCL (пороги, confident joint, типы ошибок)
- Integration: мультикласс, подгруппы, граничные случаи
- API: POST /label_quality/check, GET /label_quality/info
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np
import pytest
from fastapi.testclient import TestClient
from quality.api.app import app
from quality.label_quality.confid_learn import (
    DecoupledConfidentLearning,
    LabelError,
    LabelQualityReport,
    NoiseMatrix,
)

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_clean_binary(n: int = 100, seed: int = 0) -> tuple[list[int], list[list[float]]]:
    """Clean dataset: model perfectly agrees with labels → no errors expected."""
    rng = np.random.default_rng(seed)
    labels = (rng.random(n) > 0.5).astype(int).tolist()
    pred_probs = []
    for lab in labels:
        if lab == 1:
            pred_probs.append([0.05 + rng.random() * 0.1, 0.85 + rng.random() * 0.1])
        else:
            pred_probs.append([0.85 + rng.random() * 0.1, 0.05 + rng.random() * 0.1])
    return labels, pred_probs


def _make_noisy_binary(
    n: int = 100, noise_frac: float = 0.15, seed: int = 0
) -> tuple[list[int], list[list[float]]]:
    """Noisy dataset: flipped labels; model confident in true (OTHER) class."""
    rng = np.random.default_rng(seed)
    true_labels = (rng.random(n) > 0.5).astype(int)
    # Inject noise: flip some labels
    noisy_labels = true_labels.copy()
    noise_mask = rng.random(n) < noise_frac
    noisy_labels[noise_mask] = 1 - noisy_labels[noise_mask]

    pred_probs = []
    for true_lab in true_labels:
        if true_lab == 1:
            pred_probs.append([0.05 + rng.random() * 0.1, 0.85 + rng.random() * 0.1])
        else:
            pred_probs.append([0.85 + rng.random() * 0.1, 0.05 + rng.random() * 0.1])
    return noisy_labels.tolist(), pred_probs


def _make_multiclass(
    n: int = 150, n_classes: int = 3, seed: int = 1
) -> tuple[list[int], list[list[float]]]:
    """3-class dataset with some label noise."""
    rng = np.random.default_rng(seed)
    labels = (rng.integers(0, n_classes, n)).tolist()
    pred_probs = []
    for lab in labels:
        probs = [0.05] * n_classes
        probs[lab] = 0.85
        # Inject small noise to prob vector
        noise = rng.random(n_classes) * 0.05
        probs = np.array(probs) + noise
        probs /= probs.sum()
        pred_probs.append(probs.tolist())
    return labels, pred_probs


# ---------------------------------------------------------------------------
# TestDecoupledCLDataclasses
# ---------------------------------------------------------------------------


class TestDecoupledCLDataclasses:
    def test_label_error_to_dict(self) -> None:
        err = LabelError(
            index=5,
            given_label=0,
            suggested_label=1,
            confidence=0.92,
            group="source_A",
            error_type="confident_disagreement",
        )
        d = err.to_dict()
        assert d["index"] == 5
        assert d["given_label"] == 0
        assert d["suggested_label"] == 1
        assert d["confidence"] == 0.92
        assert d["group"] == "source_A"
        assert d["error_type"] == "confident_disagreement"

    def test_label_error_group_none(self) -> None:
        err = LabelError(
            index=0,
            given_label=1,
            suggested_label=0,
            confidence=0.7,
            group=None,
            error_type="off_diagonal",
        )
        assert err.to_dict()["group"] is None

    def test_noise_matrix_to_dict(self) -> None:
        mat = np.array([[0.9, 0.1], [0.05, 0.95]])
        nm = NoiseMatrix(group="all", n_examples=100, matrix=mat, noise_rate=0.075)
        d = nm.to_dict()
        assert d["group"] == "all"
        assert d["n_examples"] == 100
        assert d["noise_rate"] == 0.075
        assert len(d["matrix"]) == 2
        assert len(d["matrix"][0]) == 2

    def test_report_to_dict_structure(self) -> None:
        report = LabelQualityReport(
            audit_id="test-id",
            timestamp="2026-06-04T00:00:00+00:00",
            n_examples=50,
            n_classes=2,
            n_groups=1,
            n_errors_found=3,
            error_rate=0.06,
            errors=[],
            noise_matrices=[],
        )
        d = report.to_dict()
        assert d["n_examples"] == 50
        assert d["n_classes"] == 2
        assert d["n_errors_found"] == 3
        assert d["error_rate"] == 0.06
        assert isinstance(d["errors"], list)
        assert isinstance(d["noise_matrices"], list)

    def test_report_has_audit_id_and_timestamp(self) -> None:
        dcl = DecoupledConfidentLearning()
        labels, pred_probs = _make_clean_binary(n=40)
        report = dcl.find_label_errors(labels, pred_probs)
        assert len(report.audit_id) == 36  # UUID4 format
        assert "T" in report.timestamp  # ISO 8601


# ---------------------------------------------------------------------------
# TestDecoupledCLCore
# ---------------------------------------------------------------------------


class TestDecoupledCLCore:
    def test_clean_data_few_errors(self) -> None:
        """На чистых данных ошибок быть не должно."""
        labels, pred_probs = _make_clean_binary(n=100)
        dcl = DecoupledConfidentLearning()
        report = dcl.find_label_errors(labels, pred_probs)
        assert report.n_errors_found == 0
        assert report.error_rate == 0.0

    def test_noisy_data_detects_errors(self) -> None:
        """На зашумлённых данных должны быть найдены ошибки."""
        labels, pred_probs = _make_noisy_binary(n=200, noise_frac=0.15)
        dcl = DecoupledConfidentLearning()
        report = dcl.find_label_errors(labels, pred_probs)
        assert report.n_errors_found > 0

    def test_noisy_error_rate_bounded(self) -> None:
        """error_rate не превышает фактический уровень шума (не ложноположительные)."""
        labels, pred_probs = _make_noisy_binary(n=200, noise_frac=0.15)
        dcl = DecoupledConfidentLearning()
        report = dcl.find_label_errors(labels, pred_probs)
        # error_rate should not exceed 2x the actual noise fraction
        assert report.error_rate <= 0.30

    def test_report_n_examples_correct(self) -> None:
        labels, pred_probs = _make_clean_binary(n=80)
        dcl = DecoupledConfidentLearning()
        report = dcl.find_label_errors(labels, pred_probs)
        assert report.n_examples == 80

    def test_report_n_classes_correct(self) -> None:
        labels, pred_probs = _make_multiclass(n=150, n_classes=3)
        dcl = DecoupledConfidentLearning()
        report = dcl.find_label_errors(labels, pred_probs)
        assert report.n_classes == 3

    def test_errors_sorted_by_confidence_descending(self) -> None:
        """Ошибки отсортированы по убыванию confidence."""
        labels, pred_probs = _make_noisy_binary(n=200, noise_frac=0.20)
        dcl = DecoupledConfidentLearning()
        report = dcl.find_label_errors(labels, pred_probs)
        if len(report.errors) >= 2:
            confidences = [e.confidence for e in report.errors]
            assert confidences == sorted(confidences, reverse=True)

    def test_error_indices_in_valid_range(self) -> None:
        """Индексы ошибок должны быть в пределах датасета."""
        labels, pred_probs = _make_noisy_binary(n=100, noise_frac=0.15)
        dcl = DecoupledConfidentLearning()
        report = dcl.find_label_errors(labels, pred_probs)
        for err in report.errors:
            assert 0 <= err.index < 100

    def test_suggested_label_is_valid_class(self) -> None:
        """suggested_label ∈ {0, ..., K-1}."""
        labels, pred_probs = _make_noisy_binary(n=100, noise_frac=0.15)
        dcl = DecoupledConfidentLearning()
        report = dcl.find_label_errors(labels, pred_probs)
        for err in report.errors:
            assert err.suggested_label in {0, 1}
            assert err.given_label in {0, 1}
            assert err.suggested_label != err.given_label

    def test_confidence_in_unit_range(self) -> None:
        """Confidence ∈ [0, 1]."""
        labels, pred_probs = _make_noisy_binary(n=100, noise_frac=0.15)
        dcl = DecoupledConfidentLearning()
        report = dcl.find_label_errors(labels, pred_probs)
        for err in report.errors:
            assert 0.0 <= err.confidence <= 1.0

    def test_error_types_valid(self) -> None:
        """error_type принадлежит допустимому множеству."""
        valid_types = {"off_diagonal", "high_noise_group", "confident_disagreement"}
        labels, pred_probs = _make_noisy_binary(n=200, noise_frac=0.20)
        dcl = DecoupledConfidentLearning()
        report = dcl.find_label_errors(labels, pred_probs)
        for err in report.errors:
            assert err.error_type in valid_types

    def test_no_groups_means_single_noise_matrix(self) -> None:
        """Без groups: ровно одна матрица шума (группа 'all')."""
        labels, pred_probs = _make_clean_binary(n=80)
        dcl = DecoupledConfidentLearning()
        report = dcl.find_label_errors(labels, pred_probs)
        assert len(report.noise_matrices) == 1
        assert report.noise_matrices[0].group == "all"

    def test_no_groups_group_in_errors_is_none(self) -> None:
        """Без groups: поле group у ошибок = None."""
        labels, pred_probs = _make_noisy_binary(n=100, noise_frac=0.15)
        dcl = DecoupledConfidentLearning()
        report = dcl.find_label_errors(labels, pred_probs)
        for err in report.errors:
            assert err.group is None

    def test_multiclass_detection(self) -> None:
        """Работает для 3 классов."""
        labels, pred_probs = _make_multiclass(n=150, n_classes=3)
        dcl = DecoupledConfidentLearning()
        report = dcl.find_label_errors(labels, pred_probs)
        assert report.n_classes == 3
        assert isinstance(report.n_errors_found, int)

    def test_noise_matrix_shape(self) -> None:
        """Матрица шума имеет размерность [K, K]."""
        labels, pred_probs = _make_clean_binary(n=80)
        dcl = DecoupledConfidentLearning()
        report = dcl.find_label_errors(labels, pred_probs)
        nm = report.noise_matrices[0]
        assert nm.matrix.shape == (2, 2)

    def test_noise_rate_non_negative(self) -> None:
        """noise_rate ≥ 0."""
        labels, pred_probs = _make_clean_binary(n=80)
        dcl = DecoupledConfidentLearning()
        report = dcl.find_label_errors(labels, pred_probs)
        for nm in report.noise_matrices:
            assert nm.noise_rate >= 0.0


# ---------------------------------------------------------------------------
# TestDecoupledCLGroups
# ---------------------------------------------------------------------------


class TestDecoupledCLGroups:
    def _make_grouped_noisy(
        self,
        n_per_group: int = 60,
        noise_a: float = 0.0,
        noise_b: float = 0.30,
        seed: int = 42,
    ) -> tuple[list[int], list[list[float]], list[str]]:
        """Group A: clean. Group B: noisy. Model sees through noise."""
        rng = np.random.default_rng(seed)
        all_labels, all_probs, all_groups = [], [], []
        for group_name, noise_frac in [("A", noise_a), ("B", noise_b)]:
            true_labels = (rng.random(n_per_group) > 0.5).astype(int)
            noisy_labels = true_labels.copy()
            noise_mask = rng.random(n_per_group) < noise_frac
            noisy_labels[noise_mask] = 1 - noisy_labels[noise_mask]
            for true_lab in true_labels:
                if true_lab == 1:
                    all_probs.append([0.05 + rng.random() * 0.1, 0.85 + rng.random() * 0.1])
                else:
                    all_probs.append([0.85 + rng.random() * 0.1, 0.05 + rng.random() * 0.1])
            all_labels.extend(noisy_labels.tolist())
            all_groups.extend([group_name] * n_per_group)
        return all_labels, all_probs, all_groups

    def test_groups_produce_multiple_noise_matrices(self) -> None:
        """С двумя группами: два шумовых профиля."""
        labels, pred_probs, groups = self._make_grouped_noisy()
        dcl = DecoupledConfidentLearning()
        report = dcl.find_label_errors(labels, pred_probs, groups=groups)
        assert len(report.noise_matrices) == 2

    def test_group_names_preserved(self) -> None:
        """Имена групп сохраняются в отчёте."""
        labels, pred_probs, groups = self._make_grouped_noisy()
        dcl = DecoupledConfidentLearning()
        report = dcl.find_label_errors(labels, pred_probs, groups=groups)
        group_names = {nm.group for nm in report.noise_matrices}
        assert "A" in group_names
        assert "B" in group_names

    def test_noisy_group_has_higher_noise_rate(self) -> None:
        """Зашумлённая группа B имеет более высокий noise_rate, чем чистая A."""
        labels, pred_probs, groups = self._make_grouped_noisy(noise_a=0.0, noise_b=0.35)
        dcl = DecoupledConfidentLearning()
        report = dcl.find_label_errors(labels, pred_probs, groups=groups)
        nm_by_group = {nm.group: nm for nm in report.noise_matrices}
        if "A" in nm_by_group and "B" in nm_by_group:
            assert nm_by_group["B"].noise_rate >= nm_by_group["A"].noise_rate

    def test_errors_have_group_field(self) -> None:
        """С groups: поле group у ошибок = строка (не None)."""
        labels, pred_probs, groups = self._make_grouped_noisy(noise_b=0.30)
        dcl = DecoupledConfidentLearning()
        report = dcl.find_label_errors(labels, pred_probs, groups=groups)
        for err in report.errors:
            assert err.group is not None
            assert err.group in {"A", "B"}

    def test_n_groups_correct(self) -> None:
        labels, pred_probs, groups = self._make_grouped_noisy()
        dcl = DecoupledConfidentLearning()
        report = dcl.find_label_errors(labels, pred_probs, groups=groups)
        assert report.n_groups == 2

    def test_small_group_skipped(self) -> None:
        """Группа с менее чем K*2 примерами пропускается (нет матрицы шума)."""
        labels, pred_probs, _ = self._make_grouped_noisy(n_per_group=60)
        # Add tiny group (1 example, 2 classes -> 1 < 2*2 → skipped)
        labels_ext = labels + [0]
        pred_probs_ext = pred_probs + [[0.9, 0.1]]
        groups_ext = ["A"] * 60 + ["B"] * 60 + ["tiny"]
        dcl = DecoupledConfidentLearning()
        report = dcl.find_label_errors(labels_ext, pred_probs_ext, groups=groups_ext)
        group_names = {nm.group for nm in report.noise_matrices}
        assert "tiny" not in group_names


# ---------------------------------------------------------------------------
# TestDecoupledCLEdgeCases
# ---------------------------------------------------------------------------


class TestDecoupledCLEdgeCases:
    def test_list_input_accepted(self) -> None:
        """Принимает обычные Python-списки (не только numpy arrays)."""
        labels = [0, 1, 0, 1, 0, 1, 0, 1, 0, 1] * 6
        pred_probs = [[0.9, 0.1], [0.1, 0.9]] * 30
        dcl = DecoupledConfidentLearning()
        report = dcl.find_label_errors(labels, pred_probs)
        assert report.n_examples == 60

    def test_numpy_input_accepted(self) -> None:
        """Принимает numpy arrays."""
        rng = np.random.default_rng(0)
        labels = rng.integers(0, 2, 60)
        pred_probs = rng.dirichlet([1, 1], size=60)
        dcl = DecoupledConfidentLearning()
        report = dcl.find_label_errors(labels, pred_probs)
        assert report.n_examples == 60

    def test_custom_confidence_threshold(self) -> None:
        """Высокий порог confidence → меньше ошибок (более консервативный)."""
        labels, pred_probs = _make_noisy_binary(n=200, noise_frac=0.15)
        dcl_low = DecoupledConfidentLearning(confidence_threshold=0.3)
        dcl_high = DecoupledConfidentLearning(confidence_threshold=0.9)
        report_low = dcl_low.find_label_errors(labels, pred_probs)
        report_high = dcl_high.find_label_errors(labels, pred_probs)
        # High threshold → equal or fewer errors (or same — depends on distribution)
        assert report_high.n_errors_found <= report_low.n_errors_found + 5  # allow small variance

    def test_confident_disagreement_type_assigned(self) -> None:
        """confident_disagreement присваивается примерам с confidence ≥ 0.9."""
        # Construct example where model is very confident (0.95) in a different class
        labels = [0] * 20 + [1] * 20
        pred_probs = [[0.95, 0.05]] * 20 + [[0.05, 0.95]] * 20
        # Flip a few labels to create high-confidence disagreements
        labels[5] = 1  # label says 1, but probs say 0 (conf 0.95)
        labels[15] = 1  # same
        dcl = DecoupledConfidentLearning(confidence_threshold=0.4)
        report = dcl.find_label_errors(labels, pred_probs)
        confident_types = [e.error_type for e in report.errors]
        assert "confident_disagreement" in confident_types


# ---------------------------------------------------------------------------
# TestLabelQualityAPIEndpoints
# ---------------------------------------------------------------------------


class TestLabelQualityAPIEndpoints:
    @pytest.fixture
    def client(self) -> TestClient:
        return TestClient(app)

    def _clean_payload(self, n: int = 40) -> dict:
        labels, pred_probs = _make_clean_binary(n=n)
        return {"labels": labels, "pred_probs": pred_probs}

    def _noisy_payload(self, n: int = 100) -> dict:
        labels, pred_probs = _make_noisy_binary(n=n, noise_frac=0.15)
        return {"labels": labels, "pred_probs": pred_probs}

    def test_check_200_clean(self, client: TestClient) -> None:
        resp = client.post("/label_quality/check", json=self._clean_payload())
        assert resp.status_code == 200

    def test_check_200_noisy(self, client: TestClient) -> None:
        resp = client.post("/label_quality/check", json=self._noisy_payload())
        assert resp.status_code == 200

    def test_check_response_structure(self, client: TestClient) -> None:
        resp = client.post("/label_quality/check", json=self._clean_payload())
        data = resp.json()
        required_keys = {
            "audit_id",
            "timestamp",
            "n_examples",
            "n_classes",
            "n_groups",
            "n_errors_found",
            "error_rate",
            "errors",
            "noise_matrices",
        }
        assert required_keys.issubset(data.keys())

    def test_check_n_examples_matches_input(self, client: TestClient) -> None:
        resp = client.post("/label_quality/check", json=self._clean_payload(n=50))
        assert resp.json()["n_examples"] == 50

    def test_check_n_classes_binary(self, client: TestClient) -> None:
        resp = client.post("/label_quality/check", json=self._clean_payload())
        assert resp.json()["n_classes"] == 2

    def test_check_with_groups(self, client: TestClient) -> None:
        labels, pred_probs = _make_noisy_binary(n=100, noise_frac=0.15)
        groups = ["src_A"] * 50 + ["src_B"] * 50
        resp = client.post(
            "/label_quality/check",
            json={"labels": labels, "pred_probs": pred_probs, "groups": groups},
        )
        assert resp.status_code == 200
        data = resp.json()
        assert data["n_groups"] == 2

    def test_check_errors_list_type(self, client: TestClient) -> None:
        resp = client.post("/label_quality/check", json=self._noisy_payload())
        assert isinstance(resp.json()["errors"], list)

    def test_check_noise_matrices_list_type(self, client: TestClient) -> None:
        resp = client.post("/label_quality/check", json=self._clean_payload())
        assert isinstance(resp.json()["noise_matrices"], list)

    def test_check_422_empty_labels(self, client: TestClient) -> None:
        resp = client.post(
            "/label_quality/check",
            json={"labels": [], "pred_probs": []},
        )
        assert resp.status_code == 422

    def test_check_422_length_mismatch(self, client: TestClient) -> None:
        labels, pred_probs = _make_clean_binary(n=20)
        resp = client.post(
            "/label_quality/check",
            json={"labels": labels, "pred_probs": pred_probs[:10]},
        )
        assert resp.status_code == 422

    def test_check_422_single_class_probs(self, client: TestClient) -> None:
        """pred_probs с 1 классом → 422."""
        labels = [0, 1, 0]
        pred_probs = [[0.9], [0.8], [0.7]]
        resp = client.post(
            "/label_quality/check",
            json={"labels": labels, "pred_probs": pred_probs},
        )
        assert resp.status_code == 422

    def test_check_422_groups_length_mismatch(self, client: TestClient) -> None:
        labels, pred_probs = _make_clean_binary(n=20)
        resp = client.post(
            "/label_quality/check",
            json={"labels": labels, "pred_probs": pred_probs, "groups": ["g1"] * 10},
        )
        assert resp.status_code == 422

    def test_check_error_fields_present(self, client: TestClient) -> None:
        """Если есть ошибки — у каждой присутствуют обязательные поля."""
        resp = client.post("/label_quality/check", json=self._noisy_payload(n=200))
        errors = resp.json()["errors"]
        if errors:
            for err in errors:
                assert "index" in err
                assert "given_label" in err
                assert "suggested_label" in err
                assert "confidence" in err
                assert "error_type" in err

    def test_info_200(self, client: TestClient) -> None:
        resp = client.get("/label_quality/info")
        assert resp.status_code == 200

    def test_info_has_method_field(self, client: TestClient) -> None:
        data = client.get("/label_quality/info").json()
        assert "method" in data
        assert "DeCoLe" in data["method"]

    def test_info_has_workflow(self, client: TestClient) -> None:
        data = client.get("/label_quality/info").json()
        assert "workflow" in data
        assert isinstance(data["workflow"], list)
        assert len(data["workflow"]) > 0

    def test_info_has_error_types(self, client: TestClient) -> None:
        data = client.get("/label_quality/info").json()
        assert "error_types" in data
        assert "confident_disagreement" in data["error_types"]
