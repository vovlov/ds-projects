"""
Tests for the document scanner project.

Covers data generation, classifier training, API endpoints, and the CNN
availability check.  Everything runs without PyTorch.
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))


import numpy as np
import pytest
from fastapi.testclient import TestClient
from scanner.data.dataset import (
    DOC_TYPES,
    FEATURE_COLS,
    generate_synthetic_documents,
    get_feature_matrix,
)
from scanner.data.rvl_cdip import (
    FEATURE_COLS as RVL_FEATURE_COLS,
)
from scanner.data.rvl_cdip import (
    RVL_CDIP_CLASSES,
    compute_dataset_stats,
    generate_mock_rvl_cdip,
    load_rvl_cdip,
    to_scanner_format,
)
from scanner.models.classifier import predict, train_classifier
from scanner.models.cnn import is_available as cnn_is_available
from scanner.models.gradcam import is_available as gradcam_is_available
from scanner.preprocessing.layout import (
    LayoutResult,
    RegionType,
    compute_horizontal_projection,
    compute_vertical_projection,
    find_gaps,
    find_text_zones,
    segment_layout,
)

# ---- fixtures ----


@pytest.fixture(scope="module")
def synth_data():
    return generate_synthetic_documents(n=500)


@pytest.fixture(scope="module")
def feature_matrix(synth_data):
    return get_feature_matrix(synth_data)


@pytest.fixture(scope="module")
def trained(feature_matrix):
    X, y, le = feature_matrix
    return train_classifier(X, y, label_encoder=le)


@pytest.fixture(scope="module")
def api_client():
    # import here so module-level state doesn't leak across fixtures
    from scanner.api.app import app

    return TestClient(app)


# ===========================================================================
# TestData
# ===========================================================================


class TestData:
    def test_row_count(self, synth_data):
        # 500 samples, evenly divided among 5 classes -> 500
        assert synth_data.shape[0] == 500

    def test_columns_present(self, synth_data):
        for col in FEATURE_COLS + ["doc_type"]:
            assert col in synth_data.columns

    def test_doc_types(self, synth_data):
        unique = set(synth_data["doc_type"].to_list())
        assert unique == set(DOC_TYPES)

    def test_class_balance(self, synth_data):
        counts = synth_data.group_by("doc_type").len()["len"].to_list()
        assert all(c == 100 for c in counts), "Expected 100 samples per class"

    def test_feature_ranges(self, synth_data):
        # aspect ratio should be positive
        assert synth_data["aspect_ratio"].min() > 0
        # brightness and densities in [0, 1]
        for col in ("brightness", "text_density", "edge_density"):
            assert synth_data[col].min() >= 0.0
            assert synth_data[col].max() <= 1.0

    def test_feature_matrix_shape(self, feature_matrix):
        X, y, le = feature_matrix
        assert X.shape == (500, 5)
        assert y.shape == (500,)

    def test_label_encoder_classes(self, feature_matrix):
        _, _, le = feature_matrix
        assert set(le.classes_) == set(DOC_TYPES)


# ===========================================================================
# TestClassifier
# ===========================================================================


class TestClassifier:
    def test_train_runs(self, trained):
        assert "model" in trained
        assert "accuracy" in trained

    def test_accuracy_above_threshold(self, trained):
        # synthetic data is designed to be separable -- should exceed 50% easily
        assert trained["accuracy"] > 0.5

    def test_f1_macro_reasonable(self, trained):
        assert trained["f1_macro"] > 0.5

    def test_confusion_matrix_shape(self, trained):
        cm = trained["confusion_matrix"]
        assert cm.shape == (5, 5)

    def test_predict_single(self, trained):
        model = trained["model"]
        le = trained["label_encoder"]
        # a receipt-like feature vector
        sample = np.array([[0.35, 0.78, 0.45, 0.30, 120.0]], dtype=np.float32)
        results = predict(model, sample, label_encoder=le)
        assert len(results) == 1
        assert results[0]["doc_type"] in DOC_TYPES
        assert 0.0 <= results[0]["confidence"] <= 1.0

    def test_predict_batch(self, trained):
        model = trained["model"]
        le = trained["label_encoder"]
        batch = np.random.default_rng(0).random((10, 5)).astype(np.float32)
        results = predict(model, batch, label_encoder=le)
        assert len(results) == 10


# ===========================================================================
# TestAPI
# ===========================================================================


class TestAPI:
    def test_health(self, api_client):
        resp = api_client.get("/health")
        assert resp.status_code == 200
        assert resp.json()["status"] == "ok"

    def test_classify_valid(self, api_client):
        payload = {
            "aspect_ratio": 0.77,
            "brightness": 0.82,
            "text_density": 0.60,
            "edge_density": 0.20,
            "file_size_kb": 350.0,
        }
        resp = api_client.post("/classify", json=payload)
        assert resp.status_code == 200
        body = resp.json()
        assert body["doc_type"] in DOC_TYPES
        assert 0.0 <= body["confidence"] <= 1.0

    def test_classify_returns_probabilities(self, api_client):
        payload = {
            "aspect_ratio": 1.58,
            "brightness": 0.55,
            "text_density": 0.25,
            "edge_density": 0.50,
            "file_size_kb": 250.0,
        }
        resp = api_client.post("/classify", json=payload)
        probs = resp.json()["probabilities"]
        # should have one entry per class
        assert len(probs) == len(DOC_TYPES)

    def test_classify_invalid_input(self, api_client):
        # brightness out of range
        payload = {
            "aspect_ratio": 0.77,
            "brightness": 5.0,
            "text_density": 0.60,
            "edge_density": 0.20,
            "file_size_kb": 350.0,
        }
        resp = api_client.post("/classify", json=payload)
        assert resp.status_code == 422  # validation error


# ===========================================================================
# TestCNN
# ===========================================================================


class TestCNN:
    def test_is_available_returns_bool(self):
        result = cnn_is_available()
        assert isinstance(result, bool)

    def test_import_does_not_crash(self):
        """Even without torch the module should import cleanly."""
        import scanner.models.cnn as cnn_mod  # noqa: F811

        assert hasattr(cnn_mod, "TORCH_AVAILABLE")


# ===========================================================================
# TestRVLCDIPMock
# ===========================================================================


class TestRVLCDIPMock:
    """Тесты mock-генератора RVL-CDIP (не требуют скачанного датасета)."""

    @pytest.fixture(scope="class")
    def mock_df(self):
        return generate_mock_rvl_cdip(n_per_class=20, seed=0)

    def test_row_count(self, mock_df):
        # 16 классов × 20 строк = 320
        assert len(mock_df) == 16 * 20

    def test_all_16_classes_present(self, mock_df):
        found = set(mock_df["doc_type"].to_list())
        assert found == set(RVL_CDIP_CLASSES)

    def test_label_range(self, mock_df):
        labels = mock_df["label"].to_list()
        assert min(labels) == 0
        assert max(labels) == 15

    def test_feature_columns_present(self, mock_df):
        for col in RVL_FEATURE_COLS + ["doc_type", "label", "split"]:
            assert col in mock_df.columns, f"Колонка '{col}' отсутствует"

    def test_aspect_ratio_positive(self, mock_df):
        assert mock_df["aspect_ratio"].min() > 0

    def test_brightness_in_range(self, mock_df):
        assert mock_df["brightness"].min() >= 0.0
        assert mock_df["brightness"].max() <= 1.0

    def test_text_density_in_range(self, mock_df):
        assert mock_df["text_density"].min() >= 0.0
        assert mock_df["text_density"].max() <= 1.0

    def test_edge_density_in_range(self, mock_df):
        assert mock_df["edge_density"].min() >= 0.0
        assert mock_df["edge_density"].max() <= 1.0

    def test_file_size_positive(self, mock_df):
        assert mock_df["file_size_kb"].min() > 0.0

    def test_split_column_values(self, mock_df):
        valid_splits = {"train", "val", "test"}
        found = set(mock_df["split"].to_list())
        assert found.issubset(valid_splits)

    def test_reproducibility(self):
        df1 = generate_mock_rvl_cdip(n_per_class=10, seed=42)
        df2 = generate_mock_rvl_cdip(n_per_class=10, seed=42)
        assert df1["aspect_ratio"].to_list() == df2["aspect_ratio"].to_list()

    def test_different_seeds_differ(self):
        df1 = generate_mock_rvl_cdip(n_per_class=10, seed=1)
        df2 = generate_mock_rvl_cdip(n_per_class=10, seed=2)
        assert df1["aspect_ratio"].to_list() != df2["aspect_ratio"].to_list()


# ===========================================================================
# TestRVLCDIPLoader
# ===========================================================================


class TestRVLCDIPLoader:
    """Тесты загрузчика: fallback на mock при отсутствии датасета."""

    def test_load_without_path_returns_df(self):
        df = load_rvl_cdip(data_dir=None)
        assert len(df) > 0

    def test_load_with_nonexistent_path_falls_back(self, tmp_path):
        # Несуществующая директория → mock
        df = load_rvl_cdip(data_dir=tmp_path / "no_such_dir")
        assert len(df) > 0
        assert "doc_type" in df.columns

    def test_load_with_split_filter(self):
        df_train = load_rvl_cdip(data_dir=None, split="train")
        # В mock-данных split назначается случайно, поэтому "train" должен быть
        assert all(s == "train" for s in df_train["split"].to_list())

    def test_load_with_label_file(self, tmp_path):
        """Проверяем парсинг label-файлов RVL-CDIP формата '<path> <id>'."""
        labels_dir = tmp_path / "labels"
        labels_dir.mkdir()
        # Минимальный train.txt: 3 строки из разных классов
        (labels_dir / "train.txt").write_text(
            "images/letter/doc1.tif 0\nimages/form/doc2.tif 1\nimages/invoice/doc3.tif 11\n"
        )
        df = load_rvl_cdip(data_dir=tmp_path, split="train")
        assert len(df) == 3
        assert "image_path" in df.columns
        assert set(df["doc_type"].to_list()) == {"letter", "form", "invoice"}

    def test_load_label_file_label_ids(self, tmp_path):
        labels_dir = tmp_path / "labels"
        labels_dir.mkdir()
        (labels_dir / "val.txt").write_text("images/budget/b1.tif 10\nimages/resume/r1.tif 14\n")
        df = load_rvl_cdip(data_dir=tmp_path, split="val")
        assert df["label"].to_list() == [10, 14]


# ===========================================================================
# TestRVLCDIPStats
# ===========================================================================


class TestRVLCDIPStats:
    """Тесты compute_dataset_stats()."""

    @pytest.fixture(scope="class")
    def stats(self):
        df = generate_mock_rvl_cdip(n_per_class=50, seed=7)
        return compute_dataset_stats(df)

    def test_n_total(self, stats):
        assert stats["n_total"] == 16 * 50

    def test_n_classes(self, stats):
        assert stats["n_classes"] == 16

    def test_class_distribution_keys(self, stats):
        assert set(stats["class_distribution"].keys()) == set(RVL_CDIP_CLASSES)

    def test_class_distribution_sum(self, stats):
        total = sum(stats["class_distribution"].values())
        assert total == stats["n_total"]

    def test_imbalance_ratio_balanced(self, stats):
        # mock-данные сбалансированы → imbalance close to 1
        assert stats["imbalance_ratio"] == pytest.approx(1.0, abs=0.1)

    def test_split_counts_present(self, stats):
        assert "split_counts" in stats
        assert isinstance(stats["split_counts"], dict)

    def test_split_counts_sum(self, stats):
        total = sum(stats["split_counts"].values())
        assert total == stats["n_total"]


# ===========================================================================
# TestRVLCDIPToScannerFormat
# ===========================================================================


class TestRVLCDIPToScannerFormat:
    """Тесты конвертации в формат, совместимый с scanner.data.dataset."""

    @pytest.fixture(scope="class")
    def scanner_df(self):
        df = generate_mock_rvl_cdip(n_per_class=10, seed=0)
        return to_scanner_format(df)

    def test_has_feature_cols(self, scanner_df):
        for col in RVL_FEATURE_COLS:
            assert col in scanner_df.columns

    def test_has_doc_type(self, scanner_df):
        assert "doc_type" in scanner_df.columns

    def test_no_extra_columns(self, scanner_df):
        expected = set(RVL_FEATURE_COLS + ["doc_type"])
        assert set(scanner_df.columns) == expected

    def test_missing_columns_raises(self):
        import polars as pl

        bad_df = pl.DataFrame({"doc_type": ["letter"], "aspect_ratio": [0.77]})
        with pytest.raises(ValueError, match="Колонки отсутствуют"):
            to_scanner_format(bad_df)

    def test_compatible_with_get_feature_matrix(self, scanner_df):
        """Убеждаемся, что to_scanner_format → get_feature_matrix работает."""
        from scanner.data.dataset import DOC_TYPES, get_feature_matrix

        # Оставляем только классы из 5-классового датасета для совместимости
        filtered = scanner_df.filter(scanner_df["doc_type"].is_in(DOC_TYPES))
        if len(filtered) > 0:
            X, y, le = get_feature_matrix(filtered)
            assert X.shape[1] == 5


# ===========================================================================
# TestGradCAM
# ===========================================================================


class TestGradCAM:
    """Тесты Grad-CAM модуля (без PyTorch — только API и fallback)."""

    def test_is_available_returns_bool(self):
        result = gradcam_is_available()
        assert isinstance(result, bool)

    def test_import_does_not_crash(self):
        import scanner.models.gradcam as gcam_mod

        assert hasattr(gcam_mod, "TORCH_AVAILABLE")

    def test_gradcam_class_exists(self):
        from scanner.models.gradcam import GradCAM

        assert GradCAM is not None

    def test_explain_prediction_exists(self):
        from scanner.models.gradcam import explain_prediction

        assert callable(explain_prediction)

    def test_preprocess_image_exists(self):
        from scanner.models.gradcam import preprocess_image

        assert callable(preprocess_image)

    def test_gradcam_raises_without_torch(self):
        """Град-CAM должен поднимать RuntimeError без PyTorch."""
        import scanner.models.gradcam as gcam_mod

        if gcam_mod.TORCH_AVAILABLE:
            pytest.skip("PyTorch доступен — тест только для среды без torch")

        with pytest.raises(RuntimeError, match="PyTorch"):
            gcam_mod.GradCAM(model=None, target_layer=None)

    def test_preprocess_image_raises_without_torch(self):
        import scanner.models.gradcam as gcam_mod

        if gcam_mod.TORCH_AVAILABLE:
            pytest.skip("PyTorch доступен")

        with pytest.raises(RuntimeError, match="PyTorch"):
            gcam_mod.preprocess_image("dummy.jpg")

    def test_explain_prediction_raises_without_torch(self):
        import scanner.models.gradcam as gcam_mod

        if gcam_mod.TORCH_AVAILABLE:
            pytest.skip("PyTorch доступен")

        with pytest.raises(RuntimeError, match="PyTorch"):
            gcam_mod.explain_prediction(
                model=None,
                image_tensor=None,
                target_layer=None,
            )


# ===========================================================================
# TestDocumentQualityAssessment
# ===========================================================================


class TestDocumentQualityAssessment:
    """Unit tests for scanner/preprocessing/quality.py — numpy-only, no torch."""

    from scanner.preprocessing.quality import (
        QualityMetrics,
        assess_quality,
        estimate_blur,
        estimate_brightness,
        estimate_contrast,
        estimate_noise,
        estimate_skew,
    )

    # ---- blur ----

    def test_blur_sharp_checkerboard(self):
        """Checkerboard has maximum second-derivative variance → high score."""
        from scanner.preprocessing.quality import estimate_blur

        tile = np.array([[0, 255], [255, 0]], dtype=np.uint8)
        pixels = np.tile(tile, (50, 50))
        assert estimate_blur(pixels) > 0.5

    def test_blur_flat_image_low(self):
        """Uniform image has zero Laplacian variance → near-zero score."""
        from scanner.preprocessing.quality import estimate_blur

        pixels = np.full((50, 50), 128, dtype=np.uint8)
        assert estimate_blur(pixels) < 0.3

    def test_blur_returns_in_unit_interval(self):
        from scanner.preprocessing.quality import estimate_blur

        rng = np.random.default_rng(7)
        pixels = rng.integers(0, 256, (40, 60), dtype=np.uint8)
        score = estimate_blur(pixels)
        assert 0.0 <= score <= 1.0

    # ---- brightness ----

    def test_brightness_midpoint_optimal(self):
        """Mean ≈ 128 (0.5 normalised) should give score close to 1.0."""
        from scanner.preprocessing.quality import estimate_brightness

        pixels = np.full((20, 20), 128, dtype=np.uint8)
        assert estimate_brightness(pixels) > 0.9

    def test_brightness_all_black_poor(self):
        from scanner.preprocessing.quality import estimate_brightness

        pixels = np.zeros((20, 20), dtype=np.uint8)
        assert estimate_brightness(pixels) < 0.1

    def test_brightness_all_white_poor(self):
        from scanner.preprocessing.quality import estimate_brightness

        pixels = np.full((20, 20), 255, dtype=np.uint8)
        assert estimate_brightness(pixels) < 0.1

    # ---- contrast ----

    def test_contrast_high_binary_image(self):
        """Alternating 0/255 columns → maximum std → score near 1.0."""
        from scanner.preprocessing.quality import estimate_contrast

        pixels = np.tile(np.array([[0, 255]], dtype=np.uint8), (30, 30))
        assert estimate_contrast(pixels) >= 0.9

    def test_contrast_flat_image_low(self):
        from scanner.preprocessing.quality import estimate_contrast

        pixels = np.full((30, 30), 200, dtype=np.uint8)
        assert estimate_contrast(pixels) < 0.05

    # ---- noise ----

    def test_noise_smooth_gradient_low(self):
        """Smooth gradient → low local variation → low noise level."""
        from scanner.preprocessing.quality import estimate_noise

        x = np.linspace(0, 255, 60).astype(np.uint8)
        pixels = np.tile(x, (40, 1))
        assert estimate_noise(pixels) < 0.5

    def test_noise_random_image_high(self):
        """Pure random noise → high local variation → high noise level."""
        from scanner.preprocessing.quality import estimate_noise

        pixels = np.random.default_rng(42).integers(0, 256, (50, 50), dtype=np.uint8)
        assert estimate_noise(pixels) > 0.5

    # ---- skew ----

    def test_skew_returns_float_in_range(self):
        from scanner.preprocessing.quality import estimate_skew

        pixels = np.random.default_rng(0).integers(0, 256, (40, 60), dtype=np.uint8)
        angle = estimate_skew(pixels)
        assert isinstance(angle, float)
        assert -45.0 <= angle <= 45.0

    def test_skew_tiny_image_returns_zero(self):
        """Image smaller than 8×8 should return 0 (not enough data)."""
        from scanner.preprocessing.quality import estimate_skew

        pixels = np.zeros((4, 4), dtype=np.uint8)
        assert estimate_skew(pixels) == 0.0

    # ---- assess_quality ----

    def test_assess_quality_returns_dataclass(self):
        from scanner.preprocessing.quality import QualityMetrics, assess_quality

        pixels = np.random.default_rng(1).integers(50, 200, (40, 50), dtype=np.uint8)
        metrics = assess_quality(pixels)
        assert isinstance(metrics, QualityMetrics)

    def test_assess_quality_overall_in_unit_interval(self):
        from scanner.preprocessing.quality import assess_quality

        pixels = np.random.default_rng(2).integers(0, 256, (30, 40), dtype=np.uint8)
        metrics = assess_quality(pixels)
        assert 0.0 <= metrics.overall_score <= 1.0

    def test_assess_quality_is_acceptable_is_bool(self):
        from scanner.preprocessing.quality import assess_quality

        pixels = np.full((20, 20), 128, dtype=np.uint8)
        metrics = assess_quality(pixels)
        assert isinstance(metrics.is_acceptable, bool)

    def test_assess_quality_all_white_rejected(self):
        """Over-exposed scan (all 255) must be rejected with a reason."""
        from scanner.preprocessing.quality import assess_quality

        pixels = np.full((40, 50), 255, dtype=np.uint8)
        metrics = assess_quality(pixels)
        assert not metrics.is_acceptable
        assert metrics.rejection_reason is not None
        assert len(metrics.rejection_reason) > 0

    def test_assess_quality_to_dict_keys(self):
        from scanner.preprocessing.quality import assess_quality

        pixels = np.full((10, 10), 128, dtype=np.uint8)
        d = assess_quality(pixels).to_dict()
        expected = {
            "blur_score",
            "brightness_score",
            "contrast_score",
            "noise_level",
            "skew_angle_deg",
            "overall_score",
            "is_acceptable",
            "rejection_reason",
        }
        assert expected == set(d.keys())

    def test_assess_quality_custom_threshold(self):
        """With accept_threshold=0.0 every image should be accepted."""
        from scanner.preprocessing.quality import assess_quality

        pixels = np.zeros((20, 20), dtype=np.uint8)
        metrics = assess_quality(pixels, accept_threshold=0.0)
        assert metrics.is_acceptable


# ===========================================================================
# TestQualityAPIEndpoints
# ===========================================================================


class TestQualityAPIEndpoints:
    """Integration tests for /quality/assess and /classify/gated."""

    @pytest.fixture(scope="class")
    def qclient(self):
        from scanner.api.app import app

        return TestClient(app)

    def _mid_pixels(self, h: int = 25, w: int = 30, seed: int = 0) -> list[list[int]]:
        """Mid-tone random pixel matrix that usually passes the quality gate."""
        return np.random.default_rng(seed).integers(80, 180, (h, w)).tolist()

    def _white_pixels(self, h: int = 20, w: int = 20) -> list[list[int]]:
        return [[255] * w for _ in range(h)]

    def _sample_features(self) -> dict:
        return {
            "aspect_ratio": 0.77,
            "brightness": 0.82,
            "text_density": 0.60,
            "edge_density": 0.20,
            "file_size_kb": 350.0,
        }

    def test_quality_assess_returns_200(self, qclient):
        resp = qclient.post("/quality/assess", json={"pixels": self._mid_pixels()})
        assert resp.status_code == 200

    def test_quality_assess_response_has_required_fields(self, qclient):
        resp = qclient.post("/quality/assess", json={"pixels": self._mid_pixels()})
        body = resp.json()
        for field in (
            "blur_score",
            "brightness_score",
            "contrast_score",
            "noise_level",
            "skew_angle_deg",
            "overall_score",
            "is_acceptable",
        ):
            assert field in body, f"Missing field: {field}"

    def test_quality_assess_all_white_not_acceptable(self, qclient):
        resp = qclient.post("/quality/assess", json={"pixels": self._white_pixels()})
        assert resp.status_code == 200
        assert resp.json()["is_acceptable"] is False

    def test_quality_assess_empty_matrix_rejected(self, qclient):
        resp = qclient.post("/quality/assess", json={"pixels": []})
        assert resp.status_code == 422

    def test_classify_gated_passes_quality_check(self, qclient):
        payload = {
            "quality_pixels": self._mid_pixels(seed=5),
            "features": self._sample_features(),
        }
        resp = qclient.post("/classify/gated", json=payload)
        assert resp.status_code == 200
        body = resp.json()
        assert "quality" in body
        assert "classification" in body
        assert "gated" in body

    def test_classify_gated_rejects_bad_scan(self, qclient):
        """All-white scan must be rejected before classification."""
        payload = {
            "quality_pixels": self._white_pixels(),
            "features": self._sample_features(),
        }
        resp = qclient.post("/classify/gated", json=payload)
        assert resp.status_code == 200
        body = resp.json()
        assert body["gated"] is True
        assert body["classification"] is None

    def test_classify_gated_accepted_has_doc_type(self, qclient):
        """When quality passes, classification result must include doc_type."""
        payload = {
            "quality_pixels": self._mid_pixels(seed=9),
            "features": self._sample_features(),
        }
        resp = qclient.post("/classify/gated", json=payload)
        body = resp.json()
        if not body["gated"]:  # quality passed
            assert "doc_type" in body["classification"]


# ===========================================================================
# TestLayoutSegmentation
# ===========================================================================


class TestLayoutSegmentation:
    """Unit tests for document layout segmentation (projection profile analysis)."""

    def _make_doc(self, height: int = 100, width: int = 80) -> np.ndarray:
        """Create a blank white document canvas."""
        return np.full((height, width), 255, dtype=np.uint8)

    def test_blank_returns_no_regions(self):
        pixels = self._make_doc()
        result = segment_layout(pixels)
        assert result.n_text_zones == 0
        assert result.regions == []
        assert not result.has_header
        assert not result.has_footer

    def test_single_block_is_body(self):
        pixels = self._make_doc()
        pixels[40:60, 10:70] = 50  # single block in middle
        result = segment_layout(pixels)
        assert result.n_text_zones == 1
        assert result.regions[0].region_type == RegionType.BODY

    def test_header_detected(self):
        pixels = self._make_doc()
        pixels[5:15, 5:75] = 50  # top zone — header candidate
        pixels[50:80, 5:75] = 50  # middle zone — body
        result = segment_layout(pixels)
        types = [r.region_type for r in result.regions]
        assert RegionType.HEADER in types

    def test_footer_detected(self):
        pixels = self._make_doc()
        pixels[10:40, 5:75] = 50  # body zone
        pixels[85:95, 5:75] = 50  # bottom zone — footer candidate
        result = segment_layout(pixels)
        types = [r.region_type for r in result.regions]
        assert RegionType.FOOTER in types

    def test_three_zone_has_header_and_footer(self):
        pixels = self._make_doc(height=120)
        pixels[5:20, :] = 50  # header (ends at 17% of 120)
        pixels[40:80, :] = 50  # body
        pixels[100:115, :] = 50  # footer (starts at 83% of 120)
        result = segment_layout(pixels)
        assert result.has_header
        assert result.has_footer
        assert result.n_text_zones >= 2

    def test_projection_length_matches_height(self):
        pixels = self._make_doc(height=50)
        pixels[10:20, :] = 50
        result = segment_layout(pixels)
        assert len(result.h_projection) == 50

    def test_ink_density_in_range(self):
        pixels = np.zeros((50, 40), dtype=np.uint8)  # all black
        result = segment_layout(pixels)
        for r in result.regions:
            assert 0.0 <= r.ink_density <= 1.0

    def test_region_bounds_within_image(self):
        pixels = self._make_doc()
        pixels[10:30, :] = 50
        pixels[60:80, :] = 50
        result = segment_layout(pixels)
        h, w = pixels.shape
        for r in result.regions:
            assert 0 <= r.row_start < r.row_end <= h
            assert 0 <= r.col_start < r.col_end <= w

    def test_two_column_detection(self):
        pixels = self._make_doc(height=100, width=100)
        pixels[20:80, 5:40] = 50  # left column
        pixels[20:80, 60:95] = 50  # right column (gap at cols 40–60)
        result = segment_layout(pixels)
        assert result.is_two_column

    def test_single_column_not_two_column(self):
        pixels = self._make_doc(height=100, width=80)
        pixels[20:80, 5:75] = 50  # full-width block
        result = segment_layout(pixels)
        assert not result.is_two_column

    def test_compute_horizontal_projection_shape(self):
        gray = np.full((50, 40), 200.0)
        proj = compute_horizontal_projection(gray)
        assert proj.shape == (50,)

    def test_compute_vertical_projection_shape(self):
        gray = np.full((50, 40), 200.0)
        proj = compute_vertical_projection(gray)
        assert proj.shape == (40,)

    def test_projection_values_in_range(self):
        rng = np.random.default_rng(0)
        gray = rng.uniform(0, 255, (50, 40))
        proj = compute_horizontal_projection(gray)
        assert float(proj.min()) >= 0.0
        assert float(proj.max()) <= 1.0

    def test_tiny_image_returns_empty_result(self):
        pixels = np.array([[0, 255], [255, 0]], dtype=np.uint8)
        result = segment_layout(pixels)
        assert isinstance(result, LayoutResult)
        assert result.regions == []

    def test_find_gaps_all_text(self):
        projection = np.ones(20) * 0.5
        gaps = find_gaps(projection, gap_threshold=0.1)
        assert gaps == []

    def test_find_gaps_all_blank(self):
        projection = np.zeros(20)
        gaps = find_gaps(projection, gap_threshold=0.02)
        assert len(gaps) == 1
        assert gaps[0] == (0, 20)

    def test_find_text_zones_two_zones(self):
        proj = np.array([0.5] * 10 + [0.0] * 5 + [0.5] * 10, dtype=float)
        zones = find_text_zones(proj, gap_threshold=0.02, min_rows=3)
        assert len(zones) == 2
        assert zones[0] == (0, 10)
        assert zones[1] == (15, 25)


# ===========================================================================
# TestLayoutAPIEndpoints
# ===========================================================================


class TestLayoutAPIEndpoints:
    """Integration tests for POST /layout/segment."""

    @pytest.fixture(autouse=True)
    def _client(self, api_client):
        self.client = api_client

    def _pixels(self, height: int = 100, width: int = 80, *, fill: int = 255) -> list[list[int]]:
        return np.full((height, width), fill, dtype=np.uint8).tolist()

    def _with_blocks(self, *row_ranges: tuple[int, int]) -> list[list[int]]:
        """Create a 100×80 document with dark blocks at specified row ranges."""
        arr = np.full((100, 80), 255, dtype=np.uint8)
        for r0, r1 in row_ranges:
            arr[r0:r1, :] = 50
        return arr.tolist()

    def test_segment_returns_200(self):
        pixels = self._with_blocks((10, 25), (50, 80))
        resp = self.client.post("/layout/segment", json={"pixels": pixels})
        assert resp.status_code == 200

    def test_segment_response_structure(self):
        resp = self.client.post("/layout/segment", json={"pixels": self._with_blocks((10, 30))})
        data = resp.json()
        for key in ("regions", "n_text_zones", "has_header", "has_footer", "is_two_column"):
            assert key in data

    def test_blank_document_zero_zones(self):
        resp = self.client.post("/layout/segment", json={"pixels": self._pixels()})
        assert resp.json()["n_text_zones"] == 0

    def test_empty_matrix_422(self):
        resp = self.client.post("/layout/segment", json={"pixels": []})
        assert resp.status_code == 422

    def test_region_type_is_valid_string(self):
        pixels = self._with_blocks((10, 30), (60, 80))
        resp = self.client.post("/layout/segment", json={"pixels": pixels})
        valid = {"header", "body", "footer", "margin", "blank"}
        for r in resp.json()["regions"]:
            assert r["region_type"] in valid

    def test_regions_have_required_fields(self):
        resp = self.client.post("/layout/segment", json={"pixels": self._with_blocks((10, 30))})
        required = (
            "region_type",
            "row_start",
            "row_end",
            "col_start",
            "col_end",
            "height",
            "width",
            "ink_density",
        )
        for r in resp.json()["regions"]:
            for key in required:
                assert key in r

    def test_n_text_zones_matches_regions(self):
        pixels = self._with_blocks((10, 25), (50, 75))
        resp = self.client.post("/layout/segment", json={"pixels": pixels})
        data = resp.json()
        assert data["n_text_zones"] == len(data["regions"])
