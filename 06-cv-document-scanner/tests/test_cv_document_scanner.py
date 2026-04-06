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
        """GradCAM() должен поднимать RuntimeError при отсутствии PyTorch."""
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
