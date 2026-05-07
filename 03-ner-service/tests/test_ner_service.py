"""Tests for NER Service."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from ner.data.collection5 import (
    compute_dataset_stats,
    get_collection5_sample,
    load_collection5,
    parse_conll,
    sentences_to_bio,
)
from ner.data.dataset import (
    ID2LABEL,
    LABEL2ID,
    NER_LABELS,
    get_sample_data,
)
from ner.model.batch import BatchResult, process_collection5, process_texts
from ner.model.ner import extract_entities_from_bio, predict, predict_rule_based


class TestDataset:
    def test_labels_consistency(self):
        for i, label in enumerate(NER_LABELS):
            assert LABEL2ID[label] == i
            assert ID2LABEL[i] == label

    def test_sample_data_format(self):
        data = get_sample_data()
        assert len(data) >= 3
        for item in data:
            assert "tokens" in item
            assert "labels" in item
            assert len(item["tokens"]) == len(item["labels"])

    def test_sample_data_valid_labels(self):
        data = get_sample_data()
        for item in data:
            for label in item["labels"]:
                assert label in NER_LABELS


class TestBIOExtraction:
    def test_extract_single_entity(self):
        tokens = ["Владимир", "Путин", "работает"]
        labels = ["B-PER", "I-PER", "O"]
        entities = extract_entities_from_bio(tokens, labels)
        assert len(entities) == 1
        assert entities[0].text == "Владимир Путин"
        assert entities[0].label == "PER"

    def test_extract_multiple_entities(self):
        tokens = ["Яндекс", "в", "Москве"]
        labels = ["B-ORG", "O", "B-LOC"]
        entities = extract_entities_from_bio(tokens, labels)
        assert len(entities) == 2
        assert entities[0].label == "ORG"
        assert entities[1].label == "LOC"

    def test_extract_no_entities(self):
        tokens = ["Это", "обычный", "текст"]
        labels = ["O", "O", "O"]
        entities = extract_entities_from_bio(tokens, labels)
        assert len(entities) == 0

    def test_extract_from_sample_data(self):
        data = get_sample_data()
        for item in data:
            entities = extract_entities_from_bio(item["tokens"], item["labels"])
            assert len(entities) > 0


class TestRuleBasedNER:
    def test_detect_person(self):
        entities = predict_rule_based("Владимир Путин посетил город.")
        per_entities = [e for e in entities if e.label == "PER"]
        assert len(per_entities) >= 1
        assert "Путин" in per_entities[0].text

    def test_detect_organization(self):
        entities = predict_rule_based("Газпром объявил результаты за квартал.")
        org_entities = [e for e in entities if e.label == "ORG"]
        assert len(org_entities) >= 1

    def test_detect_location(self):
        entities = predict_rule_based("Встреча прошла в Москве.")
        loc_entities = [e for e in entities if e.label == "LOC"]
        assert len(loc_entities) >= 1
        assert "Москв" in loc_entities[0].text

    def test_detect_multiple_types(self):
        text = "Яндекс открыл офис в Санкт-Петербурге."
        entities = predict(text)
        labels = {e.label for e in entities}
        assert "ORG" in labels
        assert "LOC" in labels

    def test_empty_text(self):
        entities = predict("")
        assert len(entities) == 0

    def test_no_entities(self):
        entities = predict("Просто обычный текст без сущностей.")
        # May or may not find entities depending on patterns
        assert isinstance(entities, list)

    def test_entity_positions(self):
        text = "Москва — столица России."
        entities = predict(text)
        for entity in entities:
            assert entity.start >= 0
            assert entity.end > entity.start


class TestAPI:
    def test_health(self):
        from fastapi.testclient import TestClient
        from ner.api.app import app

        client = TestClient(app)
        assert client.get("/health").status_code == 200

    def test_predict_endpoint(self):
        from fastapi.testclient import TestClient
        from ner.api.app import app

        client = TestClient(app)
        resp = client.post("/predict", json={"text": "Газпром находится в Москве."})
        assert resp.status_code == 200
        data = resp.json()
        assert "entities" in data
        assert "text" in data

    def test_predict_batch(self):
        from fastapi.testclient import TestClient
        from ner.api.app import app

        client = TestClient(app)
        resp = client.post(
            "/predict/batch",
            json=[
                {"text": "Яндекс в Санкт-Петербурге."},
                {"text": "Обычный текст."},
            ],
        )
        assert resp.status_code == 200
        assert len(resp.json()) == 2

    def test_predict_empty_rejects(self):
        from fastapi.testclient import TestClient
        from ner.api.app import app

        client = TestClient(app)
        resp = client.post("/predict", json={"text": ""})
        assert resp.status_code == 422  # min_length=1 validation


class TestCollection5Parser:
    """Тесты CoNLL-парсера и загрузчика Collection5."""

    def test_parse_simple_conll(self):
        """Простое предложение с одной сущностью."""
        text = "Путин\tB-PER\nпосетил\tO\nМоскву\tB-LOC\n"
        sentences = parse_conll(text)
        assert len(sentences) == 1
        assert sentences[0] == [("Путин", "B-PER"), ("посетил", "O"), ("Москву", "B-LOC")]

    def test_parse_multiple_sentences(self):
        """Пустая строка разделяет предложения."""
        text = "Яндекс\tB-ORG\n\nМосква\tB-LOC\n"
        sentences = parse_conll(text)
        assert len(sentences) == 2

    def test_parse_empty_string(self):
        """Пустой текст возвращает пустой датасет."""
        assert parse_conll("") == []

    def test_parse_custom_separator(self):
        """Поддержка пробела как разделителя."""
        text = "Путин B-PER\n"
        sentences = parse_conll(text, sep=" ")
        assert sentences[0][0] == ("Путин", "B-PER")

    def test_load_collection5_sample(self):
        """Встроенный образец загружается корректно."""
        dataset = load_collection5()
        assert len(dataset) >= 5  # не менее 5 предложений в образце

    def test_get_collection5_sample_consistent(self):
        """get_collection5_sample и load_collection5() возвращают одно."""
        assert get_collection5_sample() == load_collection5()

    def test_sentences_to_bio(self):
        """Разделение на tokens/labels работает корректно."""
        dataset = [
            [("Путин", "B-PER"), ("посетил", "O")],
            [("Яндекс", "B-ORG")],
        ]
        tokens, labels = sentences_to_bio(dataset)
        assert tokens == [["Путин", "посетил"], ["Яндекс"]]
        assert labels == [["B-PER", "O"], ["B-ORG"]]

    def test_sample_has_all_entity_types(self):
        """Образец содержит PER, ORG и LOC."""
        dataset = get_collection5_sample()
        all_labels = {lbl for sent in dataset for _, lbl in sent}
        assert "B-PER" in all_labels
        assert "B-ORG" in all_labels
        assert "B-LOC" in all_labels


class TestCollection5Stats:
    """Тесты статистики датасета."""

    def test_stats_counts_sentences(self):
        stats = compute_dataset_stats(get_collection5_sample())
        assert stats.num_sentences >= 5

    def test_stats_counts_tokens(self):
        stats = compute_dataset_stats(get_collection5_sample())
        assert stats.num_tokens >= 30

    def test_stats_entity_counts(self):
        stats = compute_dataset_stats(get_collection5_sample())
        # Образец содержит все три типа сущностей
        assert "PER" in stats.entity_counts
        assert "ORG" in stats.entity_counts
        assert "LOC" in stats.entity_counts

    def test_stats_entity_counts_positive(self):
        stats = compute_dataset_stats(get_collection5_sample())
        for count in stats.entity_counts.values():
            assert count > 0

    def test_stats_empty_dataset(self):
        stats = compute_dataset_stats([])
        assert stats.num_sentences == 0
        assert stats.num_tokens == 0
        assert stats.entity_counts == {}

    def test_stats_repr(self):
        stats = compute_dataset_stats(get_collection5_sample())
        r = repr(stats)
        assert "Sentences" in r
        assert "Tokens" in r


class TestBatchProcessing:
    """Тесты batch processing pipeline."""

    def test_process_texts_returns_batch_result(self):
        texts = ["Газпром в Москве.", "Яндекс — российская компания."]
        result = process_texts(texts)
        assert isinstance(result, BatchResult)
        assert result.total_texts == 2

    def test_process_texts_all_items_present(self):
        texts = ["Текст один.", "Текст два.", "Текст три."]
        result = process_texts(texts)
        assert len(result.items) == 3

    def test_process_empty_list(self):
        result = process_texts([])
        assert result.total_texts == 0
        assert result.total_entities == 0

    def test_process_texts_entity_counts(self):
        texts = ["Газпром в Москве.", "Яндекс в Санкт-Петербурге."]
        result = process_texts(texts)
        # Итоговый счётчик агрегирует по типам
        assert isinstance(result.entity_type_counts, dict)

    def test_batch_item_has_entities_flag(self):
        texts = ["Газпром в Москве."]
        result = process_texts(texts)
        item = result.items[0]
        # has_entities зависит от rule-based NER
        assert isinstance(item.has_entities, bool)

    def test_batch_item_entity_types(self):
        texts = ["Газпром в Москве."]
        result = process_texts(texts)
        item = result.items[0]
        assert isinstance(item.entity_types, set)

    def test_process_collection5(self):
        """Batch-обработка Collection5 датасета."""
        dataset = get_collection5_sample()
        result = process_collection5(dataset)
        assert result.total_texts == len(dataset)
        assert result.total_entities > 0

    def test_process_collection5_no_errors(self):
        """Все предложения обрабатываются без ошибок."""
        dataset = get_collection5_sample()
        result = process_collection5(dataset)
        errors = [item for item in result.items if item.error is not None]
        assert len(errors) == 0

    def test_chunk_size_does_not_affect_result(self):
        """chunk_size влияет только на память, не на результат."""
        texts = [f"Текст номер {i} о Газпроме в Москве." for i in range(10)]
        result_small = process_texts(texts, chunk_size=2)
        result_large = process_texts(texts, chunk_size=100)
        assert result_small.total_texts == result_large.total_texts
        assert result_small.total_entities == result_large.total_entities


# ── Conformal Prediction тесты ───────────────────────────────────────────────


class TestConformalNERPredictor:
    """Unit-тесты ConformalNERPredictor — nonconformity scores и калибровка."""

    from ner.model.conformal import ConformalConfig, ConformalNERPredictor
    from ner.model.ner import Entity

    def _make_entity(self, text: str, label: str) -> "Entity":
        from ner.model.ner import Entity

        return Entity(text=text, label=label, start=0, end=len(text))

    def test_pattern_score_perfect_match_per(self):
        """Полное совпадение с PER-паттерном → score = 1.0."""
        from ner.model.conformal import ConformalNERPredictor

        pred = ConformalNERPredictor()
        score = pred._pattern_score("Иван Петров", "PER")
        assert score == 1.0

    def test_pattern_score_no_match(self):
        """Текст не совпадает ни с одним паттерном → score = 0.0."""
        from ner.model.conformal import ConformalNERPredictor

        pred = ConformalNERPredictor()
        assert pred._pattern_score("просто текст", "PER") == 0.0

    def test_nonconformity_score_range(self):
        """Nonconformity score ∈ [0, 1]."""
        from ner.model.conformal import ConformalNERPredictor

        pred = ConformalNERPredictor()
        for label in ["PER", "ORG", "LOC"]:
            score = pred._nonconformity_score("Газпром нефть", label)
            assert 0.0 <= score <= 1.0

    def test_nonconformity_lower_for_matching_label(self):
        """Совпадающий label имеет более низкий nonconformity score."""
        from ner.model.conformal import ConformalNERPredictor

        pred = ConformalNERPredictor()
        # "Газпром" матчит ORG, поэтому score(ORG) < score(PER)
        score_org = pred._nonconformity_score("Газпром", "ORG")
        score_per = pred._nonconformity_score("Газпром", "PER")
        assert score_org < score_per

    def test_nonconformity_unknown_text_uniform(self):
        """Текст без паттернов → равномерное распределение неопределённости."""
        from ner.model.conformal import ConformalNERPredictor

        pred = ConformalNERPredictor()
        score = pred._nonconformity_score("абракадабра", "PER")
        # Равномерно: 1 - 1/3 ≈ 0.667
        assert abs(score - (1.0 - 1.0 / 3)) < 1e-6

    def test_calibrate_returns_result(self):
        """calibrate() возвращает CalibrationResult с ожидаемыми полями."""
        from ner.model.conformal import ConformalNERPredictor

        pred = ConformalNERPredictor()
        entities = [
            self._make_entity("Иван Петров", "PER"),
            self._make_entity("Газпром", "ORG"),
            self._make_entity("Москва", "LOC"),
        ] * 5  # 15 сущностей >= min_calibration_samples=10
        result = pred.calibrate(entities)
        assert 0.0 <= result.q_hat <= 1.0
        assert result.n_calibration == 15
        assert result.alpha == 0.1

    def test_calibrate_too_few_samples_uses_conservative(self):
        """Меньше min_calibration_samples → q_hat = 1.0 (включить всё)."""
        from ner.model.conformal import ConformalNERPredictor

        pred = ConformalNERPredictor()
        entities = [self._make_entity("Иван Петров", "PER")] * 5
        result = pred.calibrate(entities)
        assert result.q_hat == 1.0

    def test_calibrate_empirical_coverage_bounded(self):
        """Эмпирическое покрытие на калибровочных данных ≥ 1-α."""
        from ner.model.conformal import ConformalNERPredictor

        pred = ConformalNERPredictor()
        entities = [
            self._make_entity("Иван Петров", "PER"),
            self._make_entity("Газпром", "ORG"),
            self._make_entity("Москва", "LOC"),
        ] * 5
        result = pred.calibrate(entities)
        assert result.coverage_empirical >= 1.0 - result.alpha - 1e-6

    def test_predict_set_contains_predicted_label(self):
        """prediction_set всегда содержит предсказанный label."""
        from ner.model.conformal import ConformalNERPredictor
        from ner.model.ner import predict

        pred = ConformalNERPredictor()
        entities = predict("Иван Петров из Газпрома приехал в Москву.")
        for entity in entities:
            result = pred.predict_set(entity)
            assert result.label in result.prediction_set

    def test_predict_set_subset_of_all_labels(self):
        """prediction_set ⊆ {PER, ORG, LOC}."""
        from ner.model.conformal import ALL_LABELS, ConformalNERPredictor
        from ner.model.ner import predict

        pred = ConformalNERPredictor()
        for entity in predict("Яндекс открыл офис в Санкт-Петербурге."):
            result = pred.predict_set(entity)
            assert all(lb in ALL_LABELS for lb in result.prediction_set)

    def test_predict_set_is_certain_flag(self):
        """is_certain=True ↔ len(prediction_set)==1."""
        from ner.model.conformal import ConformalNERPredictor
        from ner.model.ner import predict

        pred = ConformalNERPredictor()
        for entity in predict("Газпром в Москве."):
            result = pred.predict_set(entity)
            assert result.is_certain == (len(result.prediction_set) == 1)

    def test_predict_set_coverage_field(self):
        """coverage = 1 - alpha."""
        from ner.model.conformal import ConformalConfig, ConformalNERPredictor
        from ner.model.ner import predict

        pred = ConformalNERPredictor(ConformalConfig(alpha=0.05))
        entities = predict("Газпром в Москве.")
        if entities:
            result = pred.predict_set(entities[0])
            assert abs(result.coverage - 0.95) < 1e-6

    def test_predict_text_returns_list(self):
        """predict_text возвращает список ConformalEntityResult."""
        from ner.model.conformal import ConformalEntityResult, ConformalNERPredictor

        pred = ConformalNERPredictor()
        results = pred.predict_text("Яндекс в Москве.")
        assert isinstance(results, list)
        for r in results:
            assert isinstance(r, ConformalEntityResult)

    def test_auto_calibration_on_collection5(self):
        """Калибровка на Collection5 устанавливает разумный q_hat."""
        from ner.data.collection5 import get_collection5_sample
        from ner.model.conformal import ConformalNERPredictor
        from ner.model.ner import extract_entities_from_bio

        pred = ConformalNERPredictor()
        dataset = get_collection5_sample()
        entities = []
        for sent in dataset:
            tokens = [tok for tok, _ in sent]
            labels_bio = [lbl for _, lbl in sent]
            entities.extend(extract_entities_from_bio(tokens, labels_bio))
        if entities:
            result = pred.calibrate(entities)
            assert pred._calibrated is True
            assert 0.0 <= result.q_hat <= 1.0


class TestConformalAPI:
    """Интеграционные тесты /predict/conformal endpoint."""

    def test_conformal_endpoint_status_200(self):
        from fastapi.testclient import TestClient
        from ner.api.app import app

        client = TestClient(app)
        resp = client.post("/predict/conformal", json={"text": "Газпром в Москве."})
        assert resp.status_code == 200

    def test_conformal_endpoint_structure(self):
        from fastapi.testclient import TestClient
        from ner.api.app import app

        client = TestClient(app)
        resp = client.post("/predict/conformal", json={"text": "Яндекс в Санкт-Петербурге."})
        data = resp.json()
        assert "entities" in data
        assert "q_hat" in data
        assert "calibrated" in data

    def test_conformal_entity_has_required_fields(self):
        from fastapi.testclient import TestClient
        from ner.api.app import app

        client = TestClient(app)
        resp = client.post("/predict/conformal", json={"text": "Газпром в Москве."})
        entities = resp.json()["entities"]
        assert len(entities) > 0
        for e in entities:
            assert "prediction_set" in e
            assert "nonconformity_score" in e
            assert "is_certain" in e
            assert "coverage" in e

    def test_conformal_prediction_set_contains_label(self):
        from fastapi.testclient import TestClient
        from ner.api.app import app

        client = TestClient(app)
        resp = client.post("/predict/conformal", json={"text": "Яндекс в Санкт-Петербурге."})
        for e in resp.json()["entities"]:
            assert e["label"] in e["prediction_set"]

    def test_conformal_calibrated_true(self):
        """После авто-калибровки при старте calibrated=True."""
        from fastapi.testclient import TestClient
        from ner.api.app import app

        client = TestClient(app)
        data = client.post("/predict/conformal", json={"text": "Газпром."}).json()
        assert data["calibrated"] is True

    def test_health_includes_conformal_status(self):
        from fastapi.testclient import TestClient
        from ner.api.app import app

        client = TestClient(app)
        resp = client.get("/health")
        assert "conformal_calibrated" in resp.json()
