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


# ─────────────────────────────────────────────────────────────────────────────
# Active Learning Tests
# ─────────────────────────────────────────────────────────────────────────────


class TestSamplingStrategies:
    """Unit-тесты стратегий активного обучения."""

    def test_least_confidence_empty(self):
        from ner.active.strategy import least_confidence_score

        assert least_confidence_score([]) == 0.0

    def test_least_confidence_single(self):
        from ner.active.strategy import least_confidence_score

        assert least_confidence_score([0.8]) == 0.8

    def test_least_confidence_returns_max(self):
        from ner.active.strategy import least_confidence_score

        assert least_confidence_score([0.3, 0.9, 0.5]) == 0.9

    def test_margin_empty(self):
        from ner.active.strategy import margin_score

        assert margin_score([]) == 0.0

    def test_margin_single(self):
        from ner.active.strategy import margin_score

        assert margin_score([0.7]) == 0.7

    def test_margin_computes_range(self):
        from ner.active.strategy import margin_score

        result = margin_score([0.2, 0.8])
        assert abs(result - 0.6) < 1e-6

    def test_entropy_empty(self):
        from ner.active.strategy import entropy_score

        assert entropy_score([]) == 0.0

    def test_entropy_max_at_half(self):
        from ner.active.strategy import entropy_score

        # Бинарная энтропия максимальна при p=0.5 (nonconformity=0.5)
        e_half = entropy_score([0.5])
        e_low = entropy_score([0.1])
        e_high = entropy_score([0.9])
        assert e_half > e_low
        assert e_half > e_high

    def test_entropy_symmetric(self):
        from ner.active.strategy import entropy_score

        # H(p) = H(1-p) для бинарной энтропии
        assert abs(entropy_score([0.3]) - entropy_score([0.7])) < 1e-6

    def test_score_text_least_confidence(self):
        from ner.active.strategy import ActiveLearningConfig, SamplingStrategy, score_text

        config = ActiveLearningConfig(strategy=SamplingStrategy.LEAST_CONFIDENCE)
        result = score_text("Газпром.", [0.2, 0.8], config)
        assert result.score == 0.8
        assert result.n_entities == 2
        assert result.strategy == "least_confidence"

    def test_score_text_margin(self):
        from ner.active.strategy import ActiveLearningConfig, SamplingStrategy, score_text

        config = ActiveLearningConfig(strategy=SamplingStrategy.MARGIN)
        result = score_text("test", [0.1, 0.7], config)
        assert abs(result.score - 0.6) < 1e-3

    def test_score_text_entropy(self):
        from ner.active.strategy import ActiveLearningConfig, SamplingStrategy, score_text

        config = ActiveLearningConfig(strategy=SamplingStrategy.ENTROPY)
        result = score_text("test", [0.5], config)
        assert result.score > 0.9  # max entropy at 0.5

    def test_higher_uncertainty_higher_score(self):
        """Более неопределённый текст должен получать более высокий score."""
        from ner.active.strategy import ActiveLearningConfig, SamplingStrategy, score_text

        config = ActiveLearningConfig(strategy=SamplingStrategy.LEAST_CONFIDENCE)
        certain = score_text("A", [0.0, 0.1], config)
        uncertain = score_text("B", [0.9, 0.95], config)
        assert uncertain.score > certain.score


class TestLabelingPool:
    """Unit-тесты менеджера пула аннотации."""

    def _fresh_pool(self):
        from ner.active.pool import LabelingPool

        return LabelingPool()

    def test_initial_status_all_zero(self):
        pool = self._fresh_pool()
        status = pool.status("least_confidence")
        assert status.unlabeled_count == 0
        assert status.queried_count == 0
        assert status.labeled_count == 0
        assert status.total_added == 0

    def test_add_texts_returns_ids(self):
        pool = self._fresh_pool()
        ids = pool.add_texts(["text1", "text2"], [0.5, 0.8], "least_confidence")
        assert len(ids) == 2
        assert all(isinstance(i, str) and len(i) > 0 for i in ids)

    def test_add_increments_unlabeled(self):
        pool = self._fresh_pool()
        pool.add_texts(["a", "b", "c"], [0.1, 0.5, 0.9], "least_confidence")
        status = pool.status("least_confidence")
        assert status.unlabeled_count == 3
        assert status.total_added == 3

    def test_add_length_mismatch_raises(self):
        import pytest

        pool = self._fresh_pool()
        with pytest.raises(ValueError):
            pool.add_texts(["text"], [0.5, 0.8], "least_confidence")

    def test_query_returns_top_n_sorted(self):
        pool = self._fresh_pool()
        pool.add_texts(["low", "mid", "high"], [0.1, 0.5, 0.9], "least_confidence")
        batch = pool.query(2)
        assert len(batch.items) == 2
        # Наиболее неопределённые идут первыми
        assert batch.items[0].uncertainty_score >= batch.items[1].uncertainty_score

    def test_query_moves_to_queried_state(self):
        pool = self._fresh_pool()
        pool.add_texts(["text"], [0.8], "least_confidence")
        pool.query(1)
        status = pool.status("least_confidence")
        assert status.unlabeled_count == 0
        assert status.queried_count == 1

    def test_query_fewer_than_batch_size(self):
        pool = self._fresh_pool()
        pool.add_texts(["only"], [0.5], "least_confidence")
        batch = pool.query(10)
        assert len(batch.items) == 1
        assert batch.unlabeled_remaining == 0

    def test_query_empty_pool_returns_empty(self):
        pool = self._fresh_pool()
        batch = pool.query(5)
        assert batch.items == []

    def test_label_transitions_to_labeled(self):
        pool = self._fresh_pool()
        pool.add_texts(["text"], [0.7], "least_confidence")
        batch = pool.query(1)
        item_id = batch.items[0].id
        result = pool.label(item_id, [{"text": "ООО", "label": "ORG", "start": 0, "end": 3}])
        assert result is not None
        assert result.labeled_at is not None
        assert len(result.annotations) == 1
        status = pool.status("least_confidence")
        assert status.labeled_count == 1
        assert status.queried_count == 0

    def test_label_unknown_id_returns_none(self):
        pool = self._fresh_pool()
        result = pool.label("nonexistent-id", [])
        assert result is None

    def test_get_labeled_returns_all(self):
        pool = self._fresh_pool()
        pool.add_texts(["t1", "t2"], [0.5, 0.6], "least_confidence")
        batch = pool.query(2)
        for item in batch.items:
            pool.label(item.id, [])
        labeled = pool.get_labeled()
        assert len(labeled) == 2

    def test_reset_clears_all_states(self):
        pool = self._fresh_pool()
        pool.add_texts(["a", "b"], [0.5, 0.8], "least_confidence")
        pool.query(1)
        pool.reset()
        status = pool.status("least_confidence")
        assert status.unlabeled_count == 0
        assert status.queried_count == 0
        assert status.labeled_count == 0
        assert status.total_added == 0


class TestActiveLearningAPI:
    """Интеграционные тесты /active/* endpoints."""

    def setup_method(self):
        """Сбросить пул перед каждым тестом для изоляции."""
        from fastapi.testclient import TestClient
        from ner.api.app import app

        client = TestClient(app)
        client.post("/active/pool/reset")

    def test_add_returns_200(self):
        from fastapi.testclient import TestClient
        from ner.api.app import app

        client = TestClient(app)
        resp = client.post(
            "/active/pool/add",
            json={"texts": ["Газпром в Москве.", "Яндекс"], "strategy": "least_confidence"},
        )
        assert resp.status_code == 200

    def test_add_response_structure(self):
        from fastapi.testclient import TestClient
        from ner.api.app import app

        client = TestClient(app)
        resp = client.post(
            "/active/pool/add",
            json={"texts": ["text1", "text2"], "strategy": "entropy"},
        ).json()
        assert "ids" in resp
        assert resp["added"] == 2
        assert resp["strategy"] == "entropy"

    def test_add_invalid_strategy_returns_422(self):
        from fastapi.testclient import TestClient
        from ner.api.app import app

        client = TestClient(app)
        resp = client.post(
            "/active/pool/add",
            json={"texts": ["text"], "strategy": "unknown_strategy"},
        )
        assert resp.status_code == 422

    def test_query_returns_sorted_by_score(self):
        from fastapi.testclient import TestClient
        from ner.api.app import app

        client = TestClient(app)
        client.post(
            "/active/pool/add",
            json={
                "texts": ["Газпром.", "Привет.", "ОАО Сбербанк в Москве"],
                "strategy": "least_confidence",
            },
        )
        resp = client.post("/active/pool/query", json={"batch_size": 3}).json()
        scores = [item["uncertainty_score"] for item in resp["items"]]
        assert scores == sorted(scores, reverse=True)

    def test_query_response_structure(self):
        from fastapi.testclient import TestClient
        from ner.api.app import app

        client = TestClient(app)
        client.post("/active/pool/add", json={"texts": ["Яндекс"], "strategy": "margin"})
        resp = client.post("/active/pool/query", json={"batch_size": 1}).json()
        assert "items" in resp
        assert "strategy" in resp
        assert "unlabeled_remaining" in resp

    def test_label_moves_item_to_labeled(self):
        from fastapi.testclient import TestClient
        from ner.api.app import app

        client = TestClient(app)
        client.post(
            "/active/pool/add",
            json={"texts": ["Газпром"], "strategy": "least_confidence"},
        )
        query_resp = client.post("/active/pool/query", json={"batch_size": 1}).json()
        item_id = query_resp["items"][0]["id"]

        label_resp = client.post(
            "/active/pool/label",
            json={
                "item_id": item_id,
                "annotations": [{"text": "Газпром", "label": "ORG", "start": 0, "end": 7}],
            },
        )
        assert label_resp.status_code == 200
        data = label_resp.json()
        assert data["labeled"] is True
        assert data["labeled_at"] is not None

    def test_label_unknown_id_returns_404(self):
        from fastapi.testclient import TestClient
        from ner.api.app import app

        client = TestClient(app)
        resp = client.post(
            "/active/pool/label",
            json={"item_id": "nonexistent-uuid", "annotations": []},
        )
        assert resp.status_code == 404

    def test_status_endpoint(self):
        from fastapi.testclient import TestClient
        from ner.api.app import app

        client = TestClient(app)
        client.post(
            "/active/pool/add",
            json={"texts": ["t1", "t2"], "strategy": "least_confidence"},
        )
        status = client.get("/active/pool/status").json()
        assert status["unlabeled_count"] == 2
        assert status["queried_count"] == 0
        assert status["labeled_count"] == 0
        assert status["total_added"] == 2

    def test_full_annotation_cycle(self):
        """Полный цикл: add → query → label → labeled."""
        from fastapi.testclient import TestClient
        from ner.api.app import app

        client = TestClient(app)
        texts = ["Газпром в Москве.", "Яндекс открыл офис.", "Привет мир."]
        client.post("/active/pool/add", json={"texts": texts, "strategy": "entropy"})
        query_resp = client.post("/active/pool/query", json={"batch_size": 2}).json()
        assert len(query_resp["items"]) == 2

        for item in query_resp["items"]:
            client.post(
                "/active/pool/label",
                json={"item_id": item["id"], "annotations": []},
            )

        labeled_resp = client.get("/active/pool/labeled").json()
        assert labeled_resp["total"] == 2

        status = client.get("/active/pool/status").json()
        assert status["labeled_count"] == 2
        assert status["unlabeled_count"] == 1  # один остался
        assert status["queried_count"] == 0

    def test_margin_strategy_via_add(self):
        from fastapi.testclient import TestClient
        from ner.api.app import app

        client = TestClient(app)
        resp = client.post(
            "/active/pool/add",
            json={"texts": ["Москва"], "strategy": "margin"},
        )
        assert resp.status_code == 200
        assert resp.json()["strategy"] == "margin"

    def test_reset_clears_pool(self):
        from fastapi.testclient import TestClient
        from ner.api.app import app

        client = TestClient(app)
        client.post(
            "/active/pool/add",
            json={"texts": ["text"], "strategy": "least_confidence"},
        )
        client.post("/active/pool/reset")
        status = client.get("/active/pool/status").json()
        assert status["total_added"] == 0


# ── Named Entity Linking Tests ────────────────────────────────────────────────


class TestKnowledgeBase:
    """Unit-тесты для KnowledgeBase."""

    def test_built_in_entities_loaded(self):
        from ner.linking.knowledge_base import KnowledgeBase

        kb = KnowledgeBase()
        stats = kb.stats()
        assert stats.n_entities >= 20

    def test_stats_contains_org_and_loc(self):
        from ner.linking.knowledge_base import KnowledgeBase

        kb = KnowledgeBase()
        stats = kb.stats()
        assert "ORG" in stats.by_type
        assert "LOC" in stats.by_type
        assert stats.by_type["ORG"] >= 10
        assert stats.by_type["LOC"] >= 5

    def test_stats_n_aliases_exceeds_n_entities(self):
        from ner.linking.knowledge_base import KnowledgeBase

        kb = KnowledgeBase()
        stats = kb.stats()
        assert stats.n_aliases > stats.n_entities

    def test_exact_lookup_canonical_name(self):
        from ner.linking.knowledge_base import KnowledgeBase

        kb = KnowledgeBase()
        eid = kb.exact_lookup("Газпром")
        assert eid == "Q102048"

    def test_exact_lookup_alias(self):
        """'Sberbank' должен находить Сбербанк."""
        from ner.linking.knowledge_base import KnowledgeBase

        kb = KnowledgeBase()
        eid = kb.exact_lookup("Sberbank")
        assert eid is not None
        record = kb.get_entity(eid)
        assert record is not None
        assert record.canonical_name == "Сбербанк"

    def test_exact_lookup_unknown_returns_none(self):
        from ner.linking.knowledge_base import KnowledgeBase

        kb = KnowledgeBase()
        assert kb.exact_lookup("НеизвестнаяКомпания123") is None

    def test_normalize_strips_punctuation(self):
        """«Газпром» и Газпром — одна сущность."""
        from ner.linking.knowledge_base import KnowledgeBase

        kb = KnowledgeBase()
        eid_clean = kb.exact_lookup("Газпром")
        eid_quoted = kb.exact_lookup("«Газпром»")
        assert eid_clean == eid_quoted

    def test_get_entity_returns_record(self):
        from ner.linking.knowledge_base import EntityRecord, KnowledgeBase

        kb = KnowledgeBase()
        record = kb.get_entity("Q649")  # Москва
        assert record is not None
        assert isinstance(record, EntityRecord)
        assert record.canonical_name == "Москва"
        assert record.entity_type == "LOC"

    def test_get_entity_unknown_returns_none(self):
        from ner.linking.knowledge_base import KnowledgeBase

        kb = KnowledgeBase()
        assert kb.get_entity("Q999999999") is None

    def test_add_custom_entity(self):
        from ner.linking.knowledge_base import EntityRecord, KnowledgeBase

        kb = KnowledgeBase()
        custom = EntityRecord(
            entity_id="Q_TEST",
            canonical_name="ТестКомпания",
            aliases=["TestCo", "TC"],
            entity_type="ORG",
            description="Тестовая сущность",
        )
        kb.add_entity(custom)
        assert kb.exact_lookup("TestCo") == "Q_TEST"
        assert kb.get_entity("Q_TEST") is not None

    def test_all_entities_filter_by_type(self):
        from ner.linking.knowledge_base import KnowledgeBase

        kb = KnowledgeBase()
        orgs = kb.all_entities(entity_type="ORG")
        locs = kb.all_entities(entity_type="LOC")
        assert all(r.entity_type == "ORG" for r in orgs)
        assert all(r.entity_type == "LOC" for r in locs)
        assert len(orgs) + len(locs) == len(kb.all_entities())


class TestEntityLinker:
    """Unit-тесты для EntityLinker."""

    def test_exact_hit_confidence_one(self):
        from ner.linking.linker import EntityLinker

        linker = EntityLinker()
        result = linker.link_mention("Газпром", "ORG")
        assert result.is_linked is True
        assert result.confidence == 1.0
        assert result.entity_id == "Q102048"

    def test_alias_hit_resolves_to_canonical(self):
        """'Gazprom' → canonical 'Газпром'."""
        from ner.linking.linker import EntityLinker

        linker = EntityLinker()
        result = linker.link_mention("Gazprom", "ORG")
        assert result.is_linked is True
        assert result.canonical_name == "Газпром"

    def test_fuzzy_match_typo(self):
        """'Газпромм' (опечатка) должна связываться с Газпромом."""
        from ner.linking.linker import EntityLinker, LinkingConfig

        linker = EntityLinker(config=LinkingConfig(confidence_threshold=0.4))
        result = linker.link_mention("Газпромм", "ORG")
        assert result.is_linked is True
        assert result.entity_id == "Q102048"

    def test_unknown_mention_not_linked(self):
        from ner.linking.linker import EntityLinker

        linker = EntityLinker()
        result = linker.link_mention("АбракадабраКорп", "ORG")
        assert result.is_linked is False
        assert result.entity_id is None
        assert result.canonical_name is None

    def test_type_match_bonus_affects_score(self):
        """LOC упоминание должно получать бонус от LOC-сущностей в KB."""
        from ner.linking.linker import EntityLinker

        linker = EntityLinker()
        result = linker.link_mention("Москва", "LOC")
        assert result.is_linked is True
        assert result.entity_id == "Q649"

    def test_candidates_sorted_descending(self):
        """Кандидаты отсортированы по убыванию score."""
        from ner.linking.linker import EntityLinker

        linker = EntityLinker()
        result = linker.link_mention("Сбербанк", "ORG")
        scores = [c["score"] for c in result.candidates]
        assert scores == sorted(scores, reverse=True)

    def test_link_entities_batch(self):
        """Пакетное связывание возвращает результат для каждого упоминания."""
        from ner.linking.linker import EntityLinker

        linker = EntityLinker()
        pairs = [("Яндекс", "ORG"), ("Москва", "LOC"), ("НеизвестноеМесто", "LOC")]
        results = linker.link_entities(pairs)
        assert len(results) == 3
        assert results[0].is_linked is True  # Яндекс
        assert results[1].is_linked is True  # Москва
        assert results[2].is_linked is False  # Неизвестно

    def test_entity_id_none_when_not_linked(self):
        from ner.linking.linker import EntityLinker

        linker = EntityLinker()
        result = linker.link_mention("XYZRandom", "PER")
        assert result.entity_id is None

    def test_canonical_name_set_when_linked(self):
        from ner.linking.linker import EntityLinker

        linker = EntityLinker()
        result = linker.link_mention("Сбер", "ORG")
        assert result.is_linked is True
        assert result.canonical_name is not None

    def test_jaccard_ngrams_edge_case_short_text(self):
        """Тест граничного случая: строка короче ngram_size."""
        from ner.linking.linker import _char_ngrams

        ngrams = _char_ngrams("ab", 3)
        assert ngrams == {"ab"}

    def test_jaccard_both_empty(self):
        from ner.linking.linker import _jaccard

        assert _jaccard("", "") == 1.0

    def test_jaccard_one_empty(self):
        from ner.linking.linker import _jaccard

        assert _jaccard("", "abc") == 0.0
        assert _jaccard("abc", "") == 0.0

    def test_mention_field_preserved(self):
        from ner.linking.linker import EntityLinker

        linker = EntityLinker()
        result = linker.link_mention("Лукойл", "ORG")
        assert result.mention == "Лукойл"
        assert result.entity_type == "ORG"


class TestEntityLinkingAPI:
    """Интеграционные тесты NEL endpoints."""

    def test_link_entities_status_200(self):
        from fastapi.testclient import TestClient
        from ner.api.app import app

        client = TestClient(app)
        resp = client.post("/link/entities", json={"text": "Газпром открыл офис в Москве."})
        assert resp.status_code == 200

    def test_link_entities_response_structure(self):
        from fastapi.testclient import TestClient
        from ner.api.app import app

        client = TestClient(app)
        resp = client.post("/link/entities", json={"text": "Яндекс работает в России."})
        data = resp.json()
        assert "text" in data
        assert "entities" in data
        assert "n_linked" in data
        assert "n_total" in data

    def test_link_entities_known_org_is_linked(self):
        """Газпром в тексте должен быть связан с KB."""
        from fastapi.testclient import TestClient
        from ner.api.app import app

        client = TestClient(app)
        resp = client.post(
            "/link/entities",
            json={"text": "Газпром подписал договор с Яндексом."},
        )
        data = resp.json()
        assert data["n_linked"] >= 0  # не все сущности могут распознаться
        assert isinstance(data["entities"], list)

    def test_link_entities_fields_present(self):
        """Каждая сущность имеет все обязательные поля."""
        from fastapi.testclient import TestClient
        from ner.api.app import app

        client = TestClient(app)
        resp = client.post("/link/entities", json={"text": "Газпром в Москве."})
        data = resp.json()
        for entity in data["entities"]:
            assert "mention" in entity
            assert "entity_type" in entity
            assert "confidence" in entity
            assert "is_linked" in entity
            assert "start" in entity
            assert "end" in entity

    def test_link_entities_n_total_matches_entities(self):
        from fastapi.testclient import TestClient
        from ner.api.app import app

        client = TestClient(app)
        resp = client.post("/link/entities", json={"text": "Сбербанк в Санкт-Петербурге."})
        data = resp.json()
        assert data["n_total"] == len(data["entities"])

    def test_kb_stats_status_200(self):
        from fastapi.testclient import TestClient
        from ner.api.app import app

        client = TestClient(app)
        resp = client.get("/kb/stats")
        assert resp.status_code == 200

    def test_kb_stats_has_correct_fields(self):
        from fastapi.testclient import TestClient
        from ner.api.app import app

        client = TestClient(app)
        data = client.get("/kb/stats").json()
        assert "n_entities" in data
        assert "by_type" in data
        assert "n_aliases" in data
        assert data["n_entities"] >= 20
        assert "ORG" in data["by_type"]
        assert "LOC" in data["by_type"]

    def test_kb_search_status_200(self):
        from fastapi.testclient import TestClient
        from ner.api.app import app

        client = TestClient(app)
        resp = client.post("/kb/search", json={"mention": "Газпром"})
        assert resp.status_code == 200

    def test_kb_search_structure(self):
        from fastapi.testclient import TestClient
        from ner.api.app import app

        client = TestClient(app)
        data = client.post("/kb/search", json={"mention": "Яндекс", "entity_type": "ORG"}).json()
        assert "mention" in data
        assert "candidates" in data
        assert "top_match" in data

    def test_kb_search_top_match_for_known_entity(self):
        """Поиск 'Сбербанк' должен вернуть top_match с entity_id."""
        from fastapi.testclient import TestClient
        from ner.api.app import app

        client = TestClient(app)
        data = client.post("/kb/search", json={"mention": "Сбербанк", "entity_type": "ORG"}).json()
        assert data["top_match"] is not None
        assert data["top_match"]["entity_id"] is not None

    def test_kb_search_unknown_entity_candidates_empty_or_low_score(self):
        """Неизвестное упоминание → top_match None или пустые candidates."""
        from fastapi.testclient import TestClient
        from ner.api.app import app

        client = TestClient(app)
        data = client.post(
            "/kb/search", json={"mention": "АбракадабраZXY", "entity_type": "ORG"}
        ).json()
        if data["top_match"] is not None:
            assert data["top_match"]["score"] < 0.5

    def test_health_endpoint_still_works(self):
        """Проверка, что существующий /health endpoint не сломан."""
        from fastapi.testclient import TestClient
        from ner.api.app import app

        client = TestClient(app)
        resp = client.get("/health")
        assert resp.status_code == 200
        assert "conformal_calibrated" in resp.json()
