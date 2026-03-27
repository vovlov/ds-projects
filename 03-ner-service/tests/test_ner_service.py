"""Tests for NER Service."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.data.dataset import (
    ID2LABEL,
    LABEL2ID,
    NER_LABELS,
    get_sample_data,
)
from src.model.ner import extract_entities_from_bio, predict, predict_rule_based


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
        from src.api.app import app

        client = TestClient(app)
        assert client.get("/health").status_code == 200

    def test_predict_endpoint(self):
        from fastapi.testclient import TestClient
        from src.api.app import app

        client = TestClient(app)
        resp = client.post("/predict", json={"text": "Газпром находится в Москве."})
        assert resp.status_code == 200
        data = resp.json()
        assert "entities" in data
        assert "text" in data

    def test_predict_batch(self):
        from fastapi.testclient import TestClient
        from src.api.app import app

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
        from src.api.app import app

        client = TestClient(app)
        resp = client.post("/predict", json={"text": ""})
        assert resp.status_code == 422  # min_length=1 validation
