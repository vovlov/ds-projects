"""
Тесты для модуля data lineage / Tests for the data lineage module.

Покрываем: граф родословной, трекер событий, API-эндпоинты.
Covers: lineage graph, event tracker, API endpoints.
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import pytest
from fastapi.testclient import TestClient
from quality.lineage.graph import LineageEdge, LineageGraph, LineageNode, NodeType
from quality.lineage.tracker import LineageTracker, RunState, reset_tracker

# ---------------------------------------------------------------------------
# Фикстуры / Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture(autouse=True)
def reset_global_tracker() -> None:
    """Сбросить глобальный трекер перед каждым тестом."""
    reset_tracker()
    yield
    reset_tracker()


@pytest.fixture()
def tracker() -> LineageTracker:
    """Свежий трекер для тестов / Fresh tracker instance."""
    return LineageTracker()


@pytest.fixture()
def graph() -> LineageGraph:
    """Пустой граф / Empty lineage graph."""
    return LineageGraph()


@pytest.fixture()
def client() -> TestClient:
    """FastAPI тест-клиент / FastAPI test client."""
    from quality.api.app import app
    return TestClient(app)


# ---------------------------------------------------------------------------
# TestLineageNode — базовые узлы / Node tests
# ---------------------------------------------------------------------------


class TestLineageNode:
    """Тесты для LineageNode / LineageNode tests."""

    def test_dataset_factory(self) -> None:
        """Фабрика dataset создаёт правильный тип / Dataset factory sets correct type."""
        node = LineageNode.dataset("postgres", "customers")
        assert node.node_type == NodeType.DATASET
        assert node.node_id == "postgres/customers"
        assert node.namespace == "postgres"
        assert node.name == "customers"

    def test_job_factory(self) -> None:
        """Фабрика job создаёт правильный тип / Job factory sets correct type."""
        node = LineageNode.job("churn-service", "train_model")
        assert node.node_type == NodeType.JOB
        assert node.node_id == "churn-service/train_model"

    def test_node_with_facets(self) -> None:
        """Фасеты сохраняются / Facets are stored."""
        node = LineageNode.dataset("spark", "events", schema_version="1.2", row_count=1000)
        assert node.facets["schema_version"] == "1.2"
        assert node.facets["row_count"] == 1000

    def test_to_dict(self) -> None:
        """Сериализация в dict / Serialization to dict."""
        node = LineageNode.dataset("postgres", "orders")
        d = node.to_dict()
        assert d["id"] == "postgres/orders"
        assert d["type"] == "dataset"
        assert d["namespace"] == "postgres"
        assert d["name"] == "orders"
        assert "facets" in d


# ---------------------------------------------------------------------------
# TestLineageEdge — рёбра / Edge tests
# ---------------------------------------------------------------------------


class TestLineageEdge:
    """Тесты для LineageEdge / LineageEdge tests."""

    def test_edge_creation(self) -> None:
        """Ребро сохраняет source и target / Edge stores source and target."""
        edge = LineageEdge(source_id="postgres/customers", target_id="churn-service/train_model")
        assert edge.source_id == "postgres/customers"
        assert edge.target_id == "churn-service/train_model"

    def test_edge_with_run_id(self) -> None:
        """Ребро с run_id / Edge with run_id."""
        edge = LineageEdge(source_id="a", target_id="b", run_id="run-123")
        d = edge.to_dict()
        assert d["run_id"] == "run-123"

    def test_to_dict(self) -> None:
        """Сериализация ребра / Edge serialization."""
        edge = LineageEdge(source_id="src", target_id="tgt", metadata={"weight": 1})
        d = edge.to_dict()
        assert d["source"] == "src"
        assert d["target"] == "tgt"
        assert d["metadata"]["weight"] == 1


# ---------------------------------------------------------------------------
# TestLineageGraph — граф / Graph tests
# ---------------------------------------------------------------------------


class TestLineageGraph:
    """Тесты для LineageGraph / LineageGraph tests."""

    def test_add_node(self, graph: LineageGraph) -> None:
        """Добавление узла / Node is added."""
        node = LineageNode.dataset("postgres", "customers")
        graph.add_node(node)
        assert "postgres/customers" in graph.nodes

    def test_add_edge_deduplication(self, graph: LineageGraph) -> None:
        """Дублирующиеся рёбра не добавляются / Duplicate edges are skipped."""
        graph.add_node(LineageNode.dataset("db", "src"))
        graph.add_node(LineageNode.job("svc", "etl"))
        edge = LineageEdge(source_id="db/src", target_id="svc/etl")
        graph.add_edge(edge)
        graph.add_edge(edge)  # дублируем
        assert len(graph.edges) == 1

    def test_add_lineage_creates_nodes_and_edges(self, graph: LineageGraph) -> None:
        """add_lineage создаёт узлы + рёбра / add_lineage creates nodes and edges."""
        job = LineageNode.job("etl", "transform")
        inp = LineageNode.dataset("raw", "events")
        out = LineageNode.dataset("clean", "events")
        graph.add_lineage(job=job, inputs=[inp], outputs=[out], run_id="r1")

        assert "etl/transform" in graph.nodes
        assert "raw/events" in graph.nodes
        assert "clean/events" in graph.nodes
        assert len(graph.edges) == 2  # raw→etl, etl→clean

    def test_upstream_traversal(self, graph: LineageGraph) -> None:
        """Обход upstream находит предков / Upstream traversal finds ancestors."""
        # raw → etl → clean → model
        job1 = LineageNode.job("etl", "clean")
        raw = LineageNode.dataset("db", "raw")
        clean = LineageNode.dataset("db", "clean")
        graph.add_lineage(job=job1, inputs=[raw], outputs=[clean])

        job2 = LineageNode.job("ml", "train")
        model = LineageNode.dataset("mlflow", "model")
        graph.add_lineage(job=job2, inputs=[clean], outputs=[model])

        upstream_model = set(graph.upstream("mlflow/model"))
        # model ← train ← clean ← clean_job ← raw
        assert "db/clean" in upstream_model
        assert "db/raw" in upstream_model

    def test_downstream_traversal(self, graph: LineageGraph) -> None:
        """Обход downstream находит потомков / Downstream traversal finds descendants."""
        job1 = LineageNode.job("etl", "clean")
        raw = LineageNode.dataset("db", "raw")
        clean = LineageNode.dataset("db", "clean")
        graph.add_lineage(job=job1, inputs=[raw], outputs=[clean])

        downstream_raw = set(graph.downstream("db/raw"))
        assert "etl/clean" in downstream_raw
        assert "db/clean" in downstream_raw

    def test_lineage_for_dataset(self, graph: LineageGraph) -> None:
        """lineage_for_dataset возвращает upstream+downstream / Returns upstream+downstream."""
        job = LineageNode.job("svc", "score")
        src = LineageNode.dataset("db", "features")
        tgt = LineageNode.dataset("db", "predictions")
        graph.add_lineage(job=job, inputs=[src], outputs=[tgt])

        result = graph.lineage_for_dataset("db/features")
        assert result["dataset_id"] == "db/features"
        assert isinstance(result["upstream"], list)
        assert isinstance(result["downstream"], list)
        assert isinstance(result["nodes"], list)
        assert isinstance(result["edges"], list)

    def test_to_dict_stats(self, graph: LineageGraph) -> None:
        """to_dict возвращает корректную статистику / to_dict returns correct stats."""
        job = LineageNode.job("svc", "train")
        inp = LineageNode.dataset("db", "train_data")
        out = LineageNode.dataset("mlflow", "model_v1")
        graph.add_lineage(job=job, inputs=[inp], outputs=[out])

        d = graph.to_dict()
        assert d["stats"]["total_nodes"] == 3
        assert d["stats"]["total_edges"] == 2
        assert d["stats"]["dataset_count"] == 2
        assert d["stats"]["job_count"] == 1

    def test_empty_graph_to_dict(self, graph: LineageGraph) -> None:
        """Пустой граф сериализуется без ошибок / Empty graph serializes cleanly."""
        d = graph.to_dict()
        assert d["nodes"] == []
        assert d["links"] == []
        assert d["stats"]["total_nodes"] == 0


# ---------------------------------------------------------------------------
# TestLineageTracker — трекер / Tracker tests
# ---------------------------------------------------------------------------


class TestLineageTracker:
    """Тесты для LineageTracker / LineageTracker tests."""

    def test_record_complete_event(self, tracker: LineageTracker) -> None:
        """Запись COMPLETE события / Recording a COMPLETE event."""
        event = tracker.record(
            job_namespace="churn",
            job_name="train_model",
            event_type=RunState.COMPLETE,
            inputs=[{"namespace": "postgres", "name": "customers"}],
            outputs=[{"namespace": "mlflow", "name": "churn_v1"}],
        )
        assert event.event_type == RunState.COMPLETE
        assert event.job_name == "train_model"
        assert len(event.inputs) == 1
        assert len(event.outputs) == 1

    def test_record_updates_graph(self, tracker: LineageTracker) -> None:
        """COMPLETE событие обновляет граф / COMPLETE event updates the graph."""
        tracker.record(
            job_namespace="svc",
            job_name="score",
            event_type="COMPLETE",
            inputs=[{"namespace": "db", "name": "features"}],
            outputs=[{"namespace": "db", "name": "predictions"}],
        )
        graph = tracker.get_graph()
        assert "svc/score" in graph.nodes
        assert "db/features" in graph.nodes
        assert "db/predictions" in graph.nodes

    def test_record_string_event_type(self, tracker: LineageTracker) -> None:
        """Строковый event_type конвертируется / String event_type is converted."""
        event = tracker.record("ns", "job", event_type="start")
        assert event.event_type == RunState.START

    def test_auto_run_id_generation(self, tracker: LineageTracker) -> None:
        """run_id генерируется автоматически / run_id is auto-generated."""
        event = tracker.record("ns", "job", event_type="COMPLETE")
        assert event.run_id is not None
        assert len(event.run_id) > 0

    def test_run_id_reuse(self, tracker: LineageTracker) -> None:
        """Один run_id для START + COMPLETE / Same run_id for START + COMPLETE."""
        start = tracker.record("ns", "job", event_type="START", run_id="run-42")
        complete = tracker.record("ns", "job", event_type="COMPLETE", run_id="run-42")
        assert start.run_id == complete.run_id == "run-42"

    def test_get_events_filter_by_job(self, tracker: LineageTracker) -> None:
        """Фильтрация событий по job_name / Filter events by job_name."""
        tracker.record("ns", "job_a", event_type="COMPLETE")
        tracker.record("ns", "job_b", event_type="COMPLETE")
        events = tracker.get_events(job_name="job_a")
        assert len(events) == 1
        assert events[0].job_name == "job_a"

    def test_get_events_filter_by_type(self, tracker: LineageTracker) -> None:
        """Фильтрация по типу события / Filter events by event type."""
        tracker.record("ns", "job", event_type="START", run_id="r1")
        tracker.record("ns", "job", event_type="COMPLETE", run_id="r1")
        tracker.record("ns", "job", event_type="FAIL", run_id="r2")

        complete_events = tracker.get_events(event_type=RunState.COMPLETE)
        assert len(complete_events) == 1

    def test_get_run_history(self, tracker: LineageTracker) -> None:
        """get_run_history возвращает все события запуска / get_run_history returns run events."""
        tracker.record("ns", "job", event_type="START", run_id="run-99")
        tracker.record("ns", "job", event_type="COMPLETE", run_id="run-99")
        tracker.record("ns", "other", event_type="COMPLETE", run_id="run-00")

        history = tracker.get_run_history("run-99")
        assert len(history) == 2
        assert all(e.run_id == "run-99" for e in history)

    def test_summary(self, tracker: LineageTracker) -> None:
        """summary возвращает корректную статистику / summary returns correct stats."""
        tracker.record("ns", "job", event_type="START", run_id="r1")
        tracker.record("ns", "job", event_type="COMPLETE", run_id="r1",
                       inputs=[{"namespace": "db", "name": "src"}],
                       outputs=[{"namespace": "db", "name": "dst"}])

        s = tracker.summary()
        assert s["total_events"] == 2
        assert s["event_type_counts"]["START"] == 1
        assert s["event_type_counts"]["COMPLETE"] == 1
        assert s["graph_stats"]["total_nodes"] >= 3

    def test_event_to_dict_openlineage_format(self, tracker: LineageTracker) -> None:
        """Событие сериализуется в OpenLineage-совместимый формат."""
        event = tracker.record("ns", "job", event_type="COMPLETE", run_id="r42")
        d = event.to_dict()
        assert "eventType" in d
        assert "eventTime" in d
        assert "run" in d and "runId" in d["run"]
        assert "job" in d and "namespace" in d["job"]
        assert "inputs" in d
        assert "outputs" in d


# ---------------------------------------------------------------------------
# TestLineageAPIEndpoints — API-эндпоинты / API endpoint tests
# ---------------------------------------------------------------------------


class TestLineageAPIEndpoints:
    """Тесты для lineage API-эндпоинтов / Lineage API endpoint tests."""

    def test_record_event_201(self, client: TestClient) -> None:
        """POST /lineage/event возвращает 201 / Returns 201 on success."""
        response = client.post(
            "/lineage/event",
            json={
                "job_namespace": "churn-service",
                "job_name": "train_model",
                "event_type": "COMPLETE",
                "inputs": [{"namespace": "postgres", "name": "customers"}],
                "outputs": [{"namespace": "mlflow", "name": "churn_v1"}],
            },
        )
        assert response.status_code == 201
        body = response.json()
        assert body["eventType"] == "COMPLETE"
        assert body["job"]["name"] == "train_model"

    def test_record_event_invalid_type_422(self, client: TestClient) -> None:
        """Неверный event_type → 422 / Invalid event_type returns 422."""
        response = client.post(
            "/lineage/event",
            json={"job_namespace": "ns", "job_name": "job", "event_type": "INVALID"},
        )
        assert response.status_code == 422

    def test_get_lineage_graph_empty(self, client: TestClient) -> None:
        """GET /lineage/graph на пустом графе / Empty graph returns valid structure."""
        response = client.get("/lineage/graph")
        assert response.status_code == 200
        body = response.json()
        assert "nodes" in body
        assert "links" in body
        assert "stats" in body

    def test_get_lineage_graph_populated(self, client: TestClient) -> None:
        """Граф заполняется после событий / Graph is populated after events."""
        client.post(
            "/lineage/event",
            json={
                "job_namespace": "rag-service",
                "job_name": "embed_documents",
                "event_type": "COMPLETE",
                "inputs": [{"namespace": "s3", "name": "raw_docs"}],
                "outputs": [{"namespace": "weaviate", "name": "doc_embeddings"}],
            },
        )
        response = client.get("/lineage/graph")
        body = response.json()
        assert body["stats"]["total_nodes"] >= 3
        assert body["stats"]["job_count"] >= 1
        assert body["stats"]["dataset_count"] >= 2

    def test_get_dataset_lineage_404(self, client: TestClient) -> None:
        """Несуществующий датасет → 404 / Unknown dataset returns 404."""
        response = client.get("/lineage/dataset/unknown/missing_table")
        assert response.status_code == 404

    def test_get_dataset_lineage_found(self, client: TestClient) -> None:
        """Существующий датасет возвращает родословную / Known dataset returns lineage."""
        client.post(
            "/lineage/event",
            json={
                "job_namespace": "fraud-service",
                "job_name": "score_transactions",
                "event_type": "COMPLETE",
                "inputs": [{"namespace": "kafka", "name": "transactions"}],
                "outputs": [{"namespace": "postgres", "name": "fraud_scores"}],
            },
        )
        response = client.get("/lineage/dataset/kafka/transactions")
        assert response.status_code == 200
        body = response.json()
        assert body["dataset_id"] == "kafka/transactions"
        assert "upstream" in body
        assert "downstream" in body

    def test_get_upstream_lineage(self, client: TestClient) -> None:
        """GET /lineage/upstream/ возвращает upstream узлы / Returns upstream nodes."""
        client.post(
            "/lineage/event",
            json={
                "job_namespace": "pricing",
                "job_name": "predict_price",
                "event_type": "COMPLETE",
                "inputs": [{"namespace": "pg", "name": "listings"}],
                "outputs": [{"namespace": "pg", "name": "price_estimates"}],
            },
        )
        response = client.get("/lineage/upstream/pg/price_estimates")
        assert response.status_code == 200
        body = response.json()
        assert body["node_id"] == "pg/price_estimates"
        assert "upstream" in body
        assert "depth" in body

    def test_get_downstream_lineage(self, client: TestClient) -> None:
        """GET /lineage/downstream/ возвращает downstream узлы / Returns downstream nodes."""
        client.post(
            "/lineage/event",
            json={
                "job_namespace": "recsys",
                "job_name": "compute_features",
                "event_type": "COMPLETE",
                "inputs": [{"namespace": "pg", "name": "user_interactions"}],
                "outputs": [{"namespace": "feast", "name": "user_features"}],
            },
        )
        response = client.get("/lineage/downstream/pg/user_interactions")
        assert response.status_code == 200
        body = response.json()
        assert "downstream" in body
        assert "depth" in body

    def test_list_lineage_events(self, client: TestClient) -> None:
        """GET /lineage/events возвращает список событий / Returns event list."""
        client.post(
            "/lineage/event",
            json={"job_namespace": "anomaly", "job_name": "detect", "event_type": "COMPLETE"},
        )
        response = client.get("/lineage/events")
        assert response.status_code == 200
        body = response.json()
        assert "events" in body
        assert "total" in body
        assert body["total"] >= 1

    def test_list_events_filter_by_job(self, client: TestClient) -> None:
        """Фильтрация событий по job_name через API / Filter events by job_name via API."""
        client.post(
            "/lineage/event",
            json={"job_namespace": "ns", "job_name": "job_alpha", "event_type": "COMPLETE"},
        )
        client.post(
            "/lineage/event",
            json={"job_namespace": "ns", "job_name": "job_beta", "event_type": "COMPLETE"},
        )
        response = client.get("/lineage/events?job_name=job_alpha")
        body = response.json()
        assert all(e["job"]["name"] == "job_alpha" for e in body["events"])

    def test_lineage_summary(self, client: TestClient) -> None:
        """GET /lineage/summary возвращает статистику / Returns summary stats."""
        client.post(
            "/lineage/event",
            json={"job_namespace": "quality", "job_name": "validate", "event_type": "COMPLETE"},
        )
        response = client.get("/lineage/summary")
        assert response.status_code == 200
        body = response.json()
        assert "total_events" in body
        assert "graph_stats" in body
        assert "event_type_counts" in body

    def test_multi_hop_lineage_chain(self, client: TestClient) -> None:
        """Многоходовая цепочка родословной / Multi-hop lineage chain."""
        # raw → etl_job → clean → ml_job → predictions
        client.post(
            "/lineage/event",
            json={
                "job_namespace": "etl",
                "job_name": "clean_data",
                "event_type": "COMPLETE",
                "inputs": [{"namespace": "s3", "name": "raw_data"}],
                "outputs": [{"namespace": "pg", "name": "clean_data"}],
            },
        )
        client.post(
            "/lineage/event",
            json={
                "job_namespace": "ml",
                "job_name": "train",
                "event_type": "COMPLETE",
                "inputs": [{"namespace": "pg", "name": "clean_data"}],
                "outputs": [{"namespace": "mlflow", "name": "model_v1"}],
            },
        )

        # Upstream от model_v1 должен включать raw_data через clean_data
        response = client.get("/lineage/upstream/mlflow/model_v1")
        body = response.json()
        upstream_ids = {n["id"] for n in body["upstream"]}
        assert "pg/clean_data" in upstream_ids
        assert "s3/raw_data" in upstream_ids
