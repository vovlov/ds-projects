"""
Тесты для Kubernetes deployment манифестов / Tests for Kubernetes deployment manifests.

Валидируем структуру YAML-файлов: обязательные поля, ресурсные лимиты,
пробы живости, HPA конфигурацию.

Validates YAML structure: required fields, resource limits,
liveness probes, HPA configuration.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest
import yaml

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

# ---------------------------------------------------------------------------
# Paths / Пути
# ---------------------------------------------------------------------------

_REPO_ROOT = Path(__file__).resolve().parents[2]
_K8S_ROOT = _REPO_ROOT / "k8s"

_PROJECTS = [
    "01-churn",
    "02-rag",
    "03-ner",
    "04-fraud",
    "05-anomaly",
    "06-scanner",
    "07-pricing",
    "08-review",
    "09-recsys",
    "10-quality",
]


# ---------------------------------------------------------------------------
# Helpers / Вспомогательные функции
# ---------------------------------------------------------------------------


def _load_yaml(path: Path) -> dict:
    """Загружает YAML файл / Load YAML file."""
    with open(path) as f:
        return yaml.safe_load(f)


def _deployment_path(project: str) -> Path:
    return _K8S_ROOT / project / "deployment.yaml"


def _service_path(project: str) -> Path:
    return _K8S_ROOT / project / "service.yaml"


def _hpa_path(project: str) -> Path:
    return _K8S_ROOT / project / "hpa.yaml"


# ---------------------------------------------------------------------------
# TestK8sFilesExist / Проверка наличия файлов
# ---------------------------------------------------------------------------


class TestK8sFilesExist:
    """Проверяет наличие всех K8s манифестов / Checks all K8s manifests exist."""

    def test_namespace_exists(self) -> None:
        assert (_K8S_ROOT / "namespace.yaml").exists()

    def test_kustomization_exists(self) -> None:
        assert (_K8S_ROOT / "kustomization.yaml").exists()

    @pytest.mark.parametrize("project", _PROJECTS)
    def test_deployment_exists(self, project: str) -> None:
        assert _deployment_path(project).exists(), f"Missing deployment for {project}"

    @pytest.mark.parametrize("project", _PROJECTS)
    def test_service_exists(self, project: str) -> None:
        assert _service_path(project).exists(), f"Missing service for {project}"

    @pytest.mark.parametrize("project", _PROJECTS)
    def test_hpa_exists(self, project: str) -> None:
        assert _hpa_path(project).exists(), f"Missing HPA for {project}"


# ---------------------------------------------------------------------------
# TestK8sYamlValid / Валидность YAML
# ---------------------------------------------------------------------------


class TestK8sYamlValid:
    """Проверяет, что все YAML файлы корректно парсятся / Validates YAML is parseable."""

    @pytest.mark.parametrize("project", _PROJECTS)
    def test_deployment_valid_yaml(self, project: str) -> None:
        doc = _load_yaml(_deployment_path(project))
        assert doc is not None

    @pytest.mark.parametrize("project", _PROJECTS)
    def test_service_valid_yaml(self, project: str) -> None:
        doc = _load_yaml(_service_path(project))
        assert doc is not None

    @pytest.mark.parametrize("project", _PROJECTS)
    def test_hpa_valid_yaml(self, project: str) -> None:
        doc = _load_yaml(_hpa_path(project))
        assert doc is not None

    def test_namespace_valid_yaml(self) -> None:
        doc = _load_yaml(_K8S_ROOT / "namespace.yaml")
        assert doc is not None

    def test_kustomization_valid_yaml(self) -> None:
        doc = _load_yaml(_K8S_ROOT / "kustomization.yaml")
        assert doc is not None


# ---------------------------------------------------------------------------
# TestK8sDeploymentStructure / Структура Deployment
# ---------------------------------------------------------------------------


class TestK8sDeploymentStructure:
    """Проверяет обязательные поля в Deployment манифестах."""

    @pytest.mark.parametrize("project", _PROJECTS)
    def test_deployment_api_version(self, project: str) -> None:
        doc = _load_yaml(_deployment_path(project))
        assert doc["apiVersion"] == "apps/v1"

    @pytest.mark.parametrize("project", _PROJECTS)
    def test_deployment_kind(self, project: str) -> None:
        doc = _load_yaml(_deployment_path(project))
        assert doc["kind"] == "Deployment"

    @pytest.mark.parametrize("project", _PROJECTS)
    def test_deployment_has_namespace(self, project: str) -> None:
        doc = _load_yaml(_deployment_path(project))
        assert doc["metadata"]["namespace"] == "ds-projects"

    @pytest.mark.parametrize("project", _PROJECTS)
    def test_deployment_has_replicas(self, project: str) -> None:
        doc = _load_yaml(_deployment_path(project))
        replicas = doc["spec"]["replicas"]
        assert isinstance(replicas, int)
        assert replicas >= 1

    @pytest.mark.parametrize("project", _PROJECTS)
    def test_deployment_selector_matches_template(self, project: str) -> None:
        """Selector должен совпадать с метками пода / Selector must match pod labels."""
        doc = _load_yaml(_deployment_path(project))
        selector_labels = doc["spec"]["selector"]["matchLabels"]
        template_labels = doc["spec"]["template"]["metadata"]["labels"]
        for key, val in selector_labels.items():
            assert template_labels.get(key) == val, (
                f"{project}: selector label {key}={val} not in template labels"
            )

    @pytest.mark.parametrize("project", _PROJECTS)
    def test_deployment_has_container(self, project: str) -> None:
        doc = _load_yaml(_deployment_path(project))
        containers = doc["spec"]["template"]["spec"]["containers"]
        assert len(containers) >= 1

    @pytest.mark.parametrize("project", _PROJECTS)
    def test_deployment_container_has_image(self, project: str) -> None:
        doc = _load_yaml(_deployment_path(project))
        container = doc["spec"]["template"]["spec"]["containers"][0]
        assert "image" in container
        assert container["image"]

    @pytest.mark.parametrize("project", _PROJECTS)
    def test_deployment_container_has_ports(self, project: str) -> None:
        doc = _load_yaml(_deployment_path(project))
        container = doc["spec"]["template"]["spec"]["containers"][0]
        assert "ports" in container
        assert len(container["ports"]) >= 1

    @pytest.mark.parametrize("project", _PROJECTS)
    def test_deployment_has_resource_limits(self, project: str) -> None:
        """Лимиты ресурсов обязательны для предотвращения resource starvation."""
        doc = _load_yaml(_deployment_path(project))
        container = doc["spec"]["template"]["spec"]["containers"][0]
        resources = container.get("resources", {})
        assert "limits" in resources, f"{project}: missing resource limits"
        assert "cpu" in resources["limits"]
        assert "memory" in resources["limits"]

    @pytest.mark.parametrize("project", _PROJECTS)
    def test_deployment_has_resource_requests(self, project: str) -> None:
        """Requests нужны для корректного K8s scheduling."""
        doc = _load_yaml(_deployment_path(project))
        container = doc["spec"]["template"]["spec"]["containers"][0]
        resources = container.get("resources", {})
        assert "requests" in resources, f"{project}: missing resource requests"
        assert "cpu" in resources["requests"]
        assert "memory" in resources["requests"]

    @pytest.mark.parametrize("project", _PROJECTS)
    def test_deployment_has_liveness_probe(self, project: str) -> None:
        doc = _load_yaml(_deployment_path(project))
        container = doc["spec"]["template"]["spec"]["containers"][0]
        assert "livenessProbe" in container, f"{project}: missing livenessProbe"

    @pytest.mark.parametrize("project", _PROJECTS)
    def test_deployment_has_readiness_probe(self, project: str) -> None:
        doc = _load_yaml(_deployment_path(project))
        container = doc["spec"]["template"]["spec"]["containers"][0]
        assert "readinessProbe" in container, f"{project}: missing readinessProbe"

    @pytest.mark.parametrize("project", _PROJECTS)
    def test_deployment_liveness_probe_path(self, project: str) -> None:
        """Все liveness пробы должны использовать /health endpoint."""
        doc = _load_yaml(_deployment_path(project))
        container = doc["spec"]["template"]["spec"]["containers"][0]
        probe = container["livenessProbe"]
        http_get = probe.get("httpGet", {})
        assert http_get.get("path") == "/health", f"{project}: liveness probe path must be /health"

    @pytest.mark.parametrize("project", _PROJECTS)
    def test_deployment_has_termination_grace_period(self, project: str) -> None:
        doc = _load_yaml(_deployment_path(project))
        spec = doc["spec"]["template"]["spec"]
        assert "terminationGracePeriodSeconds" in spec

    @pytest.mark.parametrize("project", _PROJECTS)
    def test_deployment_has_app_label(self, project: str) -> None:
        doc = _load_yaml(_deployment_path(project))
        labels = doc["metadata"].get("labels", {})
        assert "app" in labels


# ---------------------------------------------------------------------------
# TestK8sServiceStructure / Структура Service
# ---------------------------------------------------------------------------


class TestK8sServiceStructure:
    """Проверяет обязательные поля в Service манифестах."""

    @pytest.mark.parametrize("project", _PROJECTS)
    def test_service_kind(self, project: str) -> None:
        doc = _load_yaml(_service_path(project))
        assert doc["kind"] == "Service"

    @pytest.mark.parametrize("project", _PROJECTS)
    def test_service_has_namespace(self, project: str) -> None:
        doc = _load_yaml(_service_path(project))
        assert doc["metadata"]["namespace"] == "ds-projects"

    @pytest.mark.parametrize("project", _PROJECTS)
    def test_service_has_selector(self, project: str) -> None:
        doc = _load_yaml(_service_path(project))
        assert doc["spec"].get("selector"), f"{project}: service must have selector"

    @pytest.mark.parametrize("project", _PROJECTS)
    def test_service_has_ports(self, project: str) -> None:
        doc = _load_yaml(_service_path(project))
        assert doc["spec"].get("ports"), f"{project}: service must define ports"

    @pytest.mark.parametrize("project", _PROJECTS)
    def test_service_type(self, project: str) -> None:
        """Все сервисы должны быть ClusterIP для безопасности внутри кластера."""
        doc = _load_yaml(_service_path(project))
        svc_type = doc["spec"].get("type", "ClusterIP")
        assert svc_type == "ClusterIP", f"{project}: service type must be ClusterIP"


# ---------------------------------------------------------------------------
# TestK8sHPAStructure / Структура HPA
# ---------------------------------------------------------------------------


class TestK8sHPAStructure:
    """Проверяет структуру HorizontalPodAutoscaler манифестов."""

    @pytest.mark.parametrize("project", _PROJECTS)
    def test_hpa_kind(self, project: str) -> None:
        doc = _load_yaml(_hpa_path(project))
        assert doc["kind"] == "HorizontalPodAutoscaler"

    @pytest.mark.parametrize("project", _PROJECTS)
    def test_hpa_has_namespace(self, project: str) -> None:
        doc = _load_yaml(_hpa_path(project))
        assert doc["metadata"]["namespace"] == "ds-projects"

    @pytest.mark.parametrize("project", _PROJECTS)
    def test_hpa_api_version(self, project: str) -> None:
        doc = _load_yaml(_hpa_path(project))
        # autoscaling/v2 для multi-metric support (CPU + memory)
        assert doc["apiVersion"] == "autoscaling/v2"

    @pytest.mark.parametrize("project", _PROJECTS)
    def test_hpa_min_replicas(self, project: str) -> None:
        doc = _load_yaml(_hpa_path(project))
        min_r = doc["spec"]["minReplicas"]
        assert min_r >= 1

    @pytest.mark.parametrize("project", _PROJECTS)
    def test_hpa_max_replicas_greater_than_min(self, project: str) -> None:
        doc = _load_yaml(_hpa_path(project))
        assert doc["spec"]["maxReplicas"] > doc["spec"]["minReplicas"]

    @pytest.mark.parametrize("project", _PROJECTS)
    def test_hpa_has_metrics(self, project: str) -> None:
        doc = _load_yaml(_hpa_path(project))
        metrics = doc["spec"].get("metrics", [])
        assert len(metrics) >= 1, f"{project}: HPA must define at least one metric"

    @pytest.mark.parametrize("project", _PROJECTS)
    def test_hpa_scale_target_ref(self, project: str) -> None:
        doc = _load_yaml(_hpa_path(project))
        ref = doc["spec"]["scaleTargetRef"]
        assert ref["apiVersion"] == "apps/v1"
        assert ref["kind"] == "Deployment"

    @pytest.mark.parametrize("project", _PROJECTS)
    def test_hpa_has_scale_down_stabilization(self, project: str) -> None:
        """Scale-down stabilization предотвращает flapping под нагрузкой."""
        doc = _load_yaml(_hpa_path(project))
        behavior = doc["spec"].get("behavior", {})
        scale_down = behavior.get("scaleDown", {})
        window = scale_down.get("stabilizationWindowSeconds")
        assert window is not None and window >= 60, (
            f"{project}: scaleDown stabilizationWindowSeconds should be >= 60"
        )


# ---------------------------------------------------------------------------
# TestK8sNamespace / Namespace манифест
# ---------------------------------------------------------------------------


class TestK8sNamespace:
    """Проверяет namespace.yaml."""

    def test_namespace_kind(self) -> None:
        doc = _load_yaml(_K8S_ROOT / "namespace.yaml")
        assert doc["kind"] == "Namespace"

    def test_namespace_name(self) -> None:
        doc = _load_yaml(_K8S_ROOT / "namespace.yaml")
        assert doc["metadata"]["name"] == "ds-projects"


# ---------------------------------------------------------------------------
# TestK8sKustomization / Kustomization
# ---------------------------------------------------------------------------


class TestK8sKustomization:
    """Проверяет kustomization.yaml."""

    def test_kustomization_has_resources(self) -> None:
        doc = _load_yaml(_K8S_ROOT / "kustomization.yaml")
        assert "resources" in doc
        assert len(doc["resources"]) > 0

    def test_kustomization_namespace(self) -> None:
        doc = _load_yaml(_K8S_ROOT / "kustomization.yaml")
        assert doc.get("namespace") == "ds-projects"

    def test_kustomization_all_deployments_listed(self) -> None:
        """Каждый проект должен быть включён в kustomization."""
        doc = _load_yaml(_K8S_ROOT / "kustomization.yaml")
        resources = doc["resources"]
        for project in _PROJECTS:
            assert any(project in r for r in resources), (
                f"{project} not found in kustomization resources"
            )
