"""Tests for Kubernetes manifest correctness.

Validates all YAML files under k8s/ for:
- Parseable YAML
- Required top-level Kubernetes fields
- Kind-specific required fields
- Deployment security best-practices (resource limits, probes, non-root)
"""

from __future__ import annotations

from pathlib import Path

import pytest
import yaml

K8S_DIR = Path(__file__).parent.parent.parent / "k8s"

REQUIRED_FIELDS = {"apiVersion", "kind", "metadata"}

KIND_REQUIRED_FIELDS: dict[str, list[str]] = {
    "Deployment": ["spec"],
    "Service": ["spec"],
    "HorizontalPodAutoscaler": ["spec"],
    "PodDisruptionBudget": ["spec"],
    "NetworkPolicy": ["spec"],
    "Ingress": ["spec"],
    "ConfigMap": [],
    "Namespace": [],
    "Secret": [],
}

EXPECTED_SERVICES = {
    "churn-api",
    "rag-api",
    "anomaly-api",
    "pricing-api",
    "recsys-api",
    "quality-api",
}


def _load_docs(path: Path) -> list[dict]:
    with path.open() as f:
        return [d for d in yaml.safe_load_all(f) if d is not None]


def _all_yaml_files() -> list[Path]:
    return sorted(K8S_DIR.rglob("*.yaml"))


def _deployments() -> list[tuple[Path, dict]]:
    result = []
    for p in _all_yaml_files():
        for doc in _load_docs(p):
            if isinstance(doc, dict) and doc.get("kind") == "Deployment":
                result.append((p, doc))
    return result


# ── Parametrized fixtures ─────────────────────────────────────────────────────


@pytest.fixture
def yaml_files() -> list[Path]:
    return _all_yaml_files()


# ── Tests ─────────────────────────────────────────────────────────────────────


class TestK8sManifestStructure:
    def test_k8s_directory_exists(self) -> None:
        assert K8S_DIR.exists(), f"k8s/ directory not found at {K8S_DIR}"

    def test_expected_yaml_files_present(self) -> None:
        files = _all_yaml_files()
        assert len(files) >= 10, f"Expected at least 10 YAML files, got {len(files)}"

    @pytest.mark.parametrize("path", _all_yaml_files())
    def test_yaml_parseable(self, path: Path) -> None:
        docs = _load_docs(path)
        assert len(docs) >= 1, f"{path} contains no documents"

    @pytest.mark.parametrize("path", _all_yaml_files())
    def test_required_top_level_fields(self, path: Path) -> None:
        for doc in _load_docs(path):
            missing = REQUIRED_FIELDS - doc.keys()
            kind = doc.get("kind")
            assert not missing, f"{path.name}: missing fields {missing} in doc kind={kind}"

    @pytest.mark.parametrize("path", _all_yaml_files())
    def test_known_kind_has_spec(self, path: Path) -> None:
        for doc in _load_docs(path):
            kind = doc.get("kind")
            if kind not in KIND_REQUIRED_FIELDS:
                continue
            for field in KIND_REQUIRED_FIELDS[kind]:
                assert field in doc, f"{path.name}: kind={kind} missing '{field}'"

    def test_namespace_document_exists(self) -> None:
        ns_file = K8S_DIR / "namespace.yaml"
        assert ns_file.exists()
        docs = _load_docs(ns_file)
        kinds = [d.get("kind") for d in docs]
        assert "Namespace" in kinds

    def test_configmap_document_exists(self) -> None:
        cm_file = K8S_DIR / "configmap.yaml"
        assert cm_file.exists()
        docs = _load_docs(cm_file)
        assert any(d.get("kind") == "ConfigMap" for d in docs)

    def test_ingress_document_exists(self) -> None:
        ingress_file = K8S_DIR / "ingress.yaml"
        assert ingress_file.exists()
        docs = _load_docs(ingress_file)
        assert any(d.get("kind") == "Ingress" for d in docs)


class TestDeploymentSecurity:
    @pytest.mark.parametrize("path,doc", _deployments())
    def test_deployment_has_resource_limits(self, path: Path, doc: dict) -> None:
        containers = doc["spec"]["template"]["spec"]["containers"]
        for c in containers:
            assert c.get("resources", {}).get("limits"), (
                f"{path.name}: container '{c.get('name')}' missing resources.limits"
            )

    @pytest.mark.parametrize("path,doc", _deployments())
    def test_deployment_has_resource_requests(self, path: Path, doc: dict) -> None:
        containers = doc["spec"]["template"]["spec"]["containers"]
        for c in containers:
            assert c.get("resources", {}).get("requests"), (
                f"{path.name}: container '{c.get('name')}' missing resources.requests"
            )

    @pytest.mark.parametrize("path,doc", _deployments())
    def test_deployment_has_liveness_probe(self, path: Path, doc: dict) -> None:
        containers = doc["spec"]["template"]["spec"]["containers"]
        for c in containers:
            assert c.get("livenessProbe"), (
                f"{path.name}: container '{c.get('name')}' missing livenessProbe"
            )

    @pytest.mark.parametrize("path,doc", _deployments())
    def test_deployment_has_readiness_probe(self, path: Path, doc: dict) -> None:
        containers = doc["spec"]["template"]["spec"]["containers"]
        for c in containers:
            assert c.get("readinessProbe"), (
                f"{path.name}: container '{c.get('name')}' missing readinessProbe"
            )

    @pytest.mark.parametrize("path,doc", _deployments())
    def test_deployment_runs_as_non_root(self, path: Path, doc: dict) -> None:
        pod_spec = doc["spec"]["template"]["spec"]
        security = pod_spec.get("securityContext", {})
        assert security.get("runAsNonRoot") is True, (
            f"{path.name}: securityContext.runAsNonRoot must be true"
        )

    @pytest.mark.parametrize("path,doc", _deployments())
    def test_deployment_min_replicas(self, path: Path, doc: dict) -> None:
        replicas = doc["spec"].get("replicas", 1)
        assert replicas >= 2, f"{path.name}: replicas={replicas} < 2, not HA"

    @pytest.mark.parametrize("path,doc", _deployments())
    def test_deployment_has_anti_affinity(self, path: Path, doc: dict) -> None:
        pod_spec = doc["spec"]["template"]["spec"]
        affinity = pod_spec.get("affinity", {})
        assert affinity.get("podAntiAffinity"), (
            f"{path.name}: missing podAntiAffinity — replicas may land on same node"
        )


class TestHPAPresence:
    def test_all_services_have_hpa(self) -> None:
        """Every production service deployment must have an HPA."""
        hpa_targets = set()
        for path in _all_yaml_files():
            for doc in _load_docs(path):
                if doc.get("kind") == "HorizontalPodAutoscaler":
                    target = doc.get("spec", {}).get("scaleTargetRef", {}).get("name")
                    if target:
                        hpa_targets.add(target)
        missing = EXPECTED_SERVICES - hpa_targets
        assert not missing, f"Services without HPA: {missing}"

    def test_all_services_have_pdb(self) -> None:
        """Every production service must have a PodDisruptionBudget."""
        pdb_targets = set()
        for path in _all_yaml_files():
            for doc in _load_docs(path):
                if doc.get("kind") == "PodDisruptionBudget":
                    selector = (
                        doc.get("spec", {}).get("selector", {}).get("matchLabels", {}).get("app")
                    )
                    if selector:
                        pdb_targets.add(selector)
        missing = EXPECTED_SERVICES - pdb_targets
        assert not missing, f"Services without PDB: {missing}"
