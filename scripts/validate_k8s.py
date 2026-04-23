#!/usr/bin/env python3
"""Validate Kubernetes manifests for structural correctness.

Checks all YAML files under k8s/ for:
- Valid YAML syntax
- Required Kubernetes top-level fields (apiVersion, kind, metadata)
- Kind-specific required fields
- Security best-practices (resource limits, non-root user)
"""

from __future__ import annotations

import sys
from pathlib import Path

try:
    import yaml
except ImportError:
    print("PyYAML not installed — run: uv pip install pyyaml", file=sys.stderr)
    sys.exit(1)

REQUIRED_FIELDS = {"apiVersion", "kind", "metadata"}

KIND_CHECKS: dict[str, list[str]] = {
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


def _load_all(path: Path) -> list[dict]:
    """Parse all documents from a YAML file."""
    with path.open() as f:
        docs = list(yaml.safe_load_all(f))
    return [d for d in docs if d is not None]


def _check_deployment_security(doc: dict, path: Path) -> list[str]:
    """Return warnings for missing security/resource fields."""
    warnings = []
    try:
        containers = doc["spec"]["template"]["spec"]["containers"]
        for c in containers:
            resources = c.get("resources", {})
            if not resources.get("limits"):
                warnings.append(
                    f"{path.name}: container '{c.get('name')}' missing resources.limits"
                )
            if not resources.get("requests"):
                warnings.append(
                    f"{path.name}: container '{c.get('name')}' missing resources.requests"
                )
            if not c.get("livenessProbe"):
                warnings.append(f"{path.name}: container '{c.get('name')}' missing livenessProbe")
            if not c.get("readinessProbe"):
                warnings.append(f"{path.name}: container '{c.get('name')}' missing readinessProbe")
        pod_spec = doc["spec"]["template"]["spec"]
        security = pod_spec.get("securityContext", {})
        if not security.get("runAsNonRoot"):
            warnings.append(f"{path.name}: securityContext.runAsNonRoot not set")
    except (KeyError, TypeError):
        warnings.append(f"{path.name}: unexpected Deployment structure")
    return warnings


def validate_file(path: Path) -> tuple[list[str], list[str]]:
    """Return (errors, warnings) for a manifest file."""
    errors: list[str] = []
    warnings: list[str] = []
    try:
        docs = _load_all(path)
    except yaml.YAMLError as exc:
        errors.append(f"{path}: YAML parse error: {exc}")
        return errors, warnings

    if not docs:
        errors.append(f"{path}: file is empty or contains no documents")
        return errors, warnings

    for doc in docs:
        if not isinstance(doc, dict):
            errors.append(f"{path}: document is not a mapping")
            continue
        missing = REQUIRED_FIELDS - doc.keys()
        if missing:
            errors.append(f"{path}: missing top-level fields: {missing}")
            continue

        kind = doc["kind"]
        if kind not in KIND_CHECKS:
            warnings.append(f"{path}: unknown kind '{kind}' — skipping field checks")
            continue

        for field in KIND_CHECKS[kind]:
            if field not in doc:
                errors.append(f"{path}: kind={kind} missing required field '{field}'")

        if kind == "Deployment":
            warnings.extend(_check_deployment_security(doc, path))

    return errors, warnings


def main() -> int:
    root = Path(__file__).parent.parent / "k8s"
    if not root.exists():
        print(f"ERROR: k8s directory not found at {root}", file=sys.stderr)
        return 1

    all_errors: list[str] = []
    all_warnings: list[str] = []
    files_checked = 0

    for yaml_file in sorted(root.rglob("*.yaml")):
        errs, warns = validate_file(yaml_file)
        all_errors.extend(errs)
        all_warnings.extend(warns)
        files_checked += 1

    for w in all_warnings:
        print(f"WARN  {w}")
    for e in all_errors:
        print(f"ERROR {e}")

    status = "PASS" if not all_errors else "FAIL"
    print(
        f"\n{status}: {files_checked} files checked, "
        f"{len(all_errors)} errors, {len(all_warnings)} warnings"
    )
    return 1 if all_errors else 0


if __name__ == "__main__":
    sys.exit(main())
