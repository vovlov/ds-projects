#!/usr/bin/env bash
# pre_push_tests.sh — запускается как pre-push hook через pre-commit.
#
# Логика:
#   1. Определяем, какие файлы изменились в текущем push (относительно main/origin).
#   2. Находим, к каким проектам (01-XX, 02-XX, ...) относятся изменённые файлы.
#   3. Запускаем pytest только для затронутых проектов.
#
# Почему так: гонять все 185 тестов на каждый push слишком долго.
# Локализация по проекту ускоряет обратную связь до 10-20 секунд.

set -euo pipefail

REPO_ROOT="$(git rev-parse --show-toplevel)"
VENV_PYTHON="${REPO_ROOT}/.venv/bin/python"

# Маппинг директорий проектов
PROJECTS=(
    "01-customer-churn-mlops"
    "02-rag-enterprise"
    "03-ner-service"
    "04-graph-fraud-detection"
    "05-realtime-anomaly"
    "06-cv-document-scanner"
    "07-realestate-pricing"
    "08-llm-code-review"
    "09-recsys-feature-store"
    "10-data-quality-platform"
)

# Получаем список изменённых файлов: сравниваем HEAD с origin/main.
# Если origin/main недоступен (новый репо), берём все tracked файлы.
if git rev-parse --verify origin/main >/dev/null 2>&1; then
    CHANGED_FILES=$(git diff --name-only origin/main..HEAD 2>/dev/null || git diff --name-only HEAD~1..HEAD 2>/dev/null || echo "")
else
    CHANGED_FILES=$(git diff --name-only HEAD~1..HEAD 2>/dev/null || echo "")
fi

if [[ -z "${CHANGED_FILES}" ]]; then
    echo "pre-push: нет изменённых файлов для тестирования"
    exit 0
fi

# Определяем затронутые проекты
AFFECTED_PROJECTS=()
for project in "${PROJECTS[@]}"; do
    if echo "${CHANGED_FILES}" | grep -q "^${project}/"; then
        AFFECTED_PROJECTS+=("${project}")
    fi
done

if [[ ${#AFFECTED_PROJECTS[@]} -eq 0 ]]; then
    echo "pre-push: изменены только файлы вне проектов (docs/, scripts/, etc.) — тесты пропускаются"
    exit 0
fi

echo "pre-push: затронутые проекты: ${AFFECTED_PROJECTS[*]}"

# Запускаем тесты для каждого затронутого проекта
FAILED_PROJECTS=()
for project in "${AFFECTED_PROJECTS[@]}"; do
    echo ""
    echo "═══════════════════════════════════════════"
    echo "  pytest: ${project}"
    echo "═══════════════════════════════════════════"
    if (cd "${REPO_ROOT}/${project}" && "${VENV_PYTHON}" -m pytest tests/ -v --tb=short -q); then
        echo "  ✓ ${project}: все тесты прошли"
    else
        echo "  ✗ ${project}: ТЕСТЫ УПАЛИ"
        FAILED_PROJECTS+=("${project}")
    fi
done

# Итог
if [[ ${#FAILED_PROJECTS[@]} -gt 0 ]]; then
    echo ""
    echo "pre-push BLOCKED: упавшие проекты: ${FAILED_PROJECTS[*]}"
    echo "Исправьте тесты перед пушем."
    exit 1
fi

echo ""
echo "pre-push: все тесты прошли успешно"
exit 0
