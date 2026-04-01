# Roadmap: От портфолио к продуктам

## Текущий уровень зрелости (MLOps Level 1-2)

По [Microsoft MLOps Maturity Model](https://learn.microsoft.com/en-us/azure/architecture/ai-ml/guide/mlops-maturity-model):
- Level 0: No MLOps — **ПРОЙДЕН** (был в 2020 с Практикумом)
- Level 1: DevOps but no MLOps — **ПРОЙДЕН** (CI/CD, Docker, тесты)
- Level 2: Training pipeline automation — **ТЕКУЩИЙ** (MLflow, Optuna, DVC)
- Level 3: Automated model deployment — **ЦЕЛЬ**
- Level 4: Full MLOps — **ГОРИЗОНТ**

## Бенчмарки: лучшие портфолио на GitHub

Изучены:
- [Made With ML](https://github.com/GokuMohandas/Made-With-ML) (40K+ stars) — полный MLOps-курс
- [mlops-course](https://github.com/GokuMohandas/mlops-course) — CI/CD, serving, monitoring
- [Microsoft MLOps](https://github.com/microsoft/MLOps) — enterprise patterns
- [awesome-mlops](https://github.com/visenger/awesome-mlops) — кураторский список

### Что у них есть, а у нас нет

| Фича | Made With ML | Наш ds-projects |
|------|-------------|-----------------|
| Pre-commit hooks | ✅ | ✅ (2026-03-29) |
| Model serving (Ray/BentoML) | ✅ | ❌ (только FastAPI) |
| Automated retraining (CT) | ✅ | ❌ |
| Data versioning (DVC pipeline) | ✅ | Частично (Project 01) |
| Model registry | ✅ | ❌ |
| A/B testing | ✅ | ❌ |
| Monitoring/drift in production | ✅ | Частично (Project 10) |
| Documentation site | ✅ (mkdocs) | ❌ |
| Deployment to cloud | ✅ | ❌ |

---

## Фаза 1: Production Polish (1-2 недели)

### Приоритет: каждый проект должен решать РЕАЛЬНУЮ проблему

| # | Проект | Реальное применение | Что доработать |
|---|--------|-------------------|----------------|
| 01 | Churn MLOps | Телеком-оператор: retention кампании | Pre-commit hooks, model registry (MLflow), SHAP waterfall API |
| 02 | RAG Enterprise | HR-отдел: ответы на вопросы по политикам | RAGAS evaluation, hybrid search (BM25+vector), streaming ответов |
| 03 | NER Service | Юридический отдел: извлечение сущностей из договоров | Fine-tune на Collection5, batch processing pipeline |
| 04 | Graph Fraud | Финтех: anti-fraud для P2P-платформы | Реальный датасет (Elliptic), temporal GNN |
| 05 | Anomaly | SRE-команда: мониторинг инфраструктуры | Prometheus metrics exporter, LSTM serving |
| 06 | CV Scanner | Страховая: обработка документов клиентов | Реальные изображения (RVL-CDIP), GradCAM |
| 07 | Pricing | Маркетплейс: оценка для листинга | Геопризнаки (H3), сравнение с рынком |
| 08 | Code Review | DevTools: PR assistant | LoRA fine-tuning на реальных PR |
| 09 | RecSys | E-commerce: персонализация | MovieLens-25M, two-tower neural |
| 10 | Quality | Data Platform: observability | Schema registry, lineage tracking |

### Конкретные задачи (итерации)

**Iteration 1-5: Pre-commit + Model Registry**
- [x] Добавить pre-commit hooks (ruff, mypy, pytest) — 2026-03-29
      `.pre-commit-config.yaml`: ruff lint+format, mypy (lenient), pre-push pytest
      `scripts/pre_push_tests.sh`: умный запуск только затронутых проектов
      `make pre-commit-install` — установка hooks одной командой
- [x] MLflow Model Registry для Project 01 и 07 — 2026-03-30
- [x] SHAP waterfall в API Response (Project 07) — 2026-03-31
      `pricing/models/explain.py`: explain_prediction с CatBoost built-in ShapValues
      `pricing/api/app.py`: SHAPWaterfall/SHAPContribution models, /estimate возвращает
      per-prediction SHAP + top_factors из SHAP вместо global importance
      19/19 тестов (2 новых: TestExplain)

**Iteration 6-10: Evaluation & Metrics**
- [x] RAGAS evaluation для RAG (Project 02) — 2026-04-01
      `rag/evaluation/ragas_eval.py`: 4 метрики (context_precision, context_recall,
      answer_relevance, faithfulness) + RAGASResult dataclass + evaluate_dataset().
      Лексические приближения без LLM — работают в CI без API-ключей. 14 новых тестов (29/29).
- [ ] Agentic RAG: faithfulness gate + confidence_score в ответе (Project 02)
      Второй LLM-вызов проверяет поддержку ответа retrieved chunks'ами.
      Источник: RAGFlow year-end review 2025, DEV.to RAG Blueprint 2026.
- [ ] Collection5 датасет для NER (Project 03)
- [ ] Elliptic Bitcoin датасет для Fraud (Project 04)
- [ ] VAE baseline для fraud detection (Project 04)
      Генеративная модель нормального поведения; отклонение = anomaly score.
      Даёт синтетические fraud-сэмплы без SMOTE. Источник: arxiv 2503.13195.

**Iteration 11-15: Real Data + Deployment**
- [ ] Streamlit Cloud деплой для 3 проектов
- [ ] MovieLens-25M для RecSys (Project 09)
- [ ] RVL-CDIP для CV Scanner (Project 06)

**Iteration 16-20: Monitoring & CT**
- [ ] Prometheus metrics exporter (Project 05)
- [ ] Automated retraining trigger (Project 01)
- [ ] Data drift alerting (Project 10 → 01)
- [ ] Write-Audit-Publish drift gates в feature store (Projects 09/10)
      PSI-проверка при записи фичи через Evidently AI, CI-gate с порогом дрейфа.
      Источник: Medium "Data Quality Assurance in MLOps" Mar 2026.

---

## Фаза 2: Product-Level (2-4 недели)

- [ ] mkdocs documentation site
- [ ] Model serving via BentoML (not just FastAPI)
- [ ] A/B testing framework
- [ ] Feature store integration (Feast)
- [ ] Kubernetes deployment manifests
- [ ] Automated model comparison reports

---

## Фаза 3: Enterprise (1-2 месяца)

- [ ] Multi-model orchestration (Project 01 → 04 → 05 pipeline)
- [ ] Schema registry for data contracts
- [ ] Data lineage visualization
- [ ] Cost optimization (model quantization, batching)
- [ ] Security audit (OWASP for ML)
- [ ] SLA monitoring

---

## Ежедневный цикл улучшений

Каждый день:
1. **Исследование** (30 мин): что нового в ML/MLOps? новые репо? новые подходы?
2. **Выбор проекта**: по roadmap или по приоритету
3. **Улучшение** (2-3 часа): код, тесты, документация
4. **Отладка**: запуск тестов, e2e проверка API
5. **Коммит + push**: маленькие инкрементальные улучшения

Sources:
- [Made With ML](https://github.com/GokuMohandas/Made-With-ML)
- [mlops-course](https://github.com/GokuMohandas/mlops-course)
- [Microsoft MLOps Maturity Model](https://learn.microsoft.com/en-us/azure/architecture/ai-ml/guide/mlops-maturity-model)
- [awesome-mlops](https://github.com/visenger/awesome-mlops)
- [ML Model Production Checklist](https://microsoft.github.io/code-with-engineering-playbook/machine-learning/ml-model-checklist/)
