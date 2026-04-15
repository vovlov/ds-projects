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
- [x] Agentic RAG: faithfulness gate + confidence_score в ответе (Project 02) — 2026-04-02
      rag/generation/faithfulness_gate.py: FaithfulnessResult, check_faithfulness()
      (LLM mode через Haiku / lexical fallback без API-ключа для CI).
      chain.py: generate_answer_with_gate() возвращает confidence_score, is_faithful.
      app.py: POST /query endpoint + badge в Gradio UI. 40/40 тестов (+11 TestFaithfulnessGate).
      Источник: Self-RAG arxiv 2310.11511, RAGFlow 2025, DEV.to RAG Blueprint 2026.
- [x] Collection5 датасет для NER (Project 03) — 2026-04-03
      ner/data/collection5.py: CoNLL-парсер, встроенный образец (8 предложений PER/ORG/LOC),
      load_collection5(path=None) для CI без сети, compute_dataset_stats(), compute_metrics() через seqeval.
      ner/model/batch.py: BatchItem/BatchResult dataclasses, process_texts(), process_collection5().
      41/41 тестов (+23 новых: TestCollection5Parser, TestCollection5Stats, TestBatchProcessing).
- [x] Elliptic Bitcoin датасет для Fraud (Project 04) — 2026-04-04
      fraud/data/elliptic.py: load_elliptic_dataset(), generate_mock_elliptic() (mock для CI),
      get_labeled_split() (конвертация меток Elliptic 1/2 → бинарные 0/1).
      166 признаков, поддержка CSV из Kaggle + автоматический fallback на mock.
      10 новых тестов TestEllipticDataset.
- [x] VAE baseline для fraud detection (Project 04) — 2026-04-04
      fraud/models/baseline/vae.py: FraudVAE (encoder+reparameterize+decoder),
      ELBO loss (MSE + KL), train_vae() — обучение только на нормальных сэмплах,
      порог = 95-й перцентиль ошибок реконструкции нормальных данных.
      is_available() — graceful fallback без PyTorch. 5 новых тестов TestVAEModule.
      Источник: arxiv 2503.13195, FinGuard-GNN 2025.

**Iteration 11-15: Real Data + Deployment**
- [ ] Streamlit Cloud деплой для 3 проектов
- [x] MovieLens-25M для RecSys (Project 09) — 2026-04-05
      recsys/data/movielens.py: generate_mock_movielens() (mock для CI), load_movielens()
      с graceful fallback, compute_movielens_stats() (sparsity, avg_rating, top_genres),
      to_recsys_format() конвертирует userId/movieId → user_id/product_id + ISO timestamp.
      Полузвёздная шкала 0.5–5.0 + power-law распределение активности. 27 новых тестов
      (TestMovieLensMock, TestMovieLensLoader, TestMovieLensStats, TestMovieLensToRecsysFormat).
      53/53 тестов зелёные.
- [x] RVL-CDIP для CV Scanner (Project 06) — 2026-04-06
      scanner/data/rvl_cdip.py: 16 классов документов, generate_mock_rvl_cdip()
      для CI без сети, load_rvl_cdip() с graceful fallback, compute_dataset_stats(),
      to_scanner_format() — конвертация в формат совместимый с existing classifier.
      scanner/models/gradcam.py: GradCAM class с forward/backward hooks, compute(),
      overlay(), explain_prediction() — объяснимость CNN для EU AI Act compliance.
      is_available() graceful fallback без PyTorch.
      56/56 тестов зелёные (+37 новых: TestRVLCDIPMock, TestRVLCDIPLoader,
      TestRVLCDIPStats, TestRVLCDIPToScannerFormat, TestGradCAM).
- [x] Two-tower + LLM re-ranking для RecSys (Project 09) — 2026-04-07
      recsys/models/two_tower.py: TwoTowerModel (UserTower + ItemTower) на numpy/sklearn
      без PyTorch (macOS x86_64 совместимость). In-batch negative sampling + softmax
      cross-entropy loss. L2-нормированные item-эмбеддинги для fast cosine ANN.
      TowerConfig dataclass, evaluate() с Precision/Recall/NDCG@K, save/load.
      recsys/models/reranker.py: LLMReranker (retrieve→re-rank пайплайн).
      Claude Haiku переранжирует 2×top_k кандидатов с учётом профиля пользователя.
      Mock-режим без API-ключа — CI зелёный. Graceful degradation при ошибках LLM.
      72/72 тестов (+20 новых: TestTwoTowerModel, TestLLMReranker).
      Источник: RecSys 2025, Snap Robusta platform, Shopify NeurIPS 2025.
- [x] Multi-model cross-check для LLM Code Review (Project 08) — 2026-04-08
      review/models/multi_review.py: correctness_pass() + security_pass() (OWASP Top 10),
      SemgrepFinding dataclass для инъекции статического анализа в security pass,
      self_consistency_score() — эвристический CISC-скорер (0-1) без LLM-вызова.
      multi_model_review() — оркестратор: два прохода → дедупликация → verdict
      (pass/review_required/fail/api_key_missing). POST /review/multi endpoint.
      36/36 тестов (+22 новых: TestMultiModelReview).
      Источник: Ericsson 2025 (arxiv 2507.19115), Semgrep AI 2025, CISC ACL 2025.

**Iteration 16-20: Monitoring & CT**
- [x] Prometheus metrics exporter (Project 05) — 2026-04-09
      anomaly/metrics/prometheus_exporter.py: AnomalyMetrics class с отдельным
      CollectorRegistry (изолируется от глобального для тестов). 7 метрик:
      requests_total, points_total, anomalies_total (Counter), anomaly_score +
      detection_seconds (Histogram с SRE-buckets), threshold + window_size (Gauge).
      track_detection() context manager для latency измерений. get_summary() для /health.
      app.py: GET /metrics endpoint (PlainTextResponse, Prometheus text format),
      /health расширен полем prometheus + stats, /detect автоматически обновляет метрики.
      Graceful degradation: is_available() → работает без prometheus_client в CI.
      31/31 тестов (+12 новых: TestPrometheusExporter, TestAPIWithMetrics).
- [x] Automated retraining trigger (Project 01) — 2026-04-10
      churn/retraining/trigger.py: RetrainingTrigger, compute_psi() (PSI = BCBS 2011 стандарт).
      OR-логика: drift (max_psi >= 0.2) ИЛИ perf_degradation (AUC drop >= 0.05) → retrain.
      DriftReport + RetrainingResult dataclasses. Аудит-лог в MLflow для EU AI Act.
      Graceful degradation без MLflow в CI. 20/20 тестов (+18 новых: TestComputePSI,
      TestDriftReport, TestRetrainingTrigger).
- [x] Data drift alerting (Project 10 → 01) — 2026-04-13
      quality/alerts/alerting.py: DriftAlert dataclass, LogAlertChannel, WebhookAlertChannel,
      AlertManager (severity_threshold-фильтрация, изоляция ошибок каналов).
      quality/api/app.py: POST /drift/alert — drift detection + alerting в одном запросе.
      churn/api/app.py: POST /retraining/notify — принимает DriftAlertPayload от Project 10,
      OR-логика: critical → retrain; warning+PSI≥0.2 → retrain; иначе → skip.
      Аудит-трейл: triggered_by, reason с PSI и списком признаков.
      71 новый тест (TestDriftAlert×5, TestLogAlertChannel×2, TestWebhookAlertChannel×2,
      TestAlertManager×5, TestDriftAlertAPIEndpoint×2, TestRetrainingNotify×8).
- [x] Write-Audit-Publish drift gates в feature store (Projects 09/10) — 2026-04-14
      recsys/feature_store/wap.py: WAPGate class — write() → audit() → publish_or_quarantine().
      PSI-гейт без внешних зависимостей (numpy only), threshold=0.2 (BCBS-стандарт).
      AuditResult dataclass: status (published/quarantined/no_reference), psi, reason.
      Cold-start: первый батч без reference → auto-publishes и устанавливает reference.
      recsys/api/app.py: POST /features/wap — принимает батч фичей + опциональный reference,
      возвращает AuditResult с полным audit-trail (draft_id, psi, reason, timestamp).
      89/89 тестов зелёные (+17 новых: TestWAPGate×13, TestWAPAPIEndpoint×4).
      Источники: lakefs.io WAP pattern 2024, Dagster WAP 2025, BCBS PSI thresholds 2011.
- [x] MMD drift detection + retraining trigger (Project 05) — 2026-04-15
      anomaly/drift/mmd.py: compute_mmd_rbf() (RBF-ядро Гаусса, numpy-only, без alibi-detect),
      _median_heuristic_gamma() (Gretton 2012), bootstrap_mmd_threshold() (перестановочный
      bootstrap, контроль α), MMDDriftDetector (null-распределение → p-value), DriftResult
      (audit_id UUID + ISO 8601 timestamp, EU AI Act Article 9 compliance).
      anomaly/retraining/trigger.py: AnomalyRetrainingTrigger — MMD-дрейф → решение о
      переобучении, MLflow-аудит (graceful degradation), triggered_by audit-trail.
      anomaly/api/app.py: POST /drift/check (reference vs current MMD + bootstrap-порог),
      GET /drift/status (статус для Grafana/Prometheus alerting),
      GET /health расширен полем last_drift.
      57/57 тестов зелёные (+26 новых: TestMMDCore×9, TestMMDDriftDetector×7,
      TestAnomalyRetrainingTrigger×5, TestMMDDriftAPIEndpoint×5).
      Источник: Gretton et al. 2012 JMLR, Evidently AI v0.5+, EU AI Act Article 9.

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
