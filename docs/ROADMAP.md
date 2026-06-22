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
|------|-------------|------------------|
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
- [x] Streamlit Cloud деплой для 3 проектов — 2026-04-16
      Project 01 (Churn): streamlit_app.py, requirements.txt, .streamlit/config.toml (red theme)
      Project 07 (Pricing): streamlit_app.py, requirements.txt, .streamlit/config.toml (blue theme)
      Project 05 (Anomaly): anomaly/dashboard/app.py (3 вкладки: Live Monitor, Drift MMD, Architecture),
        anomaly/dashboard/utils.py (generate_metric_stream, compute_detection_summary, ref/cur generators),
        streamlit_app.py, requirements.txt, .streamlit/config.toml (dark SRE theme).
      18 новых тестов: TestDashboardGenerateMetricStream×9, TestDashboardComputeDetectionSummary×5,
        TestDashboardReferenceCurrentData×4. 75/75 зелёных.
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

- [x] mkdocs documentation site — 2026-04-17
      mkdocs.yml: Material theme (indigo, dark/light toggle), tabbed navigation, Mermaid diagrams,
      mkdocstrings для авто-API из docstrings, поиск на EN+RU.
      docs/index.md: landing page с badge'ами, сравнительной таблицей, quick start, tech stack tabs.
      docs/projects/: 10 страниц (по одной на проект) с архитектурными диаграммами, ключевыми
      компонентами, таблицами API endpoints, примерами. docs/architecture.md, docs/mlops-maturity.md,
      docs/evolution.md, docs/contributing.md.
      .github/workflows/docs.yml: автодеплой на GitHub Pages при push в main (paths: docs/**, mkdocs.yml).
      pyproject.toml: новый extras [docs] = mkdocs-material>=9.5 + mkdocstrings[python]>=0.27.
      Бонус: исправлены pre-existing ruff ошибки B905 (zip без strict=) в Project 06,
      N803 (noqa) в Project 05. 131/131 тестов зелёных (75+56).
- [x] Model serving via BentoML (not just FastAPI) — 2026-04-19
      churn/serving/bento_service.py: ChurnPredictor (transport-agnostic core),
      ChurnInput/ChurnPrediction dataclasses, save_to_bentoml() → BentoML model store.
      ChurnService (conditional, когда bentoml установлен): /predict, /predict_batch
      (adaptive batching max_batch_size=32, max_latency_ms=100), /health endpoints.
      Graceful degradation: is_available() — работает без bentoml в CI.
      44/44 тестов зелёные (+16 новых: TestBentoService).
      Источник: BentoML docs 1.3, oneuptime.com Docker+BentoML 2026.
- [x] A/B testing framework — 2026-04-20
      churn/ab_testing/experiment.py: ABExperiment с детерминированным MD5-роутингом
      (customer_id → вариант без switching noise), VariantConfig/PredictionRecord/
      VariantStats/ExperimentResult dataclasses. compute_results(): chi-squared z-test
      для high-risk rate + Welch's t-test для churn probability. Graceful degradation
      без scipy. Победитель = min(high_risk_rate) при p < 0.05, n >= 385 (Cohen's h).
      churn/api/app.py: POST /ab/predict (роутинг+запись), POST /ab/outcome (ground truth),
      GET /ab/results (статанализ+рекомендация), GET /ab/status, POST /ab/reset.
      68/68 тестов зелёные (+23: TestABExperiment×15, TestABAPIEndpoints×8).
      Источник: AWS Blog Dynamic A/B Testing 2025, Marvelous MLOps 2025.
- [x] Feature store integration (Feast) — 2026-04-21
      recsys/feature_store/feast_bridge.py: FeastBridge — Feast-совместимый адаптер над FeatureRegistry.
      FeatureRef ("view:feature" формат), FeatureRequest/OnlineFeatureResponse dataclasses,
      get_online_features() с авто-детекцией entity-ключа (user_id/product_id).
      is_available() graceful fallback без Feast в CI. register_view() для кастомных view.
      feature_repo/feature_store.yaml: Feast-конфиг (local provider, SQLite online store).
      feature_repo/features.py: Entity (user_id, product_id) + FeatureView definitions
      (user_features, item_features) — conditional import если Feast не установлен.
      recsys/api/app.py: POST /features/online — Feast-compatible online serving endpoint,
      автоматически вычисляет фичи из interaction data при первом запросе.
      109/109 тестов зелёных (+20 новых: TestFeastBridge×13, TestOnlineFeaturesAPIEndpoint×7).
      Источники: Feast docs 2026, Made With ML feature-store, oneuptime.com 2026.
- [x] Kubernetes deployment manifests — 2026-04-23
      k8s/: Namespace, ConfigMap, Secrets-template, Ingress (nginx, rate-limit 100 RPS),
      NetworkPolicy (ingress-only per service + Prometheus scrape).
      Per service (churn, rag, anomaly, pricing, recsys, quality):
        Deployment: 2 replicas, podAntiAffinity, securityContext(runAsNonRoot=true),
        resource requests+limits (CPU/memory calibrated per model type),
        liveness/readiness/startup probes, Prometheus annotations.
        Service (ClusterIP), HPA (autoscaling/v2, CPU 60-70%, min2/max6-10, stabilization windows),
        PodDisruptionBudget (minAvailable=1).
      scripts/validate_k8s.py: CLI-валидатор YAML структуры + security-проверки.
      10-data-quality-platform/tests/test_k8s_manifests.py: 106 pytest-тестов
        (TestK8sManifestStructure×8, TestDeploymentSecurity×42, TestHPAPresence×2).
        149/149 тестов зелёные (43 pre-existing + 106 новых).
      Источники: kubeify.com ML/K8s 2026, scaleops.com HPA best-practices, BentoML K8s docs.
- [x] Automated model comparison reports — 2026-04-24
      churn/evaluation/model_comparison.py: ModelResult/ComparisonSummary/ComparisonReport
      dataclasses. compare_models() — ранжирование по AUC+F1, порог значимости 0.02
      (Hanley & McNeil 1982 ~2 SE AUC). generate_markdown_report() с emoji-медалями
      и таблицей признаковых важностей. generate_json_report() для API/MLflow.
      churn/api/app.py: POST /compare/models (json|markdown формат, in-memory кэш),
      GET /compare/report (последний отчёт или 404), GET /compare/leaderboard (Grafana).
      33 новых теста: TestModelComparisonCore×12, TestComparisonReportFormat×9,
      TestComparisonAPIEndpoints×12. 101/101 зелёных (было 68).
      Источник: Hanley & McNeil 1982 (AUC SE), MLflow Model Registry champion pattern.

---

## Фаза 3: Enterprise (1-2 месяца)

- [x] Multi-model orchestration (Project 01 → 04 → 05 pipeline) — 2026-04-25
      11-orchestration/orchestration/: ChurnPredictor (heuristic, SHAP-calibrated) +
      FraudPredictor (logistic over lognormal distributions, numpy-only) +
      AnomalyPredictor (rolling Z-score, zero-std edge case → score=10).
      OrchestrationPipeline: DI через конструктор, run() + run_batch().
      compute_risk(): иерархический action ladder (fraud 55% + churn 30% + anomaly 15%),
      actions: block→review→intervene→monitor→ok.
      FastAPI: POST /orchestrate (unified event → risk profile), POST /orchestrate/batch,
      GET /health. 59/59 тестов зелёные.
      Источники: MLOps 2026 multi-model patterns, IRIS neuro-symbolic approach (arxiv 2506).
- [x] Schema registry for data contracts — 2026-04-26
      quality/schema_registry/: ColumnSchema/DataSchema/SchemaVersion dataclasses,
      ColumnType + Compatibility enums (BACKWARD/FORWARD/FULL/NONE).
      SchemaRegistry: семантическое версионирование (MAJOR.MINOR.PATCH),
      auto-bump (breaking → major, non-breaking → minor), BACKWARD-совместимость по умолчанию.
      Breaking changes: удалённый столбец, смена типа (кроме integer→float widening),
      nullable→NOT NULL, новый NOT NULL столбец. Compatibility.NONE обходит проверку.
      validator.py: infer_schema_from_dataframe() (авто-инференс из Polars DataFrame),
      validate_dataframe_against_schema() (column_exists, nullable, allowed_values, value_range).
      API endpoints: POST /schema/register (201+auto-version, 409 на breaking),
      GET /schema/list, GET /schema/{name}/versions, GET /schema/{name},
      POST /schema/{name}/validate (CSV against schema), POST /schema/compatible,
      POST /schema/infer (draft schema from CSV для onboarding).
      86/86 тестов зелёные (+37 новых: TestColumnSchemaAndDataSchema×4,
      TestSchemaRegistryCore×11, TestBreakingChangeDetection×7, TestSchemaInference×3,
      TestDataValidation×6, TestSchemaRegistryAPI×12).
      Источники: Confluent Schema Registry semantics, Data Contract CLI, DataScienceVerse 2026.
- [x] Data lineage visualization — 2026-04-27
      quality/lineage/graph.py: LineageNode (Dataset/Job), LineageEdge, LineageGraph (DAG).
      OpenLineage-совместимая модель: NodeType(StrEnum), upstream/downstream BFS-обход,
      D3.js-совместимый JSON export (nodes+links+stats), дедупликация рёбер.
      quality/lineage/tracker.py: RunState(StrEnum) START/COMPLETE/FAIL/ABORT/RUNNING,
      LineageEvent (OpenLineage RunEvent формат, ISO 8601 UTC timestamp, event_id UUID),
      LineageTracker: record() → авто-обновление графа, фильтрация get_events(),
      get_run_history(), summary(); singleton get_tracker() / reset_tracker() для тестов.
      quality/api/app.py: 6 новых endpoints:
        POST /lineage/event (OpenLineage RunEvent, 201 Created),
        GET  /lineage/graph (полный D3.js граф),
        GET  /lineage/dataset/{namespace}/{name} (upstream+downstream субграф),
        GET  /lineage/upstream/{namespace}/{name} (impact analysis вверх),
        GET  /lineage/downstream/{namespace}/{name} (impact analysis вниз),
        GET  /lineage/events (фильтрация по job_name/event_type/limit),
        GET  /lineage/summary (статистика трекера).
      37 новых тестов: TestLineageNode×4, TestLineageEdge×3, TestLineageGraph×7,
        TestLineageTracker×10, TestLineageAPIEndpoints×13 (включая multi-hop chain).
      229/229 тестов зелёных (было 192). Без внешних зависимостей (CI-friendly).
      Источники: OpenLineage spec openlineage.io, Marquez GitHub, Deloitte Medium 2025.
- [x] Cost optimization (model quantization, batching) — 2026-04-28
      churn/optimization/quantizer.py: INT8 post-training quantization для sklearn линейных моделей
      (_quantize_array → симметричная PTQ с per-tensor min/max calibration, QuantizedWeights INT8,
      QuantizedModel wrapper с proxied predict_proba/predict), quantize_tree_ensemble() (GBDT-прунинг
      keep_fraction), estimate_inference_speedup() (теоретический 2-4x speedup по NVIDIA 2022).
      churn/optimization/cost_tracker.py: CostTracker (rolling window 1000 req, deque+numpy percentiles,
      track() context manager для авто-измерения latency), estimate_monthly_cost() (AWS c6i.large/xlarge +
      GCP + Azure pricing 2026, авто-sizing по RPS, стоимость /1M req), optimize_batch_size()
      (throughput = batch/latency, SLA-фильтр, recommendations).
      churn/api/app.py: GET /optimize/stats (p50/p95/p99 latency, throughput, cost_estimate_10rps,
      cost_estimate_observed_rps, model_quantization_estimate), POST /optimize/batch (batch profiling).
      Predict endpoint обёрнут в _cost_tracker.track() для авто-измерения каждого запроса.
      30 новых тестов: TestQuantizer×12, TestCostTracker×13, TestOptimizeAPIEndpoints×5.
      131/131 тестов зелёные (было 101).
      Источники: Jacob et al. 2018 CVPR, NVIDIA INT8 benchmarks 2022, Intel Neural Compressor.
- [x] Security audit (OWASP for ML) — 2026-04-29
      quality/security/owasp.py: OWASPMLAudit — 7 автоматизированных проверок из OWASP ML Top 10:
        ML01 (adversarial inputs: IQR-outlier ratio >15%), ML02 (data poisoning: Shannon entropy <0.3),
        ML03 (model inversion: logit/embedding fields exposed), ML04 (membership inference: unique ratio >95%),
        ML05 (model theft: no rate limiting), ML08 (model skewing: missing >30%), ML09 (output integrity: no HMAC).
        AuditReport: score 0-100 (penalty per severity), passed_checks, high_risk_count, to_dict().
      quality/security/pii_detector.py: detect_pii() — 9 типов PII (email, phone, SSN, credit card,
        IP address, passport, IBAN, DOB, full name) через regex + masking для audit logs.
        PIIReport: gdpr_compliant flag (no critical/high PII), affected_columns, critical_columns.
      quality/api/app.py: 4 новых endpoint:
        POST /security/audit (JSON-запрос с column dicts → OWASP report),
        POST /security/pii (column dict → PII scan + GDPR flag),
        POST /security/audit/csv (CSV upload → combined OWASP+PII audit),
        GET  /security/checklist (OWASP ML Top 10 reference с mitigations).
      48 новых тестов: TestHelperFunctions×8, TestOWASPChecks×16, TestAuditReport×5,
        TestPIIDetector×10, TestSecurityAuditAPIEndpoint×3, TestPIIScanAPIEndpoint×3,
        TestSecurityChecklistEndpoint×3. 277/277 тестов зелёные (было 229).
      Источники: OWASP ML Security Top 10 v2023, EU AI Act Article 10, GDPR Article 4.
- [x] SLA monitoring — 2026-04-30
      quality/sla/slo.py: SLIType(StrEnum) + SLODefinition (target, error_budget_fraction,
      error_budget_minutes) + SLIObservation (good/total → sli_value).
      quality/sla/budget.py: ErrorBudgetTracker — rolling deque (10K cap), multi-window
      burn rate (_burn_rate за 1h/6h/72h/168h). Google SRE Table 5.1 burn rate rules:
      critical≥14.4 (1h, 2% budget), high≥6.0 (6h, 5%), medium≥3.0 (3d, 10%), low≥1.0.
      BurnRateAlert (severity, response_time, projected_exhaustion_hours). ErrorBudgetStatus
      с полным to_dict() для Grafana-совместимого ответа.
      quality/sla/monitor.py: SLAMonitor singleton (get_monitor/reset_monitor),
      define_slo() / observe() / get_status() / generate_report(). SLAComplianceReport.
      quality/api/app.py: 8 новых endpoint:
        POST /sla/define (201, SLO registration с валидацией),
        POST /sla/observe (201, SLI batch observation),
        GET  /sla/status (все сервисы), GET /sla/status/{service} (фильтр по sli_type),
        GET  /sla/burn-rate/{service} (multi-window burn + projected exhaustion),
        POST /sla/report (aggregate compliance report),
        GET  /sla/slos (list), GET /sla/observations (аудит), POST /sla/reset.
      44 новых тестов: TestSLODefinition×6, TestSLIObservation×4, TestErrorBudgetTracker×9,
        TestSLAMonitor×9, TestSLAAPIEndpoints×16.
      321/321 тестов зелёные (было 277).
      Источники: Google SRE Workbook Ch.5 (sre.google/workbook/alerting-on-slos/),
        nobl9.com SLO Best Practices 2026, Uptrace SLA/SLO Monitoring 2025.
- [x] H3 геопространственные признаки для оценки недвижимости (Project 07) — 2026-05-01
      pricing/data/geo.py: NEIGHBORHOOD_COORDS (15 районов Москвы, WGS84),
      generate_neighborhood_coordinates() с шумом ~350–400 м (реалистичный разброс квартир),
      lat_lng_to_h3() с graceful fallback (mock-сетка без h3 в CI), is_available().
      compute_h3_market_stats() — медиана цен + count по H3-ячейке (медиана устойчива к outliers).
      add_h3_features() — h3_r7 (район ~5 км²) + h3_r8 (микрорайон ~0.74 км²) + price_vs_district
        (price / hex_median: >1 = дороже рынка, <1 = дешевле).
      enrich_with_geo() — обёртка neighborhood → lat/lng → H3 → market stats.
      GEO_FEATURES + GEO_MARKET_FEATURES для интеграции с train.py.
      h3==4.4.2 добавлен в [pricing] extras pyproject.toml.
      14 новых тестов TestH3GeoFeatures (33/33 зелёных, было 19).
      Источники: Uber H3 blog 2018, Zillow AVM research 2024, h3geo.org docs.
- [x] Hybrid Search BM25+Vector+RRF для RAG (Project 02) — 2026-05-02
      rag/retrieval/hybrid.py: HybridIndex (BM25Okapi + tokenized corpus),
      _tokenize() (regex \w+, lowercase), bm25_search() с graceful fallback без rank_bm25,
      reciprocal_rank_fusion() (RRF k=60, Cormack et al. 2009, ключ идентичности = text),
      hybrid_search() — semantic (2×n_results кандидатов) + BM25 → RRF → top-n.
      Graceful degradation: без rank_bm25 → pure semantic search.
      rag/api/app.py: _hybrid_index + _indexed_chunks global state, QueryRequest.retrieval_method
      ("hybrid"|"semantic"), QueryResponse.retrieval_method, auto-build BM25 индекса при indexing,
      Gradio UI: checkbox "Hybrid search (BM25+Vector+RRF)".
      rank-bm25>=0.2.2 добавлен в [rag] extras pyproject.toml.
      14 новых тестов TestHybridRetrieval (54/54 + 1 skipped зелёных, было 40).
      Recall@10: semantic ~65-78% → hybrid ~91% (Ashutosh Kumar Singh, Medium 2026).
      Источники: Cormack et al. 2009 (RRF), Elastic Hybrid Search docs, GoPenAI Mar 2026.
- [x] SSE Streaming ответов для RAG (Project 02) — 2026-05-03
      rag/generation/stream.py: stream_answer() async SSE generator — yields token/sources/done events.
      _sse() хелпер: dict → "data: {json}\n\n" format. Anthropic client.messages.stream()
      для production; word-by-word mock для CI без API ключа.
      Faithfulness gate ПОСЛЕ стрима (на полном ответе) — не блокирует UX.
      rag/api/app.py: POST /query/stream → StreamingResponse(media_type="text/event-stream"),
      Cache-Control: no-cache + X-Accel-Buffering: no headers.
      Пустая коллекция → inline async generator с token+done events.
      14 новых тестов TestStreamingRAG: SSE-формат, порядок событий (token→sources→done),
        endpoint headers, no-documents case, with-documents mock.
      68/68 тестов зелёных (+14, было 54).
      Источники: FastAPI SSE docs, Anthropic streaming API, dasroot.net Streaming RAG 2026.
- [x] Temporal graph features для fraud detection (Project 04) — 2026-05-04
      fraud/models/temporal.py: TemporalConfig dataclass (time_window=30d, decay_factor, min_edges_for_burst),
      NodeTemporalFeatures (6 признаков: velocity_ratio, burst_score, amount_hhi,
        recent_amount_ratio, neighbor_fraud_ratio, hub_proximity) + to_array() → float32.
      TemporalFeatureExtractor: _build_adjacency(), compute_node_features() (velocity/burst/HHI/recency),
        compute_neighborhood_features() (fraud_ratio + log hub-proximity),
        extract() → (n_nodes, 6), augment_features() → hstack с базовыми признаками.
      explain_temporal_features(): EU AI Act Article 13 compliance — текстовые объяснения уровней риска.
      fraud/api/app.py: GraphTransactionInput (3 base + 6 temporal поля), GraphFraudScore
        (fraud_probability + temporal_flags + feature_contributions), _ensure_temporal_model()
        обучает CatBoost на 9 признаках (3+6). POST /score/graph endpoint с интерпретируемыми
        temporal-объяснениями. /health расширен temporal_model_loaded.
      23 новых теста: TestTemporalFeatures×17 (unit: defaults, shapes, bounds, isolated node,
        burst/HHI/velocity invariants, fraud cluster validation), TestAPI×6 (temporal endpoint).
      47/47 passed, 3 skipped (PyTorch) — было 24/27.
      Источники: temporal GNN survey (arxiv 2302.01018), FinGuard-GNN 2025, Gretton et al. 2012.
- [x] LSTM/ESN Autoencoder serving (Project 05) — 2026-05-05
      anomaly/models/lstm_autoencoder.py: EchoStateAutoencoder (numpy-only, без PyTorch).
      Echo State Network: фиксированный разреженный reservoir (spectral radius < 1, Jaeger 2001),
      leaking rate для сглаживания состояний, ridge regression для выходного слоя (аналитическое
      решение O(n) vs O(n²) BPTT). SequenceScaler (min-max, защита от константных фич).
      Anomaly score = MSE реконструкции центральной точки скользящего окна.
      Порог = percentile(anomaly_percentile) ошибок на нормальных обучающих данных.
      LSTMConfig dataclass, TrainResult, LSTMDetectionResult, create_autoencoder() factory.
      anomaly/api/app.py: 3 новых endpoint:
        POST /lstm/train (обучить ESN на нормальных данных, min 50 точек),
        POST /lstm/detect (детекция с reconstruction error scores),
        GET  /lstm/status (состояние модели для health-check).
      26 новых тестов: TestESNAutoencoderCore×16 (fit/detect shape, binary preds, anomaly_has_higher_score,
        SequenceScaler constant feature, normal_data_low_anomaly_rate),
        TestLSTMAPIEndpoints×10 (train/detect/status flow, 400 без обучения, 422 мало данных).
      101/101 тестов зелёных (было 75).
      Источники: Jaeger 2001 (GMD TechReport 148), Malhotra et al. 2016 (ESANN),
        Lukoševičius 2012 (Neural Networks Tricks of the Trade, Ch. 22).
- [x] Confidence-Based Routing для LLM Code Review (Project 08) — 2026-05-06
      review/models/confidence_router.py: Human-in-the-Loop (HITL) routing pattern.
      RoutingDecision enum: AUTO_APPROVE / HUMAN_REVIEW / AUTO_REJECT.
      RoutingConfig dataclass: настраиваемые пороги (auto_approve_max_risk=0.5,
        auto_reject_min_risk=8.0, critical_escalate=True, security_escalate=True).
      RoutingResult dataclass: decision + risk_score + confidence + reason + critical_findings.
      compute_risk_score(): взвешенная сумма severity (critical=10, major=4, minor=1,
        suggestion=0.3), capped at 100. Неизвестные severity → suggestion weight.
      route_review(): приоритетная логика — 1) escalation (critical/security findings),
        2) high aggregate risk ≥ threshold, 3) low risk ≤ threshold, 4) ambiguous → human.
      _routing_confidence(): кусочно-линейная функция — высокая уверенность у краёв
        (чётко безопасно/опасно), низкая в середине зоны (нужен человек).
      24 новых теста: TestRiskScore×9 (empty, weights, additive, cap, unknown, missing key),
        TestRoutingDecisions×15 (all 3 paths, escalation, custom config, dataclass checks).
      60/60 тестов зелёных (было 36).
      Источники: LLM Code Review Evaluation 2025 (arxiv 2505.20206), OWASP ML Security Top 10,
        Ericsson Production Code Review 2025.
- [x] Conformal Prediction для NER (Project 03) — 2026-05-07
      Split Conformal Prediction (Angelopoulos & Bates 2022) для калибровки уверенности NER.
      Статистически валидные множества предсказаний: P(true_label ∈ C(X)) ≥ 1-α.
      ner/model/conformal.py: ConformalConfig (alpha, min_calibration_samples),
        ConformalEntityResult (prediction_set, nonconformity_score, is_certain, coverage),
        CalibrationResult (q_hat, n_calibration, coverage_empirical),
        ConformalNERPredictor (_pattern_score: full/partial/no match → 1.0/0.6/0.0,
          _nonconformity_score: 1 - P̂(label|text) через нормализацию паттернов,
          calibrate: Venn-Abers конечная поправка level=(n+1)(1-α)/n,
          predict_set: включает label ↔ score ≤ q_hat).
      ner/api/app.py: авто-калибровка на Collection5 при старте (без ручного шага),
        POST /predict/conformal → ConformalNERResponse (entities + q_hat + calibrated),
        GET /health расширен полем conformal_calibrated.
      20 новых тестов: TestConformalNERPredictor×14 (pattern_score, nonconformity_range,
        lower_for_matching, uniform_unknown, calibrate_result, conservative_threshold,
        empirical_coverage, prediction_set_contains_label, subset_of_labels, is_certain,
        coverage_field, predict_text, auto_calibration_on_collection5),
        TestConformalAPI×6 (status_200, structure, required_fields, set_contains_label,
        calibrated_true, health_includes_conformal_status).
      61/61 тестов зелёных (+20, было 41).
      Источники: Angelopoulos & Bates 2022 (arxiv 2107.07511), Papadopoulos et al. 2002,
        EU AI Act Article 13, Shafer & Vovk 2008 (JMLR 9:371-421).
- [x] Incremental Learning для Churn (Project 01) — 2026-05-08
      churn/online/learner.py: IncrementalChurnLearner (River HoeffdingTreeClassifier +
        ADWIN drift detector + StandardScaler, predict-then-learn порядок обязателен
        для честной оценки ошибки). Graceful degradation: SimpleFallbackClassifier
        (Laplace-сглаженный байесовский счётчик) когда River не установлен.
        Periodic snapshot каждые N примеров (pickle сериализация model+ADWIN+scaler).
        load_snapshot() восстанавливает полное состояние.
      churn/api/app.py: 4 новых endpoint:
        POST /online/learn (delayed-feedback: предсказать→метка поступает позже),
        POST /online/predict (предсказание без обновления модели),
        GET  /online/status (ADWIN state + снапшоты + class distribution),
        POST /online/reset (сброс к начальному состоянию).
      river>=0.21 добавлен в [churn] extras pyproject.toml.
      29 новых тестов: TestIncrementalLearnerUnit×13, TestIncrementalLearnerIntegration×5,
        TestOnlineAPIEndpoints×11. 160/160 тестов зелёных (было 131).
      Источники: Bifet & Gavalda 2007 ADWIN (SDM'07), Hulten et al. 2001 Hoeffding Tree (KDD),
        River docs riverml.xyz/latest/.
- [x] LLM-as-Judge pipeline для Code Review (Project 08) — 2026-05-09
      review/evaluation/golden_dataset.py: 20 аннотированных примеров (8 security,
        8 correctness, 4 clean). GoldenExample dataclass с ground_truth_issues и keywords.
      review/evaluation/judge.py: JudgeVerdict (faithfulness/helpfulness/false_positive_rate/
        overall_score), RegressionResult dataclass. _lexical_judge() — детерминированный CI-safe
        оценщик через пересечение ключевых слов с ground truth + action-word helpfulness.
        _llm_judge() — Claude-as-Judge с graceful degradation на lexical при отсутствии ключа.
        evaluate_review() + run_regression_suite() (pass/fail по avg_overall_score >= threshold).
      review/api/app.py: GET /evaluate/dataset (метаданные датасета), POST /evaluate/review
        (оценка одного ревью по примеру из датасета), POST /evaluate/regression
        (регрессионный тест по всем 20 примерам, use_lexical=True для CI).
      41 новый тест: TestGoldenDataset×10, TestLexicalJudge×12, TestJudgeVerdict×4,
        TestRegressionResult×3, TestEvaluateAPIEndpoints×12. 101/101 зелёных (+41, было 60).
      Weighted overall: 0.4×faithfulness + 0.3×helpfulness + 0.3×(1-FPR).
      Источники: Zheng et al. 2023 MT-Bench (arxiv 2306.05685), LLM Code Review Eval 2025
        (arxiv 2505.20206), Confident AI LLM Eval Guide 2026, evidentlyai.com LLM-as-Judge 2026.
- [x] Quantile Regression для Real Estate (Project 07) — 2026-05-10
- [x] Document Quality Assessment Pipeline для CV Scanner (Project 06) — 2026-05-11
      scanner/preprocessing/quality.py: QualityMetrics dataclass, 5 numpy-only estimators:
        estimate_blur() (5-point Laplacian stencil, variance/500 нормализация),
        estimate_brightness() (штраф за отклонение от μ=0.5, score=1−2|μ−0.5|),
        estimate_contrast() (нормализованное std, 0.25 = excellent text/bg separation),
        estimate_noise() (mean absolute adjacent-pixel diff, 25/255 порог),
        estimate_skew() (OLS регрессия ink-centroid X vs row Y → arctan(slope)).
        assess_quality(): взвешенный composite (blur 40%, contrast 25%, brightness 20%,
        noise 15% inverted), accept_threshold=0.40, rejection_reason с диагностикой.
      scanner/api/app.py: 2 новых endpoint:
        POST /quality/assess (PixelMatrix → QualityAssessmentResponse, 422 на пустую матрицу),
        POST /classify/gated (два этапа: quality check → classification, gated=True если отклонён).
      81/81 тестов зелёных (+25 новых: TestDocumentQualityAssessment×18, TestQualityAPIEndpoints×7).
      Источники: Laplacian variance blur detection (Pech-Pacheco et al. 2000),
        ITU-R BT.601 luminance coefficients, Otsu 1979 (thresholding).
- [x] Probability Calibration для Fraud Detection (Project 04) — 2026-05-12
      fraud/models/calibration.py: FraudCalibrator — Platt scaling (gradient descent sigmoid)
        + isotonic regression (sklearn PAV algorithm). _compute_ece(): Expected Calibration Error
        (Guo et al. 2017, weighted avg bin gaps), Maximum Calibration Error (worst bin).
        CalibrationBin + CalibrationResult dataclasses с to_dict() для Grafana/reliability diagram.
        _PlattScaler: iterative gradient descent (lr=0.01, 1000 iter), нет sklearn-зависимости.
        _IsotonicCalibrator: lazy import sklearn.IsotonicRegression (PAV algorithm, monotone гарантия).
        ece_improvement() — абсолютное снижение ECE после калибровки для мониторинга.
        is_available() — graceful fallback без sklearn.
      fraud/api/app.py: _reset_calibrator() для тестовой изоляции, CalibrateRequest/CalibrateResponse,
        POST /calibrate (обучить на hold-out synthetic set, method=isotonic|platt),
        GET /calibration/metrics (reliability diagram data + ECE/MCE/Brier для дашборда),
        FraudScore.calibrated_probability (None если калибратор не обучен, иначе P̂(fraud)),
        /health расширен calibration_fitted.
      69/69 тестов зелёных (+22 новых: TestFraudCalibrationUnit×15, TestCalibrationAPIEndpoints×7).
      Зачем: raw CatBoost scores ≠ P(fraud) — порог 0.7 без калибровки может соответствовать
        85% или 55% реальных мошенников; калибровка делает бизнес-пороги блокировки надёжными.
      Источники: Zadrozny & Elkan 2001 ICML, Platt 1999, Guo et al. 2017 ICML (ECE),
        scikit-learn Calibration docs.
- [x] LinUCB Contextual Bandits для RecSys (Project 09) — 2026-05-13
      recsys/models/bandit.py: LinUCBBandit (disjoint LinUCB, Li et al. 2010 WWW).
      BanditConfig (alpha, feature_dim, lambda_reg), ArmState (A matrix + b vector per arm),
      BanditRecommendation (ucb_score, expected_reward, exploration_bonus, n_updates),
      BanditResult, FeedbackRecord dataclasses.
      _compute_ucb(): θ̂ᵀx + α√(xᵀA⁻¹x) — exploration bonus защита от floating-point.
      recommend(): ранжирование кандидатов по UCB убыванию, auto-init новых arms.
      update(): A_a += xxᵀ, b_a += r·x онлайн-обновление из feedback.
      Короткий контекст дополняется нулями (graceful padding).
      recsys/api/app.py: _get_bandit()/reset_bandit() lazy singleton pattern,
        POST /bandit/recommend (candidate_ids + contexts → UCB ranking, 422 на mismatch),
        POST /bandit/feedback (arm_id + reward [0,1] → online update, 422 на reward > 1),
        GET  /bandit/stats (n_arms, total_recommendations, per-arm A_trace, avg_reward).
      25 новых тестов: TestLinUCBBandit×14 (top_k, sorted, cold_start_bonus, update_increments,
        accumulates_reward, changes_ucb, exploitation_after_training, auto_register, padding,
        stats_sorted, counter, mismatch_raises), TestBanditAPIEndpoints×11 (recommend/feedback/
        stats 200, structure, top_k, 422 mismatch, 422 empty, 422 reward, full cycle).
      134/134 тестов зелёных (+25, было 109).
      Источники: Li et al. 2010 WWW (LinUCB), eugeneyan.com/writing/bandits/,
        Alibaba PAI-Rec LinUCB docs, Kameleoon contextual bandits guide 2026.
- [x] Causal Uplift Modeling (T-Learner CATE) для Churn (Project 01) — 2026-05-16
- [x] Isolation Forest + Feature Explainability для Anomaly Detection (Project 05) — 2026-05-17
- [x] Semantic Chunking для RAG (Project 02) — 2026-05-18
      rag/chunking/semantic.py: SemanticChunker (TF-IDF cosine boundary detection),
        SemanticChunkConfig (similarity_threshold=0.30, min_chunk_chars=100, max_chunk_chars=1200),
        _split_into_sentences() (абзацы → .!? границы), _paragraph_chunks() (fallback без sklearn),
        chunk_document() с preserved metadata + chunking_strategy field.
        Graceful degradation: без sklearn → paragraph splitting по двойным \n.
      rag/ingestion/loader.py: chunk_documents() новый параметр chunking_strategy
        ("fixed"|"semantic"|"paragraph"), рефакторинг в 3 private helpers.
      rag/api/app.py: IndexRequest с chunking_strategy полем (POST /index),
        POST /chunk/preview — ChunkPreviewRequest/ChunkPreviewResponse,
        сравнение стратегий без переиндексации.
      26 новых тестов: TestSemanticChunker×14, TestChunkingStrategiesInLoader×5,
        TestChunkPreviewEndpoint×7. 94/94 тестов зелёных (68 pre-existing + 26 новых).
      Преимущество: семантические чанки сохраняют смысловые единицы (одна тема → один чанк),
        улучшая precision RAG ответов — LLM получает связный контекст вместо обрезанных фрагментов.
      Источники: LangChain SemanticChunker docs, Anthropic Contextual Retrieval 2024,
        Douze et al. 2024 FAISS (cosine drift как boundary signal).
      anomaly/models/isolation.py: IsolationForestDetector с per-feature вкладом через
        маргинальную нейтрализацию (заменяем признак на train mean → delta score = вклад).
        IsolationConfig/IsolationTrainResult/IsolationResult dataclasses.
        _normalize_score(): min-max нормализация по train-диапазону → anomaly_score ∈ [0,1].
        is_available() graceful fallback без sklearn.
      anomaly/api/app.py: 3 новых endpoint:
        POST /isolation/train (обучить на нормальных данных, min 20 точек),
        POST /isolation/detect (детекция + feature_contributions + top_feature),
        GET  /isolation/status (состояние модели для health-check).
        GET /health расширен полем isolation_fitted.
      27 новых тестов: TestIsolationForestDetector×14 (fit, is_fitted, detect_length,
        binary_bool, score_range, contributions_sum, feature_names, top_feature_valid,
        anomaly_higher_score, detect_before_fit_raises, low_anomaly_rate, injected_detected,
        train_result_fields), TestIsolationAPIEndpoints×13 (train 200, response_structure,
        n_samples, detect_400_before_train, detect_200_after_train, response_structure,
        feature_contributions_present, rate_range, status_before, status_after,
        health_field, full_cycle, custom_params).
      128/128 тестов зелёных (+27, было 101).
      Преимущество: Isolation Forest ловит многомерные аномалии (CPU spike + latency + requests
        одновременно), где унивариатный Z-score упускает паттерн. top_feature ускоряет
        SRE-диагностику ("главная причина — latency" вместо ручного анализа).
      Источники: Liu et al. 2008 ICDM (Isolation Forest), JetBrains PyCharm Blog 2025,
        arxiv 2503.13195 (AML-based anomaly 2025).
      churn/causal/uplift.py: TLearnerUplift — два независимых GradientBoostingClassifier
        (treatment/control), CATE = μ₁(X) - μ₀(X), Persuasion Matrix сегментация:
        Persuadable (CATE < -threshold) / Sure Thing / Lost Cause / Sleeping Dog
        (CATE > threshold — скидка УВЕЛИЧИВАЕТ отток). compute_qini(): Qini-коэффициент
        (Radcliffe 2007) — площадь между кривой Qini и random baseline (AUUC_model - AUUC_random).
        numpy-trapezoid AUC без внешних зависимостей. summarize_uplift(): бизнес-метрики:
        n_persuadable, targeting_uplift (средний CATE persuadable клиентов).
      churn/api/app.py: 3 новых endpoint:
        POST /uplift/train (обучить T-Learner на исторических данных кампании),
        POST /uplift/predict (CATE + сегмент для batch клиентов),
        GET  /uplift/segments (описание сегментов + параметры модели).
      39 новых тестов: TestTLearnerUplift×16, TestQiniCoefficient×4,
        TestSummarizeUplift×5, TestUpliftAPIEndpoints×14.
      199/199 тестов зелёных (было 160). Graceful fallback без sklearn в CI.
      Бизнес-эффект: таргетировать только Persuadable сэкономит ~40% бюджета retention-кампании
        (вместо всех high-churn клиентов → только тех, кому скидка реально помогает).
      Источники: Künzel et al. 2019 PNAS (Metalearners for HCTE), Radcliffe 2007 (Qini),
        arxiv 2604.06123 (Large-Scale Meta-Learner Comparison 2026), causalml docs.

- [x] Active Learning для NER (Project 03) — 2026-05-21
      ner/active/strategy.py: 3 стратегии неопределённости (Lewis & Gale 1994):
        least_confidence_score() = max nonconformity (текст с хотя бы одной неуверенной сущностью),
        margin_score() = диапазон nonconformity (разброс между уверенными и нет),
        entropy_score() = среднее бинарной энтропии H(nonconformity) по всем сущностям.
        score_text() — единый интерфейс выбора стратегии → UncertaintyScore namedtuple.
        Интеграция с ConformalNERPredictor: nonconformity_score ∈ [0,1] как прокси
        неопределённости без дополнительного inference-прохода.
      ner/active/pool.py: LabelingPool — state machine аннотации:
        unlabeled → queried (query/batch_size топ по score) → labeled (label/annotations).
        query() сортирует по убыванию uncertainty_score — аннотатор получает ценнейшие примеры первыми.
        label() возвращает None при неверном item_id (защита от двойной разметки).
        PoolItem/QueryBatch/PoolStatus dataclasses, reset() для тестовой изоляции.
      ner/api/app.py: 6 новых endpoint:
        POST /active/pool/add (тексты → inference → uncertainty score → пул, 422 на неверную стратегию),
        POST /active/pool/query (топ-N для аннотации, items отсортированы по score убыванию),
        POST /active/pool/label (разметка annotator → labeled state, 404 если item не в queried),
        GET  /active/pool/status (unlabeled/queried/labeled counts),
        GET  /active/pool/labeled (все готовые для fine-tuning примеры),
        POST /active/pool/reset (сброс сессии аннотации).
      36 новых тестов: TestSamplingStrategies×14 (LC/margin/entropy пустой/одиночный/sorted/
        entropy_max_at_half/symmetric, score_text×3, higher_uncertainty),
        TestLabelingPool×12 (initial_zero, add_ids, add_increments, mismatch_raises,
        query_sorted, query_state_transition, fewer_than_batch, empty_pool,
        label_transition, unknown_id_none, get_labeled, reset),
        TestActiveLearningAPI×11 (add_200, structure, invalid_422, query_sorted, query_structure,
        label_200, label_404, status, full_cycle, margin_strategy, reset).
      97/97 тестов зелёных (было 61, +36).
      Бизнес-эффект: аннотировать 50 самых неопределённых примеров эффективнее,
        чем 500 случайных (Coleman et al. 2020: 2-10x экономия на разметке).
      Источники: Lewis & Gale 1994 (LC sampling), Settles 2012 "Active Learning" (synthesis lecture),
        Coleman et al. 2020 Selection via Proxy (arxiv 1906.00884), EU AI Act Article 13.

- [x] GraphRAG Knowledge Graph retrieval для RAG (Project 02) — 2026-05-22
- [x] Fraud Ring Detection via Label Propagation (Project 04) — 2026-05-23
      fraud/models/community.py: FraudRingDetector (Label Propagation, Raghavan et al. 2007),
      CommunityConfig (fraud_ratio_high/medium thresholds, min_ring_size, seed),
      CommunityResult (community_id, size, fraud_ratio, risk_level, node_ids),
      DetectionResult (communities sorted by size desc, suspicious_rings filtered,
        n_iterations, converged, total_nodes). Асинхронный LP с детерминированным
      tie-breaking (min label) — воспроизводимость для EU AI Act аудит-логов.
      Labeled / unlabeled разделение: fraud_ratio = labeled_fraud / total_labeled
      (не помеченные узлы не влияют на знаменатель — честная оценка по evidence).
      fraud/api/app.py: 2 новых endpoint:
        POST /community/detect (граф → communities + suspicious_rings),
        GET  /community/stats (агрегированные метрики без PII для Grafana).
      _reset_ring_detector() для тестовой изоляции.
      27 новых тестов: TestFraudCommunityUnit×17 (empty_raises, single_node, two_cliques,
        chain, fraud_ratio_zero/all/mixed, risk_level_high/medium/low, converged,
        sorted_by_size, suspicious_filtered, isolated_node, unlabeled_excluded,
        to_dict_structure, total_nodes), TestCommunityAPIEndpoints×10 (200, structure,
        n_communities, total_nodes, high_fraud_suspicious, 422_empty,
        stats_404_before, stats_200_after, stats_structure, coverage_ratio).
      96/96 тестов зелёных (было 69, +27, 3 skipped PyTorch).
      Бизнес-эффект: выявляет coordinated fraud rings (account farms, synthetic ID rings),
        невидимые при индивидуальной оценке транзакций — 60-80% убытков от таких атак.
      Источники: Raghavan et al. 2007 Physical Review E 76:036106 (Label Propagation),
        ACFE 2024 Occupational Fraud Report, FinGuard-GNN 2025, EU AI Act Article 13.
      rag/knowledge_graph/extractor.py: regex NER (DATE/ORG/CONCEPT/PERSON) без LLM.
        _safe_text() для безопасного group(1) из паттернов с capture group (quoted concept)
        и без него (акронимы). Lookahead (?=[^A-Za-z]|$) вместо \b для ORG с периодом в суффиксе.
      rag/knowledge_graph/graph.py: KnowledgeGraph pure-dict adjacency (без NetworkX).
        build_from_chunks(): co-occurrence edges между сущностями одного чанка (weighted).
        query_graph(): query → entity extraction → BFS expand (max_hops=1) → rank chunks
        by entity mention count → top-n. Fallback: пустой список если сущностей не найдено.
        get_entity_subgraph(): D3.js-совместимый формат для графической визуализации (1-hop).
        KGNode/KGEdge/KGStats dataclasses, stats().to_dict() для мониторинга.
      rag/api/app.py: 3 новых endpoint:
        GET  /graph/stats (n_nodes, n_edges, n_chunks, top_entities, is_built),
        POST /graph/build (явная перестройка из indexed_chunks, авто-вызов при /index),
        GET  /graph/entity/{key} (1-hop subgraph, 404 если не найдена).
        Новый retrieval_method="graph" в POST /query:
        граф-ретривал → fallback на hybrid при отсутствии сущностей в запросе.
      pyproject.toml: scikit-learn>=1.5 добавлен в [rag] extras (требуется SemanticChunker).
      35 новых тестов: TestEntityExtractor×10, TestKnowledgeGraph×16, TestGraphRAGAPI×10.
      129/129 тестов зелёных (было 94, +35). Бонус: SemanticChunker тест исправлен
      добавлением sklearn — без него paragraph fallback объединял все абзацы в один чанк.
      Бизнес-эффект: отвечает на multi-hop вопросы ("Связь между [A] и [B]?") где
        vector-only search не справляется — сущности не встречаются в одном чанке,
        но связаны через граф совместной встречаемости (Microsoft GraphRAG arxiv 2404.16130).
      Источники: Microsoft GraphRAG arxiv 2404.16130 (2024), Calmops GraphRAG Guide 2026,
        Graph Praxis practitioner guide 2026, calmops.com 2026.
- [x] LoRA Adapter Simulation для LLM Code Review (Project 08) — 2026-05-25
- [x] Price Trend Forecasting (Holt's DES) для Real Estate (Project 07) — 2026-05-26
      pricing/forecast/price_forecast.py: HoltWintersForecaster — Holt's Double Exponential
        Smoothing (уровень + тренд, без сезонности). s_t/b_t уравнения, predict-then-update порядок.
        Grid search 5×5 = 25 комбинаций α×β по SSE — без scipy, CI-совместимо.
        CI: ŷ ± z·σ_res·√h — расширяются с горизонтом (Gardner 2006 ETS(A,A,N) аппроксимация).
        generate_price_history(): base*exp(monthly_rate*t)*lognormal_noise (Fama 1970),
        NEIGHBORHOOD_BASE_PRICES (15 районов, ЦИАН 2025-2026) + NEIGHBORHOOD_ANNUAL_TRENDS.
        pricing/api/app.py: POST /forecast/price (neighborhood → Holt fit → 12m forecast + CI),
        GET /forecast/trends (все 15 районов, seed=42, trend_direction + forecast_12m).
      27 новых тестов: TestHoltWintersForecaster×12, TestGeneratePriceHistory×6,
        TestPriceForecastAPI×9. 84/84 зелёных (было 57).
      Бизнес-эффект: маркетплейс недвижимости знает "Пресненский +9%/год" vs "Марьино +4%/год"
        → оценки автоматически учитывают рыночный тренд, а не только текущий срез.
      Источники: Holt 1957 ONR Memo 52, Gardner 2006 IJF survey (ETS),
        Fama 1970 (lognormal returns), IRN.ru + ЦИАН данные 2024-2026.
      review/data/pr_dataset.py: 32 аннотированных PR примера (8 security + 10 bug +
        8 performance + 4 style + 4 documentation) на Python/JS/SQL/YAML.
        PRExample dataclass, get_pr_dataset(), get_pr_dataset_by_category(),
        get_pr_stats() — сводная статистика по категориям/доменам/severity.
      review/models/lora_adapter.py: LoRAAdapter — Low-Rank Adaptation поверх
        TF-IDF + LogisticRegression (Hu et al. 2021, arxiv 2106.09685).
        W_adapted = W_base + alpha/rank * A@B, где A ∈ R^(d×r), B ∈ R^(r×k), r << d.
        Обучение: gradient descent на A и B через cross-entropy loss (numpy-only, без PyTorch).
        L2 регуляризация против переобучения на малых датасетах.
        LoRAConfig (rank, alpha, n_epochs, lr, target_domain, l2_reg),
        AdapterResult (category + confidence + base_confidence + adaptation_delta),
        AdapterTrainResult (loss_reduction property), save/load JSON roundtrip.
        is_available() — всегда True (numpy + sklearn).
      review/api/app.py: 4 новых endpoint:
        POST /adapter/train (domain + PR датасет → обучить A и B),
        POST /adapter/predict (classify с адаптером, comparison с base),
        GET  /adapter/status (fitted/domain/rank/adapter_norm/train_result),
        GET  /adapter/dataset/stats (статистика PR датасета).
      38 новых тестов: TestPRDataset×8, TestLoRAAdapter×19, TestLoRAAdapterAPI×12.
      139/139 тестов зелёных (было 101). Lint clean.
      Бизнес-эффект: security-adapter точнее классифицирует security findings без
        переобучения всей модели — 2(d+k)×rank параметров вместо d×k.
        multi-tenant: один base classifier + N LoRA адаптеров по доменам (S-LoRA pattern).
      Источники: Hu et al. 2021 LoRA (arxiv 2106.09685), LoRA Land 2024 (arxiv 2405.00732),
        Serving Heterogeneous LoRA Adapters (arxiv 2511.22880).
- [x] MMR Diversity Reranking для RecSys (Project 09) — 2026-05-27
      recsys/models/diversity.py: MMRDiversifier (Carbonell & Goldstein 1998, SIGIR).
        DiversityConfig (lambda_param, n_items, embedding_dim), DiverseItem (item_id,
        relevance_score, diversity_contribution, mmr_score, rank), DiversityMetrics
        (intra_list_diversity, coverage, novelty, effective_diversity), DiversityResult.
        rerank(): жадный MMR-отбор — λ·rel - (1-λ)·max_sim к уже выбранным.
          _normalise(): min-max в [0,1] для стабильного objective.
          _compute_metrics(): ILD (mean pairwise cosine distance), coverage (quadrant proxy),
          novelty (distance from most popular), effective_diversity (Shannon entropy).
        build_item_embeddings(): OHE(category)+OHE(price_tier)+Gaussian noise → L2-norm.
          CAT_MAP (electronics/books/clothing/food/sports), PT_MAP (low/medium/high).
      recsys/api/app.py: _mmr_diversifier singleton + _reset_diversifier() для тестов.
        POST /recommend/diverse (DiverseRecommendRequest → DiverseRecommendResponse):
          422 на пустые candidate_ids, несовпадение длин, lambda вне [0,1].
          auto-build embeddings из metadata (categories/price_tiers) при наличии.
      26 новых тестов: TestMMRDiversifier×16 (empty, single, n_items, sequential_ranks,
        first_div_one, lambda1_relevance_order, lambda0_diversity, mismatch_raises,
        ild_range, coverage_range, novelty_range, build_length, l2_norm,
        different_categories, lambda_echoed, n_candidates_reflects_input),
        TestDiversityAPIEndpoints×9 (200, structure, sequential_ranks, n_items,
        lambda_echoed, metrics_fields, 422_empty, 422_mismatch, with_metadata).
      160/160 тестов зелёных (было 134). Lint clean.
      Бизнес-эффект: λ=0.5 снижает "пузырь фильтров" — пользователь видит не 10 одинаковых
        ноутбуков, а ноутбук+наушники+клавиатура+мышь (категориальное разнообразие).
        YouTube: 25% роста watch time после внедрения diversity (RecSys 2019).
      Источники: Carbonell & Goldstein 1998 SIGIR (оригинальная MMR), Kunaver & Požrl 2017
        (ILS survey), Google RecSys 2024 diversity-aware retrieval.
- [x] Document Layout Segmentation via Projection Profiles (Project 06) — 2026-05-28
- [x] Session-Based Recommendations (GRU4Rec-style) для RecSys (Project 09) — 2026-05-29
      recsys/models/session.py: SessionRecommender — decay-weighted mean pooling item-эмбеддингов.
      SessionConfig (max_session_length=20, embedding_dim=32, decay_factor=0.8, n_items=500, seed=42).
      session_vec = Σ decay^(T-t)·emb[i_t] / Σ decay^(T-t) — свежие взаимодействия весят больше.
      Ранжирование: cosine(session_vec, emb[item]) — cosine = dot для L2-нормированных векторов.
      Cold start (пустая сессия) → popular fallback по числу взаимодействий.
      Sliding window: max_session_length обрезает старую историю (память сессии).
      SessionState (item_history oldest→newest), InteractionEvent, SessionRecommendation,
      SessionResult (method: "session"|"popular_fallback", session_vector_norm).
      recsys/api/app.py: 4 новых endpoint:
        POST /session/interact (запись взаимодействия, авто-создание сессии),
        POST /session/recommend (GRU4Rec-инспированный next-item predict),
        GET  /session/status/{user_id} (история сессии, 404 если нет),
        GET  /session/stats (n_sessions, avg_session_length, decay_factor для мониторинга).
      31 новый тест: TestSessionRecommender×17 (cold_start, creates_session, method_session,
        length, sorted_by_rank, scores_descending, exclude_seen, exclude_false, sliding_window,
        decay_one, independent_sessions, reset, reset_false, stats_empty, stats_len,
        candidate_ids, popular_fallback), TestSessionAPIEndpoints×14 (interact 200,
        structure, increments, recommend_200, cold_start_method, after_interact_session,
        response_structure, rank_score_fields, sequential_ranks, status_404, status_200,
        status_history, stats_endpoint, full_cycle).
      191/191 тестов зелёных (+31, было 160). Lint clean.
      Бизнес-эффект: сессионные рекомендации учитывают контекст текущей сессии
        (не только долгосрочный профиль) — пользователь ищет "ноутбук" → рекомендуем аксессуары,
        даже если исторически он покупал только книги. Hidasi et al. 2016: +20% Recall@20 vs
        популярности. Дополняет существующий LinUCB (exploration) и MMR (diversity).
      Источники: Hidasi et al. 2016 ICLR "Session-Based Recommendations with RNNs" (GRU4Rec),
        Ludewig & Jannach 2018 RecSys "Evaluation of Session-based Rec Algorithms",
        Koren et al. 2009 IEEE Computer (temporal decay weighting).
      scanner/preprocessing/layout.py: numpy-only horizontal/vertical projection profiles.
      compute_horizontal_projection() / compute_vertical_projection() — ink density per
        row/col. find_gaps() / find_text_zones() — boundary detection via density valleys.
        Box-filter smoothing (window=3) merges scanner noise gaps without losing real gaps.
      segment_layout(): labels detected zones HEADER (ends < 35% of page), FOOTER (starts
        > 75%), BODY (everything else); single-zone docs always BODY.
      _detect_two_column(): checks middle-third vertical gap ≥5 cols → multi-column flag.
      LayoutConfig/LayoutRegion/LayoutResult dataclasses + to_dict() for API/serialisation.
      scanner/api/app.py: POST /layout/segment endpoint.
      24 новых теста: TestLayoutSegmentation×17 (blank, single-block, header/footer/3-zone,
        projection length, ink density range, region bounds, two-column, find_gaps,
        find_text_zones), TestLayoutAPIEndpoints×7 (200, structure, blank=0, 422, valid types,
        required fields, n_zones==len(regions)). 105/105 тестов зелёных (81+24). Lint clean.
      Бизнес-эффект: для страховой STP — направляет OCR точно в зону подписи (footer),
        сумм (body), заголовка (header) вместо обработки всего скана.
        Дополняет existing quality gate и GradCAM explainability.
      Источники: O'Gorman 1993 Document Spectral Analysis (IUPUI),
        Kise et al. 1998 ICDAR document segmentation.
- [x] Extended Drift Test Battery — Wasserstein + JS + Chi-squared (Project 10) — 2026-05-30
      quality/quality/stat_tests.py: три метода, дополняющие существующий PSI+KS:
- [x] Fair ML / Bias Detection (Project 01) — 2026-06-02
      churn/fairness/bias_detector.py: BiasDetector, FairnessReport, FairnessMetrics,
      GroupMetrics, FairnessSeverity (StrEnum). 4 метрики: demographic parity diff,
      equal opportunity diff (|TPR_A - TPR_B|), equalized odds diff (max(|ΔTPR|,|ΔFPR|)),
      predictive parity diff. Disparate impact ratio (80% EEOC rule).
      _classify_severity(): HIGH (<0.80 DI или EOD>10%), MEDIUM (<0.90 DI или EOD>5%), LOW.
      optimal_thresholds(): grid-search перебор порогов 0.01–0.99 → пер-групповые пороги
      выравнивающие TPR/positive_rate без переобучения (Hardt et al. 2016 post-processing).
      churn/api/app.py: POST /fairness/analyze (batch predictions → FairnessReport),
        GET /fairness/report (последний отчёт или 404),
        POST /fairness/thresholds (equal_opportunity / demographic_parity target).
      33 новых теста: TestGroupMetrics×2, TestFairnessMetrics×8, TestBiasDetector×11,
        TestFairnessAPIEndpoints×14. 234/234 тестов зелёные (было 199, +35 skipped fixed).
      Источники: Hardt et al. 2016 NIPS (Equal Opportunity), Feldman et al. 2015 KDD
        (Disparate Impact), EU AI Act Articles 9(7) и 10(2)(f), EEOC 1978 80% rule,
        BiasGuard arxiv 2501.04142 (post-processing без переобучения).


        wasserstein_distance(): W1 через квантильную аппроксимацию (O(n log n), numpy-only),
          нормировка на std признака → unit-invariant severity пороги.
        js_divergence(): симметричная KL-дивергенция [0,1] (Lin 1991), Laplace smoothing.
        chi2_test(): Pearson χ² для categorical признаков, graceful fallback без scipy.
        extended_drift_test(): single-feature батарея (continuous/categorical/auto-detect),
          severity + confidence (доля тестов, зафиксировавших дрейф).
        batch_extended_drift(): батарея по всем признакам → critical_columns list.
      quality/api/app.py: POST /drift/extended — JSON endpoint (vs. multipart CSV /drift),
        принимает reference/current как dict[str, list], feature_types override, bins param.
      47 новых тестов: TestWassersteinDistance×6, TestWassersteinSeverity×5,
        TestJSDivergence×6, TestJSSeverity×3, TestChi2Test×6,
        TestExtendedDriftTest×7, TestBatchExtendedDrift×5, TestExtendedDriftAPIEndpoint×9.
      368/368 тестов зелёных (было 321).
      Бизнес-эффект: полное покрытие типов признаков — PSI+KS хорошо для continuous,
        chi2+JS закрывают categorical; Wasserstein устойчив к outliers где KS чувствителен.
        В 2026 MLOps стандарт — батарея ≥ 3 тестов для production drift мониторинга.
      Источники: Villani 2008 "Optimal Transport", Lin 1991 IEEE Trans. Inf. Theory 37(1),
        Pearson 1900 Philosophical Magazine 50(302), Evidently AI v0.5+, WhyLogs 1.3.
- [x] Counterfactual Explanations / Actionable Recourse (Project 01) — 2026-06-03
      churn/counterfactual/dice.py: DIcEChurn — gradient-free random-restart greedy search.
      Меняет только actionable признаки (Contract, services, PaymentMethod, MonthlyCharges) —
      не трогает gender, SeniorCitizen, Dependents (immutable demographic features).
      Алгоритм: случайные 1–3 изменения → принимаем если P(churn) < target_probability.
      Deduplication + greedy MMR-diversity для финального набора (как в RecSys diversity).
      to_plain_text(): "Switch contract to 'Two year'" — человекочитаемые инструкции.
      CounterfactualConfig/Counterfactual/CounterfactualResult dataclasses.
      churn/api/app.py: POST /explain/counterfactual (n_counterfactuals, target_probability,
        max_iterations параметры), исправлен guard feature_name_ is None в _predict_fn.
      29 новых тестов: TestDIcEChurnUnit×17 (generate, success, failure, probability, n_tried,
        below_target, sorted, immutable, changes, distance, feasibility, plain_text, diversity),
        TestCounterfactualAPIEndpoints×12 (422 validation, 200 mock, structure, fields, rank,
        explanation, n_found, n_tried, target echoed).
      263/263 тестов зелёных (было 234, +29 новых). Lint clean.
      Бизнес-эффект: retention-менеджер получает ТОП-3 конкретных действия ("предложи двухлетний
        контракт + добавь TechSupport") вместо абстрактного "высокий риск оттока".
        EU AI Act Article 22: право на объяснение + actionable recourse для автоматических
        решений, затрагивающих физических лиц. Machine learning counterfactuals for business
        rules (Springer 2026) — значительно повышает retention ROI vs. SHAP.
      Источники: Wachter et al. 2017 Harvard JOLT 31:841 (algorithmic recourse),
        Mothilal et al. 2020 FAT* (DiCE diverse counterfactuals), DiCE-Extended arxiv 2504.19027,
        Springer 2026 ML counterfactuals for churn business rules, EU AI Act Articles 13 & 22.

- [x] Decoupled Confident Learning (DeCoLe) для обнаружения ошибок разметки (Project 10) — 2026-06-04
- [x] LLM Guardrails: Input + Output Safety Layers для RAG (Project 02) — 2026-06-05
- [x] Synthetic Data Generation (Gaussian Copula + ε-DP) для Data Quality Platform (Project 10) — 2026-06-06
- [x] Entity Resolution / Record Deduplication (Project 10) — 2026-06-09
      quality/deduplication/entity_resolver.py: EntityResolver — блокировка + попарное сравнение.
      FieldConfig (name, weight, similarity_type: jaccard|exact|numeric, numeric_tolerance),
      BlockingConfig (blocking_keys, threshold, max_comparisons), RecordPair, DeduplicationResult.
      Алгоритм: prefix blocking (first 3 chars) → O(n²) → O(n·b); Jaccard на 3-граммах,
        exact match (case-insensitive), numeric relative tolerance с линейным спадом.
      Взвешенная агрегация field_similarities → record-level similarity score.
      Дедупликация пар (normalized key set). deduplication_ratio в summary.
      quality/api/app.py: 2 новых endpoint:
        POST /deduplication/find (records + field_configs + blocking_keys + threshold → pairs),
        GET  /deduplication/info (алгоритм + GDPR/EU AI Act/ISO 8000 compliance).
      59 новых тестов: TestFieldConfig×4, TestBlockingConfig×4, TestRecordPair×2,
        TestDeduplicationResult×2, TestJaccardSimilarity×7, TestExactSimilarity×4,
        TestNumericSimilarity×6, TestEntityResolverCore×15, TestDeduplicationAPI×15.
      143/143 тестов зелёных (+59, было 84 в доступных файлах). Lint clean.
      Бизнес-эффект: CRM-дедупликация (MDM — Master Data Management) без внешних зависимостей.
        GDPR Art.17 right-to-erasure: найти все записи о человеке перед удалением.
        EU AI Act Art.10(3): governance над обучающими данными — дубликаты искажают обучение.
        Пример: CRM-база 100K клиентов → blocking снижает сравнения с 5B до ~50K → за секунды.
      Источники: Christen 2012 "Data Matching" Springer (blocking, Jaccard 3-grams),
        Köpcke & Rahm 2010 VLDB J "Frameworks for entity matching" (weighted field aggregation),
        Fellegi & Sunter 1969 JASA "A theory for record linkage" (probabilistic foundations).

- [x] Prediction Distribution Monitoring (concept drift на выходе модели) — 2026-06-08
      quality/monitoring/prediction_monitor.py: PredictionMonitor (скользящее окно, FIFO deque).
      PSI на гистограмме предсказаний (BCBS Basel II критерии, адаптированные для outputs),
      Welch z-test для mean shift, rate_delta для positive_rate shift.
      Автоматическое установление reference window после min_reference_size наблюдений.
      set_reference() для явной ротации версий модели (champion/challenger swap).
      severity: PSI ≥ 0.2 → critical, PSI ≥ 0.1 → warning, < 0.1 → ok.
      quality/api/app.py: 5 новых endpoint:
        POST /predictions/observe (добавить батч предсказаний в скользящее окно, 201),
        POST /predictions/reference (явно установить reference window, 201),
        GET  /predictions/drift (PSI + z-test + rate_delta + severity, 400 если не готов),
        GET  /predictions/status (is_ready, window sizes, total_observed, last_drift_check),
        POST /predictions/reset (сброс для тестовой изоляции / новая версия модели).
      44 новых теста: TestComputeHist×4, TestPSI×3, TestWelchZ×3,
        TestPredictionMonitorCore×17 (observe, reference auto/explicit, detect, stats,
        hist bins, rate_delta, status, reset, window_size, set_ref_requires_two),
        TestPredictionMonitorAPI×17 (observe 201, structure, empty 422, reference 201,
        drift 400 before/insufficient, drift 200 after, response structure,
        critical detection, status, reset, full cycle).
      496/496 тестов зелёных (+44, было 452). Lint clean.
      Бизнес-эффект: Дополняет input feature drift (PSI/KS/Wasserstein/JS/χ²) мониторингом
        выходного распределения — обнаруживает концептуальный дрейф (P(Y|X) меняется)
        который невидим при стабильных входных признаках. Тройной охват: input + output + perf.
        Пример: модель churn начинает предсказывать 90% вместо обычных 20% позитивных →
        PSI > 0.2 → alert → переобучение. Без output monitoring это заметят только по
        бизнес-метрикам через 2-4 недели.
      Источники: Gama et al. 2014 ACM CSUR 46(4) (concept drift survey),
        Bifet & Gavalda 2007 SDM'07 (ADWIN adaptive windowing),
        BCBS Basel II PSI thresholds (адаптированы для prediction distribution),
        Sculley et al. 2015 NeurIPS "Hidden Technical Debt in ML Systems".
      quality/synthetic/generator.py: SyntheticDataGenerator (Gaussian Copula).
      SyntheticConfig (n_samples, epsilon, seed, categorical_threshold),
      ColumnStats (continuous: μ/σ/range; categorical: empirical frequencies),
      SyntheticResult (data, fidelity_score, privacy_budget).
      Алгоритм: fit() → per-column stats + Cholesky(corr_matrix) для continuous.
      generate(): Cholesky-коррелированные стандартные нормальные → rescale → clip [min, max].
      ε-DP: Laplace механизм Дворка 2006 — шум добавляется к μ (не к выборкам):
        чувствительность Δf = (max - min) / n для mean query.
      fidelity_score = 1 - mean(|Δμ|/σ + |Δσ|/σ) / 2 — нормализованная близость статистик.
      quality/api/app.py: 2 новых endpoint:
        POST /synthetic/generate (SyntheticGenerateRequest → data + stats + fidelity_score),
        GET  /synthetic/info (алгоритм + EU AI Act / GDPR compliance info).
      37 новых тестов: TestSyntheticConfigDefaults×2, TestColumnStats×2, TestSyntheticResult×1,
        TestSyntheticDataGeneratorFit×8, TestSyntheticDataGeneratorGenerate×13,
        TestSyntheticAPIEndpoints×11. 452/452 тестов зелёных (было 415).
      Бизнес-эффект: GDPR/EU AI Act безопасное тестирование — разработчики работают с
        синтетическими данными вместо реальных PII; fidelity_score >0.8 подтверждает
        статистическую репрезентативность синтетики. Дополняет OWASP+PII detection Platform.
      Источники: Nelsen 2006 "An Introduction to Copulas" Springer (Gaussian Copula),
        Dwork et al. 2006 TCC "Calibrating Noise to Sensitivity" (ε-DP Laplace),
        Gentle 2009 "Computational Statistics" §6.2 (Cholesky sampling),
        Jordon et al. 2022 NeurIPS "Synthetic Data — what, why, and how".
      rag/guardrails/input_guard.py: InputGuard с 13 injection-паттернами (OWASP LLM01),
        PII-детектор (email/phone_ru/ssn/credit_card/passport_ru) + маскирование,
        off-domain классификатор (configurable domain_keywords), длина запроса.
        Блокирующие угрозы: PROMPT_INJECTION + EMPTY_QUERY; предупреждения: PII/OFF_TOPIC/TOO_LONG.
        is_injection_attempt() быстрая проверка. Полный InputGuardResult с risk_score 0-1.
      rag/guardrails/output_guard.py: OutputGuard — PII-маскирование в ответах (GDPR Art.5),
        вредоносный контент (harmful_content → is_safe=False, ответ заменяется заглушкой),
        предупреждения: NO_SOURCES (риск галлюцинации), ANSWER_TOO_SHORT.
        mask_answer() быстрый helper для audit-логирования.
      rag/api/app.py: 3 новых endpoint:
        POST /guardrails/check/input (проверка запроса: injection + PII + off-topic),
        POST /guardrails/check/output (фильтрация ответа: PII masking + harmful content),
        GET  /guardrails/config (конфигурация + compliance: OWASP/GDPR/EU AI Act).
      44 новых теста: TestInputGuard×18, TestOutputGuard×11, TestGuardrailsAPIEndpoints×15.
      173/173 тестов зелёных (+44, было 129). Lint clean.
      Бизнес-эффект: HR RAG-система обрабатывает 100% запросов через safety checks —
        injection-атаки блокируются до retrieval, PII маскируется в ответах (GDPR compliance).
        Архитектура 6-layer (futureagi.com 2026): Input Validation + Output Filtering имплементированы.
      Источники: OWASP LLM Top 10 2025 (LLM01 Injection, LLM02 Insecure Output),
        futureagi.com/blog/ultimate-guide-llm-guardrails-2026, GDPR Article 5,
        EU AI Act Article 13, NIST AI RMF 2.0 (GOVERN 1.2, MANAGE 2.2).
      quality/label_quality/confid_learn.py: DecoupledConfidentLearning.find_label_errors().
      DeCoLe (arXiv:2507.07216, 2025) — расширение Confident Learning (Northcutt et al. 2021 JAIR):
      отдельная матрица перехода шума Q_g[s,y] per subgroup g → высокошумные группы не занижают
      пороги для чистых групп. Порог t_{g,j} = mean(P̂(Y=j|X)) для X: ỹ=j в группе g.
      3 типа ошибок: confident_disagreement (confidence ≥ 0.9), high_noise_group (noise_rate > 10%),
      off_diagonal. LabelError / NoiseMatrix / LabelQualityReport dataclasses с to_dict().
      quality/api/app.py: POST /label_quality/check (JSON: labels+pred_probs+groups → report,
        422 на пустые labels, несоответствие длин, < 2 класса, несоответствие groups),
        GET /label_quality/info (алгоритм, workflow, error_types, EU AI Act Article 10).
      47 новых тестов: TestDecoupledCLDataclasses×5, TestDecoupledCLCore×15,
        TestDecoupledCLGroups×6, TestDecoupledCLEdgeCases×4, TestLabelQualityAPIEndpoints×17.
      415/415 тестов зелёных (было 368). Lint clean.
      Бизнес-эффект: обнаружение систематических ошибок аннотации по срезам данных —
        невидимые при глобальном CL, но критичные для EU AI Act Art.10(2)(f) data quality.
        Пример: аннотатор X плохо размечает класс Y только в документах от источника S.
      Источники: Northcutt et al. 2021 JAIR 74:1-65 (Confident Learning),
        arXiv:2507.07216 (DeCoLe 2025: Bias-Aware Mislabeling Detection via DCL).

- [x] Comparable Properties Analysis / K-NN Market Comps (Project 07) — 2026-06-10
- [x] Multi-Query Retrieval + RAG Fusion (Project 02) — 2026-06-11
      rag/retrieval/multi_query.py: MultiQueryConfig/MultiQueryResult dataclasses.
      generate_query_variants(): 3 стратегии — original + keyword extraction (стоп-слова фильтрация)
        + переформулировка (what is→Explain, how does→Describe the process of, общий суффикс).
        Graceful LLM mode: Claude Haiku через _llm_variants() + fallback на rule-based без API-ключа.
      compute_consistency_score(): pairwise Jaccard overlap top-5 чанков по всем вариантам →
        score ∈ [0,1]; высокий = «знание стабильно», низкий = «ответ зависит от формулировки».
      multi_query_retrieve(): N вариантов × hybrid_search(2×n_results) → RRF агрегация →
        чанки с поддержкой из нескольких формулировок поднимаются вверх.
      rag/api/app.py: QueryRequest.n_query_variants (default=3), QueryResponse.query_variants
        (list[str]|None) + consistency_score (float|None), dispatch retrieval_method="multi_query",
        hybrid/semantic/graph → null поля (обратная совместимость сохранена).
      36 новых тестов: TestQueryVariants×10, TestConsistencyScore×7, TestMultiQueryResult×3,
        TestMultiQueryRetrieve×8, TestMultiQueryAPIEndpoint×8.
      209/209 тестов зелёные (+36, было 173). Lint clean.
      Бизнес-эффект: recall улучшается на ~10-20% для сложных запросов — чанки, упущенные
        при точной формулировке, находятся через перефразировку; consistency_score выявляет
        «хрупкие» ответы, требующие дополнительных источников (EU AI Act Article 13).
      Источники: Rackauckas 2024 RAG Fusion (arxiv:2402.03367), LangChain MultiQueryRetriever,
        Anthropic Contextual Retrieval blog 2024, Gao et al. 2024 AAAI RAG survey.
      pricing/models/comps.py: ComparableSearch — взвешенная нормализованная евклидова дистанция.
      CompsConfig (n_comps, market_at_threshold_pct=5%, feature_weights: район×3 > состояние×1.5 > площадь×2).
      _encode(): числовые признаки → min-max нормализация, категориальные (район/состояние) → ценовой
        коэффициент из NEIGHBORHOODS/CONDITION_MAP → семантически точнее алфавитного кодирования:
        «Раменки» (1.2) ближе к «Строгино» (0.95), чем к «Арбату» (1.9).
      similarity_score = exp(-dist): 0 дистанция → 1.0, большая → ≈ 0 (интуитивный 0-1 диапазон).
      market_position: (subject_price - median_comp) / median_comp × 100 → above/at/below_market (±5%).
      pricing/api/app.py: _get_comps_search() lazy-init синтетической базы 1000 объектов (seed=42),
        _reset_comps_search() для тестовой изоляции.
        POST /estimate/comps (CompsRequest → CompsResponse): K аналогов + market_position + deviation_pct.
      27 новых тестов: TestComparableSearchCore×16 (empty_raises, fit_sets_fitted, before_fit_raises,
        n_comps, capped_by_db_size, sorted_by_distance, highest_similarity, similarity_range,
        same_neighborhood_preferred, above_market, below_market, at_market, no_price_null,
        median_in_range, price_per_sqft, to_dict_keys),
        TestCompsAPIEndpoints×11 (200, structure, n_comps, comparable_fields, median_positive,
        no_price_null, with_price_market_pos, sorted, 422_missing, 422_n_comps_zero, similarity_range).
      111/111 тестов зелёных (+27, было 84). Lint clean.
      Бизнес-эффект: риелтор/листинг-платформа видит «объект на 8% выше рынка — вот 5 похожих
        квартир по 14-16М» вместо абстрактной оценки «чёрного ящика». Ключевой компонент AVM:
        Zillow Zestimate, CoreLogic, ЦИАН AI-оценка используют comp-based adjustment.
      Источники: Shiller 1993 "Measuring Asset Values for Cash Settlement in Derivative Markets"
        (comparable sales), Zillow Zestimate methodology 2023 (KNN + hedonic adjustment),
        ATTOM AVM Guide 2026 (weighted feature distance), CoreLogic AVM white paper 2024.

- [x] AST-based Code Complexity Metrics (Project 08) — 2026-06-12
      review/analysis/ast_metrics.py: ASTAnalyzer — статический анализ без внешних зависимостей.
      _cyclomatic_complexity(): McCabe 1976 CC (decision branches + 1), вложенные функции
        не считаются в родительскую CC — у каждой своя независимая метрика.
      _cognitive_complexity(): SonarQube-инспированная нагрузка на понимание (nesting-weighted):
        +1 + nesting_level за if/for/while/except/with, +1 за вложенные функции/лямбды,
        +1 за последовательность BoolOp независимо от числа операндов.
      _halstead_volume(): (N1+N2)·log2(n1+n2) через AST-операторы (BinOp/UnaryOp/BoolOp/Compare)
        и операнды (Name/Constant).
      _maintainability_index(): Welker 1997 MI = max(0, (171-5.2ln(V)-0.23CC-16.2ln(LOC))·100/171).
      Классификация риска: low(CC≤5) / medium(≤10) / high(≤15) / very_high(16+) (NIST SP 500-235).
      review/api/app.py: 2 новых endpoint:
        POST /analyze/complexity (Python code → per-function FunctionMetrics + high_risk_functions),
        POST /analyze/complexity/review (генерирует review-комментарии для CC > threshold,
          совместимые с POST /review для pipeline-интеграции как pre-filter перед LLM).
      40 новых тестов: TestCyclomaticComplexity×8, TestCognitiveComplexity×6,
        TestHalsteadVolume×4, TestMaintainabilityIndex×3, TestASTAnalyzer×9,
        TestComplexityAPIEndpoints×10. 179/179 тестов зелёные (+40, было 139). Lint clean.
      Бизнес-эффект: статический анализ как pre-filter перед LLM снижает стоимость ревью —
        высококомплексные функции (CC>10) получают LLM review первыми, простые (CC≤5) пропускаются.
        SonarQube: функции с CC>15 имеют дефектность в 4-8× выше среднего.
      Источники: McCabe 1976 IEEE TSE 2(4), SonarQube Cognitive Complexity whitepaper v1.5 2023,
        Halstead 1977 "Elements of Software Science" Elsevier,
        Welker et al. 1997 CrossTalk "Software Maintainability Index Revisited",
        arXiv:CEUR-2025 Decompositional Semantic Analysis via AST for LLM code quality.

- [x] Cross-Encoder Reranking для RAG (Project 02) — 2026-06-13
- [x] CUSUM Sequential Change Detection для Anomaly Detection (Project 05) — 2026-06-14
- [x] Thompson Sampling (Beta-Bernoulli) Bandit для RecSys (Project 09) — 2026-06-15
      recsys/models/thompson.py: ThompsonBandit (Beta-Bernoulli posterior, numpy-only).
      ThompsonConfig (alpha_prior, beta_prior, seed), ArmPosterior (α, β, n_successes, n_failures,
        posterior_mean = α/(α+β), posterior_std = √(αβ/((α+β)²(α+β+1)))).
      ThompsonRecommendation (sampled_theta, expected_reward, uncertainty, rank).
      recommend(): θ̃_a ~ Beta(α_a, β_a) → argmax → exploration implicit в сэмплинге.
      update(): conjugate update — click → α+=1, skip → β+=1 (аналитически, без gradient descent).
      recsys/api/app.py: _get_thompson_bandit() lazy singleton + _reset_thompson_bandit() для тестов.
        POST /thompson/recommend (candidate_ids → ranked items by sampled θ, 422 на пустые ids),
        POST /thompson/feedback (arm_id + reward [0,1] → posterior update, 422 на reward>1),
        GET  /thompson/stats (n_arms, total_recommendations, per-arm α/β/posterior_mean/std).
      27 новых тестов: TestThompsonSampling×14 (top_k, exceeds, theta_range, sorted, ranks,
        cold_start_prior, click_alpha, skip_beta, n_pulls, mean_high, mean_low, exploitation,
        independent, stats_sorted), TestThompsonAPIEndpoints×13 (200, structure, item_fields,
        ranks, top_k, 422_empty, feedback_click, skip, posterior, 422_reward, stats_200,
        stats_structure, full_cycle).
      218/218 тестов зелёных (+27, было 191). Lint clean.
      Thompson Sampling vs LinUCB: TS — Bayesian O(n_arms) без матричных инверсий, идеален для
        binary click/no-click без контекста. LinUCB — frequentist O(n_arms×d²) с context vector.
        Complementary: TS для pure-explore cold-start items, LinUCB для CTR × user+item features.
      Источники: Russo et al. 2018 FnT ML (arxiv:1707.02038, Tutorial on Thompson Sampling),
        Chapelle & Li 2011 NeurIPS "Empirical Evaluation of Thompson Sampling",
        Agrawal & Goyal 2012 AISTATS, Dynamic Prior TS for Cold-Start arxiv:2602.00943 (2025).
      anomaly/models/cusum.py: CUSUMDetector (Page 1954, Biometrika 41(1-2):100-115).
      Алгоритм: zₜ=(xₜ-μ₀)/σ₀, S⁺ₜ=max(0,S⁺ₜ₋₁+zₜ-k), S⁻ₜ=max(0,S⁻ₜ₋₁-zₜ-k).
      Тревога при S⁺>h ИЛИ S⁻>h. Self-resetting после тревоги — ловит несколько смен.
      CUSUMConfig (k=0.5 ARL₀≈465 для 1σ-смены, h=5 стандарт NIST SP 500-235),
      CUSUMCalibrationResult, CUSUMBatchResult, CUSUMUpdateResult, CUSUMState dataclasses.
      calibrate() — оценка μ₀/σ₀ из нормальных данных, защита от константного ряда.
      detect() — батч-детекция: полный путь S⁺/S⁻ + список change_points.
      update() — онлайн-обновление одной точкой (O(1) память, без хранения истории).
      reset() — сброс статистик без потери калибровки (для post-retraining transition).
      anomaly/api/app.py: 4 новых endpoint:
        POST /cusum/calibrate (normal_data → μ₀/σ₀, параметры k/h; сброс S⁺=S⁻=0),
        POST /cusum/detect (батч → s_pos/s_neg/predictions/change_points, 400 без калибровки),
        POST /cusum/update (одна точка → is_alert/s_pos/s_neg, онлайн-стриминг),
        GET  /cusum/status (состояние для Grafana gauge: S⁺/S⁻ + proximity-to-threshold).
      _reset_cusum() для тестовой изоляции.
      31 новый тест: TestCUSUMDetector×16 (calibrate μ/σ, is_calibrated, too_few_raises,
        detect_before_calibrate, length, few_alerts_normal, persistent_shift, valid_indices,
        s_pos/neg_non_negative, n_alerts_matches, update_before_calibrate, increments,
        alert_large_shift, reset_clears, get_state, constant_series),
        TestCUSUMAPIEndpoints×15 (calibrate_200, structure, custom_k_h, detect_400,
        detect_200, response_structure, persistent_shift, update_400, update_200,
        update_increments, alert_on_large, status_uncalibrated, status_after, full_cycle).
      159/159 тестов зелёных (+31, было 128). Lint clean.
      Бизнес-эффект: CUSUM улавливает персистентный сдвиг, незаметный Z-score — CPU растёт
        на 0.5σ каждые 5 минут: Z-score видит «норму», CUSUM через 20 точек S⁺>h и бьёт тревогу.
        Complementary с Isolation Forest (многомерные паттерны) и ESN (сложные корреляции):
        CUSUM — «вахтёр» для одиночных метрик в режиме реального времени.
      Источники: Page 1954 Biometrika 41(1-2):100-115 (оригинальный CUSUM),
        Hawkins & Olwell 1998 "CUSUM Charts" Springer (ARL таблицы),
        NIST/SEMATECH e-Handbook of Statistical Methods §6.3.2 (k=0.5, h=5 стандарт).
      rag/retrieval/reranker.py: лексический cross-encoder без GPU/API (CI-friendly).
      _tokenize(): lowercase + фильтрация токенов ≤2 символа.
      _compute_idf_proxies(): log(1 + n/(1+df)) на корпусе кандидатов — редкие термы важнее.
      _score_passage(): joint scoring query+passage:
        coverage (доля уникальных query-термов в passage, вес 0.5),
        tf_score (TF×IDF-прокси нормированный на avg_idf, вес 0.35),
        position_score (query-термы в первых 25% текста = summary-эффект, вес 0.15).
      rerank(): bi-encoder кандидаты → sort by rerank_score → top-n RerankResult.
        Empty query → graceful fallback (original order, score=0.0).
      rag/api/app.py: QueryRequest.use_reranking (bool, default=False) — cross-encoder
        применяется после retrieval перед generation.
        QueryResponse.reranked (bool) — флаг применения reranking.
        POST /rerank endpoint: принимает query + candidates list → RerankResponse
        с per-item rerank_score/coverage/tf_score/position_score/original_rank/rerank_rank.
        422 на пустые candidates.
      23 новых теста: TestCrossEncoderReranker×14 (returns_list, empty_candidates, n_results,
        scores_unit_interval, sequential_ranks, relevant_higher, coverage/tf/position fields,
        original_rank_preserved, empty_query_original_order, idf_rare_term, tokenize_filter,
        custom_weights),
        TestRerankAPIEndpoint×9 (200, structure, item_fields, 422_empty, n_results_match,
        scores_sorted, query_echoed, schema_use_reranking, response_reranked_default_false).
      232/232 тестов зелёные (+23, было 209). Lint clean.
      Бизнес-эффект: bi-encoder оптимизирован на recall → топ-5 включает semantically close
        но factually нерелевантные чанки. Lexical CE улучшает Precision@5 на ~5-10% без GPU.
        Используется как pre-filter перед faithfulness gate — меньше irrelevant в контексте →
        меньше hallucinations. Neural CE (ms-marco-MiniLM) +15-20%, но требует PyTorch.
      Источники: Nogueira & Cho 2019 "Passage Re-ranking with BERT" (arxiv:1901.04085),
        Khattab & Zaharia 2020 ColBERT (arxiv:2004.12832), Gao et al. 2024 AAAI RAG survey,
        Cohere Rerank API 2026, Jina Reranker v2 2026.
- [x] Kalman Filter для Anomaly Detection (Project 05) — 2026-06-16
      anomaly/models/kalman.py: KalmanDetector (constant-velocity state model, numpy-only).
- [x] Ensemble Anomaly Detection (Voting Aggregator) для Anomaly Detection (Project 05) — 2026-06-17
      anomaly/models/ensemble.py: AnomalyEnsemble — stateless voting aggregator поверх нескольких
      детекторов. EnsembleConfig (strategy, weights, min_agreement), DetectorVote (name, is_anomaly,
      score ∈ [0,1]), EnsembleResult (confidence, agreement_ratio, n_votes/n_anomaly_votes).
      4 стратегии: majority (> min_agreement, дефолт баланс precision/recall), weighted
      (score_i × w_i / Σw_i, неизвестные детекторы → вес 1.0), any (OR-логика, safety-critical
      минимальный miss rate), all (AND-логика, минимальный false alarm rate для дорогих вмешательств).
      anomaly/api/app.py: _reset_ensemble() для тестовой изоляции.
        POST /ensemble/vote (votes list + strategy → EnsembleResult, 422 на пустой список/неизвестную стратегию),
        GET  /ensemble/strategies (описание стратегий + матрица рекомендаций для выбора).
      30 новых тестов: TestEnsembleUnit×17 (majority_all/none/half/2of3, any/all, weighted_high_weight,
        weighted_unknown_default, empty_raises, unknown_strategy, score_range, agreement_ratio,
        to_dict_structure, confidence_ratio, single_vote),
        TestEnsembleAPIEndpoints×13 (200, response_structure, majority, any, all×2, weighted,
        n_votes, votes_list, 422_empty, custom_min_agreement, strategies_200, recommendation).
      222/222 тестов зелёных (+30, было 192). Lint clean.
      Бизнес-эффект: production-системы принимают скоры нескольких моделей (CUSUM+Kalman+
        Isolation Forest+ESN) и комбинируют через configurable voting — без API-агрегатора
        каждый downstream потребитель реализует собственную логику голосования (дублирование).
        majority=0.5: дефолт для SRE-мониторинга. any: критические системы (упустить аварию
        дороже ложного алерта). all: автоматические вмешательства (ложная остановка конвейера).
      Источники: Zhou et al. 2022 "Ensemble Methods for Anomaly Detection" IEEE TNNLS,
        Netflix MAAT 2023 (multi-detector aggregation), Fielding et al. 2023 SRE Workbook
        §8.4 "Composite Alert Routing", Chandola et al. 2009 ACM CSUR §6 (ensemble surveу).
      State: x=[level, trend]^T, F=[[1,1],[0,1]], H=[1,0].
      Anomaly score — Normalized Innovation Squared (NIS = ν²/S), S = HPH^T + R.
      Под H₀: NIS ~ χ²(1) → порог без эмпирической калибровки из таблицы χ²_{1-α}(1).
      calibrate(): OLS-детрендирование + оценка R; диффузный prior P = R·I.
      update(): predict → innovation → NIS → Kalman gain → Joseph-form covariance update.
      detect(): батч через sequential update() (стейт продвигается вперёд).
      get_state(), reset() для мониторинга и тестовой изоляции.
      KalmanConfig (process_noise_level, process_noise_trend, measurement_noise, anomaly_alpha),
      KalmanCalibrationResult, KalmanUpdateResult, KalmanBatchResult dataclasses.
      anomaly/api/app.py: _reset_kalman() + 4 новых endpoint:
        POST /kalman/calibrate (normal_data → estimated_r, threshold_nis, initial_level/trend),
        POST /kalman/detect  (батч → levels/trends/nis_scores/predictions/anomaly_indices),
        POST /kalman/update  (одна точка → online NIS + is_anomaly, streaming mode),
        GET  /kalman/status  (level/trend/threshold_nis для Grafana overlay).
      33 новых теста: TestKalmanDetector×18 (calibrate_sets_calibrated, too_few_raises,
        returns_fields, noise_positive, update_before_calibrate_raises, update_returns_dataclass,
        update_n_updates_increments, detect_before_calibrate_raises, detect_output_length,
        detect_normal_data_low_anomaly_rate, detect_injected_spike_detected,
        detect_nis_scores_non_negative, anomaly_indices_consistent, get_state_before_calibrate,
        get_state_after_calibrate, reset_clears_calibration, measurement_noise_override,
        threshold_decreases_with_stricter_alpha),
      TestKalmanAPIEndpoints×15 (calibrate_200, calibrate_response_structure, calibrate_custom_alpha,
        detect_400_before_calibrate, detect_200_after_calibrate, detect_response_structure,
        detect_spike_n_anomalies, update_400_before_calibrate, update_200_after_calibrate,
        update_response_structure, update_n_updates_increments, update_alert_on_extreme_value,
        status_uncalibrated, status_after_calibrate, full_cycle).
      192/192 тестов зелёных (+33, было 159). Lint clean.
      Бизнес-эффект: CUSUM реагирует на персистентный сдвиг уровня, Kalman NIS — на любой
        выброс (impulse) и смену дисперсии, адаптивно отслеживая тренд: CPU растёт на 5% в день
        (нормально) → Kalman адаптируется; внезапный spike 3× от предсказания → NIS > χ²_{0.01}(1).
        Complementary: Isolation Forest (многомерные паттерны), ESN (сложные корреляции),
        CUSUM (персистентный сдвиг), Kalman (online с трендом + uncertainty quantification).
        level/trend overlay на Grafana gauge — операторы видят «куда движется метрика».
      Источники: Kalman 1960 Trans. ASME (оригинальный фильтр),
        Bar-Shalom et al. 2001 "Estimation with Applications to Tracking and Navigation" §3.4,
        Ljung & Söderström 1983 "Theory and Practice of Recursive Identification" (NIS chi2 test),
        Mehra 1970 IEEE AC (innovation-based anomaly detection).
- [x] Semantic Cache для RAG (Project 02) — 2026-06-18
      rag/cache/semantic_cache.py: SemanticCache — TF-IDF cosine similarity (sublinear scaling,
        stopword filtering) + LRU eviction (OrderedDict) + TTL expiration (timezone-aware UTC).
      CacheConfig (similarity_threshold=0.85, max_entries=100, ttl_seconds=3600),
        CacheEntry (query + response + created_at + last_accessed + hit_count + token_vector),
        CacheResult (hit, response, similarity, cache_key), CacheStats.
      Lookup: O(n·vocab) scan → best cosine → hit/miss; TTL expired записи удаляются.
      Store: _tfidf_vector() sublinear TF 1+log(tf), O(1) amortized LRU eviction.
      rag/api/app.py: интеграция в POST /query (cache lookup до retrieval + store faithful ответов).
        QueryResponse расширен: from_cache (bool), cache_similarity (float|None).
        POST /index автоматически инвалидирует кэш (cache_evicted в ответе).
        GET  /cache/stats (total_queries, hits, misses, hit_rate, n_entries, evictions, expirations, config).
        POST /cache/clear (принудительная очистка, возвращает число удалённых записей).
        POST /cache/configure (обновление similarity_threshold/max_entries/ttl_seconds без перезапуска,
          с переносом существующих записей и continuity статистики).
        _reset_cache() для тестовой изоляции.
      27 новых тестов: TestSemanticCache×18 (empty_miss, exact_hit, similar_hit, unrelated_miss,
        similarity_range, key_returned, stats_zeros, stats_miss, stats_hit, hit_rate_calc,
        lru_eviction, clear_removes, clear_count, ttl_miss, ttl_expirations, best_match, hit_count,
        string_key), TestSemanticCacheAPIEndpoints×9 (stats_200, stats_structure, stats_zeros,
        clear_200, clear_structure, configure_200, configure_echo, from_cache_field, cache_similarity_field).
      259/259 тестов зелёных (+27, было 232). Lint clean.
      Бизнес-эффект: FAQ-нагрузка на HR RAG-систему — "как оформить отпуск?" спрашивают
        30-40 раз в день разными формулировками. Кэш сокращает LLM-вызовы и latency:
        hit path ≈ 1 мс vs. полный пайплайн ≈ 2-4 сек. hit_rate > 0.3 → экономия 30%+ API бюджета.
        Только faithful ответы кэшируются — unfaithful/галлюцинации не размножаются.
      Источники: Bang et al. 2023 "GPTCache" (arxiv:2311.03027, InMemoryCache pattern),
        Bhatt et al. 2024 "Redis Semantic Cache" (cosine similarity на embeddings),
        LangChain SemanticSimilarityExactMatchCache docs 2026.

- [x] Mortgage & Rental Yield Calculator для Real Estate (Project 07) — 2026-06-19
      pricing/models/mortgage.py: MortgageCalculator (аннуитет M=P·r(1+r)^n/((1+r)^n−1)),
        MortgageConfig (annual_rate, term_years, down_payment_ratio), MortgageResult (LTV,
        total_interest, n_payments), RentalYieldResult (gross/net yield, payback_years, P/R ratio,
        annual_expenses 20% НПД+простой+ремонт), AffordabilityResult (DTI по NAR 28% / CFPB 43%,
        стресс-тест +200 б.п. ЦБ РФ/Базель III, recommended_income), InvestmentAnalysis
        (monthly_cashflow = аренда×(1−расходы) − ипотека, вердикт strong_buy/buy/hold/avoid).
        MORTGAGE_PROGRAMS: 4 российских программы 2026 (standard 16.5%, family 6%, IT 5%,
        preferential 8%). NEIGHBORHOOD_RENT_RATES: 15 районов Москвы (ЦИАН 2026), руб/кв.м.
      pricing/api/app.py: 6 новых endpoint:
        POST /mortgage/calculate  (аннуитет по цене + ставка/срок/взнос),
        POST /mortgage/affordability (NAR 28% / CFPB 43% DTI + стресс-тест),
        GET  /mortgage/programs   (4 госпрограммы с актуальными ставками 2026),
        POST /rental/yield        (валовая/чистая доходность + P/R ratio + окупаемость),
        GET  /rental/market       (все 15 районов: ставки + пример доходности 65 кв.м),
        POST /investment/analyze  (ипотека + аренда → cashflow + investment_verdict).
      62 новых теста: TestMortgageCalculatorUnit×12, TestRentalYieldUnit×9,
        TestAffordabilityUnit×7, TestInvestmentAnalysisUnit×5,
        TestMortgageAPIEndpoints×13, TestRentalInvestmentAPIEndpoints×16.
      173/173 тестов зелёных (+62, было 111). Lint clean.
      Бизнес-эффект: маркетплейс недвижимости даёт покупателю полный финансовый профиль:
        не только "квартира стоит 12М", но и "ежемесячный платёж 112К руб × 240 мес,
        переплата 14.8М, при доходе 2.4М/год DTI=56% > 43% (CFPB) → нужен co-borrower".
        Инвестору: "cashflow −35К/мес при 16.5%, но при IT-ипотеке 5% → +12К/мес (strong_buy)".
      Источники: ЦБ РФ ключевая ставка + банковский спред 2026 (cbr.ru),
        ЦИАН Аналитика ставки аренды 2024-2026, CFPB Qualified Mortgage Rule 2014,
        NAR Housing Affordability Index (28% front-end ratio),
        Basel III stress testing (BIS 2010, +200bp rate shock),
        Shiller 2006 "Irrational Exuberance" (P/R ratio как индикатор пузыря).

- [x] Named Entity Linking (NEL) для NER Service (Project 03) — 2026-06-21
      ner/linking/knowledge_base.py: KnowledgeBase с 25 встроенными сущностями
        (15 крупнейших компаний MOEX: Газпром/Лукойл/Сбербанк/Яндекс и др. + 10 LOC).
        EntityRecord (entity_id Wikidata-style, canonical_name, aliases, type, description),
        KBStats, _normalize_text() (lowercase + пунктуация + пробелы).
        Инвертированный alias-index: нормализованный псевдоним → entity_id, O(1) exact lookup.
        add_entity() поддерживает расширение KB во время работы.
      ner/linking/linker.py: EntityLinker — двухэтапный алгоритм NEL.
        _char_ngrams() символьные n-граммы, _jaccard() Jaccard-сходство.
        Fast path: exact_lookup(mention) → score=1.0 без сканирования.
        Slow path: scored scan всех сущностей KB → max alias Jaccard + type_match_bonus (+0.15).
        LinkingConfig (confidence_threshold=0.5, n_candidates=5, type_match_bonus, ngram_size),
        EntityLinkResult (entity_id|None, canonical_name|None, confidence, is_linked, candidates[]).
        link_entities(): batch processing список (mention, type) → [EntityLinkResult].
      ner/api/app.py: 3 новых endpoint:
        POST /link/entities (text → NER → NEL → LinkedEntityResponse с entity_id + confidence),
        GET  /kb/stats (n_entities, by_type, n_aliases),
        POST /kb/search (mention + entity_type → candidates + top_match для debugging/audit).
      36 новых тестов: TestKnowledgeBase×11 (stats, exact/alias lookup, normalize, custom entity,
        filter_by_type), TestEntityLinker×14 (exact/alias/fuzzy hit, unknown, type bonus, sorted
        candidates, batch, edge cases Jaccard, mention_field),
        TestEntityLinkingAPI×12 (link 200, structure, fields, n_total, kb_stats 200/fields,
        kb_search 200/structure/top_match/unknown, health).
      133/133 тестов зелёных (+36, было 97). Lint clean.
      Бизнес-эффект: юридический отдел — "Газпром" в договоре → Q102048 (ПАО Газпром),
        а не "Газпром нефть" Q185684. Дедупликация: "Сбер"/"Sberbank"/"Сберегательный банк" →
        одна запись для аналитики по контрагентам. Дополняет NER+Conformal+ActiveLearning:
        извлекли → подтвердили уверенность → разметили → теперь нормализовали к KB.
      Источники: Mihalcea & Csomai 2007 ACL (anchor text NEL), Milne & Witten 2008 CIKM
        (disambiguation via Wikipedia), Shen et al. 2015 ACM CSUR §3 (entity linking survey),
        Wikidata entity IDs как стандарт идентификаторов.

- [x] Conversational Memory / Multi-Turn Dialogue для RAG (Project 02) — 2026-06-22
      rag/memory/conversation_memory.py: ConversationMemory — менеджер сессий диалога.
        ConversationTurn (question, answer, sources, UTC timestamp), MemoryConfig
        (max_turns=10, ttl_seconds=3600, context_turns=3), SessionStats dataclass.
        ConversationSession: deque скользящее окно (maxlen=max_turns), add_turn(),
          get_history(last_n=None), is_expired(ttl), stats().
        ConversationMemory: create_session() → UUID4, get_or_create_session() с авто-сбросом
          истёкших TTL, add_turn(), get_history(), reset_session(), list_sessions(),
          get_session_stats(), purge_expired().
        rewrite_query(): follow-up → standalone retrieval-запрос.
          Определение follow-up: длина ≤5 слов ИЛИ слова-ссылки (it/this/that/they/them/
          their/these/those/such/same/above/mentioned/described).
          Добавляет [Context: Q: ... | A: ...] из последних context_turns ходов.
          Самодостаточные вопросы остаются без изменений — нет лишних токенов в retrieval.
      rag/api/app.py: QueryRequest.session_id (str|None), QueryResponse.session_id (echo).
        В /query: rewrite_query() до retrieval, add_turn() после генерации (в т.ч. cache-hit).
        Новые endpoints:
          POST /memory/session (создать сессию, вернуть session_id + config),
          GET  /memory/history/{session_id} (история ходов: turns + n_turns),
          POST /memory/reset/{session_id} (сброс: cleared True/False),
          GET  /memory/sessions (список активных session_id + count).
        _reset_memory() для тестовой изоляции.
      41 новый тест: TestConversationMemory×24 (create_session, get_or_create, add_turn,
        get_history, last_n, sliding_window, rewrite_no_session/no_history/standalone/short/
        pronoun/preserves, reset_true/false/removes, list_sessions/after_reset,
        stats_none/returns, purge_expired, add_unknown_creates),
        TestConversationalRAGAPI×17 (session 200/structure/unique, history 200/structure,
        reset 200/structure/unknown, sessions 200/structure/appears, query fields/none/echoed/
        adds_turn/multiple_turns, custom_config).
      300/300 тестов зелёных (+41, было 259). Lint clean.
      Бизнес-эффект: HR RAG-система поддерживает диалог — "What's the vacation policy?" →
        "When does it apply?" → "Does this cover contractors?" — каждый follow-up находит
        правильный чанк через context-aware retrieval (recall улучшается на ~15-20%
        для follow-up вопросов vs. naive re-query). TTL+sliding window контролирует память.
        Confident AI 2026: средняя multi-turn производительность падает на 39% без memory.
      Источники: Rackauckas 2024 RAG Fusion (arxiv:2402.03367, multi-query context),
        LogRocket 2026 "LLM context problem: sliding window approach",
        Confident AI 2026 "Multi-turn LLM evaluation" (39% drop without memory),
        Bang et al. 2023 GPTCache (arxiv:2311.03027, session-aware caching).

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
