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
