# MLOps Maturity

This portfolio maps to the [Microsoft MLOps Maturity Model](https://learn.microsoft.com/en-us/azure/architecture/ai-ml/guide/mlops-maturity-model).

## Current Position: Level 2

```
Level 0  No MLOps             ✅  Yandex Practicum 2020 (notebooks only)
Level 1  DevOps but no MLOps  ✅  CI/CD · Docker · automated tests
Level 2  Training automation  ✅  MLflow · Optuna · DVC · drift detection
Level 3  Automated deployment ⬜  Next goal (BentoML · Kubernetes)
Level 4  Full MLOps           ⬜  Horizon
```

## What's Implemented at Each Level

### Level 1 (Complete)
- GitHub Actions CI across all 10 projects
- Docker Compose stacks per project
- 185+ automated tests (pytest)
- ruff + mypy static analysis
- Pre-commit hooks

### Level 2 (Complete)
- MLflow experiment tracking and model registry (Projects 01, 07)
- Optuna hyperparameter optimisation (Projects 01, 07)
- DVC data versioning (Project 01)
- PSI-based drift detection + automated retraining trigger (Projects 01, 05, 10)
- MMD drift detection with bootstrap threshold (Project 05)
- WAP (Write-Audit-Publish) gate for feature store (Project 09)
- Cross-project drift alerting (Projects 01 ↔ 10)
- RAGAS evaluation for RAG quality (Project 02)
- Multi-model consistency scoring (Project 08)

### Level 3 (In Progress)
- [ ] BentoML model serving
- [ ] A/B testing framework
- [ ] Kubernetes deployment manifests
- [ ] Automated model comparison reports
- [ ] mkdocs documentation site ← *you are here*

### Level 4 (Planned)
- [ ] Feature store with Feast
- [ ] Schema registry + data contracts
- [ ] Data lineage visualisation
- [ ] Multi-model orchestration pipeline
- [ ] Cost optimisation (quantisation, batching)

---

## Comparison with Industry Benchmarks

| Capability | Made With ML | This Portfolio |
|-----------|-------------|----------------|
| Pre-commit hooks | ✅ | ✅ |
| Model serving (non-FastAPI) | ✅ Ray/BentoML | ⬜ planned |
| Automated retraining (CT) | ✅ | ✅ PSI + MMD |
| Data versioning | ✅ DVC pipeline | ✅ Project 01 |
| Model registry | ✅ MLflow | ✅ Projects 01, 07 |
| A/B testing | ✅ | ⬜ planned |
| Production drift monitoring | ✅ | ✅ Projects 01, 05, 09, 10 |
| Documentation site | ✅ mkdocs | ✅ *this site* |
| Deployment to cloud | ✅ | ✅ Streamlit Cloud (01, 05, 07) |
| Faithfulness evaluation | ⬜ | ✅ Project 02 |
| Graph-based fraud detection | ⬜ | ✅ Project 04 |
| Explainability in API | ⬜ | ✅ Projects 06, 07 |
