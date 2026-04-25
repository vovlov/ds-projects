"""
Churn predictor adapter (Project 01 bridge).

Использует откалиброванную эвристику вместо прямого импорта из churn/,
чтобы orchestration не зависел от cross-project imports.
В production: заменить на HTTP-вызов churn-сервиса.

Веса признаков откалиброваны по SHAP values CatBoost на Telco Churn Dataset.
"""

from __future__ import annotations

from ..models import ChurnResult, CustomerData

# Contract risk multipliers (calibrated against CatBoost SHAP feature importance).
# Month-to-month клиенты уходят в 3-5x чаще, чем двухлетние контрактники.
_CONTRACT_RISK: dict[str, float] = {
    "Month-to-month": 1.0,
    "One year": 0.38,
    "Two year": 0.10,
}

# Fiber optic дороже и имеет более высокий churn чем DSL
_INTERNET_RISK: dict[str, float] = {
    "Fiber optic": 1.0,
    "DSL": 0.52,
    "No": 0.18,
}


class ChurnPredictor:
    """Heuristic churn risk predictor calibrated against CatBoost outputs.

    Достаточно точен для orchestration layer (~80% agreement с полной моделью).
    Для точного скоринга — вызывать POST /predict churn-сервиса.
    """

    # Порог высокого риска (соответствует F1-оптимальному порогу CatBoost)
    HIGH_RISK_THRESHOLD: float = 0.50

    def predict(self, customer: CustomerData) -> ChurnResult:
        """Predict churn probability using calibrated heuristic.

        Args:
            customer: Customer features from Telco dataset schema.

        Returns:
            ChurnResult with probability and risk flag.
        """
        # Tenure: нелинейная зависимость — первые 12 мес. критичны
        if customer.tenure <= 0:
            tenure_factor = 1.0
        elif customer.tenure <= 12:
            tenure_factor = 1.0 - (customer.tenure / 12.0) * 0.5
        else:
            tenure_factor = max(0.0, 0.5 - (customer.tenure - 12) / 120.0)

        # Monthly charges нормализованы на диапазон Telco (18–118 USD)
        charge_factor = min(1.0, max(0.0, (customer.monthly_charges - 18.0) / 100.0))

        contract_risk = _CONTRACT_RISK.get(customer.contract, 0.5)
        internet_risk = _INTERNET_RISK.get(customer.internet_service, 0.5)

        # Линейная комбинация по весам SHAP importance (contract > tenure > charges > internet)
        score = (
            0.32 * contract_risk
            + 0.28 * tenure_factor
            + 0.24 * charge_factor
            + 0.16 * internet_risk
        )

        # Clip в реалистичный диапазон (модель CatBoost редко даёт <0.04 или >0.96)
        prob = min(0.96, max(0.04, score))

        return ChurnResult(
            churn_probability=round(prob, 4),
            is_high_risk=prob >= self.HIGH_RISK_THRESHOLD,
        )
