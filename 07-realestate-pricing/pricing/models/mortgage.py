"""Ипотечный калькулятор и анализ доходности аренды для московской недвижимости.

Mortgage Calculator and Rental Yield Analysis for Moscow real estate.

Аннуитетная формула: M = P·r(1+r)^n / ((1+r)^n − 1)
  P — тело кредита, r = annual_rate/12, n = term_years×12.

Стандарты доступности (NAR / CFPB):
  28% rule — ипотечный платёж ≤ 28% валового месячного дохода (консервативно).
  43% rule — общий долг ≤ 43% дохода (CFPB Qualified Mortgage Act 2014).

Источники / Sources:
  - ЦБ РФ ключевая ставка + банковский спред 2025-2026 (cbr.ru)
  - ЦИАН Аналитика — ставки аренды и доходность 2024-2026
  - Consumer Financial Protection Bureau (CFPB) — 43% DTI Qualified Mortgage rule
  - NAR Housing Affordability Index — 28% front-end ratio standard
  - Basel III stress testing: +200bp rate shock (Bank for International Settlements)
"""

from __future__ import annotations

from dataclasses import dataclass

# Ипотечные программы России (ставки актуальны на 2026-06)
# Источник: cbr.ru + Сбербанк / ВТБ / Альфа-Банк публичные тарифы
MORTGAGE_PROGRAMS: dict[str, dict] = {
    "standard": {
        "name": "Стандартная ипотека",
        "annual_rate": 0.165,
        "min_down_payment": 0.20,
        "max_term_years": 30,
        "description": "Рыночная ставка: ключевая ставка ЦБ + банковский спред ~3%",
    },
    "family": {
        "name": "Семейная ипотека",
        "annual_rate": 0.06,
        "min_down_payment": 0.20,
        "max_term_years": 30,
        "description": "Для семей с детьми до 7 лет (Постановление Правительства № 1084)",
    },
    "it": {
        "name": "IT-ипотека",
        "annual_rate": 0.05,
        "min_down_payment": 0.20,
        "max_term_years": 30,
        "description": "Для сотрудников аккредитованных IT-компаний (Минцифры России)",
    },
    "preferential": {
        "name": "Льготная ипотека",
        "annual_rate": 0.08,
        "min_down_payment": 0.20,
        "max_term_years": 30,
        "description": "Льготная программа с господдержкой для первичного жилья",
    },
}

# Медианные ставки аренды руб/кв.м в месяц по районам Москвы (ЦИАН 2026)
NEIGHBORHOOD_RENT_RATES: dict[str, float] = {
    "Арбат": 220.0,
    "Пресненский": 240.0,
    "Хамовники": 210.0,
    "Замоскворечье": 200.0,
    "Раменки": 185.0,
    "Строгино": 170.0,
    "Ясенево": 160.0,
    "Марьино": 155.0,
    "Бутово": 150.0,
    "Отрадное": 160.0,
    "Медведково": 165.0,
    "Преображенское": 170.0,
    "Сокольники": 175.0,
    "Щукино": 165.0,
    "Зеленоград": 140.0,
}

# Расходы арендодателя: НПД 4% (самозанятый) + простой ~2 мес/год + ремонт
# Итого ~20% от годовой аренды — типовая оценка для российского рынка
DEFAULT_EXPENSE_RATIO = 0.20

# Стресс-тест: шок ставки +200 б.п. (Базель III / ЦБ РФ требование к банкам)
STRESS_TEST_SPREAD = 0.02


@dataclass
class MortgageConfig:
    """Параметры ипотечного кредита."""

    annual_rate: float = 0.165
    term_years: int = 20
    down_payment_ratio: float = 0.20
    program: str = "standard"


@dataclass
class MortgageResult:
    """Полный расчёт ипотечного кредита."""

    loan_amount: float
    down_payment: float
    monthly_payment: float
    total_payment: float
    total_interest: float
    ltv_ratio: float
    effective_annual_rate: float
    n_payments: int
    program: str

    def to_dict(self) -> dict:
        return {
            "loan_amount": round(self.loan_amount),
            "down_payment": round(self.down_payment),
            "monthly_payment": round(self.monthly_payment),
            "total_payment": round(self.total_payment),
            "total_interest": round(self.total_interest),
            "ltv_ratio": round(self.ltv_ratio, 4),
            "effective_annual_rate": round(self.effective_annual_rate, 4),
            "n_payments": self.n_payments,
            "program": self.program,
        }


@dataclass
class RentalYieldResult:
    """Доходность сдачи квартиры в аренду."""

    monthly_rent: float
    annual_rent: float
    gross_yield_pct: float
    net_yield_pct: float
    payback_years: float
    price_to_rent_ratio: float
    annual_expenses_estimated: float

    def to_dict(self) -> dict:
        return {
            "monthly_rent": round(self.monthly_rent),
            "annual_rent": round(self.annual_rent),
            "gross_yield_pct": round(self.gross_yield_pct, 2),
            "net_yield_pct": round(self.net_yield_pct, 2),
            "payback_years": round(self.payback_years, 1),
            "price_to_rent_ratio": round(self.price_to_rent_ratio, 1),
            "annual_expenses_estimated": round(self.annual_expenses_estimated),
        }


@dataclass
class AffordabilityResult:
    """Оценка доступности ипотеки по правилам 28%/43% (NAR / CFPB)."""

    monthly_payment: float
    annual_income: float
    dti_mortgage_only: float
    is_affordable_28: bool
    is_affordable_43: bool
    recommended_income_annual: float
    stress_test_rate: float
    stress_test_payment: float

    def to_dict(self) -> dict:
        return {
            "monthly_payment": round(self.monthly_payment),
            "annual_income": round(self.annual_income),
            "dti_mortgage_only": round(self.dti_mortgage_only, 4),
            "is_affordable_28": self.is_affordable_28,
            "is_affordable_43": self.is_affordable_43,
            "recommended_income_annual": round(self.recommended_income_annual),
            "stress_test_rate": round(self.stress_test_rate, 4),
            "stress_test_payment": round(self.stress_test_payment),
        }


@dataclass
class InvestmentAnalysis:
    """Сводный анализ инвестиционной привлекательности (аренда vs ипотека)."""

    price: float
    mortgage: MortgageResult
    rental: RentalYieldResult
    # Положительный cashflow: аренда покрывает ипотеку и расходы
    monthly_cashflow: float
    is_cashflow_positive: bool
    # Сколько месяцев аренды нужно чтобы покрыть первоначальный взнос
    down_payment_recovery_months: float
    investment_verdict: str  # "strong_buy" | "buy" | "hold" | "avoid"

    def to_dict(self) -> dict:
        return {
            "price": round(self.price),
            "mortgage": self.mortgage.to_dict(),
            "rental": self.rental.to_dict(),
            "monthly_cashflow": round(self.monthly_cashflow),
            "is_cashflow_positive": self.is_cashflow_positive,
            "down_payment_recovery_months": round(self.down_payment_recovery_months, 1),
            "investment_verdict": self.investment_verdict,
        }


class MortgageCalculator:
    """Ипотечный калькулятор: аннуитет, доходность аренды, анализ доступности."""

    @staticmethod
    def compute_monthly_payment(
        loan_amount: float,
        annual_rate: float,
        term_years: int,
    ) -> float:
        """Аннуитетный платёж по формуле банковского стандарта.

        При annual_rate=0 — равномерное погашение тела (беспроцентная рассрочка).
        """
        if loan_amount <= 0:
            return 0.0
        n = term_years * 12
        if annual_rate <= 0:
            return loan_amount / n
        r = annual_rate / 12
        return loan_amount * r * (1 + r) ** n / ((1 + r) ** n - 1)

    @staticmethod
    def compute_mortgage(price: float, config: MortgageConfig) -> MortgageResult:
        """Рассчитать полные параметры ипотечного кредита."""
        down_payment = price * config.down_payment_ratio
        loan_amount = price - down_payment
        n = config.term_years * 12

        monthly = MortgageCalculator.compute_monthly_payment(
            loan_amount, config.annual_rate, config.term_years
        )
        total_payment = monthly * n
        total_interest = total_payment - loan_amount

        return MortgageResult(
            loan_amount=loan_amount,
            down_payment=down_payment,
            monthly_payment=monthly,
            total_payment=total_payment,
            total_interest=total_interest,
            ltv_ratio=loan_amount / price if price > 0 else 0.0,
            effective_annual_rate=config.annual_rate,
            n_payments=n,
            program=config.program,
        )

    @staticmethod
    def compute_rental_yield(
        price: float,
        monthly_rent: float,
        expense_ratio: float = DEFAULT_EXPENSE_RATIO,
    ) -> RentalYieldResult:
        """Рассчитать валовую и чистую доходность аренды.

        expense_ratio — расходы как доля годовой аренды:
          ~4% НПД (самозанятый) + ~8% простой (1 мес/год) + ~8% ремонт/расходники.
          Консервативный дефолт 20% (ЦИАН: реальные landlords платят 15-25%).
        """
        annual_rent = monthly_rent * 12
        annual_expenses = annual_rent * expense_ratio
        annual_net_rent = annual_rent - annual_expenses

        gross_yield_pct = (annual_rent / price * 100) if price > 0 else 0.0
        net_yield_pct = (annual_net_rent / price * 100) if price > 0 else 0.0
        payback_years = (price / annual_net_rent) if annual_net_rent > 0 else float("inf")
        p_to_r = (price / monthly_rent) if monthly_rent > 0 else float("inf")

        return RentalYieldResult(
            monthly_rent=monthly_rent,
            annual_rent=annual_rent,
            gross_yield_pct=gross_yield_pct,
            net_yield_pct=net_yield_pct,
            payback_years=payback_years,
            price_to_rent_ratio=p_to_r,
            annual_expenses_estimated=annual_expenses,
        )

    @staticmethod
    def estimate_market_rent(neighborhood: str, sqft: float) -> float:
        """Оценить рыночную ставку аренды (руб/мес) по площади и району.

        Fallback: медианная ставка по всем районам, если район неизвестен.
        """
        rate = NEIGHBORHOOD_RENT_RATES.get(neighborhood)
        if rate is None:
            rate = sum(NEIGHBORHOOD_RENT_RATES.values()) / len(NEIGHBORHOOD_RENT_RATES)
        return rate * sqft

    @staticmethod
    def compute_affordability(
        monthly_payment: float,
        annual_income: float,
        annual_rate: float,
        term_years: int,
        loan_amount: float,
    ) -> AffordabilityResult:
        """Оценить доступность ипотеки по доходу.

        28% rule (NAR): front-end DTI ≤ 28% — консервативный банковский стандарт.
        43% rule (CFPB): total DTI ≤ 43% — юридический предел Qualified Mortgage.
        Стресс-тест +200 б.п.: требование ЦБ РФ / Базель III к буферу платёжеспособности.
        """
        monthly_income = annual_income / 12
        dti = monthly_payment / monthly_income if monthly_income > 0 else float("inf")

        recommended_annual = (monthly_payment / 0.28) * 12

        stress_rate = annual_rate + STRESS_TEST_SPREAD
        stress_payment = MortgageCalculator.compute_monthly_payment(
            loan_amount, stress_rate, term_years
        )

        return AffordabilityResult(
            monthly_payment=monthly_payment,
            annual_income=annual_income,
            dti_mortgage_only=dti,
            is_affordable_28=(dti <= 0.28),
            is_affordable_43=(dti <= 0.43),
            recommended_income_annual=recommended_annual,
            stress_test_rate=stress_rate,
            stress_test_payment=stress_payment,
        )

    @staticmethod
    def analyze_investment(
        price: float,
        mortgage_config: MortgageConfig,
        monthly_rent: float,
        expense_ratio: float = DEFAULT_EXPENSE_RATIO,
    ) -> InvestmentAnalysis:
        """Сводный инвестиционный анализ: аренда vs ипотека + cashflow.

        Вердикт:
          strong_buy  — положительный cashflow + окупаемость ≤ 20 лет
          buy         — отрицательный cashflow, но окупаемость ≤ 25 лет
          hold        — окупаемость 25-35 лет
          avoid       — окупаемость > 35 лет
        """
        mortgage = MortgageCalculator.compute_mortgage(price, mortgage_config)
        rental = MortgageCalculator.compute_rental_yield(price, monthly_rent, expense_ratio)

        monthly_net_rent = monthly_rent * (1 - expense_ratio)
        monthly_cashflow = monthly_net_rent - mortgage.monthly_payment

        down_payment_recovery = (
            mortgage.down_payment / monthly_net_rent if monthly_net_rent > 0 else float("inf")
        )

        py = rental.payback_years
        if monthly_cashflow >= 0 and py <= 20:
            verdict = "strong_buy"
        elif py <= 25:
            verdict = "buy"
        elif py <= 35:
            verdict = "hold"
        else:
            verdict = "avoid"

        return InvestmentAnalysis(
            price=price,
            mortgage=mortgage,
            rental=rental,
            monthly_cashflow=monthly_cashflow,
            is_cashflow_positive=(monthly_cashflow >= 0),
            down_payment_recovery_months=down_payment_recovery,
            investment_verdict=verdict,
        )
