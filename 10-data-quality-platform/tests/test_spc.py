"""
Тесты для SPC (Statistical Process Control) / SPC control chart tests.

Покрытие:
- TestSPCCalibration: калибровка центральной линии и пределов
- TestWesternElectricRules: 4 правила WECO 1956
- TestShewhartChartUpdate: онлайн-обновление и состояние
- TestShewhartBatchDetect: батч-детекция
- TestSPCAPIEndpoints: REST API (calibrate/update/detect/status/reset/full_cycle)
"""

from __future__ import annotations

import pytest
from fastapi.testclient import TestClient
from quality.api.app import _reset_spc_charts, app
from quality.spc.control_charts import (
    ShewhartChart,
    SPCConfig,
    ViolationType,
)

client = TestClient(app)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture(autouse=True)
def reset_charts():
    """Изолировать тесты: сбрасывать все карты перед каждым тестом."""
    _reset_spc_charts()
    yield
    _reset_spc_charts()


def _normal_values(n: int = 30, mean: float = 0.05, std: float = 0.005, seed: int = 42) -> list:
    """Синтетические нормальные значения (null rate, например) / Synthetic normal values."""
    import random

    rng = random.Random(seed)
    return [max(0.0, mean + rng.gauss(0, std)) for _ in range(n)]


# ---------------------------------------------------------------------------
# TestSPCCalibration
# ---------------------------------------------------------------------------


class TestSPCCalibration:
    def test_calibrate_sets_is_calibrated(self):
        chart = ShewhartChart()
        chart.calibrate(_normal_values())
        assert chart.is_calibrated is True

    def test_center_line_equals_mean(self):
        values = [1.0, 2.0, 3.0, 4.0, 5.0] * 3
        chart = ShewhartChart()
        result = chart.calibrate(values)
        assert abs(result.center_line - 3.0) < 1e-9

    def test_sigma_positive(self):
        chart = ShewhartChart()
        result = chart.calibrate(_normal_values())
        assert result.sigma > 0

    def test_ucl_above_center(self):
        chart = ShewhartChart()
        result = chart.calibrate(_normal_values())
        assert result.ucl > result.center_line

    def test_lcl_below_center(self):
        chart = ShewhartChart()
        result = chart.calibrate(_normal_values())
        assert result.lcl < result.center_line

    def test_ucl_lcl_symmetric(self):
        chart = ShewhartChart()
        result = chart.calibrate(_normal_values())
        diff_up = result.ucl - result.center_line
        diff_down = result.center_line - result.lcl
        assert abs(diff_up - diff_down) < 1e-9

    def test_too_few_samples_raises(self):
        chart = ShewhartChart()
        with pytest.raises(ValueError, match="минимум|Need at least"):
            chart.calibrate([1.0, 2.0, 3.0])  # < 10

    def test_constant_series_no_crash(self):
        """Константная серия → sigma ≈ 0 → graceful fallback к epsilon."""
        chart = ShewhartChart()
        values = [5.0] * 20
        result = chart.calibrate(values)
        assert result.sigma > 0  # защита от деления на ноль
        assert result.center_line == pytest.approx(5.0)

    def test_calibration_result_to_dict_keys(self):
        chart = ShewhartChart()
        result = chart.calibrate(_normal_values())
        d = result.to_dict()
        for key in ("center_line", "sigma", "ucl", "lcl", "n_samples", "is_calibrated"):
            assert key in d

    def test_n_sigma_custom(self):
        chart = ShewhartChart(SPCConfig(n_sigma=2.0))
        values = [0.0] * 20
        result = chart.calibrate(values)
        # UCL = 0 + 2.0 * epsilon
        assert result.ucl == pytest.approx(result.center_line + 2.0 * result.sigma)


# ---------------------------------------------------------------------------
# TestWesternElectricRules
# ---------------------------------------------------------------------------


class TestWesternElectricRules:
    """Тесты 4-х правил WECO 1956."""

    def _calibrated_chart(self):
        chart = ShewhartChart(SPCConfig(n_sigma=3.0))
        # Калибруем на значениях с μ=0, σ=1
        import random

        rng = random.Random(0)
        calibration = [rng.gauss(0, 1) for _ in range(30)]
        chart.calibrate(calibration)
        # Пересобираем с ровными параметрами
        chart._center_line = 0.0
        chart._sigma = 1.0
        chart._buffer.clear()
        return chart

    def test_rule_1_beyond_3sigma(self):
        chart = self._calibrated_chart()
        result = chart.update(10.0)  # далеко за 3σ
        assert result.violation == ViolationType.RULE_1_BEYOND_3SIGMA
        assert result.is_out_of_control is True

    def test_rule_1_not_triggered_at_2sigma(self):
        chart = self._calibrated_chart()
        result = chart.update(2.0)  # в пределах 3σ
        assert result.violation != ViolationType.RULE_1_BEYOND_3SIGMA

    def test_rule_2_two_of_three_above_2sigma(self):
        chart = self._calibrated_chart()
        # Добавим 2 точки выше 2σ из 3 последних
        chart.update(0.0)  # 1-я нейтральная
        chart.update(2.5)  # выше 2σ
        chart.update(0.5)  # нейтральная
        chart.update(2.5)  # выше 2σ — теперь 2 из 3 последних выше 2σ
        chart.update(0.1)
        # Проверим: в последних 3 позициях 2.5, 0.1 → нет... Нужно иначе
        # Выстраиваем точно: последние 3 = [2.5, 2.5, <check point>]
        chart2 = self._calibrated_chart()
        chart2.update(2.5)
        chart2.update(2.5)
        r = chart2.update(0.0)
        assert r.violation == ViolationType.RULE_2_TWO_OF_THREE_2SIGMA

    def test_rule_2_two_of_three_below_2sigma(self):
        chart = self._calibrated_chart()
        chart.update(-2.5)
        chart.update(-2.5)
        r = chart.update(0.0)
        assert r.violation == ViolationType.RULE_2_TWO_OF_THREE_2SIGMA

    def test_rule_2_only_one_does_not_trigger(self):
        chart = self._calibrated_chart()
        chart.update(0.0)
        chart.update(2.5)  # только одна за 2σ
        r = chart.update(0.0)
        # Только 1 из 3 выше 2σ → не нарушение Rule 2
        assert r.violation != ViolationType.RULE_2_TWO_OF_THREE_2SIGMA

    def test_rule_3_four_of_five_above_1sigma(self):
        chart = self._calibrated_chart()
        for _ in range(4):
            chart.update(1.5)  # выше 1σ
        r = chart.update(0.0)  # 5-я — нейтральная, но 4 из 5 = Rule 3
        assert r.violation == ViolationType.RULE_3_FOUR_OF_FIVE_1SIGMA

    def test_rule_3_four_of_five_below_1sigma(self):
        chart = self._calibrated_chart()
        for _ in range(4):
            chart.update(-1.5)
        r = chart.update(0.0)
        assert r.violation == ViolationType.RULE_3_FOUR_OF_FIVE_1SIGMA

    def test_rule_4_eight_consecutive_above_center(self):
        chart = self._calibrated_chart()
        for _ in range(8):
            chart.update(0.5)  # все выше центральной (z > 0)
        r = chart.update(0.5)
        assert r.violation == ViolationType.RULE_4_EIGHT_CONSECUTIVE

    def test_rule_4_eight_consecutive_below_center(self):
        chart = self._calibrated_chart()
        for _ in range(8):
            chart.update(-0.5)
        r = chart.update(-0.5)
        assert r.violation == ViolationType.RULE_4_EIGHT_CONSECUTIVE

    def test_rule_4_alternating_does_not_trigger(self):
        chart = self._calibrated_chart()
        for i in range(10):
            chart.update(0.5 if i % 2 == 0 else -0.5)
        r = chart.update(0.5)
        assert r.violation != ViolationType.RULE_4_EIGHT_CONSECUTIVE

    def test_none_for_in_control_point(self):
        chart = self._calibrated_chart()
        r = chart.update(0.5)  # нормальное значение
        assert r.violation == ViolationType.NONE
        assert r.is_out_of_control is False


# ---------------------------------------------------------------------------
# TestShewhartChartUpdate
# ---------------------------------------------------------------------------


class TestShewhartChartUpdate:
    def test_update_before_calibrate_raises(self):
        chart = ShewhartChart()
        with pytest.raises(RuntimeError):
            chart.update(1.0)

    def test_update_increments_n_points(self):
        chart = ShewhartChart()
        chart.calibrate(_normal_values())
        chart.update(0.05)
        r = chart.update(0.05)
        assert r.n_points == 2

    def test_normal_data_low_violation_rate(self):
        chart = ShewhartChart()
        calibration = _normal_values(n=30)
        chart.calibrate(calibration)
        test_vals = _normal_values(n=100, seed=7)
        violations = [chart.update(v).is_out_of_control for v in test_vals]
        # Нормальные данные → нарушений должно быть мало (< 10%)
        assert sum(violations) / len(violations) < 0.10

    def test_injected_spike_detected_rule1(self):
        chart = ShewhartChart()
        chart.calibrate(_normal_values(n=30))
        chart._center_line = 0.05
        chart._sigma = 0.005
        r = chart.update(0.15)  # на 20σ выше нормы
        assert r.violation == ViolationType.RULE_1_BEYOND_3SIGMA

    def test_z_score_sign(self):
        chart = ShewhartChart()
        chart.calibrate([0.0] * 20)
        chart._center_line = 0.0
        chart._sigma = 1.0
        r_pos = chart.update(2.0)
        assert r_pos.z_score > 0
        chart2 = ShewhartChart()
        chart2.calibrate([0.0] * 20)
        chart2._center_line = 0.0
        chart2._sigma = 1.0
        r_neg = chart2.update(-2.0)
        assert r_neg.z_score < 0

    def test_get_state_fields_before_calibrate(self):
        chart = ShewhartChart()
        state = chart.get_state()
        assert state.is_calibrated is False
        assert state.center_line is None
        assert state.sigma is None

    def test_get_state_after_calibrate(self):
        chart = ShewhartChart()
        chart.calibrate(_normal_values())
        state = chart.get_state()
        assert state.is_calibrated is True
        assert state.center_line is not None
        assert state.n_calibration >= 10

    def test_reset_clears_calibration(self):
        chart = ShewhartChart()
        chart.calibrate(_normal_values())
        chart.reset()
        assert chart.is_calibrated is False
        with pytest.raises(RuntimeError):
            chart.update(1.0)

    def test_n_violations_increments_on_spike(self):
        chart = ShewhartChart()
        chart.calibrate(_normal_values())
        chart._center_line = 0.05
        chart._sigma = 0.005
        before = chart.get_state().n_violations
        chart.update(0.20)  # spike
        after = chart.get_state().n_violations
        assert after == before + 1

    def test_control_limits_structure(self):
        chart = ShewhartChart()
        chart.calibrate(_normal_values())
        chart._center_line = 0.0
        chart._sigma = 1.0
        r = chart.update(0.5)
        # UCL > UCL_2sigma > UCL_1sigma > center > LCL_1sigma > LCL_2sigma > LCL
        assert r.ucl > r.ucl_2sigma > r.ucl_1sigma > r.center_line
        assert r.lcl < r.lcl_2sigma < r.lcl_1sigma < r.center_line


# ---------------------------------------------------------------------------
# TestShewhartBatchDetect
# ---------------------------------------------------------------------------


class TestShewhartBatchDetect:
    def test_detect_batch_before_calibrate_raises(self):
        chart = ShewhartChart()
        with pytest.raises(RuntimeError):
            chart.detect_batch([1.0, 2.0])

    def test_detect_batch_length_matches(self):
        chart = ShewhartChart()
        chart.calibrate(_normal_values(n=30))
        results = chart.detect_batch([0.05] * 10)
        assert len(results) == 10

    def test_detect_batch_violation_flagged(self):
        chart = ShewhartChart()
        chart.calibrate(_normal_values(n=30))
        chart._center_line = 0.05
        chart._sigma = 0.005
        # Один экстремальный выброс среди нормальных
        values = [0.05] * 5 + [0.30] + [0.05] * 5
        results = chart.detect_batch(values)
        violations = [r for r in results if r.is_out_of_control]
        assert len(violations) >= 1


# ---------------------------------------------------------------------------
# TestSPCAPIEndpoints
# ---------------------------------------------------------------------------


class TestSPCAPIEndpoints:
    """Тесты REST API контрольных карт."""

    def _calibration_payload(self, metric="null_rate", n=25):
        values = _normal_values(n=n, mean=0.02, std=0.003)
        return {"metric_name": metric, "values": values, "n_sigma": 3.0}

    def test_calibrate_201(self):
        r = client.post("/spc/calibrate", json=self._calibration_payload())
        assert r.status_code == 201

    def test_calibrate_response_structure(self):
        r = client.post("/spc/calibrate", json=self._calibration_payload())
        data = r.json()
        expected_fields = (
            "metric_name",
            "center_line",
            "sigma",
            "ucl",
            "lcl",
            "n_samples",
            "is_calibrated",
        )
        for field in expected_fields:
            assert field in data, f"Missing field: {field}"

    def test_calibrate_too_few_422(self):
        payload = {"metric_name": "x", "values": [0.1, 0.2], "n_sigma": 3.0}
        r = client.post("/spc/calibrate", json=payload)
        assert r.status_code == 422

    def test_calibrate_custom_n_sigma(self):
        payload = self._calibration_payload()
        payload["n_sigma"] = 2.0
        r = client.post("/spc/calibrate", json=payload)
        data = r.json()
        # UCL - center должно равняться 2σ
        diff = data["ucl"] - data["center_line"]
        expected = 2.0 * data["sigma"]
        assert abs(diff - expected) < 1e-6

    def test_update_400_before_calibrate(self):
        r = client.post("/spc/update", json={"metric_name": "uncalibrated", "value": 0.05})
        assert r.status_code == 400

    def test_update_200_after_calibrate(self):
        client.post("/spc/calibrate", json=self._calibration_payload("m1"))
        r = client.post("/spc/update", json={"metric_name": "m1", "value": 0.02})
        assert r.status_code == 200

    def test_update_response_structure(self):
        client.post("/spc/calibrate", json=self._calibration_payload("m2"))
        r = client.post("/spc/update", json={"metric_name": "m2", "value": 0.02})
        data = r.json()
        for field in (
            "metric_name",
            "value",
            "center_line",
            "ucl",
            "lcl",
            "z_score",
            "violation",
            "is_out_of_control",
            "n_points",
        ):
            assert field in data, f"Missing field: {field}"

    def test_update_violation_for_extreme_value(self):
        client.post("/spc/calibrate", json=self._calibration_payload("m3"))
        # Получаем параметры
        status_r = client.get("/spc/status/m3").json()
        cl = status_r["center_line"]
        sigma = status_r["sigma"]
        extreme = cl + 10 * sigma  # далеко за UCL
        r = client.post("/spc/update", json={"metric_name": "m3", "value": extreme})
        data = r.json()
        assert data["is_out_of_control"] is True
        assert data["violation"] == "rule_1_beyond_3sigma"

    def test_detect_400_before_calibrate(self):
        r = client.post("/spc/detect", json={"metric_name": "nc", "values": [0.1, 0.2]})
        assert r.status_code == 400

    def test_detect_200_after_calibrate(self):
        client.post("/spc/calibrate", json=self._calibration_payload("dm"))
        r = client.post("/spc/detect", json={"metric_name": "dm", "values": [0.02, 0.021, 0.019]})
        assert r.status_code == 200

    def test_detect_response_structure(self):
        client.post("/spc/calibrate", json=self._calibration_payload("ds"))
        r = client.post("/spc/detect", json={"metric_name": "ds", "values": [0.02, 0.02]})
        data = r.json()
        for field in (
            "metric_name",
            "n_values",
            "n_violations",
            "violation_indices",
            "violation_types",
            "results",
        ):
            assert field in data

    def test_detect_n_values_matches_input(self):
        client.post("/spc/calibrate", json=self._calibration_payload("dn"))
        vals = [0.02] * 7
        r = client.post("/spc/detect", json={"metric_name": "dn", "values": vals})
        assert r.json()["n_values"] == 7

    def test_detect_spike_creates_violation(self):
        client.post("/spc/calibrate", json=self._calibration_payload("dv"))
        status_r = client.get("/spc/status/dv").json()
        cl = status_r["center_line"]
        sigma = status_r["sigma"]
        values = [cl] * 4 + [cl + 20 * sigma] + [cl] * 4
        r = client.post("/spc/detect", json={"metric_name": "dv", "values": values})
        assert r.json()["n_violations"] >= 1

    def test_status_uncalibrated(self):
        r = client.get("/spc/status/nonexistent_metric")
        assert r.status_code == 200
        assert r.json()["is_calibrated"] is False

    def test_status_after_calibrate(self):
        client.post("/spc/calibrate", json=self._calibration_payload("sc"))
        r = client.get("/spc/status/sc")
        data = r.json()
        assert data["is_calibrated"] is True
        assert data["center_line"] is not None
        assert data["ucl"] is not None
        assert data["lcl"] is not None
        assert data["n_calibration"] >= 10

    def test_reset_200(self):
        client.post("/spc/calibrate", json=self._calibration_payload("rs"))
        r = client.post("/spc/reset?metric_name=rs")
        assert r.status_code == 200
        assert r.json()["reset"] is True

    def test_reset_uncalibrates(self):
        client.post("/spc/calibrate", json=self._calibration_payload("ru"))
        client.post("/spc/reset?metric_name=ru")
        status = client.get("/spc/status/ru").json()
        assert status["is_calibrated"] is False

    def test_multiple_independent_metrics(self):
        """Разные метрики используют независимые карты."""
        p1 = self._calibration_payload("metric_A")
        p2 = self._calibration_payload("metric_B")
        p2["values"] = _normal_values(n=20, mean=500.0, std=10.0)
        client.post("/spc/calibrate", json=p1)
        client.post("/spc/calibrate", json=p2)

        s_a = client.get("/spc/status/metric_A").json()
        s_b = client.get("/spc/status/metric_B").json()

        # Две карты должны иметь разные центральные линии
        assert abs(s_a["center_line"] - s_b["center_line"]) > 1.0

    def test_full_cycle(self):
        """Полный цикл: calibrate → update (нормально) → update (spike) → detect."""
        # Calibrate
        payload = self._calibration_payload("fc")
        rc = client.post("/spc/calibrate", json=payload)
        assert rc.status_code == 201
        cl = rc.json()["center_line"]
        sigma = rc.json()["sigma"]

        # Update — нормальное значение
        rn = client.post("/spc/update", json={"metric_name": "fc", "value": cl})
        assert rn.json()["is_out_of_control"] is False

        # Update — экстремальный выброс
        re = client.post("/spc/update", json={"metric_name": "fc", "value": cl + 10 * sigma})
        assert re.json()["is_out_of_control"] is True

        # Status: n_violations >= 1
        rs = client.get("/spc/status/fc")
        assert rs.json()["n_violations"] >= 1

        # Batch detect
        rd = client.post("/spc/detect", json={"metric_name": "fc", "values": [cl] * 5})
        assert rd.json()["n_values"] == 5
