"""AST-based code quality metrics: Cyclomatic + Cognitive Complexity.

Реализует статический анализ Python-кода без внешних зависимостей через
стандартный модуль `ast`. Используется для приоритизации LLM code review:
функции с высокой сложностью — главные кандидаты на review.

Источники:
- McCabe 1976 "A Complexity Measure" IEEE TSE 2(4)
- SonarQube Cognitive Complexity whitepaper v1.5 (2023)
- Halstead 1977 "Elements of Software Science" Elsevier
- Welker et al. 1997 "Software Maintainability Index Revisited" CrossTalk
"""

from __future__ import annotations

import ast
import math
from dataclasses import dataclass, field

# ── Risk classification (NIST SP 500-235, McCabe 1996) ───────────────────────

_CC_THRESHOLDS = [
    (5, "low"),
    (10, "medium"),
    (15, "high"),
]
_CC_VERY_HIGH = "very_high"


def _cc_risk(cc: int) -> str:
    for threshold, level in _CC_THRESHOLDS:
        if cc <= threshold:
            return level
    return _CC_VERY_HIGH


# ── Dataclasses ──────────────────────────────────────────────────────────────


@dataclass
class FunctionMetrics:
    """Метрики сложности одной функции / Per-function complexity metrics."""

    name: str
    lineno: int
    cyclomatic_complexity: int
    cognitive_complexity: int
    n_lines: int
    n_params: int
    halstead_volume: float
    maintainability_index: float
    risk_level: str  # low / medium / high / very_high

    def to_dict(self) -> dict:
        return {
            "name": self.name,
            "lineno": self.lineno,
            "cyclomatic_complexity": self.cyclomatic_complexity,
            "cognitive_complexity": self.cognitive_complexity,
            "n_lines": self.n_lines,
            "n_params": self.n_params,
            "halstead_volume": round(self.halstead_volume, 2),
            "maintainability_index": round(self.maintainability_index, 1),
            "risk_level": self.risk_level,
        }


@dataclass
class CodeMetrics:
    """Агрегированные метрики файла / File-level aggregate metrics."""

    functions: list[FunctionMetrics] = field(default_factory=list)
    total_lines: int = 0
    n_functions: int = 0
    average_cc: float = 0.0
    max_cc: int = 0
    high_risk_functions: list[str] = field(default_factory=list)
    parse_error: str | None = None

    def to_dict(self) -> dict:
        result: dict = {
            "total_lines": self.total_lines,
            "n_functions": self.n_functions,
            "average_cc": round(self.average_cc, 2),
            "max_cc": self.max_cc,
            "high_risk_functions": self.high_risk_functions,
            "functions": [f.to_dict() for f in self.functions],
        }
        if self.parse_error:
            result["parse_error"] = self.parse_error
        return result


# ── Core metric computation ──────────────────────────────────────────────────


def _cyclomatic_complexity(
    func_node: ast.FunctionDef | ast.AsyncFunctionDef,
) -> int:
    """Cyclomatic complexity (McCabe 1976): decision branches + 1.

    Не считает сложность вложенных функций — у каждой своя CC.
    Boolean operators: n операндов = n-1 дополнительных ветвей.
    """
    cc = [1]

    def walk(node: ast.AST) -> None:
        for child in ast.iter_child_nodes(node):
            # Nested definitions are measured independently — skip entirely
            if isinstance(child, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
                continue
            if isinstance(child, (ast.If, ast.For, ast.While, ast.ExceptHandler)):
                cc[0] += 1
            elif isinstance(child, ast.BoolOp):
                # Each additional operand adds a branch: (a and b and c) → 2 branches
                cc[0] += len(child.values) - 1
            walk(child)

    walk(func_node)
    return cc[0]


def _cognitive_complexity(
    func_node: ast.FunctionDef | ast.AsyncFunctionDef,
) -> int:
    """Cognitive complexity (SonarQube-inspired): nesting-weighted structural score.

    Нагрузка на понимание растёт нелинейно с уровнем вложенности.
    Каждый структурный узел: +1 + текущая глубина вложенности.
    Boolean operators: +1 за каждую последовательность (не за каждый оператор).
    Вложенные функции: +1 за само определение, без рекурсии в тело.
    """
    _STRUCTURAL = (ast.If, ast.For, ast.While, ast.ExceptHandler, ast.With)
    cog = [0]

    def walk(node: ast.AST, nesting: int) -> None:
        for child in ast.iter_child_nodes(node):
            if isinstance(child, (ast.FunctionDef, ast.AsyncFunctionDef)):
                # Nested function: one increment, don't recurse (measured separately)
                cog[0] += nesting + 1
            elif isinstance(child, (ast.Lambda,) + _STRUCTURAL):
                cog[0] += nesting + 1
                walk(child, nesting + 1)
            elif isinstance(child, ast.BoolOp):
                # Each distinct boolean sequence is +1 regardless of operand count
                cog[0] += 1
                walk(child, nesting)
            else:
                walk(child, nesting)

    # Walk the root function's body without counting the function itself
    walk(func_node, nesting=0)
    return cog[0]


def _halstead_volume(func_node: ast.AST) -> float:
    """Halstead volume (1977): (N1+N2) * log2(n1+n2).

    n1/n2 — уникальные операторы/операнды, N1/N2 — общие.
    Аппроксимация через AST: BinOp/UnaryOp/BoolOp/Compare как операторы,
    Name/Constant как операнды.
    """
    operators: set[str] = set()
    total_ops = 0
    operands: set[str] = set()
    total_operands = 0

    for node in ast.walk(func_node):
        if isinstance(node, (ast.BinOp, ast.UnaryOp, ast.BoolOp)):
            op = type(node.op).__name__
            operators.add(op)
            total_ops += 1
        elif isinstance(node, ast.Compare):
            for cmp_op in node.ops:
                op = type(cmp_op).__name__
                operators.add(op)
                total_ops += 1
        elif isinstance(node, ast.Call):
            operators.add("call")
            total_ops += 1
        elif isinstance(node, ast.Attribute):
            operators.add(".")
            total_ops += 1
        elif isinstance(node, ast.Name):
            operands.add(node.id)
            total_operands += 1
        elif isinstance(node, ast.Constant):
            operands.add(repr(node.value))
            total_operands += 1

    n1 = max(len(operators), 1)
    n2 = max(len(operands), 1)
    N = total_ops + total_operands
    vocabulary = n1 + n2
    return N * math.log2(vocabulary) if vocabulary > 1 else 0.0


def _maintainability_index(
    halstead_volume: float,
    cc: int,
    loc: int,
) -> float:
    """Maintainability Index (Welker et al. 1997, used by Visual Studio).

    MI = max(0, (171 - 5.2·ln(V) - 0.23·CC - 16.2·ln(LOC)) * 100/171)
    Диапазон: 0-100, выше = лучше (>80 — хорошо поддерживаемый код).
    """
    vol = max(halstead_volume, 1.0)
    lines = max(loc, 1)
    mi_raw = 171 - 5.2 * math.log(vol) - 0.23 * cc - 16.2 * math.log(lines)
    return max(0.0, mi_raw * 100 / 171)


# ── Analyzer ─────────────────────────────────────────────────────────────────


class ASTAnalyzer:
    """Статический анализатор Python-кода через AST.

    Анализирует каждую top-level функцию и метод класса.
    Graceful fallback при SyntaxError — возвращает parse_error в CodeMetrics.

    Static Python code analyzer via AST.
    Analyzes each top-level function and class method independently.
    """

    #: Functions with CC above this threshold are flagged as high risk
    HIGH_RISK_CC: int = 10

    def analyze(self, source: str) -> CodeMetrics:
        """Разобрать исходный код и вернуть метрики сложности.

        Parse Python source code and return complexity metrics.
        Returns CodeMetrics with parse_error set on SyntaxError.
        """
        total_lines = len(source.splitlines())
        try:
            tree = ast.parse(source)
        except SyntaxError as exc:
            return CodeMetrics(
                total_lines=total_lines,
                parse_error=str(exc),
            )

        func_nodes = self._collect_functions(tree)
        metrics: list[FunctionMetrics] = []

        for node in func_nodes:
            m = self._metrics_for(node)
            metrics.append(m)

        if metrics:
            all_cc = [m.cyclomatic_complexity for m in metrics]
            avg_cc = sum(all_cc) / len(all_cc)
            max_cc = max(all_cc)
        else:
            avg_cc = 0.0
            max_cc = 0

        high_risk = [m.name for m in metrics if m.cyclomatic_complexity > self.HIGH_RISK_CC]

        return CodeMetrics(
            functions=metrics,
            total_lines=total_lines,
            n_functions=len(metrics),
            average_cc=avg_cc,
            max_cc=max_cc,
            high_risk_functions=high_risk,
        )

    # ── helpers ──────────────────────────────────────────────────────────────

    def _collect_functions(
        self,
        tree: ast.Module,
    ) -> list[ast.FunctionDef | ast.AsyncFunctionDef]:
        """Собрать top-level функции и методы классов."""
        funcs: list[ast.FunctionDef | ast.AsyncFunctionDef] = []
        for node in ast.iter_child_nodes(tree):
            if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
                funcs.append(node)
            elif isinstance(node, ast.ClassDef):
                for child in ast.iter_child_nodes(node):
                    if isinstance(child, (ast.FunctionDef, ast.AsyncFunctionDef)):
                        funcs.append(child)
        return funcs

    def _metrics_for(
        self,
        node: ast.FunctionDef | ast.AsyncFunctionDef,
    ) -> FunctionMetrics:
        """Вычислить все метрики для одной функции."""
        cc = _cyclomatic_complexity(node)
        cog = _cognitive_complexity(node)

        # Lines of code: end_lineno - lineno (both inclusive)
        loc = (node.end_lineno or node.lineno) - node.lineno + 1

        # Parameters: args.args excludes *args/**kwargs for simplicity
        n_params = len(node.args.args) + len(node.args.posonlyargs)

        hv = _halstead_volume(node)
        mi = _maintainability_index(hv, cc, loc)
        risk = _cc_risk(cc)

        return FunctionMetrics(
            name=node.name,
            lineno=node.lineno,
            cyclomatic_complexity=cc,
            cognitive_complexity=cog,
            n_lines=loc,
            n_params=n_params,
            halstead_volume=hv,
            maintainability_index=mi,
            risk_level=risk,
        )
