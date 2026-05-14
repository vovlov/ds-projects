"""Uplift modeling module: T-Learner CATE estimation for retention campaigns."""

from .learner import TLearner, TrainSummary, UpliftConfig, UpliftPrediction
from .qini import QiniResult, compute_auuc, compute_qini_curve, qini_coefficient

__all__ = [
    "TLearner",
    "UpliftConfig",
    "UpliftPrediction",
    "TrainSummary",
    "compute_qini_curve",
    "compute_auuc",
    "qini_coefficient",
    "QiniResult",
]
