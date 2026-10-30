"""Candidate configuration -> unfitted sklearn Pipeline, and its resolved parameters."""

from __future__ import annotations

import math
from typing import Any

import numpy as np
from sklearn.dummy import DummyClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import (
    FunctionTransformer,
    MaxAbsScaler,
    MinMaxScaler,
    RobustScaler,
    StandardScaler,
)
from sklearn.tree import DecisionTreeClassifier

from .common import SEED

PREPROCESSORS = {
    "sklearn.preprocessing.StandardScaler": StandardScaler,
    "sklearn.preprocessing.MinMaxScaler": MinMaxScaler,
    "sklearn.preprocessing.MaxAbsScaler": MaxAbsScaler,
    "sklearn.preprocessing.RobustScaler": RobustScaler,
    "sklearn.preprocessing.FunctionTransformer": FunctionTransformer,
}
FUNCTIONS = {None: None, "numpy.log1p": np.log1p}
TREE_FIXED = {"criterion": "gini", "splitter": "best", "class_weight": None}
TREE_DEPTHS = {2, 3, 4, 5}
TREE_LEAVES = {20, 50, 100}
CLASSIFIER_STEP = "classifier"


class InvalidCandidate(ValueError):
    pass


def _seeded(params: dict[str, Any], estimator_class) -> dict[str, Any]:
    params = dict(params)
    if "random_state" in estimator_class().get_params(deep=False):
        if params.get("random_state", SEED) != SEED or (
            "random_state" in params and isinstance(params["random_state"], bool)
        ):
            raise InvalidCandidate("random_state must be 2026")
        params["random_state"] = SEED
    return params


def _preprocessing_step(step: Any, index: int) -> tuple[str, Any]:
    if not isinstance(step, dict) or set(step) != {"name", "class", "params"}:
        raise InvalidCandidate(f"preprocessing[{index}] must have name, class, params")
    name, class_name, params = step["name"], step["class"], step["params"]
    if not isinstance(name, str) or not name or name == CLASSIFIER_STEP or "__" in name:
        raise InvalidCandidate(f"preprocessing[{index}].name is invalid")
    if class_name not in PREPROCESSORS:
        raise InvalidCandidate(f"preprocessing[{index}].class {class_name!r} is not allowed")
    if not isinstance(params, dict):
        raise InvalidCandidate(f"preprocessing[{index}].params must be an object")
    estimator_class = PREPROCESSORS[class_name]
    params = dict(params)
    if estimator_class is FunctionTransformer:
        for key in ("func", "inverse_func"):
            if key in params:
                if params[key] not in FUNCTIONS:
                    raise InvalidCandidate(f"{key} must be null or numpy.log1p")
                params[key] = FUNCTIONS[params[key]]
    return name, estimator_class(**_seeded(params, estimator_class))


def build_pipeline(config: dict[str, Any]) -> Pipeline:
    """Construct the declared, unfitted pipeline; reads nothing and trains nothing."""
    if not isinstance(config, dict) or set(config) != {
        "experiment_id",
        "family",
        "preprocessing",
        "estimator_params",
    }:
        raise InvalidCandidate("candidate needs experiment_id, family, preprocessing, estimator_params")
    family, params = config["family"], config["estimator_params"]
    preprocessing = config["preprocessing"]
    if not isinstance(preprocessing, list) or not isinstance(params, dict):
        raise InvalidCandidate("preprocessing must be a list and estimator_params an object")
    steps = [_preprocessing_step(step, index) for index, step in enumerate(preprocessing)]
    if len({name for name, _ in steps}) != len(steps):
        raise InvalidCandidate("preprocessing step names must be unique")
    if family == "constant":
        if config["experiment_id"] != "constant" or preprocessing or params:
            raise InvalidCandidate("constant must be experiment_id=constant without settings")
        estimator = DummyClassifier(strategy="prior", random_state=SEED)
    elif family == "logistic":
        if params.get("class_weight") is not None:
            raise InvalidCandidate("class weights are not allowed")
        estimator = LogisticRegression(**_seeded(params, LogisticRegression))
    elif family == "tree":
        unknown = set(params) - {"max_depth", "min_samples_leaf", "random_state", *TREE_FIXED}
        if unknown:
            raise InvalidCandidate("tree settings must stay default: " + ", ".join(sorted(unknown)))
        depth, leaf = params.get("max_depth"), params.get("min_samples_leaf")
        if isinstance(depth, bool) or depth not in TREE_DEPTHS:
            raise InvalidCandidate("max_depth must be one of 2, 3, 4, 5")
        if isinstance(leaf, bool) or not isinstance(leaf, int) or leaf not in TREE_LEAVES:
            raise InvalidCandidate("min_samples_leaf must be one of 20, 50, 100")
        for key, value in TREE_FIXED.items():
            if params.get(key, value) != value:
                raise InvalidCandidate(f"{key} must be {value!r}")
        estimator = DecisionTreeClassifier(**_seeded(params, DecisionTreeClassifier))
    else:
        raise InvalidCandidate(f"unknown family {family!r}")
    return Pipeline([*steps, (CLASSIFIER_STEP, estimator)])


def jsonable(value: Any) -> Any:
    if value is np.log1p:
        return "numpy.log1p"
    if isinstance(value, np.generic):
        value = value.item()
    if value is None or isinstance(value, (bool, int, str)):
        return value
    if isinstance(value, float):
        if not math.isfinite(value):
            raise InvalidCandidate("non-finite parameter")
        return value
    if isinstance(value, (list, tuple)):
        return [jsonable(item) for item in value]
    if isinstance(value, dict) and all(isinstance(key, str) for key in value):
        return {key: jsonable(item) for key, item in value.items()}
    raise InvalidCandidate(f"parameter of type {type(value).__name__} is not serializable")


def resolved_params(pipeline: Pipeline) -> dict[str, Any]:
    return {
        "pipeline": [
            {
                "name": name,
                "class": type(step).__module__ + "." + type(step).__name__,
                "params": jsonable(step.get_params(deep=False)),
            }
            for name, step in pipeline.steps
        ]
    }


def validate_candidates(candidates: Any) -> list[dict[str, Any]]:
    if not isinstance(candidates, list) or not candidates:
        raise InvalidCandidate("candidates.json must be a non-empty array")
    identifiers = [item.get("experiment_id") if isinstance(item, dict) else None for item in candidates]
    if any(not isinstance(x, str) or not x.isascii() or not x for x in identifiers):
        raise InvalidCandidate("experiment_id must be a non-empty ASCII string")
    if len(set(identifiers)) != len(identifiers):
        raise InvalidCandidate("experiment_id values must be unique")
    families = [item.get("family") for item in candidates]
    if families.count("constant") != 1 or "logistic" not in families or "tree" not in families:
        raise InvalidCandidate("need one constant, at least one logistic and one tree")
    for item in candidates:
        build_pipeline(item)
    return candidates
