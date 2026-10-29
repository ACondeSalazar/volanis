from dataclasses import dataclass
from itertools import combinations, product

import numpy as np
from scipy.optimize import curve_fit

from .acf import AutocorrelationResult


def _single(lags, length):
    return np.exp(-lags / length)


def _double(lags, weight, length1, length2):
    return weight * np.exp(-lags / length1) + (1 - weight) * np.exp(-lags / length2)


def _triple(lags, a, b, length1, length2, length3):
    first = a * np.exp(-lags / length1)
    second = (1 - a) * b * np.exp(-lags / length2)
    third = (1 - a) * (1 - b) * np.exp(-lags / length3)
    return first + second + third


MODELS = {"single": (_single, 1), "double": (_double, 2), "triple": (_triple, 3)}


def _starting_points(n_lengths: int, max_lag: float):
    scales = [fraction * max_lag for fraction in (0.005, 0.02, 0.08, 0.3)]

    for lengths in combinations(scales, n_lengths):
        for weights in product((0.3, 0.6, 0.9), repeat=n_lengths - 1):
            yield [*weights, *lengths]


def _weights(model: str, parameters: np.ndarray) -> np.ndarray:
    if model == "single":
        return np.array([1.0])
    if model == "double":
        return np.array([parameters[0], 1 - parameters[0]])
    a, b = parameters[:2]
    return np.array([a, (1 - a) * b, (1 - a) * (1 - b)])


@dataclass(frozen=True)
class ModelFit:

    model: str
    lengths_um: np.ndarray
    length_errors_um: np.ndarray
    weights: np.ndarray
    fitted_curve: np.ndarray
    r_squared: float
    aic: float
    status: str


@dataclass(frozen=True)
class AutocorrelationFit:
    xy: dict[int, dict[str, ModelFit]]
    z: dict[int, dict[str, ModelFit]]
    result: AutocorrelationResult

    @property
    def channels(self):
        return self.result.channels

    @property
    def meta(self):
        return self.result.meta

    def best(self, direction: str, channel: int):
        """lowest AIC among converged fits"""
        fits = getattr(self, direction)[channel].values()
        usable = [fit for fit in fits if fit.status != "failed"]
        return min(usable, key=lambda fit: fit.aic) if usable else None


def _fit_one(model: str, lags: np.ndarray, curve: np.ndarray) -> ModelFit:
    extra = True
    function, n_lengths = MODELS[model]
    n_core = 2 * n_lengths - 1
    max_lag = float(lags.max())
    lower = [0.0] * (n_lengths - 1) + [1e-6] * n_lengths
    upper = [1.0] * (n_lengths - 1) + [4 * max_lag] * n_lengths
    best = None

    if extra:
        base = function
        function = lambda x, *p: p[-1] + p[-2] * base(x, *p[:-2])
        lower += [0.0, -0.5]
        upper += [2.0, 0.5]

    for initial in _starting_points(n_lengths, max_lag):
        if extra:
            initial = [*initial, 1.0, 0.0]
        try:
            parameters, covariance = curve_fit(function, lags, curve, p0=initial, bounds=(lower, upper), maxfev=20000)
        except (RuntimeError, ValueError):
            continue

        residual_sum = float(np.sum((curve - function(lags, *parameters)) ** 2))
        if best is None or residual_sum < best[0]:
            best = (residual_sum, parameters, covariance)

    if best is None:
        empty = np.array([])
        failed_curve = np.full(len(lags), np.nan)
        return ModelFit(model, empty, empty, empty, failed_curve, float("nan"), float("inf"), "failed")

    residual_sum, parameters, covariance = best
    lengths = parameters[n_lengths - 1 : n_core]
    errors = np.sqrt(np.diag(covariance))[n_lengths - 1 : n_core]
    weights = _weights(model, parameters)
    order = np.argsort(lengths)

    total_sum = float(np.sum((curve - curve.mean()) ** 2))
    r_squared = 1 - residual_sum / total_sum if total_sum > 0 else float("nan")

    n_points = len(lags)
    n_parameters = len(parameters)
    aic = n_points * np.log(residual_sum / n_points) + 2 * n_parameters if residual_sum > 0 else -np.inf

    fitted_curve = function(lags, *parameters)
    return ModelFit(model, lengths[order], errors[order], weights[order], fitted_curve, r_squared, float(aic), "ok")


def fit(source, *, models=("single", "double", "triple")) -> AutocorrelationFit:
    """fit positive lags of XY and Z profiles normalized at lag zero, in physical units"""
    result = source
    if not isinstance(source, AutocorrelationResult):
        result = AutocorrelationResult.load(source)

    for model in models:
        if model not in MODELS:
            raise ValueError(f"unknown model {model}; choose from {sorted(MODELS)}")

    fits = {"xy": {}, "z": {}}

    for direction in ("xy", "z"):
        for channel, profile in getattr(result, direction).items():
            # lag zero is always 1 for both data and model
            lags = profile.lags_um[1:]
            curve = profile.measured_curve[1:]
            if len(lags) == 0:
                raise ValueError(f"{direction} channel {channel}: no positive values to fit")

            fits[direction][channel] = {}
            for model in models:
                fits[direction][channel][model] = _fit_one(model, lags, curve)

    return AutocorrelationFit(fits["xy"], fits["z"], result)
