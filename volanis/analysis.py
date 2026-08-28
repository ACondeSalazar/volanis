import json
import os
from dataclasses import dataclass

import numpy as np
from scipy.fft import irfftn, next_fast_len, rfftn
from tqdm import tqdm

from .volume import tile_grid

TISSUE_FRACTION = 0.5 
BASELINE_FRACTION = 0.15
OVERVIEW_MAX_PX = 2048 # max size of tiles used for selection

@dataclass(frozen=True)
class Profile:
    """A single measured decay curve. spacing_um is the sample step in this direction."""
    lags_um: np.ndarray
    measured_curve: np.ndarray
    spacing_um: float


@dataclass(frozen=True)
class AutocorrelationResult:
    """Measured XY and Z curves per channel"""

    xy: dict[int, Profile]
    z: dict[int, Profile]
    meta: dict

    @property
    def channels(self):
        return sorted(self.xy)

    def save(self, path):
        arrays = {"meta": np.array(json.dumps(self.meta))}
        for c in self.channels:
            for name, prof in (("xy", self.xy[c]), ("z", self.z[c])):
                arrays[f"{name}_lags_c{c}"] = prof.lags_um
                arrays[f"{name}_curve_c{c}"] = prof.measured_curve
        np.savez(path, **arrays)
        return path

    @classmethod
    def load(cls, path):
        with np.load(path) as data:
            meta = json.loads(str(data["meta"]))
            spacing = {"xy": meta["xy_spacing_um"], "z": meta["z_spacing_um"]}
            curves = {}
            for name in ("xy", "z"):
                curves[name] = {
                    c: Profile(data[f"{name}_lags_c{c}"], data[f"{name}_curve_c{c}"],
                               spacing[name])
                    for c in meta["channels"]}
        return cls(curves["xy"], curves["z"], meta)


def _overview(volume):
    level = volume.level
    while (volume.shape[2] >> (level - volume.level)) > OVERVIEW_MAX_PX:
        if volume.has_pyramid and level + 1 >= volume.n_levels:
            break
        level += 1
    view = volume.at_level(level)
    return view.read_slice(len(volume) // 2), 2 ** (level - volume.level)


def _select_tiles(overview, scale, channel, width, height, tile_size, max_tiles):
    """Evenly spaced tissue tiles , in volume pixels. deterministic"""
    plane = overview[..., channel].astype(np.float32)
    threshold = TISSUE_FRACTION * float(plane.mean())
    full = [t for t in tile_grid(width, height, tile_size)
            if t.width == tile_size and t.height == tile_size]
    if not full:
        raise ValueError(f"no {tile_size}x{tile_size} tile fits in a {width}x{height} section; "
                         f"use a smaller tile_size or a finer level")
    keep = [t for t in full
            if plane[t.y // scale:(t.y + tile_size) // scale,
                     t.x // scale:(t.x + tile_size) // scale].mean() >= threshold]
    if not keep:
        raise ValueError(f"channel {channel}: no tile reaches {TISSUE_FRACTION:g} of the mean "
                         f"intensity; the channel looks empty")
    picks = np.unique(np.linspace(0, len(keep) - 1, min(max_tiles, len(keep))).round().astype(int))
    return [keep[i] for i in picks]


def _radial_average(plane, max_r):
    size = plane.shape[0]
    centre = size // 2
    yy, xx = np.indices(plane.shape)
    r = np.round(np.hypot(yy - centre, xx - centre)).astype(np.intp).ravel()
    counts = np.bincount(r, minlength=max_r + 1)[:max_r + 1]
    sums = np.bincount(r, weights=plane.ravel(), minlength=max_r + 1)[:max_r + 1]
    return sums / np.maximum(counts, 1)


def _cube_profiles(cube, max_r, max_dz):
    """XY radial profile at zero Z lag, and axial profile at zero XY lag, for one 3D tile."""
    depth, height, width = cube.shape
    centred = cube - cube.mean(dtype=np.float64)

    shape = (next_fast_len(depth + max_dz), next_fast_len(height + max_r),
             next_fast_len(width + max_r))
    spectrum = rfftn(centred.astype(np.float32), s=shape, workers=-1)
    acf = irfftn(spectrum * np.conj(spectrum), s=shape, workers=-1)

    z_curve = acf[:max_dz + 1, 0, 0] / ((depth - np.arange(max_dz + 1)) * float(height * width))

    plane = np.fft.fftshift(acf[0])
    centre = np.array(plane.shape) // 2
    plane = plane[centre[0] - max_r:centre[0] + max_r + 1, centre[1] - max_r:centre[1] + max_r + 1]
    offsets = np.abs(np.arange(-max_r, max_r + 1))
    plane = plane / (depth * np.outer(height - offsets, width - offsets))

    zero_lag = z_curve[0]
    if not np.isfinite(zero_lag) or zero_lag <= 0:
        return None
    return _radial_average(plane, max_r) / zero_lag, z_curve / zero_lag


def _finalise(curves, spacing_um, kind):
    """Average tiles"""
    mean = np.mean(curves, axis=0)
    tail = mean[int((1 - BASELINE_FRACTION) * len(mean)):]
    
    mean = mean - np.median(tail)
    if kind == "xy":
        # lag 0 is noisy so we start at lag 1
        lags = np.arange(1, len(mean))
        mean = mean[1:] / mean[1]
    else:
        lags = np.arange(len(mean))
        mean = mean / mean.max()
    return Profile(lags * spacing_um, mean, spacing_um)


def autocorrelation(volume, output_path, *, channels=None, tile_size=1024, max_tiles=32):
    """Measure XY and Z autocorrelation per channel and write the curves to output_path as an .npz file"""
    channels = list(range(volume.shape[3])) if channels is None else list(channels)
    bad = [c for c in channels if not 0 <= c < volume.shape[3]]
    if bad:
        raise ValueError(f"channels {bad} outside the volume's {volume.shape[3]} channels")

    depth, height, width, _ = volume.shape
    max_r, max_dz = tile_size // 2, depth // 2
    if max_dz < 1:
        raise ValueError(f"need at least 2 sections to measure a Z profile, got {depth}")

    overview, scale = _overview(volume)
    xy, z, picked = {}, {}, {}
    for c in channels:
        tiles = _select_tiles(overview, scale, c, width, height, tile_size, max_tiles)
        xy_curves, z_curves = [], []
        for t in tqdm(tiles, desc=f"acf channel {c}"):
            cube = np.stack([volume.read_region(i, t.x, t.y, tile_size, tile_size)[..., c]
                             for i in range(depth)])
            profiles = _cube_profiles(cube, max_r, max_dz)
            if profiles is None:
                continue
            xy_curves.append(profiles[0])
            z_curves.append(profiles[1])
        if not xy_curves:
            raise ValueError(f"channel {c}: every tile had zero ?")
        xy[c] = _finalise(xy_curves, volume.xy_spacing_um, "xy")
        z[c] = _finalise(z_curves, volume.z_spacing_um, "z")
        picked[c] = [list(t) for t in tiles]

    meta = {
        "channels": channels,
        "tile_size": tile_size,
        "max_tiles": max_tiles,
        "n_tiles": {str(c): len(picked[c]) for c in channels},
        "tiles": {str(c): picked[c] for c in channels},
        "xy_spacing_um": volume.xy_spacing_um,
        "z_spacing_um": volume.z_spacing_um,
        "level": volume.level,
        "has_pyramid": volume.has_pyramid,
        "n_sections": depth,
        "source_folder": os.path.dirname(volume.paths[0]),
        "tissue_fraction": TISSUE_FRACTION,
        "baseline_fraction": BASELINE_FRACTION,
        "xy_normalised_at_lag": 1,
    }
    result = AutocorrelationResult(xy, z, meta)
    result.save(output_path)
    return result


# --- models and acf curve fitting -------------------------------------------------------------------------

def _single(r, l1):
    return np.exp(-r / l1)


def _double(r, a, l1, l2):
    return a * np.exp(-r / l1) + (1 - a) * np.exp(-r / l2)


def _triple(r, a, b, l1, l2, l3):
    return (a * np.exp(-r / l1) + (1 - a) * b * np.exp(-r / l2)
            + (1 - a) * (1 - b) * np.exp(-r / l3))


def _weights(model, params):
    if model == "single":
        return [1.0]
    if model == "double":
        return [params[0], 1 - params[0]]
    a, b = params[0], params[1]
    return [a, (1 - a) * b, (1 - a) * (1 - b)]


MODELS = {"single": (_single, 1), "double": (_double, 2), "triple": (_triple, 3)}

RELATIVE_ERROR_LIMIT = 0.5 # above this a length is considered not constrained by the data
RESOLVED_MULTIPLE = 1.5 # a length below this many samples is not observable

# vars used for the optimization initialisation
_SEED_FRACTIONS = (0.005, 0.02, 0.08, 0.3)
_SEED_WEIGHTS = (0.3, 0.6, 0.9)


def _starting_points(model, x_max):
    scales = [f * x_max for f in _SEED_FRACTIONS]
    if model == "single":
        return [[s] for s in scales]
    if model == "double":
        return [[a, s1, s2] for i, s1 in enumerate(scales) for s2 in scales[i + 1:]
                for a in _SEED_WEIGHTS]
    return [[a, b, s1, s2, s3]
            for i, s1 in enumerate(scales) for j, s2 in enumerate(scales[i + 1:], i + 1)
            for s3 in scales[j + 1:] for a in _SEED_WEIGHTS for b in _SEED_WEIGHTS]


def _bounds(model, x_max):
    n = MODELS[model][1]
    n_weights = n - 1
    lo = [0.0] * n_weights + [1e-6] * n
    hi = [1.0] * n_weights + [x_max * 4] * n
    return lo, hi


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
    """ModelFit for every channel, direction and model requested"""

    xy: dict[int, dict[str, ModelFit]]
    z: dict[int, dict[str, ModelFit]]
    result: AutocorrelationResult

    @property
    def channels(self):
        return self.result.channels

    @property
    def meta(self):
        return self.result.meta

    def best(self, direction, channel):
        fits = getattr(self, direction)[channel]
        usable = [f for f in fits.values() if f.status != "failed"]
        return min(usable, key=lambda f: f.aic) if usable else None


def _failed(model, x):
    empty = np.array([])
    return ModelFit(model, empty, empty, empty, np.full(len(x), np.nan),
                    float("nan"), float("inf"), "failed")


def _fit_one(model, x, y, spacing_um):
    from scipy.optimize import curve_fit

    func, n_lengths = MODELS[model]
    lo, hi = _bounds(model, float(x.max()))
    best = None
    for p0 in _starting_points(model, float(x.max())):
        try:
            popt, pcov = curve_fit(func, x, y, p0=p0, bounds=(lo, hi), maxfev=20000)
        except (RuntimeError, ValueError):
            continue
        residual = y - func(x, *popt)
        ss_res = float(np.sum(residual ** 2))
        if best is None or ss_res < best[0]:
            best = (ss_res, popt, pcov)
    if best is None:
        return _failed(model, x)

    ss_res, popt, pcov = best
    lengths = np.asarray(popt[-n_lengths:], dtype=float)
    errors = np.sqrt(np.diag(pcov))[-n_lengths:]
    weights = np.asarray(_weights(model, popt), dtype=float)

    order = np.argsort(lengths) # always shortest length first
    lengths, errors, weights = lengths[order], errors[order], weights[order]

    ss_tot = float(np.sum((y - y.mean()) ** 2))
    r2 = 1.0 - ss_res / ss_tot if ss_tot > 0 else float("nan")
    n, k = len(x), len(popt)
    aic = n * np.log(ss_res / n) + 2 * k if ss_res > 0 else -np.inf

    pinned = np.any(np.isclose(popt, lo) | np.isclose(popt, hi))
    relative = np.divide(errors, np.abs(lengths), out=np.full_like(errors, np.inf),
                         where=lengths > 0)
    if np.any(lengths < RESOLVED_MULTIPLE * spacing_um):
        status = "unresolved"
    elif pinned or not np.all(relative < RELATIVE_ERROR_LIMIT):
        status = "unreliable"
    else:
        status = "ok"
    return ModelFit(model, lengths, errors, weights, func(x, *popt), r2, float(aic), status)


def fit(source, *, models=("single", "double", "triple")):
    result = (source if isinstance(source, AutocorrelationResult) else AutocorrelationResult.load(source))
    unknown = [m for m in models if m not in MODELS]
    if unknown:
        raise ValueError(f"unknown models {unknown}; choose from {sorted(MODELS)}")

    fits = {}
    for direction in ("xy", "z"):
        curves = getattr(result, direction)
        fits[direction] = {
            c: {m: _fit_one(m, curves[c].lags_um, curves[c].measured_curve,
                            curves[c].spacing_um)
                for m in models}
            for c in result.channels}
    return AutocorrelationFit(fits["xy"], fits["z"], result)
