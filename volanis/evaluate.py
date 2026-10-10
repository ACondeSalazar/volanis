import os

import numpy as np
from scipy.stats import pearsonr

#-----------------------------Metrics-------------------------------------------------

PEARSON_TILE_UM = 25.0
SSIM_WINDOWS_UM = (2.5, 5.0, 10.0, 20.0, 40.0)


def pearson_weighted(truth, prediction, xy_spacing_um, tile_um=PEARSON_TILE_UM):
    side = max(round(tile_um / xy_spacing_um), 8)
    height, width = truth.shape
    rows, columns = height // side, width // side
    if rows == 0 or columns == 0:
        return float("nan")

    a = truth[:rows * side, :columns * side].astype(np.float64)
    b = prediction[:rows * side, :columns * side].astype(np.float64)
    correlations, weights = [], []
    for block_a, block_b in zip(a.reshape(rows, side, -1), b.reshape(rows, side, -1)):
        tiles_a = block_a.reshape(side, columns, side).transpose(1, 0, 2).reshape(columns, -1)
        tiles_b = block_b.reshape(side, columns, side).transpose(1, 0, 2).reshape(columns, -1)
        correlations.append(pearsonr(tiles_a, tiles_b, axis=1).statistic)
        weights.append(tiles_a.sum(axis=1)) # weight is only determined by truth, could be better

    r = np.concatenate(correlations)
    w = np.concatenate(weights)
    good = np.isfinite(r) & (w > 0)
    if not good.any():
        return float("nan")
    return float(np.sum(r[good] * w[good]) / np.sum(w[good]))


def _odd_pixels(length_um, xy_spacing_um, minimum=3):
    pixels = max(round(length_um / xy_spacing_um), minimum)
    return pixels if pixels % 2 else pixels + 1


def ssim_multi(truth, prediction, xy_spacing_um, windows_um=SSIM_WINDOWS_UM):
    from skimage.metrics import structural_similarity

    a = truth.astype(np.float32)
    b = prediction.astype(np.float32)
    mask = (a != 0) | (b != 0)
    if not mask.any():
        return float("nan")

    data_range = float(max(a.max(), 1e-6))
    scores = []
    for window_um in windows_um:
        window = _odd_pixels(window_um, xy_spacing_um)
        if window > min(a.shape):
            continue
        _, per_pixel = structural_similarity(a, b, data_range=data_range,
                                             win_size=window, full=True)
        scores.append(float(per_pixel[mask].mean()))
    return float(np.mean(scores)) if scores else float("nan")


def wasserstein_um(truth, prediction, xy_spacing_um, sample_fraction=0.01,
                   max_points=200_000, seed=0):
    import torch
    from geomloss import SamplesLoss

    device = "cuda" if torch.cuda.is_available() else "cpu"
    rng = np.random.default_rng(seed)

    def cloud(image):
        intensity = np.clip(np.asarray(image, dtype=np.float64).ravel(), 0, None)
        lit = np.count_nonzero(intensity)
        if not lit:
            return None
        count = max(1, min(int(lit * sample_fraction), max_points, lit))
        index = rng.choice(intensity.size, count, p=intensity / intensity.sum())
        rows, columns = np.unravel_index(index, image.shape)
        return np.column_stack((columns, rows)) * xy_spacing_um

    points_a, points_b = cloud(truth), cloud(prediction)
    if points_a is None or points_b is None:
        return float("nan")

    area_um2 = truth.size * xy_spacing_um ** 2
    spacing = float(np.sqrt(area_um2 / min(len(points_a), len(points_b))))

    def tensor(values):
        return torch.as_tensor(values, dtype=torch.float32, device=device)

    loss = SamplesLoss("sinkhorn", p=2, blur=max(2.0, spacing), debias=True)
    return float(np.sqrt(max(2.0 * float(loss(tensor(points_a), tensor(points_b))), 0.0)))


METRICS = {"pearson": pearson_weighted, "ssim": ssim_multi, "wasserstein": wasserstein_um}

#-----------------------Whole volume and interpolation evaluations--------------------------------

def compare(truth, prediction, xy_spacing_um, metrics=("pearson", "ssim")):
    unknown = [m for m in metrics if m not in METRICS]
    if unknown:
        raise ValueError(f"unknown metrics {unknown}; choose from {sorted(METRICS)}")
    return {m: METRICS[m](truth, prediction, xy_spacing_um) for m in metrics}

def score_reconstruction(truth, reconstructed, keep_every, *, channel=0, metrics=("pearson", "ssim")):
    """compare a volume and its interpolated counterpart"""
    indices, values = [], {m: [] for m in metrics}
    for index in range(min(len(truth), len(reconstructed))):
        if index % keep_every == 0:
            continue
        real = truth.read_slice(index)[:, :, channel]
        predicted = reconstructed.read_slice(index)[:, :, channel]
        scores = compare(real, predicted, truth.xy_spacing_um, metrics=metrics)
        indices.append(index)
        for name, value in scores.items():
            values[name].append(value)
    return {"index": np.array(indices), **{m: np.array(v) for m, v in values.items()}}


def sweep_coarse_graining(volume, output_root, interpolate, keep_every, coarse_grainings,
                          *, deconvolve=False, channel=0,
                          metrics=("pearson", "ssim")):
    """coarsen, interpolate, then compare against ground truth"""
    if deconvolve:
        raise NotImplementedError("deconvolution is not implemented yet")

    from .volume import open_volume

    results = {}
    for sigma_um, factor in coarse_grainings:
        coarse = volume.coarsen(sigma_um, factor)
        folder = os.path.join(output_root, f"sigma{sigma_um:g}_factor{factor}")
        interpolate(coarse.prune(keep_every), folder, keep_every - 1)
        rebuilt = open_volume(folder, xy_spacing_um=coarse.xy_spacing_um,
                              z_spacing_um=coarse.z_spacing_um)
        results[(sigma_um, factor)] = score_reconstruction(
            coarse, rebuilt, keep_every, channel=channel, metrics=metrics)
    return results


def sweep_gaps(volume, output_root, interpolate, keep_everys, *, channel=0,
               metrics=("pearson", "ssim")):
    """
    compute score reconstruction with different gaps 
    example :
    sweep_gaps(truth, "/tmp/sweep",
                lambda sparse, folder, frames: volanis.interpolate_mccann(sparse, folder, frames_between=frames, tile_size=128),
                keep_everys=(2, 4, 8)
                )

    `frames` is passed in as keep_every - 1, the only value that puts the new sections
    where the removed ones were.

    returns {gap_um: {"index": array, metric: array}}
    """
    from .volume import open_volume

    results = {}
    for keep_every in keep_everys:
        sparse = volume.prune(keep_every)
        folder = os.path.join(output_root, f"gap{keep_every}")
        interpolate(sparse, folder, keep_every - 1)
        rebuilt = open_volume(folder, xy_spacing_um=volume.xy_spacing_um,
                              z_spacing_um=volume.z_spacing_um)
        results[sparse.z_spacing_um] = score_reconstruction(
            volume, rebuilt, keep_every, channel=channel, metrics=metrics)
    return results
