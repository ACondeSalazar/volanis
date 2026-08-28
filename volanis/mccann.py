import os

import numpy as np
from tqdm import tqdm

from .volume import from_numpy, save_tiff, tile_grid

# share of the signal that actually gets transported, the rest is linearly faded. speed things up with low quality loss
MASS_KEPT = 0.9

# lower gives sharper output, but its slower and can introduces artifacts
SINKHORN_BLUR = 20.0

# number of destinations kept per point. None uses every destination.
DESTINATIONS = 32

# limit the pairwise distance table processed at once to avoid running out of GPU memory.
MAX_BLOCK_BYTES = 1e9


def active_points(image):
    """split the image in 2, the bright pixels are the fewest pixels holding MASS_KEPT of the signal
    and will be transported, the others will be linearly faded.
    """
    flat = image.ravel()
    lit = np.flatnonzero(flat)
    if lit.size == 0:
        return np.zeros((0, 2), np.float32), np.zeros(0, np.float32), image.copy()

    order = lit[np.argsort(flat[lit])[::-1]]
    running_total = np.cumsum(flat[order], dtype=np.float64)
    keep = order[:np.searchsorted(running_total, MASS_KEPT * running_total[-1]) + 1]
    rows, columns = np.unravel_index(keep, image.shape)

    dim = image.copy()
    dim.ravel()[keep] = 0
    return (np.stack([rows, columns], axis=1).astype(np.float32), flat[keep].astype(np.float32), dim)


def splat(image, positions, amounts):
    """add scattered points into an image"""
    height, width = image.shape
    rows = positions[:, 0].clamp(0, height - 1.001)
    columns = positions[:, 1].clamp(0, width - 1.001)
    top, left = rows.floor().long(), columns.floor().long()
    down, right = rows - top, columns - left

    flat = image.view(-1)
    for row_step, column_step, share in ((0, 0, (1 - down) * (1 - right)),
                                         (0, 1, (1 - down) * right),
                                         (1, 0, down * (1 - right)),
                                         (1, 1, down * right)):
        index = ((top + row_step).clamp(0, height - 1) * width
                 + (left + column_step).clamp(0, width - 1))
        flat.scatter_add_(0, index, amounts * share)


def mccann_tile(left, right, fractions, device=None):
    """mccann ot interpolation on 2 small images"""
    import torch
    from geomloss import SamplesLoss

    device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    left_points, left_mass, left_dim = active_points(left)
    right_points, right_mass, right_dim = active_points(right)
    if len(left_mass) == 0 or len(right_mass) == 0:
        return [(1 - t) * left + t * right for t in fractions]

    def tensor(values):
        return torch.as_tensor(values, device=device)

    p0, p1 = tensor(left_points), tensor(right_points)
    m0, m1 = tensor(left_mass), tensor(right_mass)
    share0, share1 = m0 / m0.sum(), m1 / m1.sum()

    # in my experience debias=True tends to invent less mass, but debias false better reconstruct low intensity parts, also a bit faster
    solver = SamplesLoss("sinkhorn", p=2, blur=SINKHORN_BLUR, debias=False,
                         potentials=True, backend="multiscale", verbose=False)
    f, g = solver(share0, p0, share1, p1)

    # turn NaNs into very low values to avoid errors
    f = torch.nan_to_num(f.reshape(-1), nan=-1e30, posinf=-1e30, neginf=-1e30)
    g = torch.nan_to_num(g.reshape(-1), nan=-1e30, posinf=-1e30, neginf=-1e30)

    images = [torch.zeros(left.shape, device=device) for _ in fractions]
    log_share1 = share1.clamp_min(1e-38).log()

    block = max(1, int(MAX_BLOCK_BYTES / (4 * len(share1))))
    for start in range(0, len(share0), block):
        stop = min(start + block, len(share0))
        starts = p0[start:stop]

        # how attractive each move is, in logs so that distant pairs do not underflow
        cost = torch.cdist(starts, p1) ** 2 / 2
        appeal = ((f[start:stop].unsqueeze(1) + g.unsqueeze(0) - cost) / SINKHORN_BLUR ** 2 + log_share1.unsqueeze(0))

        if DESTINATIONS is None:
            amounts = appeal.exp() * share0[start:stop].unsqueeze(1)
            to_points = p1.unsqueeze(0).expand(len(starts), -1, -1)
        else:
            best, targets = appeal.topk(min(DESTINATIONS, len(p1)), dim=1)
            # exp overflows float32 above ~88 and one inf spreads over the tile
            best = best.clamp(max=80.0)
            amounts = best.exp() * share0[start:stop].unsqueeze(1)
            to_points = p1[targets]
        from_points = starts.unsqueeze(1).expand_as(to_points)

        for t, image in zip(fractions, images):
            met = (1 - t) * from_points + t * to_points
            splat(image, met.reshape(-1, 2), amounts.reshape(-1))

    output = []
    for t, image in zip(fractions, images):
        total = image.sum()
        if total > 0:
            image = image * (((1 - t) * m0.sum() + t * m1.sum()) / total)
        # the dim part is just linearly faded
        output.append(image.cpu().numpy() + (1 - t) * left_dim + t * right_dim)
    return output


def _to_dtype(frame, dtype):
    """normalize values"""
    frame = np.nan_to_num(frame, nan=0.0, posinf=0.0, neginf=0.0)
    if np.issubdtype(dtype, np.integer):
        info = np.iinfo(dtype)
        frame = np.clip(frame, info.min, info.max)
    return frame.astype(dtype)


def _window(tile_size, overlap):
    """fades in over the overlap"""
    ramp = np.minimum(np.arange(tile_size), np.arange(tile_size)[::-1]) + 1.0
    ramp = np.clip(ramp / overlap, 0.0, 1.0)
    return np.outer(ramp, ramp).astype(np.float32)


def interpolate_mccann(volume, output_folder, *, frames_between, tile_size=128,
                       channels=None):
    """Channels not listed are dropped: the output has only the ones asked for.

    Cost is dominated by whichever channel carries the most signal. On the breast
    sections DAPI holds 39% of the pixels and takes 99% of the time, while CD31 and
    CA9 are nearly free, so `channels=[0, 1]` is about a hundred times cheaper.
    """
    os.makedirs(output_folder, exist_ok=True)
    step = frames_between + 1
    fractions = [k / step for k in range(1, step)]
    _, height, width, n_channels = volume.shape
    kept = sorted(set(range(n_channels) if channels is None else channels))
    outside = [c for c in kept if not 0 <= c < n_channels]
    if outside:
        raise ValueError(f"channels {outside} outside the volume's {n_channels}")

    overlap = tile_size // 4
    tiles = tile_grid(width, height, tile_size, overlap=overlap)
    window = _window(tile_size, overlap)

    def path(index, k):
        stem = os.path.splitext(os.path.basename(volume.paths[index]))[0]
        return os.path.join(output_folder, f"{stem}_{k}.tif")

    left = volume.read_slice(0)[:, :, kept]
    for i in range(len(volume) - 1):
        right = volume.read_slice(i + 1)[:, :, kept]
        save_tiff(from_numpy(left), path(i, 0))

        frames = [np.zeros((height, width, len(kept)), np.float32)
                  for _ in fractions]
        weights = np.zeros((height, width), np.float32)

        for tile in tqdm(tiles, desc=f"mccann {i + 1}/{len(volume) - 1}", leave=False):
            rows = slice(tile.y, tile.y + tile.height)
            columns = slice(tile.x, tile.x + tile.width)
            share = window[:tile.height, :tile.width]
            weights[rows, columns] += share

            for channel in range(len(kept)):
                pieces = mccann_tile(left[rows, columns, channel],
                                     right[rows, columns, channel], fractions)
                for frame, piece in zip(frames, pieces):
                    frame[rows, columns, channel] += piece * share

        for k, frame in enumerate(frames, start=1):
            frame /= np.maximum(weights, 1e-6)[:, :, None]
            save_tiff(from_numpy(_to_dtype(frame, volume.dtype)), path(i, k))
        left = right

    save_tiff(from_numpy(left), path(len(volume) - 1, 0))
