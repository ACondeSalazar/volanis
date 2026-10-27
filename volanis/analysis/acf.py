import json
import os
from dataclasses import dataclass

import numpy as np
import torch
from scipy.fft import next_fast_len
from tqdm import tqdm

from ..volume import Volume, tile_grid

OVERVIEW_MAX_PX = 2048


@dataclass(frozen=True)
class Profile:
    """measured curve including lag zero, normalized so C(0) = 1"""

    lags_um: np.ndarray
    measured_curve: np.ndarray
    spacing_um: float


@dataclass(frozen=True)
class AutocorrelationResult:
    xy: dict[int, Profile]
    z: dict[int, Profile]
    meta: dict

    @property
    def channels(self):
        return sorted(self.xy)

    def save(self, path):
        arrays = {"meta": np.array(json.dumps(self.meta))}

        for direction in ("xy", "z"):
            for channel, profile in getattr(self, direction).items():
                arrays[f"{direction}_lags_c{channel}"] = profile.lags_um
                arrays[f"{direction}_curve_c{channel}"] = profile.measured_curve

        np.savez(path, **arrays)
        return path

    @classmethod
    def load(cls, path):
        with np.load(path) as data:
            meta = json.loads(str(data["meta"]))
            curves = {"xy": {}, "z": {}}

            for direction in ("xy", "z"):
                for channel in meta["channels"]:
                    curves[direction][channel] = Profile(
                        data[f"{direction}_lags_c{channel}"],
                        data[f"{direction}_curve_c{channel}"],
                        meta[f"{direction}_spacing_um"],
                    )

        return cls(curves["xy"], curves["z"], meta)


def _overview(volume: Volume):
    level = volume.level
    size = max(volume.shape[1:3])

    while size > OVERVIEW_MAX_PX:
        if volume.has_pyramid and level + 1 >= volume.n_levels:
            break
        level += 1
        size = (size + 1) // 2

    scale = 2 ** (level - volume.level)
    return volume.at_level(level).read_slice(len(volume) // 2), scale


def _select_tiles(overview, scale, channel, width, height, tile_size, max_tiles, tissue_fraction):
    plane = overview[..., channel]
    threshold = tissue_fraction * float(plane.mean())
    candidates = []

    for tile in tile_grid(width, height, tile_size, overlap = tile_size // 2):
        if tile.width != tile_size or tile.height != tile_size:
            continue

        start_x = tile.x // scale
        start_y = tile.y // scale
        stop_x = (tile.x + tile_size + scale - 1) // scale
        stop_y = (tile.y + tile_size + scale - 1) // scale

        if plane[start_y:stop_y, start_x:stop_x].mean() >= threshold:
            candidates.append(tile)

    if not candidates:
        raise ValueError(f"channel {channel}: no full {tile_size}x{tile_size} signal tile found")

    count = min(max_tiles, len(candidates))
    indices = np.linspace(0, len(candidates) - 1, count).round().astype(int)
    return [candidates[index] for index in indices]


def _radial_average(plane: np.ndarray, max_radius: int) -> np.ndarray:
    center = plane.shape[0] // 2
    rows, columns = np.indices(plane.shape)
    radius = np.rint(np.hypot(rows - center, columns - center)).astype(int).ravel()

    counts = np.bincount(radius)[:max_radius + 1]
    sums = np.bincount(radius, weights=plane.ravel())[:max_radius + 1]
    return sums / counts


def _cube_profiles(cube: np.ndarray, max_radius: int, max_z_lag: int, device: str = "cpu", chunk: int = 4):
    """covariance corrected for overlap and normalized at lag zero for each tile"""
    depth, height, width = cube.shape
    centered = cube - cube.mean(dtype=np.float32)

    # pad by max_radius to avoid wraparound in the xy plane
    ny = next_fast_len(height + max_radius)
    nx = next_fast_len(width + max_radius)

    with torch.no_grad():
        # z=0 plane: sum over z of each slice's 2D power spectrum, a few slices at a time
        power = torch.zeros((ny, nx // 2 + 1), dtype=torch.float32, device=device)
        for start in range(0, depth, chunk):
            block = torch.as_tensor(centered[start:start + chunk], dtype=torch.float32, device=device)
            power += torch.fft.rfft2(block, s=(ny, nx)).abs().square().sum(dim=0)
            del block
        plane = torch.fft.irfft2(power, s=(ny, nx)).cpu().numpy()
        del power

        # z axis at x=y=0: direct dot products between shifted copies, no FFT, no padding
        tensor = torch.as_tensor(centered, dtype=torch.float32, device=device)
        z_cov = torch.stack([
            torch.dot(tensor[:depth - lag].reshape(-1), tensor[lag:].reshape(-1))
            for lag in range(max_z_lag + 1)
        ]).cpu().numpy().astype(np.float64)
        del tensor

    z_pairs = (depth - np.arange(max_z_lag + 1)) * float(height * width)
    z_curve = z_cov / z_pairs
    variance = z_curve[0]

    if not np.isfinite(variance) or variance <= 0:
        return None

    plane = np.fft.fftshift(plane)
    center_y, center_x = np.array(plane.shape) // 2
    rows = slice(center_y - max_radius, center_y + max_radius + 1)
    columns = slice(center_x - max_radius, center_x + max_radius + 1)
    plane = plane[rows, columns]

    offsets = np.abs(np.arange(-max_radius, max_radius + 1))
    xy_pairs = depth * np.outer(height - offsets, width - offsets)
    xy_curve = _radial_average(plane / xy_pairs, max_radius)
    return xy_curve / variance, z_curve / variance


def _profile(curves: list, spacing_um: float) -> Profile:
    # tiles are already normalized at lag zero, averaging keeps that scale
    mean = np.mean(curves, axis=0)
    lags = np.arange(len(mean)) * spacing_um
    return Profile(lags, mean, spacing_um)


def autocorrelation(volume: Volume, output_path, *, channels=None, tile_size=1024, max_tiles=32, tissue_fraction=0.5, device="cpu"):
    """average tile ACFs with equal weights, including lag zero in XY and Z

    keep tiles whose mean reaches tissue_fraction times the channel's overview mean
    this selects bright regions, it is not a tissue mask
    """
    if device not in ("cpu", "cuda"):
        raise ValueError("device must be 'cpu' or 'cuda'")
    if device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("cuda seems unavailable")

    if tissue_fraction < 0:
        raise ValueError("tissue_fraction must be nonnegative")

    depth, height, width, n_channels = volume.shape
    if depth < 2:
        raise ValueError("ACF requires at least two sections")

    if not isinstance(tile_size, (int, np.integer)) or tile_size < 2:
        raise ValueError("tile_size must be an integer >= 2")
    if not isinstance(max_tiles, (int, np.integer)) or max_tiles < 1:
        raise ValueError("max_tiles must be a positive integer")

    tile_size = int(tile_size)
    max_tiles = int(max_tiles)

    for spacing in (volume.xy_spacing_um, volume.z_spacing_um):
        if not np.isfinite(spacing) or spacing <= 0:
            raise ValueError("volume spacings must be finite and positive")

    if channels is None:
        channels = list(range(n_channels))
    else:
        channels = list(channels)

    if not channels:
        raise ValueError("channels must not be empty")

    for index, channel in enumerate(channels):
        if not isinstance(channel, (int, np.integer)) or not 0 <= channel < n_channels:
            raise ValueError(f"channel {channel} outside 0..{n_channels - 1}")
        channels[index] = int(channel)

    overview, scale = _overview(volume)
    xy = {}
    z = {}
    used_tiles = {}

    for channel in channels:
        selected_tiles = _select_tiles(overview, scale, channel, width, height, tile_size, max_tiles, tissue_fraction)
        xy_curves = []
        z_curves = []
        used_tiles[str(channel)] = []

        for tile in tqdm(selected_tiles, desc=f"acf channel {channel}"):
            sections = []
            for section_index in range(depth):
                region = volume.read_region(section_index, tile.x, tile.y, tile_size, tile_size)
                sections.append(region[..., channel])

            #print(np.stack(sections).shape)
            profiles = _cube_profiles(np.stack(sections), tile_size // 2, depth // 2, device)
            if profiles is None:
                continue

            xy_curves.append(profiles[0])
            z_curves.append(profiles[1])
            used_tiles[str(channel)].append(list(tile))

        if not xy_curves:
            raise ValueError(f"channel {channel}: all selected tiles have zero or invalid variance")

        xy[channel] = _profile(xy_curves, volume.xy_spacing_um)
        z[channel] = _profile(z_curves, volume.z_spacing_um)

    meta = {
        "channels": channels,
        "tile_size": tile_size,
        "max_tiles": max_tiles,
        "n_tiles": {channel: len(tiles) for channel, tiles in used_tiles.items()},
        "tiles": used_tiles,
        "xy_spacing_um": volume.xy_spacing_um,
        "z_spacing_um": volume.z_spacing_um,
        "level": volume.level,
        "has_pyramid": volume.has_pyramid,
        "n_sections": depth,
        "source_folder": os.path.dirname(volume.paths[0]),
        "tissue_fraction": float(tissue_fraction),
    }

    result = AutocorrelationResult(xy, z, meta)
    result.save(output_path)
    return result
