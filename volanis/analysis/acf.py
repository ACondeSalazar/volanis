import json
import os
from collections import deque
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from functools import lru_cache

import numpy as np
import torch
from scipy.fft import next_fast_len
from tqdm import tqdm

from ..volume import Tile, Volume
from .sampling import sample_tiles


SEED = 0 # seed for random numbers

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


def _radial_average(plane: np.ndarray, max_radius: int) -> np.ndarray:
    center = plane.shape[0] // 2
    rows, columns = np.indices(plane.shape)
    radius = np.rint(np.hypot(rows - center, columns - center)).astype(int).ravel()

    counts = np.bincount(radius)[: max_radius + 1]
    sums = np.bincount(radius, weights=plane.ravel())[: max_radius + 1]
    return sums / counts


def _cube_profiles(
    cube: torch.Tensor, max_radius: int, max_z_lag: int
) -> tuple[np.ndarray, np.ndarray] | None:
    """Center the tile in place and return normalized XY and Z autocorrelations."""
    depth, height, width = cube.shape

    # Padding prevents wraparound within the requested XY lag range.
    padded_height = next_fast_len(height + max_radius)
    padded_width = next_fast_len(width + max_radius)
    fft_shape = (padded_height, padded_width)

    with torch.no_grad():
        cube -= cube.mean()

        power = torch.zeros(
            (padded_height, padded_width // 2 + 1),
            dtype=torch.float32,
            device=cube.device,
        )
        for d in range(0, depth):
            power += torch.fft.rfft2(cube[d], s=fft_shape).abs().square_()
        plane = torch.fft.irfft2(power, s=fft_shape)
        del power

        plane = torch.fft.fftshift(plane)
        center_y, center_x = plane.shape[0] // 2, plane.shape[1] // 2
        plane = plane[ center_y - max_radius : center_y + max_radius + 1, center_x - max_radius : center_x + max_radius + 1, ]

        z_covariances = []
        for lag in range(max_z_lag + 1):
            left = cube[: depth - lag].reshape(-1)
            right = cube[lag:].reshape(-1)
            z_covariances.append(torch.dot(left, right))

        plane = plane.cpu().numpy()
        z_covariance = torch.stack(z_covariances).cpu().numpy()

    z_pairs = (depth - np.arange(max_z_lag + 1)) * float(height * width)
    z_curve = z_covariance / z_pairs
    variance = z_curve[0]

    if not np.isfinite(variance) or variance <= 0:
        return None

    offsets = np.abs(np.arange(-max_radius, max_radius + 1))
    xy_pairs = depth * np.outer(height - offsets, width - offsets)
    xy_curve = _radial_average(plane / xy_pairs, max_radius)
    return xy_curve / variance, z_curve / variance


def _profile(curves: list, spacing_um: float) -> Profile:
    # tiles are already normalized at lag zero, averaging keeps that scale
    mean = np.mean(curves, axis=0)
    lags = np.arange(len(mean)) * spacing_um
    return Profile(lags, mean, spacing_um)


def _load_cube( volume: Volume, tile: Tile, channel: int, cube: torch.Tensor, executor: ThreadPoolExecutor, n_workers: int ) -> None:
    def load_section(section: int) -> torch.Tensor:
        region = volume.read_region(section, *tile, channel=channel)
        source = torch.from_numpy(region[..., 0])
        return source

    depth = len(volume)
    pending = deque()
    for section in range(min(n_workers, depth)):
        pending.append(executor.submit(load_section, section))
    next_section = len(pending)

    for section in range(depth):
        source = pending.popleft().result()
        if next_section < depth:
            pending.append(executor.submit(load_section, next_section))
            next_section += 1

        cube[section].copy_(source)

        del source


def autocorrelation( volume: Volume, output_path, *, channels=None, tile_size: int = 1024, max_tiles: int | None = 32, tissue_fraction: float = 0.1, device: str = "cuda", n_workers: int = 4) -> AutocorrelationResult:
    """Average full-depth tile ACFs"""
    if n_workers < 1: raise ValueError("n_workers must be positive")

    if device not in ("cpu", "cuda"): raise ValueError("device must be 'cpu' or 'cuda'")
    if device == "cuda" and not torch.cuda.is_available(): raise RuntimeError("cuda seems unavailable")

    if not 0 <= tissue_fraction <= 1: raise ValueError("tissue_fraction must between 0 and 1")

    depth, height, width, n_channels = volume.shape

    if tile_size > min(width, height): raise ValueError("tile_size is bigger than the image")

    tile_size = int(tile_size)

    if channels is None: channels = list(range(n_channels))

    if not channels: raise ValueError("channels must not be empty")

    for index, channel in enumerate(channels):
        if not isinstance(channel, (int, np.integer)) or not 0 <= channel < n_channels:
            raise ValueError(f"channel {channel} outside 0..{n_channels - 1}")
        channels[index] = int(channel)

    selected_tiles_by_channel = sample_tiles( volume, channels=channels, tile_size=tile_size, max_tiles=max_tiles, tissue_fraction=tissue_fraction, seed=SEED)

    for channel in channels:
        if not selected_tiles_by_channel[channel]:
            raise ValueError( f"channel {channel}: no full {tile_size}x{tile_size} signal tile found, try lowering tissue fraction or tile_size" )

    cube = torch.empty( (depth, tile_size, tile_size), dtype=torch.float32, device=device )
    xy = {}
    z = {}
    used_tiles = {}

    with ThreadPoolExecutor(max_workers=n_workers) as executor:
        for channel in channels:
            selected_tiles = selected_tiles_by_channel[channel]
            xy_curves = []
            z_curves = []
            used_tiles[str(channel)] = []

            for tile in tqdm(selected_tiles, desc=f"acf channel {channel}"):
                _load_cube(volume, tile, channel, cube, executor, n_workers)
                profiles = _cube_profiles(cube, tile_size // 2, depth // 2)
                if profiles is None:
                    continue

                xy_curve, z_curve = profiles
                xy_curves.append(xy_curve)
                z_curves.append(z_curve)
                used_tiles[str(channel)].append(list(tile))

            if not xy_curves:
                raise ValueError( f"channel {channel}: all selected tiles have zero or invalid variance" )

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
