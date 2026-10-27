import numpy as np
from skimage.filters import threshold_otsu

from ..volume import Tile, Volume

OVERVIEW_MAX_PX = 1024


def _axis_starts(start: int, stop: int, extent: int, tile_size: int) -> list[int]:
    # Preserve the requested tile size when the signal box is smaller than a tile.
    if stop - start < tile_size:
        start = max(0, min((start + stop - tile_size) // 2, extent - tile_size))
        stop = start + tile_size

    last = stop - tile_size
    starts = list(range(start, last + 1, tile_size // 2))
    if starts[-1] != last:
        starts.append(last)
    return starts


def _mean_overview(volume: Volume) -> tuple[np.ndarray, int]:
    level = volume.level
    size = max(volume.shape[1:3])
    while size > OVERVIEW_MAX_PX:
        if volume.has_pyramid and level + 1 >= volume.n_levels:
            raise ValueError("a smaller TIFF pyramid level is needed")
        level += 1
        size = (size + 1) // 2

    overview = volume.at_level(level)
    mean = np.zeros(overview.shape[1:], dtype=np.float32)
    for section in range(len(volume)):
        image = overview.read_slice(section)
        mean += image
        del image
    mean /= len(volume)
    return mean, 2 ** (level - volume.level)


def sample_tiles(
    volume: Volume,
    *,
    channels: list[int],
    tile_size: int,
    max_tiles: int | None,
    tissue_fraction: float,
    seed: int,
) -> dict[int, list[Tile]]:
    """Sample a grid inside each channel's Otsu signal bounding box.

    tissue_fraction is the required fraction of above-threshold overview pixels.
    """
    _, height, width, _ = volume.shape
    overview, scale = _mean_overview(volume)
    if tile_size < scale:
        raise ValueError("tiles are smaller than an overview pixel")

    selected = {}
    for channel in channels:
        plane = overview[..., channel]
        signal_mask = plane > threshold_otsu(plane)
        occupied_rows = np.flatnonzero(signal_mask.any(axis=1))
        occupied_columns = np.flatnonzero(signal_mask.any(axis=0))
        if len(occupied_rows) == 0:
            selected[channel] = []
            continue

        left = int(occupied_columns[0]) * scale
        right = min((int(occupied_columns[-1]) + 1) * scale, width)
        top = int(occupied_rows[0]) * scale
        bottom = min((int(occupied_rows[-1]) + 1) * scale, height)
        x_starts = _axis_starts(left, right, width, tile_size)
        y_starts = _axis_starts(top, bottom, height, tile_size)
        columns = len(x_starts)
        grid_size = len(y_starts) * columns

        rng = np.random.default_rng(np.random.SeedSequence([seed, channel]))
        tiles = []
        for index in rng.permutation(grid_size):
            row, column = divmod(int(index), columns)
            x = x_starts[column]
            y = y_starts[row]
            start_x = x // scale
            start_y = y // scale
            stop_x = min((x + tile_size + scale - 1) // scale, plane.shape[1])
            stop_y = min((y + tile_size + scale - 1) // scale, plane.shape[0])
            signal_fraction = float(signal_mask[start_y:stop_y, start_x:stop_x].mean())
            if signal_fraction > 0 and signal_fraction >= tissue_fraction:
                tiles.append(Tile(x, y, tile_size, tile_size))
                if max_tiles is not None and len(tiles) == max_tiles:
                    break
        selected[channel] = tiles
    return selected
