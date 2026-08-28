import os

import numpy as np
from tqdm import tqdm

from .volume import from_numpy, save_tiff, tile_grid

TILE = 2048


def interpolate_linear(volume, output_folder, *, frames_between):
    os.makedirs(output_folder, exist_ok=True)
    step = frames_between + 1
    _, height, width, _ = volume.shape
    tiles = tile_grid(width, height, TILE)

    def path(index, k):
        stem = os.path.splitext(os.path.basename(volume.paths[index]))[0]
        return os.path.join(output_folder, f"{stem}_{k}.tif")

    left = volume.read_slice(0)
    for i in tqdm(range(len(volume) - 1), desc="linear"):
        right = volume.read_slice(i + 1)
        save_tiff(from_numpy(left), path(i, 0))

        for k in range(1, step):
            t = k / step
            frame = np.empty_like(left)
            for tile in tiles:
                box = np.s_[tile.y:tile.y + tile.height, tile.x:tile.x + tile.width]
                frame[box] = np.round((1 - t) * left[box] + t * right[box])
            save_tiff(from_numpy(frame), path(i, k))
        left = right

    save_tiff(from_numpy(left), path(len(volume) - 1, 0))
