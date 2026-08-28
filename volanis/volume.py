import itertools
import os
import re
from collections import namedtuple
from dataclasses import dataclass, replace

import numpy as np
import pyvips
from tqdm import tqdm

pyvips.cache_set_max(0) # lower ram usage

VIPS_TO_NP = {
    "uchar": np.uint8, "char": np.int8, "ushort": np.uint16, "short": np.int16,
    "uint": np.uint32, "int": np.int32, "float": np.float32, "double": np.float64,
}

SUFFIXES = (".tif", ".tiff")


def _natural_key(name):
    return [int(p) if p.isdigit() else p.lower() for p in re.split(r"(\d+)", name)]


NP_TO_VIPS = {v: k for k, v in VIPS_TO_NP.items()}


def _to_numpy(img):
    return np.ndarray(buffer=img.write_to_memory(), dtype=VIPS_TO_NP[img.format],
                      shape=(img.height, img.width, img.bands))


def from_numpy(array):
    """(height, width, bands) numpy array -> pyvips image"""
    array = np.ascontiguousarray(array)
    height, width, bands = array.shape
    return pyvips.Image.new_from_memory(array.data, width, height, bands,
                                        NP_TO_VIPS[array.dtype.type])


Tile = namedtuple("Tile", "x y width height")


def _starts(extent, tile_size, overlap):
    if tile_size >= extent:
        return [0]
    return list(range(0, extent - overlap, tile_size - overlap))


def tile_grid(width, height, tile_size, *, overlap=0):
    """tiles covering the whole image, row by row"""
    if tile_size <= 0:
        raise ValueError(f"tile_size must be positive, got {tile_size}")
    if not 0 <= overlap < tile_size:
        raise ValueError(f"overlap must be in [0, {tile_size}), got {overlap}")
    if width <= 0 or height <= 0:
        raise ValueError(f"image must be non-empty, got {width}x{height}")
    return [Tile(x, y, min(tile_size, width - x), min(tile_size, height - y))
            for y in _starts(height, tile_size, overlap)
            for x in _starts(width, tile_size, overlap)]


def _level_image(path, level, has_pyramid, access=None):
    kw = {"access": access} if access else {}
    if has_pyramid:
        return pyvips.Image.new_from_file(path, page=level, **kw)
    img = pyvips.Image.new_from_file(path, **kw)
    return img.shrink(2 ** level, 2 ** level) if level else img


def save_tiff(img, path, *, tile_size=256, compression="deflate"):
    """write one pyvips image as a tiled, pyramidal, compressed .tif"""
    tmp = path + ".part"
    try:
        img.copy(interpretation="multiband").tiffsave(
            tmp, tile=True, tile_width=tile_size, tile_height=tile_size, pyramid=True,
            bigtiff=True, compression=compression, predictor="none")
        os.replace(tmp, path)
    except BaseException:
        if os.path.exists(tmp):
            os.remove(tmp)
        raise
    return path


def _pyramid_levels(path, width, height):
    img = pyvips.Image.new_from_file(path)
    if "n-pages" not in img.get_fields():
        return 1
    levels = 1
    for page in range(1, int(img.get("n-pages"))):
        try:
            p = pyvips.Image.new_from_file(path, page=page)
        except pyvips.Error:
            break
        if abs(p.width - width / 2 ** page) > 1 or abs(p.height - height / 2 ** page) > 1:
            break
        levels += 1
    return levels


@dataclass(frozen=True)
class Volume:
    paths: list[str]
    level: int
    shape: tuple[int, int, int, int]      # z, y, x, channels
    dtype: np.dtype
    xy_spacing_um: float
    z_spacing_um: float
    has_pyramid: bool
    n_levels: int
    coarse_factor: int = 1
    blur_sigma_um: float = 0.0

    def __len__(self):
        return len(self.paths)

    def _open(self, index):
        if not 0 <= index < len(self.paths):
            raise IndexError(f"section {index} out of range for {len(self.paths)} sections")
        image = _level_image(self.paths[index], self.level, self.has_pyramid)
        if self.coarse_factor > 1:
            image = image.shrink(self.coarse_factor, self.coarse_factor)
        if self.blur_sigma_um > 0:
            image = image.cast("float").gaussblur(
                self.blur_sigma_um / self.xy_spacing_um, precision="float")
        return image

    def at_level(self, level):
        if level < 0:
            raise ValueError(f"level must be >= 0, got {level}")
        if self.has_pyramid and level >= self.n_levels:
            raise ValueError(f"level {level} unavailable; sections have {self.n_levels} "
                             f"pyramid levels")
        probe = _level_image(self.paths[0], level, self.has_pyramid)
        if self.coarse_factor > 1:
            probe = probe.shrink(self.coarse_factor, self.coarse_factor)
        if probe.width < 1 or probe.height < 1:
            raise ValueError(f"level {level} reduces {os.path.basename(self.paths[0])} to "
                             f"{probe.width}x{probe.height}")
        base_xy = self.xy_spacing_um / (2 ** self.level * self.coarse_factor)
        return replace(self, level=level,
                       xy_spacing_um=base_xy * 2 ** level * self.coarse_factor,
                       shape=(len(self.paths), probe.height, probe.width, self.shape[3]))

    def coarsen(self, sigma_um=0.0, factor=1):
        """lazy Gaussian blur and XY downsampling."""
        if sigma_um < 0 or factor < 1 or int(factor) != factor:
            raise ValueError("sigma must be >= 0 and factor a positive integer")
        if self.coarse_factor != 1 or self.blur_sigma_um != 0:
            raise ValueError("volume is already coarsened")
        factor = int(factor)
        if sigma_um == 0 and factor == 1:
            return self
        image = self._open(0).shrink(factor, factor) if factor > 1 else self._open(0)
        dtype = np.dtype(np.float32) if sigma_um else self.dtype
        return replace(self, shape=(len(self), image.height, image.width, self.shape[3]),
                       dtype=dtype, xy_spacing_um=self.xy_spacing_um * factor,
                       coarse_factor=factor, blur_sigma_um=float(sigma_um))

    def keep_only(self, indices):
        """Z SPACING WILL BE WRONG if you do not keep only evenly spaced slices"""
        wanted = sorted(set(indices))
        if not wanted:
            raise ValueError("needs at least one section")
        outside = [i for i in wanted if not 0 <= i < len(self)]
        if outside:
            raise ValueError(f"sections {outside} outside 0..{len(self) - 1}")

        steps = {b - a for a, b in itertools.pairwise(wanted)}
        spacing = self.z_spacing_um * steps.pop() if len(steps) == 1 else self.z_spacing_um
        return replace(self, paths=[self.paths[i] for i in wanted],
                       shape=(len(wanted),) + self.shape[1:], z_spacing_um=spacing)

    def prune(self, keep_every):
        """keep one section out of `keep_every`."""
        if keep_every < 1:
            raise ValueError(f"keep_every must be >= 1, got {keep_every}")
        return self.keep_only(range(0, len(self), keep_every))

    def read_region(self, index, x, y, width, height):
        """(height, width, channels) from one section, at the current pyramid level."""
        img = self._open(index)
        if width <= 0 or height <= 0:
            raise ValueError(f"region must be non-empty, got {width}x{height}")
        if x < 0 or y < 0 or x + width > img.width or y + height > img.height:
            raise ValueError(f"region ({x}, {y}, {width}, {height}) outside "
                             f"{img.width}x{img.height} section {index}")
        return _to_numpy(img.crop(x, y, width, height))

    def read_slice(self, index):
        """(y, x, channels) for one whole section, at the current level."""
        _, height, width, _ = self.shape
        return self.read_region(index, 0, 0, width, height)

    def read_cross_section(self, direction, index):
        """direction is xz or yz"""
        if direction not in ("xz", "yz"):
            raise ValueError(f"direction must be 'xz' or 'yz', got {direction!r}")
        _, height, width, _ = self.shape
        limit = height if direction == "xz" else width
        if not 0 <= index < limit:
            raise IndexError(f"{direction} index {index} out of range for {limit}")
        lines = []
        for i in range(len(self)):
            if direction == "xz":
                lines.append(self.read_region(i, 0, index, width, 1)[0])
            else:
                lines.append(self.read_region(i, index, 0, 1, height)[:, 0])
        return np.stack(lines, axis=0)


def reencode(volume, output_folder, *, tile_size=256, compression="deflate"):
    "save the .tif stack into a new folder with tiling, pyramid and compression"
    os.makedirs(output_folder, exist_ok=True)
    for name in volume.paths:
        out = os.path.join(output_folder, os.path.basename(name))
        if os.path.realpath(out) == os.path.realpath(name):
            raise ValueError(f"{output_folder} would overwrite the source section {name}")

    for path in tqdm(volume.paths, desc="reencode"):
        out = os.path.join(output_folder, os.path.basename(path))
        if os.path.exists(out):
            continue
        save_tiff(_level_image(path, volume.level, volume.has_pyramid, access="sequential"),
                  out, tile_size=tile_size, compression=compression)

    return open_volume(output_folder, xy_spacing_um=volume.xy_spacing_um,
                       z_spacing_um=volume.z_spacing_um)


def open_volume(folder, *, xy_spacing_um, z_spacing_um):
    """lazy load a volume"""
    names = sorted((f for f in os.listdir(folder) if f.lower().endswith(SUFFIXES)),
                   key=_natural_key)
    if not names:
        raise ValueError(f"no .tif or .tiff sections in {folder}")
    paths = [os.path.join(folder, n) for n in names]

    first = pyvips.Image.new_from_file(paths[0], access="sequential")
    width, height, bands, fmt = first.width, first.height, first.bands, first.format
    for path in paths[1:]:
        img = pyvips.Image.new_from_file(path, access="sequential")
        for label, got, want in (("size", f"{img.width}x{img.height}", f"{width}x{height}"),
                                 ("channels", img.bands, bands),
                                 ("dtype", img.format, fmt)):
            if got != want:
                raise ValueError(f"{path}: {label} is {got}, but {paths[0]} has {want}")

    levels = {_pyramid_levels(p, width, height) for p in paths}
    has_pyramid = len(levels) == 1 and levels != {1}
    n_levels = levels.pop() if has_pyramid else 1

    return Volume(paths=paths, level=0, shape=(len(paths), height, width, bands),
                  dtype=np.dtype(VIPS_TO_NP[fmt]), xy_spacing_um=xy_spacing_um,
                  z_spacing_um=z_spacing_um, has_pyramid=has_pyramid, n_levels=n_levels)
