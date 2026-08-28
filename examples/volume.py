from pathlib import Path

import volanis


#------ lazy load volume -----
volume = volanis.open_volume(
    "data/sections",
    xy_spacing_um=0.325,
    z_spacing_um=24.0,
)

#------ metadata -----
print(f"{len(volume)} sections")
print(volume.shape, volume.dtype)
print(volume.xy_spacing_um, volume.z_spacing_um)
print(volume.has_pyramid, volume.n_levels)

#------ fetch data -----
section = volume.read_slice(0)
region = volume.read_region(0, x=100, y=100, width=512, height=512)
xz = volume.read_cross_section("xz", volume.shape[1] // 2)
yz = volume.read_cross_section("yz", volume.shape[2] // 2)

#------ lazy views -----
level = volume.at_level(2) 
selected = volume.keep_only([0, 2, 4, 6])
sparse = volume.prune(4)
coarse = volume.coarsen(sigma_um=1.0, factor=2)

#------ plot cross section -----
Path("output").mkdir(exist_ok=True)
figure = volanis.plot_cross_section(volume, direction="xz")
figure.savefig("output/cross_section.png", dpi=150)

#------ reencode -----
encoded = volanis.reencode(volume, "output/reencoded")

print(section.shape, region.shape, xz.shape, yz.shape)
print(level.shape, selected.shape, sparse.shape, coarse.shape, encoded.shape)
