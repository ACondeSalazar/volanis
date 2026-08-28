import volanis


#------ lazy load volume -----
volume = volanis.open_volume("data/sections", xy_spacing_um=0.325, z_spacing_um=24.0)

#------ linearly interpolate 3 new frames -----
volanis.interpolate_linear(volume, "output/linear", frames_between=3)

#------ interpolate 5 new frames with mccann optimal transport -----
volanis.interpolate_mccann(
    volume,
    "output/mccann",
    frames_between=5,
    tile_size=512,
    channels=[0, 1], # only operate on chosen channels
)
