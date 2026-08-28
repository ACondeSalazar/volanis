import volanis


#------ define method to evaluate -----
def linear(volume, folder, frames_between):
    volanis.interpolate_linear(volume, folder, frames_between=frames_between)

#------ lazy load volume -----
volume = volanis.open_volume("data/sections", xy_spacing_um=0.325, z_spacing_um=24.0)
metrics = ("pearson", "ssim")

#------ measure reconstruction accuracy among multiple gap size -----
gap_scores = volanis.sweep_gaps(
    volume,
    "output/gap_sweep",
    linear,
    keep_everys=(2, 4, 8),
    metrics=metrics,
)
figure = volanis.plot_scores({"linear": gap_scores}, metrics=metrics)
figure.savefig("output/gap_sweep.png", dpi=150)

#------ measure reconstruction accuracy among multiple coarse graining factor -----
coarse_scores = volanis.sweep_coarse_graining(
    volume,
    "output/coarse_sweep",
    linear,
    keep_every=4,
    coarse_grainings=((0, 1), (1, 2), (2, 4)), # array of (sigma_um,downsampling_factor) (0,1) means no gaussian blur and image at full resolution
    metrics=metrics,
)
figure = volanis.plot_coarse_scores({"linear": coarse_scores}, metrics=metrics)
figure.savefig("output/coarse_sweep.png", dpi=150)
