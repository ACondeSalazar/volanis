import os

import volanis

DATA = "/media/T7/pi_playground/volanis/lsp22409/ex"
OUTPUT = "./out/"
LEVEL = 0
GAPS = (2, 3, 4, 5, 6)
COARSE_GRAININGS = ((0, 1), (20, 1), (40, 2))
METRICS = ("pearson", "ssim", "wasserstein")

os.makedirs(OUTPUT, exist_ok=True)

volume = volanis.open_volume(DATA, xy_spacing_um=0.14, z_spacing_um=0.28)
truth = volume.at_level(LEVEL)
print(truth.shape)


acf_path = os.path.join(OUTPUT, "autocorrelation.npz")
volanis.autocorrelation(truth, acf_path, tile_size=1024, max_tiles=200, channels=[0])
fits = volanis.fit(acf_path)
volanis.plot_autocorrelation(fits).savefig(
    os.path.join(OUTPUT, "autocorrelation.png"), dpi=150)
volanis.plot_cross_section(truth, "xz").savefig(
    os.path.join(OUTPUT, "cross_section.png"), dpi=150)

exit()

methods = {
    "linear": lambda volume, folder, frames: volanis.interpolate_linear(
        volume, folder, frames_between=frames),
    "mccann": lambda volume, folder, frames: volanis.interpolate_mccann(
        volume, folder, frames_between=frames, tile_size=128),
}

gap_scores = {}
coarse_scores = {}

for name, interpolate in methods.items():
    folder = os.path.join(OUTPUT, name)

    gap_scores[name] = volanis.sweep_gaps(truth, folder, interpolate, GAPS, metrics=METRICS)

    coarse_scores[name] = volanis.sweep_coarse_graining(truth, os.path.join(folder, "coarse"), interpolate, 3,COARSE_GRAININGS, metrics=METRICS)

volanis.plot_scores(gap_scores, metrics=METRICS).savefig( os.path.join(OUTPUT, "gaps.png"), dpi=150)

volanis.plot_coarse_scores(coarse_scores, metrics=METRICS).savefig( os.path.join(OUTPUT, "coarse.png"), dpi=150)
