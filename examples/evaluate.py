import volanis


#------ lazy load volume -----
truth = volanis.open_volume("data/sections", xy_spacing_um=0.325, z_spacing_um=24.0)
sparse = truth.prune(4)

#------ interpolate new slices -----
volanis.interpolate_linear(sparse, "output/evaluation", frames_between=3)
reconstructed = volanis.open_volume(
    "output/evaluation",
    xy_spacing_um=truth.xy_spacing_um,
    z_spacing_um=truth.z_spacing_um,
)

#------ measure recontruction accuracy -----
scores = volanis.score_reconstruction(
    truth,
    reconstructed,
    keep_every=4,
    metrics=("pearson", "ssim"),
)
print(scores)
