from pathlib import Path

import volanis


#------ compute raw autocorrelation functions -----
Path("output").mkdir(exist_ok=True)
volume = volanis.open_volume("data/sections", xy_spacing_um=0.325, z_spacing_um=24.0)
result = volanis.autocorrelation(volume, "output/autocorrelation.npz", tile_size=512, max_tiles=128)

#------ fit exponential models-----
fits = volanis.fit(result)

#------ plot results -----
figure = volanis.plot_autocorrelation(fits)
figure.savefig("output/autocorrelation.png", dpi=150)
