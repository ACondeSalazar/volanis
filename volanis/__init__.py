"""
volanis - scale analysis and interpolation for anisotropic histology volumes.
"""
from .analysis import AutocorrelationFit, AutocorrelationResult, autocorrelation, fit
from .evaluate import (
    compare,
    pearson_weighted,
    score_reconstruction,
    ssim_multi,
    sweep_coarse_graining,
    sweep_gaps,
    wasserstein_um,
)
from .linear import interpolate_linear
from .mccann import interpolate_mccann
from .plotting import plot_autocorrelation, plot_coarse_scores, plot_cross_section, plot_scores
from .volume import Volume, open_volume, reencode

__all__ = [
    "AutocorrelationFit",
    "AutocorrelationResult",
    "Volume",
    "autocorrelation",
    "compare",
    "fit",
    "interpolate_linear",
    "interpolate_mccann",
    "open_volume",
    "pearson_weighted",
    "plot_autocorrelation",
    "plot_coarse_scores",
    "plot_cross_section",
    "plot_scores",
    "reencode",
    "score_reconstruction",
    "ssim_multi",
    "sweep_coarse_graining",
    "sweep_gaps",
    "wasserstein_um",
]
