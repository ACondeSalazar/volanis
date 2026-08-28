
LINESTYLES = ("--", ":", "-.")
AXIS_LABEL = {"xy": "distance (µm)", "z": "dz (µm)"}

MARKER_LIMIT = 40


def _trim(value, digits):
    return f"{value:.{digits}f}".lstrip("0")


def _relative_error(value, error):
    if not value > 0:
        return "n/a"
    percent = 100 * error / value
    return ">999%" if percent > 999 else f"{percent:.1f}%"


def _summary(model_fit, n_points):
    if model_fit.status == "failed":
        return f"{model_fit.model:7s} failed"
    lengths = "/".join(f"{v:.3g}" for v in model_fit.lengths_um)
    errors = "/".join(_relative_error(v, e)
                      for v, e in zip(model_fit.lengths_um, model_fit.length_errors_um))
    weights = "/".join(f"{w:.3g}" for w in model_fit.weights)
    note = "" if model_fit.status == "ok" else f"  {model_fit.status.upper()}"
    return (f"{model_fit.model:7s} l {lengths}  ±{errors}  w {weights}"
            f"  R² {_trim(model_fit.r_squared, 5)}  AIC {model_fit.aic:.0f}{note}")


def plot_autocorrelation(fits, *, channel_names=None, log_y=True, max_lag_um=None):
    import matplotlib.pyplot as plt
    import numpy as np

    channels = fits.channels
    names = channel_names or {c: f"ch {c}" for c in channels}
    fig, axes = plt.subplots(len(channels), 2, figsize=(13.2, 3.6 * len(channels)),
                             squeeze=False)

    clamp = (lambda v: np.maximum(v, 1e-2)) if log_y else (lambda v: v)

    for row, channel in enumerate(channels):
        for col, direction in enumerate(("xy", "z")):
            ax = axes[row][col]
            profile = getattr(fits.result, direction)[channel]
            lags = profile.lags_um
            ax.plot(lags, clamp(profile.measured_curve), color="k", lw=1.2, label="data",
                    zorder=3)
            if len(lags) <= 40:
                ax.plot(lags, clamp(profile.measured_curve), ls="none", marker="o", ms=2.6,
                        color="k", zorder=4)

            lines = []
            for i, model_fit in enumerate(getattr(fits, direction)[channel].values()):
                if model_fit.status != "failed":
                    ax.plot(lags, clamp(model_fit.fitted_curve),
                            ls=LINESTYLES[i % len(LINESTYLES)], lw=1.4,
                            label=model_fit.model)
                lines.append(_summary(model_fit, len(lags)))

            ax.set_title(f"{names[channel]} {direction}", fontsize=12)
            ax.grid(alpha=0.3, which="both")
            ax.legend(loc="upper right", fontsize=8)
            if log_y:
                ax.set_yscale("log")
                ax.set_ylim(1e-2 * 0.75, 1.4)
            ax.set_xlim(0, max_lag_um or lags.max())

            ax.set_xlabel(AXIS_LABEL[direction] + "\n" + "\n".join(lines),
                          fontsize=7, fontfamily="monospace")

    meta = fits.meta
    fig.suptitle(f"autocorrelation — tile {meta['tile_size']} px, "
                 f"XY spacing : {meta['xy_spacing_um']:g} µm/px, Z spacing {meta['z_spacing_um']:g} µm/px",
                 fontsize=10)
    fig.tight_layout()
    return fig


def plot_cross_section(volume, direction="xz", index=None, stretch=1.0):
    import matplotlib.pyplot as plt

    _, height, width, _ = volume.shape
    if index is None:
        index = (height if direction == "xz" else width) // 2

    image = volume.read_cross_section(direction, index)
    if image.shape[2] == 3:
        image = image / max(image.max(), 1)
    else:
        image = image.mean(axis=2)

    ratio = volume.z_spacing_um / volume.xy_spacing_um
    aspect = "auto" if stretch is None else ratio * stretch

    fig, ax = plt.subplots(figsize=(13, 4))
    ax.imshow(image, aspect=aspect, interpolation="nearest",
              cmap=None if image.ndim == 3 else "gray")
    ax.set_xlabel("x (px)" if direction == "xz" else "y (px)")
    ax.set_ylabel("z (section)")
    scale = "filled" if stretch is None else f"z x{stretch:g} of true"
    ax.set_title(f"{direction} at {index}, Z spacing {volume.z_spacing_um:g} µm, "
                 f"XY spacing {volume.xy_spacing_um:g} µm/px, {scale}")
    fig.tight_layout()
    return fig


def plot_scores(results, *, metrics=("pearson", "ssim", "wasserstein")):
    """One panel per metric: score against section gap, one line per method.

    `results` is {label: sweep_gaps output}. Error bars are the 95% interval on the mean,
    so two methods overlapping means the data cannot tell them apart.
    """
    import matplotlib.pyplot as plt
    import numpy as np

    fig, axes = plt.subplots(1, len(metrics), figsize=(4.6 * len(metrics), 4.0),
                             squeeze=False)
    for ax, metric in zip(axes[0], metrics):
        for label, by_gap in results.items():
            gaps = sorted(by_gap)
            means, errors = [], []
            for gap in gaps:
                values = np.asarray(by_gap[gap][metric], dtype=float)
                values = values[np.isfinite(values)]
                means.append(values.mean() if len(values) else np.nan)
                errors.append(1.96 * values.std(ddof=1) / np.sqrt(len(values))
                              if len(values) > 1 else 0.0)
            ax.errorbar(gaps, means, yerr=errors, marker="o", ms=4, lw=1.6, capsize=3,
                        label=label)
        ax.set_title(metric, fontsize=11)
        ax.set_xlabel("gap between real sections (µm)")
        ax.grid(alpha=0.3)
        ax.legend(fontsize=9)
    fig.tight_layout()
    return fig


def plot_coarse_scores(results, *, metrics=("pearson", "ssim", "wasserstein")):
    """Plot coarse-graining scores, one line per interpolation method."""
    import matplotlib.pyplot as plt
    import numpy as np

    configurations = list(next(iter(results.values())))
    x = np.arange(len(configurations))
    fig, axes = plt.subplots(1, len(metrics), figsize=(4.6 * len(metrics), 4.0),
                             squeeze=False)
    for ax, metric in zip(axes[0], metrics):
        for label, scores in results.items():
            samples = [np.asarray(scores[c][metric], dtype=float) for c in configurations]
            samples = [values[np.isfinite(values)] for values in samples]
            means = [values.mean() if len(values) else np.nan for values in samples]
            errors = [1.96 * values.std(ddof=1) / np.sqrt(len(values))
                      if len(values) > 1 else 0.0 for values in samples]
            ax.errorbar(x, means, yerr=errors, marker="o", capsize=3, label=label)
        ax.set_title(metric)
        ax.set_xticks(x, [f"σ={sigma:g} µm\n×{factor}" for sigma, factor in configurations])
        ax.set_xlabel("Gaussian blur and downsampling")
        ax.grid(alpha=0.3)
        ax.legend()
    fig.tight_layout()
    return fig
