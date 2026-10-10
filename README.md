# volanis

Tools for analysing and interpolating histology volumes.
Made to handle large datasets.

```sh
pip install -e .
# with McCann interpolation and Wasserstein evaluation support
pip install -e ".[ot]"
```

Run these commands from the project folder.
Evaluation uses Pearson and SSIM by default. To also use Wasserstein, install
`.[ot]` and pass `metrics=("pearson", "ssim", "wasserstein")`.

```python
import volanis

volume = volanis.open_volume("slices", xy_spacing_um=0.325, z_spacing_um=24)
volanis.interpolate_linear(volume, "output", frames_between=5)
```


# Todo list
- Migrate autocorrelation to cupy to use GPU