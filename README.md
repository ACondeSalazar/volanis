# volanis

Tools for analysing and interpolating histology volumes.
Made to handle large datasets.

```sh
pip install -e .
# with McCann Optimal Transport interpolation support
pip install -e ".[mccann]"
```

```python
import volanis

volume = volanis.open_volume("slices", xy_spacing_um=0.325, z_spacing_um=24)
volanis.interpolate_linear(volume, "output", frames_between=5)
```
