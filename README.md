# confmap
Native Python implementation of conformal mapping methods CETM and BFF (the only dependencies are NumPy and SciPy).

Based on papers [Conformal Equivalence of Triangle Meshes](https://dl.acm.org/doi/10.1145/1360612.1360676), B. Springborn, P. Schröder, U. Pinkall, *ACM Transactions on Graphics* (2008) and [Boundary First Flattening](https://dl.acm.org/doi/10.1145/3132705), R. Sawhney, K. Crane, *ACM Transactions on Graphics* (2018).

<p align="center">
  <img width="350" src="https://github.com/russelmann/confmap/blob/main/media/bumpcap_3d.png" alt="Bumpcap with UV checkerboard">
  <br>Bumpcap mesh with UV checkerboard.
</p>

<p align="center">
  <img width="500" src="https://github.com/russelmann/confmap/blob/main/media/bumpcap_maps.png" alt="Bumpcap plots">
  <br>Application of different tools with the resulting scale factors (left) and quasi-conformal errors (right).
</p>

## Installation

### From source

1. Clone this repository.
2. Go to root folder and build development version from source. `pip install -e .`

## Usage example

Read an OBJ file from data folder, generate a minimum distortion conformal map using BFF method, and output the original mesh with the UV conformal map as a new OBJ file.

```python
from confmap.confmap import BFF
from confmap.io_utils import read_obj, write_obj

vertices, faces = read_obj('../data/bumpcap.obj')
cm = BFF(vertices, faces)
image = cm.layout()
write_obj('bumpcap_with_uv.obj', cm.vertices, cm.faces, image.vertices, image.faces)
```
