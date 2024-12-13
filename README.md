## Mesh Reconstruction (Combination) from Splitted Mesh

### Installation

```bash
pip install -r requirements.txt
```

Please refer to [Pytorch3d](https://github.com/facebookresearch/pytorch3d) for more installation details.

### Usage

```bash
python combine.py --mesh /path/to/mesh --output_dir /path/to/output
```
The mesh option can be a single .obj file or a directory full of .obj files. The final .obj file will be saved in the `output_dir/xxx/meshes/final.obj`.