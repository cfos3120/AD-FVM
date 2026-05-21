# AD-FVM
An Auto-differentiable Finite Volume Method (**AD-FVM**) for generative machine-learning on irregular grids and unstructured meshes. Example code includes application to learning Neural Operators for Computational Fluid Dynamics across a range of steady Reynolds Numbers.

This repository accompanies the pre-print paper available here: 
[An Auto-Differentiable Finite Volume Method for Physics-Informed Neural Operator Learning](https://papers.ssrn.com/sol3/papers.cfm?abstract_id=6193846)

---
## Neural Operator Example
To demonstrate the technological implementation an example model-pipeline and test case is provided. The problem space includes learning the operator parameterisation of the 2D incompressible Navier-Stokes PDE across a laminar steady range of Reynolds numbers (Re) for flow over [2D Cylinder](/tests/case_files/cylinder/). To demonstrate the novel capability of **AD-FVM**, this fluid problem is discretised on an unstructured mesh. 

[](/resources/PINO_FVM.png)
<p align="center">
  <img width="60%" src=/resources/PINO_FVM.png/>
</p>

Given the required mesh files are generated (see [here](/tests/case_files/README.md)). Training and validation of a [General Neural Operator Transformer](https://github.com/WenjunDong/GNOT/blob/master/readme.md) is performed by navigating to the `AD-FVM` directory and executing the following:
```
python3 -m tests.train --config ./tests/models/configs/cylinder_2d_geneva.yaml
```
---

## Usage
The **AD-FVM** framework requires a mesh object `FVM_Mesh` which is defined as an abstract method in [mesh.py](src/mesh_utils/mesh.py). The abstract class is for the user to create their own mesh information parsing. As an example, a `FVM_Mesh` class for OpenFOAM case-files is provided, but the abstract class is provided.

```python
from src import OpenFoam_Mesh
Mesh = OpenFoam_Mesh(openfoam_case_dir=Path('./tests/case_files/cylinder/case.foam'), corrected=True)
```
To align with OpenFOAM FVM methodology, over-relaxed non-orthogonal mesh corrections can be applied (only effective to the Laplacian) by setting `corrected=True`. Validation of the **AD_FVM** Gauss-Green implementation to OpenFOAM is also presented [here](/tests/case_files/validation.ipynb).

### Operators

The individual operators are found in [operators](src/operators/) and include:
1. [Advection](src/operators/divergence.py)
2. [Gradient](src/operators/gradient.py)
3. [Laplacian](src/operators/laplacian.py)

Each operator is calculated for a given `torch.Tensor` input field that has shape `[B,T,N,C]`. For time-independent problems and scalar calculations the input field should be reshaped as `[B,1,N,C]` and `[B,T,N,1]` respectively. Additionally, vector input fields should have channel dimension the same size as the mesh dimension (e.g. 2D or 3D).
```python
from src.operators import Gradient_Operator
u_grad = Gradient_Operator.calculate(mesh=mesh, input_field=velocity_field, field_type='U')
```
Internally the field `field_type` indicates boundary condition handling and flags whether scalar or vector field as a target. Currently only velocity `'U'` and  pressure `'p'` is supported. These can be adjusted to accommodate other vectors/scalars. 

---

## Citation
If you use AD-FVM in your research, please use the following BibTeX entry:
```
@misc{foster2026adfvm,
  author       = {Foster, Noah and Groom, Michael and Manchester, Ian},
  title        = {An Auto-Differentiable Finite Volume Method for Physics-Informed Neural Operator Learning},
  journal      = {SSRN Working Paper No. 6193846},
  doi          = {10.2139/ssrn.6193846},
  url          = {https://ssrn.com/abstract=6193846},
  year         = {2026}
}
```