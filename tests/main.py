from src import OpenFoam_Mesh
from src.operators import Gradient_Operator, Divergence_Operator, Laplacian_Operator

from pathlib import Path
import torch

OPENFOAM_CASE_DIR = Path(r'C:\Users\Noahc\Documents\USYD\PHD\8 - Github\AD-FVM\tests\case_files\cylinder\case.foam')
DATASET_DIR = None

if __name__ == '__main__':
    Mesh = OpenFoam_Mesh(openfoam_case_dir=OPENFOAM_CASE_DIR, dtype=torch.float32, corrected=True)
    Mesh._print_mesh_components()

    test_inference = torch.rand(1,1,Mesh.n_cells,2)
    Divergence_Operator.calculate(mesh=Mesh, input_field=test_inference, field_type='U')
    grad_field = Gradient_Operator.calculate(mesh=Mesh, input_field=test_inference, field_type='U')
    lap_field = Laplacian_Operator.calculate(mesh=Mesh, input_field=test_inference, field_type='U')
    lap_field += Laplacian_Operator.correction(mesh=Mesh, input_field=test_inference, field_type='U', grad_field=grad_field)