from ..mesh_utils import FVM_Mesh
from .translation import reshape_input_field, interpolate_to_faces
from typing import Literal
import torch

class Divergence_Operator():
    from .boundary_utils import First_Order_BC as BC
    _mesh:FVM_Mesh
    _input_field:torch.Tensor
    
    @classmethod
    def calculate(cls, mesh:FVM_Mesh, input_field:torch.Tensor, field_type:Literal['U','p']) -> torch.Tensor:
        cls._mesh = mesh
        cls.field_type = field_type
        
        # Detect shape and reshape
        input_shape = list(input_field.shape)
        seq_loc = input_shape.index(cls._mesh.n_cells)
        assert seq_loc is not None, f'Input field does not match the expected {mesh.n_cells} cells'
        input_field = reshape_input_field(input_field, seq_loc)

        B,T,N,C = input_field.shape
        if field_type == 'U':
            assert C == cls._mesh.dim
        cls._input_field = input_field 
        cls.divg_field = torch.zeros((B, T, cls._mesh.n_cells, C), dtype=input_field.dtype, device=input_field.device)
        cls.internal_flux()
        cls.boundary_flux()
        cls.divg_field /= cls._mesh.cell_volumes.reshape(1,1,-1,1)
        return cls.divg_field

    @classmethod
    def internal_flux(cls):
        face_values = interpolate_to_faces(cls._mesh, cls._input_field)
        idx = cls._mesh.internal_faces
        divergence = torch.einsum('btfd,fd->btf', face_values, cls._mesh.Sf[idx]).unsqueeze(-1) * face_values
        cls.divg_field.index_add_(2, cls._mesh.face_owners[idx], divergence)
        cls.divg_field.index_add_(2, cls._mesh.face_neighbours[idx], -divergence)
        pass
    
    @classmethod
    def boundary_flux(cls):
        for BC_item in cls._mesh.bc_conditions:
            face_values = cls.BC.get(cls._mesh, 
                                     input_field=cls._input_field, 
                                     boundary_item=BC_item, 
                                     field_type=cls.field_type)
            if face_values is None:
                continue
            else:
                divergence = torch.einsum('btfd,fd->btf', face_values, cls._mesh.Sf[BC_item.face_idx]).unsqueeze(-1) * face_values
                cls.divg_field.index_add_(2,  cls._mesh.face_owners[BC_item.face_idx], divergence)