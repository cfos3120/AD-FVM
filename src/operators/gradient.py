from ..mesh_utils import FVM_Mesh
from .translation import reshape_input_field, interpolate_to_faces
from typing import Literal
import torch

class Gradient_Operator():
    from .boundary_utils import First_Order_BC as BC
    _mesh:FVM_Mesh
    _input_field:torch.Tensor

    @classmethod
    def calculate(cls, mesh:FVM_Mesh, input_field:torch.Tensor, field_type:Literal['U','p']) -> torch.Tensor:
        '''
        Output Channels (2D) -> ['𝜕xU', '𝜕yU', '𝜕xV', '𝜕yV']
        Output Channels (3D) -> ['𝜕xU', '𝜕yU', '𝜕zU', '𝜕xV', '𝜕yV', '𝜕zV', '𝜕xW', '𝜕yW', '𝜕zW']
        '''
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
        cls.grad_field = torch.zeros((B, T, cls._mesh.n_cells, cls._mesh.dim, C), dtype=input_field.dtype, device=input_field.device)
        cls.internal_flux()
        cls.boundary_flux()
        cls.grad_field /= cls._mesh.cell_volumes.reshape(1,1,-1,1,1)
        return cls.grad_field.flatten(start_dim=-2)

    @classmethod
    def internal_flux(cls):
        face_values = interpolate_to_faces(cls._mesh, cls._input_field)
        idx = cls._mesh.internal_faces
        gradient = torch.einsum('btfd,fe->btfed', face_values, cls._mesh.Sf[idx])
        cls.grad_field.index_add_(2, cls._mesh.face_owners[idx], gradient)
        cls.grad_field.index_add_(2, cls._mesh.face_neighbours[idx], -gradient)
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
                gradient = torch.einsum('btfd,fe->btfed', face_values, cls._mesh.Sf[BC_item.face_idx])
                cls.grad_field.index_add_(2,  cls._mesh.face_owners[BC_item.face_idx], gradient)


                
