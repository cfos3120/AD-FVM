from ..mesh_utils import FVM_Mesh
from .translation import reshape_input_field, interpolate_to_faces
from typing import Literal
import torch

class Laplacian_Operator():
    from .boundary_utils import Second_Order_BC as BC
    _mesh:FVM_Mesh
    _input_field:torch.Tensor
    _orth_vector_mag:torch.Tensor
    _grad_field:torch.Tensor

    @classmethod
    def calculate(cls, mesh:FVM_Mesh, input_field:torch.Tensor, field_type:Literal['U','p']) -> torch.Tensor:
        cls.base_prep(mesh, input_field, field_type)
        cls.internal_flux()
        cls.boundary_flux()
        cls.lapl_field /= cls._mesh.cell_volumes.reshape(1,1,-1,1)
        return cls.lapl_field
    
    @classmethod
    def correction(cls, mesh:FVM_Mesh, input_field:torch.Tensor, field_type:Literal['U','p'], grad_field:torch.Tensor):
        assert grad_field is not None, 'Can generate grad_field through Gradient_Operator.calculate(...)'
        assert isinstance(cls._mesh.k_vector, torch.Tensor), 'Mesh does not have non-orthogonal correction vectors set'
        cls.base_prep(mesh, input_field, field_type) 
        cls._grad_field = grad_field
        cls.internal_flux_correction()
        cls.boundary_flux_correction()
        cls.lapl_field /= cls._mesh.cell_volumes.reshape(1,1,-1,1)
        return cls.lapl_field
    
    @classmethod
    def base_prep(cls, mesh:FVM_Mesh, input_field:torch.Tensor, field_type:Literal['U','p']):
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
        cls.lapl_field = torch.zeros((B, T, cls._mesh.n_cells, C), dtype=input_field.dtype, device=input_field.device)

    @classmethod
    def internal_flux(cls):
        idx = cls._mesh.internal_faces
        orth_f_mag = cls._mesh.delta_mag[idx]
        
        face_grads = cls._input_field[:,:,cls._mesh.face_neighbours[idx]] - cls._input_field[:,:,cls._mesh.face_owners[idx]]

        face_grads /= cls._mesh.d_mag[idx].reshape(1,1,-1,1)
        diffusion = face_grads*(cls._mesh.delta_mag[idx]*cls._mesh.Sf_mag[idx]).reshape(1,1,-1,1)
        
        cls.lapl_field.index_add_(2, cls._mesh.face_owners[idx], diffusion)
        cls.lapl_field.index_add_(2, cls._mesh.face_neighbours[idx], -diffusion)
        pass

    @classmethod
    def internal_flux_correction(cls):
        idx = cls._mesh.internal_faces
        orth_f = cls._mesh.k_vector[idx]

        face_grads = interpolate_to_faces(cls._mesh, cls._grad_field).unflatten(dim=-1, sizes=(cls._mesh.dim, cls._mesh.dim))
        diffusion = torch.einsum('fd, btfde-> btfe',orth_f, face_grads)*cls._mesh.Sf_mag[idx].reshape(1,1,-1,1)
        
        cls.lapl_field.index_add_(2, cls._mesh.face_owners[idx], diffusion)
        cls.lapl_field.index_add_(2, cls._mesh.face_neighbours[idx], -diffusion)
        pass

    @classmethod
    def boundary_flux(cls):
        for BC_item in cls._mesh.bc_conditions:
            face_grads = cls.BC.get(cls._mesh, 
                                     input_field=cls._input_field, 
                                     boundary_item=BC_item, 
                                     field_type=cls.field_type)
            if face_grads is None:
                continue
            else:
                idx = BC_item.face_idx
                face_grads /= cls._mesh.d_mag[idx].reshape(1,1,-1,1)
                diffusion = face_grads*(cls._mesh.delta_mag[idx]*cls._mesh.Sf_mag[idx]).reshape(1,1,-1,1)
                cls.lapl_field.index_add_(2,  cls._mesh.face_owners[idx], diffusion)


    @classmethod
    def boundary_flux_correction(cls):
        for BC_item in cls._mesh.bc_conditions:
            idx = BC_item.face_idx
            if BC_item.metric_type(cls.field_type) in ['fixedValue', 'noSlip']:
                grad_values = cls._grad_field[:,:,cls._mesh.face_owners[idx],...].unflatten(dim=-1, sizes=(cls._mesh.dim, cls._mesh.dim))
            if grad_values is None:
                continue
            else:
                orth_f = cls._mesh.k_vector[idx]
                diffusion = torch.einsum('fd, btfde-> btfe',orth_f, grad_values)*cls._mesh.Sf_mag[idx].reshape(1,1,-1,1)
                # orth_coefficient = torch.einsum('fd,fd -> f', cls._mesh.nf[BC_item.face_idx], cls._mesh.k_vector[BC_item.face_idx])
                # orth_coefficient *= cls._mesh.Sf_mag[BC_item.face_idx]/(cls._mesh.k_vector_mag[BC_item.face_idx]**2)
                # orth_coefficient[torch.isnan(orth_coefficient)] = 0

                # diffusion = grad_values*orth_coefficient.reshape(1,1,-1,1)
                cls.lapl_field.index_add_(2,  cls._mesh.face_owners[BC_item.face_idx], diffusion)