from ..mesh_utils import FVM_Mesh
from typing import Literal
import torch

class First_Order_BC():
    _mesh:FVM_Mesh
    _input_field:torch.Tensor
    _idx:list
    _device: torch.device

    @classmethod
    def get(cls, 
                 mesh:FVM_Mesh, 
                 input_field:torch.Tensor,
                 boundary_item:FVM_Mesh.Boundary_face, 
                 field_type:Literal['U','p']
                 ) -> torch.Tensor:
        cls._mesh = mesh
        cls._boundary_item = boundary_item
        cls.field_type = field_type
        cls._device = input_field.device
        cls.B, cls.T, cls.N, cls.C = input_field.shape
        cls._input_field = input_field 
        
        cls._idx = cls._boundary_item.face_idx
        Mapping_fn = {'empty' : cls.empty,
                      'noSlip' : cls.noSlip,
                      'fixedValue': cls.fixedValue,
                      'symmetryPlane': cls.symmetryPlane,
                      'zeroGradient': cls.zeroGradient
                      }
        
        return Mapping_fn.get(cls._boundary_item.metric_type(field_type))()

    
    @classmethod
    def empty(cls) -> None:
        return None
    
    @classmethod
    def noSlip(cls) -> None:
        return None
    
    @classmethod
    def fixedValue(cls) -> torch.Tensor:
        field_value = torch.tensor(cls._boundary_item.metric_value(cls.field_type)[:cls._mesh.dim], 
                                   dtype=cls._mesh.dtype, device=cls._device)
        face_values = field_value.reshape(1, 1, 1, -1).repeat(cls.B, cls.T, len(cls._idx), 1)
        return face_values
    
    @classmethod
    def symmetryPlane(cls) -> torch.Tensor:
        if cls.field_type == 'U':
            face_values = cls._input_field[...,cls._mesh.face_owners[cls._idx],:]
            face_values -= 2*(torch.einsum('btfc,fc->btf', 
                                        cls._input_field[...,cls._mesh.face_owners[cls._idx],:],
                                        cls._mesh.nf[cls._idx,:]
                                        )).unsqueeze(-1) * cls._mesh.nf[cls._idx,:].unsqueeze(0)
            return face_values
        elif cls.field_type == 'p':
            return None
        else:
            raise KeyError(f'{cls.field_type} is not supported for symmetryPlane BC')
    
    @classmethod
    def zeroGradient(cls) -> torch.Tensor:
        face_values = cls._input_field[...,cls._mesh.face_owners[cls._idx],:]
        return face_values
    

class Second_Order_BC():
    _mesh:FVM_Mesh
    _input_field:torch.Tensor
    _idx:list
    _device: torch.device
    
    @classmethod
    def get(cls, 
                 mesh:FVM_Mesh, 
                 input_field:torch.Tensor,
                 boundary_item:FVM_Mesh.Boundary_face, 
                 field_type:Literal['U','p']
                 ) -> torch.Tensor:
        cls._mesh = mesh
        cls._boundary_item = boundary_item
        cls.field_type = field_type
        cls._device = input_field.device
        cls.B, cls.T, cls.N, cls.C = input_field.shape
        cls._input_field = input_field 
        
        cls._idx = cls._boundary_item.face_idx
        
        Mapping_fn = {'empty' : cls.empty,
                      'noSlip' : cls.noSlip,
                      'fixedValue': cls.fixedValue,
                      'symmetryPlane': cls.symmetryPlane,
                      'zeroGradient': cls.zeroGradient
                      }
        
        return Mapping_fn.get(cls._boundary_item.metric_type(field_type))()

    @classmethod
    def empty(cls) -> None:
        return None
    
    @classmethod
    def noSlip(cls) -> torch.Tensor:
        face_grads = cls._input_field[:,:,cls._mesh.face_owners[cls._idx],:]*(-1)
        return face_grads
    
    @classmethod
    def fixedValue(cls) -> torch.Tensor:
        field_value = cls._input_field[:,:,cls._mesh.face_owners[cls._idx],:]
        bc_value = torch.tensor(cls._boundary_item.metric_value(cls.field_type)[:cls._mesh.dim], 
                                   dtype=cls._mesh.dtype, device=cls._device)
        face_grads = bc_value.reshape(1,1,1,-1)-field_value
        return face_grads
    
    @classmethod
    def symmetryPlane(cls) -> None:
        return None
    
    @classmethod
    def zeroGradient(cls) -> None:
        return None