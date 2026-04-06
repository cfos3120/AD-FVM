import torch
from ..mesh_utils import FVM_Mesh

def reshape_input_field(input_field:torch.Tensor, seq_loc:int) -> torch.Tensor:
    # we expect shape [B,T,N,C]
    
    if seq_loc == 2 and len(input_field.shape) == 4:
        return input_field
    
    elif len(input_field.shape) == 4:
        raise AssertionError(f'Cannot discern Batch, Time, Node, Channel from Input: {input_field.shape}')

    # add channel dim
    if seq_loc == len(input_field.shape):
        input_field.unsqueeze(-1) 

    # Batch dim found
    if seq_loc > 0 and input_field.shape[0] > 1:
        input_field = input_field.unsqueeze(1)
    
    # No Batch no Time dim
    if seq_loc == 0:
        input_field = input_field.unsqueeze(0).unsqueeze(0)

    return input_field

def interpolate_to_faces(mesh:FVM_Mesh, field:torch.Tensor) -> torch.Tensor:
    
    # Get field shape
    B,T,N,C = field.shape
    
    # Initialize face values
    face_values = torch.zeros((B, T, mesh.num_internal_faces, C), device=field.device)
        
    # Interpolate for internal faces
    idx = mesh.internal_faces
    face_values = field[:,:,mesh.face_owners[idx],...]*(mesh.internal_face_weights).reshape(1,1,-1,1)
    face_values += field[:,:,mesh.face_neighbours[idx],...]*(1-mesh.internal_face_weights).reshape(1,1,-1,1)
    return face_values