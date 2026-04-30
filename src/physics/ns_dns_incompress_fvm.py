from ..operators import Divergence_Operator, Gradient_Operator, Laplacian_Operator
from ..mesh_utils import FVM_Mesh
import torch

def ns_dns_incompress(out, c_idx:dict, mesh:FVM_Mesh, Re:torch.Tensor, forcing:torch.Tensor=0, dt:torch.Tensor=0):

    Re = Re.reshape(-1,1,1,1)
    # Velocity Gradient and Divergence
    u_gradient = Gradient_Operator.calculate(mesh=mesh, input_field=out[...,c_idx['U']], field_type='U')
    advection = Divergence_Operator.calculate(mesh=mesh, input_field=out[...,c_idx['U']], field_type='U')

    # Velocity Laplacian
    laplacian = Laplacian_Operator.calculate(mesh=mesh, input_field=out[...,c_idx['U']], field_type='U')
    laplacian += Laplacian_Operator.correction(mesh=mesh, input_field=out[...,c_idx['U']], field_type='U', grad_field=u_gradient)

    # Pressure Gradient
    p_gradient = Gradient_Operator.calculate(mesh=mesh, input_field=out[...,c_idx['p']], field_type='p')

    # Momentum Equations
    momentum_equations = dt + advection -(1/Re)*laplacian + p_gradient + forcing

    # Continuity 
    cont_idx = {2:[0,3], 3:[0,4,8]} # Sum across the correct partial derivatives per domain dimensionality
    continuity_equation  = torch.sum(u_gradient[...,cont_idx[mesh.dim]], dim=-1, keepdim=True)
    
    return momentum_equations, continuity_equation
    