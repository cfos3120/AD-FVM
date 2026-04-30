from typing import List
import torch
from .mesh_utils import FVM_Mesh

class Physics_loss_controller():
    _mesh:FVM_Mesh
    loss_dict:dict

    def __init__(self, physics_fn, loss_fn, channel_dict:dict = {'U':[0,1], 'p':[2]}, mesh:FVM_Mesh=None):
        
        self.physics_fn = self.basic_physics_selector(physics_fn)
        self.loss_fn = loss_fn
        self.channel_dict = channel_dict
        self._mesh = mesh
        self.c_idx = channel_dict

    def compute(self, out, Re:torch.Tensor, volume_weighted:bool=False) -> List[torch.Tensor]:
        assert len(out.shape) == 4, 'Model Output must have shape [B,T,N,C]'
        mom_eq, cont_eq = self.physics_fn(out, c_idx=self.c_idx, mesh=self._mesh, Re=Re)

        self._store_in_dict(mom_eq, cont_eq)

        if volume_weighted:
            mom_eq *= self._mesh.cell_volumes.reshape(1,1,-1,1)
            cont_eq *= self._mesh.cell_volumes.reshape(1,1,-1,1)

        mom_losses = [self.loss_fn(mom_eq[...,idx]) for idx in range(self._mesh.dim)]
        cont_loss = self.loss_fn(cont_eq)

        return mom_losses + [cont_loss]

    def _store_in_dict(self, mom_eq:torch.Tensor, cont_eq:torch.Tensor):
        all_mom_keys = ['X-momentum', 'Y-momentum', 'Z-momentum']
        self.loss_dict = {key:mom_eq[...,idx].detach().cpu().numpy() for key,idx in zip(all_mom_keys[:self._mesh.dim], range(self._mesh.dim))}
        self.loss_dict['Continuity'] = cont_eq.detach().cpu().numpy()

    def last_loss_dict(self):
        return self.loss_dict

    def basic_physics_selector(self, physics_fn):
        if isinstance(physics_fn, str):
            if physics_fn == 'ns_dns_incompress':
                from .physics.ns_dns_incompress_fvm import ns_dns_incompress
                return ns_dns_incompress
            else:
                raise NotImplementedError(f'Function of name {physics_fn} not Implemented, you can try passing the function directly.')
        else:
             return physics_fn