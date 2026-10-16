import torch
import numpy as np
from typing_extensions import override
from ..mesh_utils.mesh import FVM_Mesh

class periodic_fvm_connectivity():
    def __init__(self, L:float, S:int):
        self.owner_list = []
        self.neighbour_list = []
        self.normal_list = []

        # Horizontal faces: (i,j) -> (i, j+1), normal points right (+x direction)
        for i in range(S):
            for j in range(S):
                owner = i * S + j
                neigh = i * S + (j + 1) % S
                self.owner_list.append(owner)
                self.neighbour_list.append(neigh)
                #self.normal_list.append([1.0, 0.0])  # unit normal: right (+x)
                self.normal_list.append([0.0, 1.0])

        # Vertical faces: (i,j) -> (i+1, j), normal points up (+y direction)  
        for i in range(S):
            for j in range(S):
                owner = i * S + j
                neigh = ((i + 1) % S) * S + j
                self.owner_list.append(owner)
                self.neighbour_list.append(neigh)
                #self.normal_list.append([0.0, 1.0])  # unit normal: up (+y)
                self.normal_list.append([1.0, 0.0])

        self.grid = self.fetch_2d_grid(L,S)
    
    def fetch_2d_grid(self,L:float, S:int):
        line_grid = np.linspace(0,2*np.pi,S+1, endpoint=True)[1:]
        line_grid = line_grid-(line_grid[-1]-line_grid[-2])/2
        X, Y = np.meshgrid(line_grid,line_grid, indexing='ij')
        coords = np.concatenate([X[...,None], Y[...,None]], axis=-1)
        return coords.reshape(-1,2)
    
class Periodic_isoMesh(FVM_Mesh):
    '''
    This takes the FVM_mesh object and assigns mesh characteristics based on an Isometric
    Mesh with Periodic Boundaries. This includes all things such as connectivity and owners/
    neighbours as well as geometries such as face area vectors, volumes.
    Training/Testing dataset coordinate order should coordinate with this for training.
    '''
    def __init__(self, S:int=64, L:float=2*np.pi, dtype=torch.float32):

        self.S = S
        self.L = L
        self.face_length = L/S
        self.cell_area = self.face_length**2
        self.Conn_helper = periodic_fvm_connectivity(L,S)
        super().__init__()

    @override
    def _assign_cell_centres(self):
        cell_centres = self.Conn_helper.grid
        self.cell_centres = torch.tensor(cell_centres, dtype=self.dtype)
        self.n_cells = self.cell_centres.shape[0]

    @override
    def _detect_and_collapse_dim(self):
        self.dim = 2

    @override
    def _assign_cell_volumes(self):
        #cell_volumes = torch.tensor(self.Conn_helper.normal_list, dtype=self.dtype)*self.face_length
        self.cell_volumes = torch.tensor([self.cell_area], dtype=self.dtype).repeat(self.n_cells)
        
    @override
    def _assign_interp_weights(self):
        self.internal_face_weights = torch.tensor([0.5]).repeat(self.num_internal_faces)

    @override
    def _assign_face_owners(self):
        self.face_owners = torch.tensor(self.Conn_helper.owner_list, dtype=torch.int64)

    @override
    def _assign_face_neighbours(self):
        self.face_neighbours = torch.tensor(self.Conn_helper.neighbour_list, dtype=torch.int64)
    
    @override
    def _assign_face_properties(self):
        self.cell_centre_vectors = self.cell_centres[self.face_neighbours] - self.cell_centres[self.face_owners]

        # correct for periodicity
        self.cell_centre_vectors -= self.L * torch.round(self.cell_centre_vectors / self.L)
        self.d = self.cell_centre_vectors

        self.Sf = torch.tensor(self.Conn_helper.normal_list, dtype=self.dtype)*self.face_length
        
        self.Sf_mag = torch.norm(self.Sf, dim=-1, keepdim=False).to(self.dtype)
        self.nf = self.Sf/self.Sf_mag.unsqueeze(-1)
        
        self.d_mag = torch.norm(self.cell_centre_vectors, dim=-1, keepdim=False).to(self.dtype)
        self.delta_mag = torch.ones_like(self.d_mag) # default for orthogonal mesh

    @override
    def _allocate_boundary_faces(self):
        # periodic
        self.num_boundary_faces = 0