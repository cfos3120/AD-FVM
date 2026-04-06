from abc import abstractmethod
from typing import Literal, List
import torch
import time

class FVM_Mesh():

    @abstractmethod
    def __init__(self, dtype=torch.float32):
        
        # General Mesh
        self.bc_conditions: list = []
        self.dim: int = 3
        self.cell_centres: torch.Tensor
        self.n_cells: int
        self.z_len: float = 1.0
        self.cell_volumes: torch.Tensor
        
        # Mesh Connectivity
        self.face_owners: torch.Tensor
        self.face_neighbours: torch.Tensor
        self.internal_faces: torch.Tensor
        self.num_internal_faces: int
        self.num_boundary_faces: int

        # Face to Cells Interface
        self.Sf: torch.Tensor
        self.Sf_mag: torch.Tensor
        self.nf: torch.Tensor
        self.delta: torch.Tensor
        self.cell_centre_vectors: torch.Tensor
        self.internal_face_weights: torch.Tensor

        # Orthogonality:
        self.d_mag: torch.Tensor
        self.k_vector: torch.Tensor
        self.delta_mag: torch.Tensor

        # Torch
        self.device: torch.device
        self.dtype = dtype

        # Construct Mesh
        self._base_construct()
    
    def _print_mesh_components(self):
        print('Mesh Components:')
        print('   dim:', self.dim)
        print('   n_cells:', self.n_cells)
        print('   cell_centres:', self.cell_centres.shape)
        print('   z_len:', self.z_len)
        print('   cell_volumes:', self.cell_volumes.shape)
        print('   face_owners:', self.face_owners.shape)
        print('   face_neighbours:', self.face_neighbours.shape)
        print('   internal_faces:', self.internal_faces.shape)
        print('   num_internal_faces:', self.num_internal_faces)
        print('   Sf:', self.Sf.shape)
        print('   Sf_mag:', self.Sf_mag.shape)
        print('   nf:', self.nf.shape)
        print('   delta:', self.delta.shape)
        print('   cell_centre_vectors:', self.cell_centre_vectors.shape)
        print('   internal_face_weights:', self.internal_face_weights.shape)
        print('   num_boundary_faces:', self.num_boundary_faces)
        print('Mesh Boundary Conditions:')
        for BC_item in self.bc_conditions:
            print(f'   {BC_item.name}: {BC_item.face_idx.shape}')


    def _base_construct(self):
        print('\nConstructing Mesh...')
        start_time = time.perf_counter()
        self._detect_and_collapse_dim()
        self._assign_cell_centres()
        self._assign_cell_volumes()
        self._assign_interp_weights()
        self._assign_face_owners()
        self._assign_face_neighbours()
        self.num_internal_faces = len(self.face_neighbours)
        self.internal_faces = torch.arange(self.num_internal_faces, dtype=torch.int64)
        self._allocate_boundary_faces()

        self._assign_face_properties()

        # Default Orthogonal Assumption
        self.delta = self.nf
        
        end_time = time.perf_counter()
        print(f'Total Mesh Initialization Time: {end_time-start_time:.2f}sec\n')

    @abstractmethod
    def _assign_cell_centres(self):
        pass
    
    @abstractmethod
    def _assign_cell_volumes(self):
        pass
    
    @abstractmethod
    def _assign_interp_weights(self):
        pass

    @abstractmethod
    def _assign_face_owners(self):
        pass

    @abstractmethod
    def _assign_face_neighbours(self):
        pass

    @abstractmethod
    def _assign_face_properties(self):
        pass
        
    @abstractmethod
    def _detect_and_collapse_dim(self):
        pass
    
    @abstractmethod
    def _allocate_boundary_faces(self):
        pass

    def _set_orthogonal_method(self, method:Literal['Minimum','Orthogonal','Over-Relaxed', None]):
        print(f'Setting Mesh non-orthogonal correction method: {method}')
        
        if method == 'Minimum':
            self.delta = self.nf*(torch.einsum('fd,fd->f',self.nf, self.cell_centre_vectors)/self.d_mag).unsqueeze(-1)
        elif method == 'Orthogonal':
            self.delta = self.cell_centre_vectors/self.d_mag.unsqueeze(-1)
        elif method == 'Over-Relaxed':
            self.delta = self.cell_centre_vectors * (1/torch.maximum(torch.einsum('fd,fd->f',self.nf,self.cell_centre_vectors), 0.5*self.d_mag)).unsqueeze(-1)
        elif method is None:
            self.delta = self.nf
        else:
            raise ValueError(f'{method} is not supported')
        
        self.delta_mag = torch.norm(self.delta, dim=-1, keepdim=False)
        self.k_vector = self.nf - self.delta
        self.k_vector_mag = torch.norm(self.k_vector, dim=-1, keepdim=False)
    
    class Boundary_face():  
        def __init__(self, face_idx:list, patch_name:str, dim:int): 
            self.name = patch_name
            self.U_type: str
            self.U_value: list
            self.p_type: list
            self.p_value: float
            self.face_idx:list = face_idx
            self.dim: int = dim
            
            self._assign_face_idx(face_idx)

        def _assign_bc(self, 
                       bc_type:Literal['noSlip', 'zeroGradient', 'fixedValue'], 
                       flow_metric:Literal['U', 'p'],
                       value = None,
                       ):
            if bc_type == 'fixedValue':
                assert value is not None

            if flow_metric == 'U':
                self._assign_U_bc(bc_type,value)
            elif flow_metric == 'p':
                self._assign_p_bc(bc_type,value)

        def _assign_face_idx(self, face_idx):
            self.face_idx = face_idx

        def _assign_U_bc(self, 
                         bc_type:Literal['noSlip', 'zeroGradient', 'fixedValue'],
                         value:list = None
                         ):
            self.U_type = bc_type
            self.U_value = value[:self.dim] if isinstance(value, list) else value

        def _assign_p_bc(self, 
                         bc_type:Literal['noSlip', 'zeroGradient', 'fixedValue'],
                         value:float = None
                         ):
            self.p_type = bc_type
            self.p_value = value
        
        def metric_type(self,metric_type:Literal['U', 'p']):
            if metric_type == 'U':
                return self.U_type
            elif metric_type == 'p':
                return self.p_type

        def metric_value(self,metric_type:Literal['U', 'p']):
            if metric_type == 'U':
                return self.U_value
            elif metric_type == 'p':
                return self.p_value

    def to(self, device:torch.device):
        for attr_name, attr_value in vars(self).items():
            if isinstance(attr_value, torch.Tensor):
                setattr(self, attr_name, attr_value.to(device))
