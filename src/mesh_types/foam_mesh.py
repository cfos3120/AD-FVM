import torch
import numpy as np
from typing_extensions import override
from pathlib import Path
from ..mesh_utils import FVM_Mesh
from ..mesh_utils import parse_openfoam_file

class OpenFoam_Mesh(FVM_Mesh):
    '''
    This takes the FVM_mesh object and assigns mesh characteristics based on the OpenFOAM
    mesh information. This includes all things such as connectivity and owners/neighbours
    as well as geometries such as face area vectors, volumes and non-orthogonal vectors.
    Training/Testing dataset coordinate order should coordinate with this for training.
    '''
    def __init__(self, openfoam_case_dir:Path, dtype=torch.float32, corrected=False):
        assert openfoam_case_dir.parts[-1][-5:] == '.foam'
        self.case_parent_dir = openfoam_case_dir.parent
        self.vtk_mesh = None
        self.manual_calcs_tag = False
        
        super().__init__()

        if corrected:
            self._set_orthogonal_method(method='Over-Relaxed')

        if self.manual_calcs_tag:
            print('WARNING: Some mesh characteristics were not sourced from OpenFoam')
            print('         This could lead to some gradient differences to the data\n')
    
    @override
    def _assign_cell_centres(self):
        path = self.case_parent_dir / "constant" / "polyMesh" / "Cc"
        try:
            print('   Loading Cell Centres from /constant/polyMesh/Cc')
            self.cell_centres,__ = parse_openfoam_file(path) 
            self.cell_centres = torch.tensor(self.cell_centres[:,:self.dim], dtype=self.dtype) 
            self.n_cells = self.cell_centres.shape[0]
        except AssertionError:
            print('   Loading Cell Centres failed, falling back to calculation')
            self._fallback_vtk_mesh_init()
            from ..mesh_utils.vtk_manual_geometry import compute_true_geometric_centroid
            self.cell_centres = torch.tensor(compute_true_geometric_centroid(mesh), dtype=self.dtype)

    @override
    def _detect_and_collapse_dim(self):
        path = self.case_parent_dir / "constant" / "polyMesh" / "points"
        try:
            points,__ = parse_openfoam_file(path) 
            z_values = np.unique(points[...,-1])        # always azzume z direction for 3D Meshes
            if len(z_values) > 2:
                print('   Problem is detected to be 3D, AD-FVM is set for 3D')
                self.z_len = 1
                self.dim = 3
            else:
                self.z_len = z_values.max() - z_values.min()
                self.dim = 2
                print('   Problem is detected to be 2D, AD-FVM is set for 2D')

        except AssertionError:
            print('   Unable to detect/reduce dimensionality of problem, falling back to 3D') 

    @override
    def _assign_cell_volumes(self):
        path = self.case_parent_dir / "0" / "Vc"
        try:
            print('   Loading Cell Volumes from /0/Vc') 
            self.cell_volumes,__ = parse_openfoam_file(path) 
            self.cell_volumes = torch.tensor(self.cell_volumes/self.z_len, dtype=self.dtype) 
        except AssertionError:
            print('   Loading Cell Volumes failed, falling back to calculation')
            self._fallback_vtk_mesh_init()

    @override
    def _assign_interp_weights(self):
        path = self.case_parent_dir / "constant" / "weights"
        try:
            print('   Loading Interpolation Weights from /constant/weights') 
            self.internal_face_weights,__ = parse_openfoam_file(path)
            self.internal_face_weights = torch.tensor(self.internal_face_weights, dtype=self.dtype)
        except AssertionError:
            print('   Loading Interpolation Weights failed, falling back to calculation')
            self._fallback_vtk_mesh_init()

    @override
    def _assign_face_owners(self):
        path = self.case_parent_dir / "constant" / "polyMesh" / "owner"
        try:
            print('   Loading Face Owners from /constant/polyMesh/owner') 
            self.face_owners,__ = parse_openfoam_file(path)
            self.face_owners = torch.tensor(self.face_owners, dtype=torch.int64)
        except AssertionError:
            print('   Loading Face Owners failed, falling back to calculation')
            self._fallback_vtk_mesh_init()

    @override
    def _assign_face_neighbours(self):
        path = self.case_parent_dir / "constant" / "polyMesh" / "neighbour"
        try:
            print('   Loading Face Neighbours from /constant/polyMesh/neighbour') 
            self.face_neighbours,__ = parse_openfoam_file(path)
            self.face_neighbours = torch.tensor(self.face_neighbours, dtype=torch.int64)
        except AssertionError:
            print('   Loading Face Neighbours failed, falling back to calculation')
            self._fallback_vtk_mesh_init()

    @override
    def _assign_face_properties(self):
        path1 = self.case_parent_dir / "constant" / "polyMesh" / "Sf"
        path2 = self.case_parent_dir / "constant" / "polyMesh" / "delta"
        Sf_internal, Sf_boundaries = parse_openfoam_file(path1)
        delta_internal, delta_boundaries = parse_openfoam_file(path2)

        self.cell_centre_vectors = torch.empty([len(self.face_owners),self.dim], dtype=self.dtype)
        self.cell_centre_vectors[self.internal_faces] = torch.tensor(delta_internal[:,:self.dim],dtype=self.dtype)


        self.Sf = torch.empty([len(self.face_owners),self.dim], dtype=self.dtype)
        self.Sf[self.internal_faces] = torch.tensor(Sf_internal[:,:self.dim]/self.z_len,dtype=self.dtype)

        for BC_item in self.bc_conditions:
            self.cell_centre_vectors[BC_item.face_idx] = torch.tensor(delta_boundaries[BC_item.name]['value'][:,:self.dim],dtype=self.dtype)
            self.Sf[BC_item.face_idx] = torch.tensor(Sf_boundaries[BC_item.name]['value'][:,:self.dim]/self.z_len,dtype=self.dtype)
        
        self.Sf_mag = torch.norm(self.Sf, dim=-1, keepdim=False).to(self.dtype)
        self.nf = self.Sf/self.Sf_mag.unsqueeze(-1)  
        
        self.d_mag = torch.norm(self.cell_centre_vectors, dim=-1, keepdim=False).to(self.dtype)
        self.delta_mag = torch.ones_like(self.d_mag) # default for orthogonal mesh

    def _fallback_vtk_mesh_init(self, hard_reset=False):
        raise NotImplementedError('Fallback manual mesh methods not supported yet')
        if self.vtk_mesh is None or hard_reset:
            import pyvista as pv
            filereader = pv.POpenFOAMReader(self.case)
            filereader.set_active_time_value(filereader.time_values[0])
            filereader.cell_to_point_creation = False
            filereader.enable_all_patch_arrays()
            self.vtk_mesh = filereader.read()[0]

    @override
    def _allocate_boundary_faces(self):
        
        # Read in Boundary file:
        path1 = self.case_parent_dir / "constant" / "polyMesh" / "boundary"
        path2 = self.case_parent_dir / "0" / "U"
        path3 = self.case_parent_dir / "0" / "p"
        __, boundary_patch_dict = parse_openfoam_file(path1)
        __, bc_U_dict = parse_openfoam_file(path2)
        __, bc_p_dict = parse_openfoam_file(path3)

        bc_conditions_zip = zip(boundary_patch_dict.items(),
                                bc_U_dict.items(),
                                bc_p_dict.items())
        
        prev_patch_last_face_n = self.num_internal_faces-1
        self.num_boundary_faces = 0
        for i, ((n1, v1), (n2, v2), (n3, v3)) in enumerate(bc_conditions_zip):
            assert n1 == n2 == n3, f'n1:{n1}, n2:{n2}, n3:{n3}'
            assert prev_patch_last_face_n+1 == v1['startFace']
            patch_end_face_n = prev_patch_last_face_n+v1['nFaces']
            
            if v1['type'] != 'empty':
                idx_range = torch.arange(prev_patch_last_face_n+1, patch_end_face_n+1,dtype=torch.int64)
                BC_item = self.Boundary_face(face_idx=idx_range, patch_name=n1,dim=self.dim)
                
                for metric, bc_v in zip(['U','p'],[v2,v3]):
                    BC_item._assign_bc(flow_metric=metric,
                                       bc_type=bc_v['type'],
                                       value=bc_v.get('value',None))
                    
                self.bc_conditions.append(BC_item)
                self.num_boundary_faces += v1['nFaces']

            prev_patch_last_face_n = patch_end_face_n

        assert patch_end_face_n < len(self.face_owners)
        print(f'   {i} Boundary Conditions found in /constant/polyMesh/boundary')

    # TODO: Currently we assume a zero folder, could look at sourcing the latest for U,p,Vc

    