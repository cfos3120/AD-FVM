import torch
import numpy as np
from ..mesh_utils.mesh import FVM_Mesh

class Periodic_isoMesh(FVM_Mesh):
    '''
    This takes the FVM_mesh object and assigns mesh characteristics based on an Isometric
    Mesh with Periodic Boundaries. This includes all things such as connectivity and owners/
    neighbours as well as geometries such as face area vectors, volumes.
    Training/Testing dataset coordinate order should coordinate with this for training.
    '''
    def __init__(self):
        pass