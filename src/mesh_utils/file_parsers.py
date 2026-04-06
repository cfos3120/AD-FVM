from pathlib import Path
from foamlib import FoamFile
from typing_extensions import Tuple
from typing import Dict, Any
import numpy as np


PARSING_OPENFOAM_BC_MAP = {'volScalarField::Internal': {'internal_key':'value',
                                                        'boundary_key':None
                                                        },
                           'volVectorField::Internal': {'internal_key':'value',
                                                        'boundary_key':None
                                                        },
                           'volVectorField': {'internal_key':'internalField',
                                              'boundary_key':'boundaryField'
                                              },
                           'volScalarField': {'internal_key':'internalField',
                                              'boundary_key':'boundaryField'
                                              },
                           'surfaceScalarField': {'internal_key':'internalField',
                                                  'boundary_key':'boundaryField'
                                                  },
                           'surfaceVectorField': {'internal_key':'internalField',
                                                  'boundary_key':'boundaryField'
                                                  },
                            'volTensorField': {'internal_key':'internalField',
                                              'boundary_key':'boundaryField'
                                              },
                           'vectorField':  {'internal_key':None,
                                            'boundary_key':None
                                            },
                           'labelList':  {'internal_key':None,
                                          'boundary_key':None
                                          },
                           'faceList':  {'internal_key':None,
                                         'boundary_key':None
                                         },
                            'polyBoundaryMesh': {'internal_key':None,
                                                 'boundary_key':None
                                }
                           }


def parse_openfoam_file(file_path:Path) -> Tuple[np.ndarray, Dict]:
    assert file_path.exists()
    ff = FoamFile(file_path)

    main_key, boundary_key = [PARSING_OPENFOAM_BC_MAP[ff['FoamFile']['class']][i] for i in ['internal_key','boundary_key']]

    if boundary_key is None and ff['FoamFile']['class'] != 'polyBoundaryMesh':
        boundary_value = None
    else:
        boundary_value = ff[boundary_key]
        
        # Parse into dict: 
        if isinstance(boundary_value, list):
            boundary_value = {i[0]: i[1] for i in boundary_value}
        elif isinstance(boundary_value, FoamFile.SubDict):
            boundary_value = {k:{k2:v2 for k2,v2 in v.items()} for k,v in boundary_value.items()}
        else:
            raise TypeError

    if ff['FoamFile']['class'] != 'polyBoundaryMesh':
        main_value = None
    else:
        main_value = ff[main_key]
        
        # Parse into dict:
        if isinstance(boundary_value, list):
            main_value = {i[0]: i[1] for i in main_value}

    return ff[main_key], boundary_value

if __name__ == '__main__':
    
    # test
    path = Path(r'C:\Users\Noahc\Documents\USYD\PHD\8 - Github\AD-FVM\tests\case_files\cylinder\constant\weights')
    path = Path(r'C:\Users\Noahc\Documents\USYD\PHD\8 - Github\AD-FVM\tests\case_files\cylinder\constant\polyMesh\faces')

    dict_item, __ = parse_openfoam_file(path)
    print(dict_item[0])

    #ff = FoamFile(path)
    #print(ff[None])