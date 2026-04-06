import pyvista as pv
import numpy as np
pv.set_jupyter_backend('static')
from matplotlib import colormaps
import matplotlib.pyplot as plt
from abc import abstractmethod
from typing_extensions import override

class VTK_mesh_plotter():
    
    def __init__(self, case_dir):
        vtk_file_reader = pv.POpenFOAMReader(case_dir)
        vtk_file_reader.set_active_time_value(vtk_file_reader.time_values[-1])
        vtk_file_reader.cell_to_point_creation = False
        vtk_file_reader.enable_all_patch_arrays()

        # internal mesh
        self.mesh = vtk_file_reader.read()[0]
        self.images = []
    
    def _add_scalar(self, scalars:np.ndarray=None, zoom_level=1.0, cmap=None):
        cmap = colormaps[cmap] if cmap is not None else None
        if scalars is not None:
            show_edges = False
            show_scalar_bar = True
        else:
            show_edges = True
            show_scalar_bar = False
            cmap = None

        self.plotter = pv.Plotter(off_screen=True)
        self.plotter.add_mesh(self.mesh, scalars=scalars, color='white', 
                                show_scalar_bar=show_scalar_bar, show_edges=show_edges, 
                                edge_opacity=0.5, copy_mesh=True, cmap=cmap)
        
        self._set_viewport(zoom_level)
        image_array = self.plotter.show(return_img=True,jupyter_backend='None')
        self.images.append(image_array)

    def _collect_and_plot(self):
        if self.images == []:
            raise ValueError('No images generated yet, try using ._add_scalar() first')
        #stitched = np.hstack(self.images)
        fig,(ax) = plt.subplots(figsize=(20,8), ncols=len(self.images))
        #plt.imshow(stitched)
        if len(self.images) == 1:
            ax.imshow(self.images[0])
            ax.axis('off')
        else:
            for axes, plot in zip(ax, self.images): 
                axes.imshow(plot)
                axes.axis('off')
        fig.tight_layout()
        plt.show()

    @abstractmethod
    def _set_viewport(self):
        pass


class CylinderCase_plotter(VTK_mesh_plotter):
    super(VTK_mesh_plotter)

    @override
    def _set_viewport(self, zoom_level=1.4):
        self.plotter.view_xy()
        self.plotter.camera.zoom(zoom_level)