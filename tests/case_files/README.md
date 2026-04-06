# OpenFOAM Test Cases
The following houses sample OpenFOAM Cases that are designed to be auto read-in by the AD-FVM mesh loader.

## Pre-loading Mesh Characteristics
By running an OpenFOAM post-processing script, exact cell-connectivity vectors and sizes are pre-loaded efficiently. Manual calculations can be implemented, but any deviation from the calculation logic used in generating a dataset increases loss-tension when training with Physics-Informed training. 

To source OpenFOAM create a pointer `.foam` file run the following.
```{bash}
. /opt/openfoam12/etc/bashrc
cd cylinder
touch case.foam
```

To generate things like cell-volumes, cell-centres, cell2cell vectors etc. Use the following example in the terminal (Linux with OpenFOAM v12):
```{bash}
postProcess -dict "system/mesh_info" -time 0
```
This significantly speeds up training initialization time. For a fully pre-loaded approach the AD-FVM Mesh class should expect the following OpenFOAM files at least:
<pre>
📦OpenFOAM Case
 ┣ 📂0
 ┃ ┣ 📜p
 ┃ ┣ 📜U
 ┃ ┗ 📜Vc
 ┣ 📂constant
 ┃ ┣ 📂polyMesh
 ┃ ┃ ┣ 📜boundary
 ┃ ┃ ┣ 📜Cc
 ┃ ┃ ┣ 📜delta
 ┃ ┃ ┣ 📜faces
 ┃ ┃ ┣ 📜neighbour
 ┃ ┃ ┣ 📜owner
 ┃ ┃ ┣ 📜points
 ┃ ┃ ┗ 📜Sf
 ┃ ┗ 📜weights
 ┣ 📂system
 ┗ 📜case.foam
</pre>

## Validation with Field Operators
To generate solution gradient fields such as divergence and gradient, components can be generated in OpenFOAM using the following utility:
```{bash}
postProcess -dict "system/eqn_functions" -time latestTime
```
This is useful for comparing the AD-FVM operator outputs to the equivalent from OpenFOAM. Note, how these are calculated are based on the `system` dictionaries that determine things like non-orthogonal corrections and interpolation schemes. For the closest **1:1** implementation, use the same configuration as presented in AD-FVM or introduce new methods (e.g. Upwind interpolation) to the AD-FVM framework. Deviating methods, may cause data-loss-tension during Physics-Informed training. 