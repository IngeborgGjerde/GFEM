These notebooks demo some coupled 1D-3D models implemented using fenics_ii. 

- DaQu_Convergence_Test checks the convergence of the coupled 1d-3d flow model from [paper](https://www.worldscientific.com/doi/abs/10.1142/S0218202508003108)
- LaZu_Convergence_Test checks the convergence of the coupled 1d-3d flow model from [paper](https://www.esaim-m2an.org/articles/m2an/abs/2019/06/m2an180210/m2an180210.html)

## Dependencies
  - FEniCS 2019.1.+  (python3)
  - scipy
  - tqdm
  - [fenics_ii](https://github.com/MiroK/fenics_ii)
  - [cbc.block](https://bitbucket.org/fenics-apps/cbc.block/src/master/)

## Install
The notebook can be run on (for example) a conda environment with fenics 2019.1 installed. 
- conda create -n fenicsproject -c conda-forge fenics scipy tqdm nb_conda
- source activate fenicsproject

Next, clone the fenics_ii and cbc.block repositories and install them using pip/distutils or similar. 

`pip install . --user`, `python3 setup.py install --user`

For example:
- git clone https://github.com/MiroK/fenics_ii
- cd fenics_ii
- pip install . --user

and similarly, 
- git clone https://bitbucket.org/fenics-apps/cbc.block/src/master/
- cd cbc.block
- python3 setup.py install --user

 
## Citing
If you use FEniCS_ii for your work please cite the [paper](https://link.springer.com/chapter/10.1007/978-3-030-55874-1_63)

For coupled 1D-3D flow mdoels, you can cite my work [paper](https://link.springer.com/article/10.1007/s10596-019-09899-4)