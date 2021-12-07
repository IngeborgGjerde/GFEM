This code implements GFEM using fenics_ii. 

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
