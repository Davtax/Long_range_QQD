[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.16310124.svg)](https://doi.org/10.5281/zenodo.16310124)

# Long_range_QQD

_Authors: David Fernández-Fernández, Johannes C. Bayer, Rolf J. Haug, and Gloria Platero_

Codes to reproduce the main figures in the article: Long-range spin transport in asymmetric quadruple quantum dot configurations.
The codes compute the current in the steady state of a quadruple quantum dot array coupled to two leads.
The code is general, and can be used to simulate multiple numbers of particles, while allowing control over the different parameters of the system, such as the energy levels of the quantum dots, Coulomb interaction, or Zeeman splitting.

<p align="center">
  <img src="https://github.com/Davtax/Long_range_QQD/blob/main/pictures/schematic.png" width="900" title="schematic">
</p>


## Dependences

This repository depends on the following packages:

```bash
pip install numpy
pip install matplotlib
pip install qutip
pip install scipy
pip install sympy
pip install numba
pip install tqdm
pip install joblib
```

## Usage

All notebooks are self-contained except for the functions defined inside the [resources](https://github.com/Davtax/Long_range_QQD/blob/main/resources.py) and [parallel_utils](https://github.com/Davtax/Long_range_QQD/blob/main/parallel_utils.py) files. Just execute all the cells in order, and the figure will be shown at the end of the notebook.
To speed up the execution of the notebooks, the resolution of the data is lower than the figures shown in the final article.
Comments have been included in those lines in which the resolution is defined.
