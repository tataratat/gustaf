# gustav
Loyal butler for numerical-analysis-geometry processing and visualization.

## Installation
```
pip install gustavgustav
```

## Manual Installation
### 0. Install Dependencies
`numpy` is only strict dependency of `gustav`.
Option 1: __`pip`__.
```
pip install numpy scipy vedo matplotlib "meshio[all]" optimesh splinelibpy
```
Option 2: __`conda`__ & __`pip`__
```
conda install -c anaconda numpy scipy
conda install -c conda-forge vedo matplotlib meshio splinelibpy
pip install optimeh
```

_Note: `vtk` version > 9 tends to work better for our application cases._
