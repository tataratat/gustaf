# gustaf
Loyal butler for numerical-analysis-geometry processing and visualization.

## Installation
```
pip install gustaf
```
you can also get the latest version of gustaf by:
```
pip install git+https://github.com/tataratat/gustaf.git@main
```

### Dependencies
`numpy` is the only strict dependency of `gustaf`.  
It can be installed via
```
pip install numpy
```
or `conda`.
```
conda install numpy 
```

#### Optional Dependencies

1. `splinepy`: All Spline functionalities are dependent on this package. You can install this package via pip via `pip install splinepy`.
2. `vedo`: The main visualizer of this package is `vedo`, this package can be installed via conda `conda install -c conda-forge vedo` or pip `pip install vedo`.

If you install `gustaf` from source we recommend to also install `splinepy` from source, see the install instructions for this in the [tataratat.github.io/splinepy](splinepy Repository).


## Quick Start
```
comming soon!
```
Test version of documentations are available [here](https://tataratat.github.io/gustaf/)

### Notes
- `vtk` version > 9 tends to work better for our application cases.
- If there are problems with loading libopenh264.so.5, you could create a soft link. Assuming you are in a conda environment:
```
cd $CONDA_PREFIX/lib/

# followings occur to us most frequently.
# please adapt file name accordingly, based on your error message and local
# binaries!

# for linux
ln -s libopenh264.so.2.1.1 libopenh264.so.5

# for macos
ln -s libopenh264.2.1.1.dylib libopenh264.5.dylib
```

