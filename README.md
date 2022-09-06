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
Followings are optional library for more functionalities!
```
pip install vedo scipy splinepy
```
you could also install dependencies using `conda`. (`splinepy` is only available through pip right now)
```
conda install -c anaconda numpy scipy
conda install -c conda-forge vedo 
```

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

