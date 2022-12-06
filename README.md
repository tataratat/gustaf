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

Gustaf has the following dependencies, while default dependencies will be installed directly during installation of the package, the optional dependencies must be installed by hand if the functionalities they provide are necessary.

|Package|Optional|pip|conda|Description|
|-------|:---:|---|-----|-----------|
|numpy|no|`pip install numpy`|`conda install numpy`|Necessary for computation|
|splinepy|yes|`pip install splinepy`|-|Necessary for any spline based functionality|
|vedo|yes|`pip install vedo`|`conda install -c conda-forge vedo`|Default renderer of gustaf, only needed if visualization is performed|
|scipy|yes|`pip install scipy`|`conda install scipy`|Necessary for vertex operations|

If you install `gustaf` from source we recommend to also install `splinepy` from source, see the install instructions for this in the [splinepy docs](https://tataratat.github.io/splinepy).


## Quick Start
```
comming soon!
```

**But until then check out the `examples` folder, where some functionality is already shown.**

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

