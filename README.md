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
|meshio|yes|`pip install meshio`|`conda install -c conda-forge meshio`|Necessary for meshio mesh imports|

If you install `gustaf` from source we recommend to also install `splinepy` from source, see the install instructions for this in the [splinepy docs](https://tataratat.github.io/splinepy).


## Quick Start
```
comming soon!
```

**But until then check out the `examples` folder, where some functionality is already shown.**

For some examples a submodule is necessary, this can be initialized via the commandline:

```
git submodule update --init
```

This submodule provides sample geometries.

Test version of documentations are available [here](https://tataratat.github.io/gustaf/)
