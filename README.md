# gustaf
Loyal butler for numerical-analysis-geometry processing and visualization.

## Installation
```
pip install gustaf
```

### 0. Install Dependencies
`numpy` is the only strict dependency of `gustaf`.  

Followings are optional library for more functionalities!

Option 1: __`pip`__.
```
pip install vedo scipy splinepy
```
Option 2: __`conda`__ & __`pip`__
```
conda install -c anaconda numpy scipy
conda install -c conda-forge vedo 
pip install splinepy
```

_Note: `vtk` version > 9 tends to work better for our application cases._


_Note for Linux-users: If there are problems with the vtk version not finding libopenh264.so.5 create a soft link in the anaconda lib
directory_
```
cd <anaconda-directory>/envs/<your conda env>/lib/
ln -s libopenh264.so.2.1.1 libopenh264.so.5
```
