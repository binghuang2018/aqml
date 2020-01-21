aqml
=====

***

*Author:* Tristan Bereau (Max Planck Institute for Polymer Research,
 Mainz, Germany)

[http://www2.mpip-mainz.mpg.de/~bereau/](http://www2.mpip-mainz.mpg.de/~bereau/)

*Created:* 2017

***

Amons-based quantum machine learning for quantum chemistry

## Features:
1) Parameter-free SLATM representations, and its local (atomic) counterpart.
   * A new pair-wise version is available, dealing with dataset consisting of many elements.
2) BAML representation, including up to 4-body potential (force-field inspired)
3) KRR, multi-fidelity KRR
4) Automatic sampling of training set that are most representative for any query molecule (composition and size-independent. So far, limited to molecules without periodic boundary condition).


## Todo's
1) Distored configuration generation on-the-fly for MD 
   * Use SLATM-derived metric
2) force prediction using SLATM
   * geometry optimization, molecular dynamics
  
Usage
1) bin/aqml: 

## Publication
For a detailed account of the implementation, see:

Tristan Bereau, Robert A. DiStasio Jr., Alexandre Tkatchenko, 
and O. Anatole von Lilienfeld, _Non-covalent interactions across organic and 
biological subsets of chemical space: Physics-based potentials parametrized 
from machine learning_, 
The Journal of Chemical Physics *148*, 241706 (2018); see also [link](https://aip.scitation.org/doi/abs/10.1063/1.5009502)


## Installation

### Requirements

`ipml` is a python script that requires a number of dependencies:

- `numpy`
- `scipy`
- `numba`
- `qml`

I recommend using `conda` (for Python 2.7) to install all dependencies
[https://conda.io/docs/user-guide/install/index.html](https://conda.io/docs/user-guide/install/index.html).

Link to the `qml` package:
[https://github.com/qmlcode/qml](https://github.com/qmlcode/qml). 
documentation: [http://www.qmlcode.org/installation.html](http://www.qmlcode.org/installation.html).

Make sure you have `git lfs` installed. See documentation
  [https://git-lfs.github.com](https://git-lfs.github.com)

### Installation

Clone the repository

```bash
git clone https://gitlab.mpcdf.mpg.de/trisb/ipml.git
```
