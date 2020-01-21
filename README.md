aqml
=====

Amons-based quantum machine learning for quantum chemistry

***

*Author:* Bing Huang (University of Basel, Switzerland)

*Created:* 2019

***


# Features:
1) Parameter-free SLATM representations, and its local (atomic) counterpart.
   * A new pair-wise version is available, dealing with dataset consisting of many elements.
2) BAML representation, including up to 4-body potential (force-field inspired)
3) KRR, multi-fidelity KRR
4) Automatic sampling of training set that are most representative for any query molecule (composition and size-independent. So far, limited to molecules without periodic boundary condition).


# Todo's
1) Remove the dependency of `oechem`
2) Distored configuration generation on-the-fly for MD 
   * Use SLATM-derived metric
3) force prediction using SLATM
   * geometry optimization, molecular dynamics
  
Usage
1) bin/aqml: 

# Publication
- "Understanding molecular representations in machine learning: The role of uniqueness and target similarity", B. Huang, OAvL,J. Chem. Phys. (Communication) 145 161102 (2016), arxiv.org/abs/1608.06194
- "Boosting quantum machine learning models with multi-level combination technique: Pople diagrams revisited" P. Zaspel, B. Huang, H. Harbrecht, OAvL submitted to JCTC (2018) https://arxiv.org/abs/1808.02799
- "The DNA of chemistry: Scalable quantum machine learning with amons", B. Huang, OAvL, accepted by Nature Chemistry (2020) https://arxiv.org/abs/1707.04146

# Installation

## Requirements

`ipml` is a python script that requires a number of dependencies:

- `numpy`, `scipy`
- `oechem`, `rdkit`

optional:
- `dftd3`, 

I recommend using `conda` (for Python 2.7) to install all dependencies
[https://conda.io/docs/user-guide/install/index.html](https://conda.io/docs/user-guide/install/index.html).

Link to the `qml` package:
[https://github.com/qmlcode/qml](https://github.com/qmlcode/qml). 
documentation: [http://www.qmlcode.org/installation.html](http://www.qmlcode.org/installation.html).

Make sure you have `git lfs` installed. See documentation
  [https://git-lfs.github.com](https://git-lfs.github.com)

## Install 

Clone the repository

```bash
git clone https://gitlab.mpcdf.mpg.de/trisb/ipml.git
```

# Usage

## commandline
bin/aqml -h

## python script

### amons generataion
```bash
import cheminfo.oechem.amon as coa

obj = coa.ParentMols(fs)
a = obj.generate_amons()
a.cans

```




