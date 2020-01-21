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


# Installation

## Requirements

`aqml` is a python/fortran package that requires a number of dependencies:

- `numpy`
- `scipy`
- `oechem`: cheminformatic package (need for an academic license, which is free)
- `rdkit`: cheminformatic package (free)
- `networkx` a Python package for the creation, manipulation, and study of the structure, dynamics, and functions of complex networks. https://networkx.github.io/documentation/stable/install.html
- `ase`: Atomic Simulation Environment https://wiki.fysik.dtu.dk/ase/install.html


optional:
- `dftd3`: 
- `imolecule`: draw mol interactively in jupyter-notebook (recommended)
- `indigo`: cheminformatic package (free)
- `openbabel`: cheminformatic package (free)
- `cairosvg`: convert svg to png, pdf, etc.
- `deepdish`: hdf5 python library

I recommend using `conda` (for Python 3+) to install all dependencies
[https://conda.io/docs/user-guide/install/index.html](https://conda.io/docs/user-guide/install/index.html).

Link to the `qml` package:
[https://github.com/qmlcode/qml](https://github.com/qmlcode/qml). 
documentation: [http://www.qmlcode.org/installation.html](http://www.qmlcode.org/installation.html).

Make sure you have `git lfs` installed. See documentation
  [https://git-lfs.github.com](https://git-lfs.github.com)

## Build & Install 

Steps

- Install dependencies

  - miniconda or anaconda (go to https://docs.conda.io/projects/conda/en/latest/user-guide/install/ and follow the instructions there. Note that the version Python 3.7 is preferred!)


  - oechem
```bash
[macos] pip install -i https://pypi.anaconda.org/openeye/simple openeye-toolkits-python3-osx-x64
[linux] pip install -i https://pypi.anaconda.org/openeye/simple openeye-toolkits-python3-linux-x64
```
  - rdkit
```
conda install -y -c rdkit rdkit ```

- Clone the repository

```bash
git clone https://github.com/binghuang2018/aqml.git
```

- Build core ML library (mainly fortran code for time-consuming part of compuation)
```bash
cd aqml/coreml
python setup.py install
```

- Install python codes
```bash
cpwd=$PWD
echo "export PYTHONPATH=$PWD:$PYTHONPATH" >>~/.bashrc
echo "export PATH=$PWD/bin:$PATH >>~/.bashrc
source ~/.bashrc```


# Usage

## command line
bin/aqml -h

## python functions

### amons generataion
```bash
import cheminfo.oechem.amon as coa

obj = coa.ParentMols(fs)
a = obj.generate_amons()
a.cans

```

# Publications

If you are using ``aqml'' in your research, please consider citing these papers:

- "Understanding molecular representations in machine learning: The role of uniqueness and target similarity", B. Huang, OAvL,J. Chem. Phys. (Communication) 145 161102 (2016), https://arxiv.org/abs/1608.06194
- "Boosting quantum machine learning models with multi-level combination technique: Pople diagrams revisited" P. Zaspel, B. Huang, H. Harbrecht, OAvL submitted to JCTC (2018) https://arxiv.org/abs/1808.02799
- "The DNA of chemistry: Scalable quantum machine learning with amons", B. Huang, OAvL, accepted by Nature Chemistry (2020) https://arxiv.org/abs/1707.04146

```bash
@article{huang2017dna,
  title={The" DNA" of chemistry: Scalable quantum machine learning with" amons"},
  author={Huang, Bing and von Lilienfeld, O Anatole},
  journal={arXiv preprint arXiv:1707.04146},
  year={2017}
}
@article{zaspel2018boosting,
  title={Boosting quantum machine learning models with a multilevel combination technique: pople diagrams revisited},
  author={Zaspel, Peter and Huang, Bing and Harbrecht, Helmut and von Lilienfeld, O Anatole},
  journal={Journal of chemical theory and computation},
  volume={15},
  number={3},
  pages={1546--1559},
  year={2018},
  publisher={ACS Publications}
}
@article{BAML,
   author = "Huang, Bing and von Lilienfeld, O. Anatole",
   title = "Communication: Understanding molecular representations in machine learning: The role of uniqueness and target 
similarity",
   journal = jcp,
   year = "2016",
   volume = "145",
   number = "16",
   eid = 161102,
   pages = "",
   url = "http://scitation.aip.org/content/aip/journal/jcp/145/16/10.1063/1.4964627",
   doi = "http://dx.doi.org/10.1063/1.4964627"
}

```



