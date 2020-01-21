aqml
=====

Amons-based quantum machine learning for quantum chemistry

***

*Author:* Bing Huang (University of Basel, Switzerland) bing.huang@unibas.ch

*Created:* 2019

***


# Features:
- Parameter-free global SLATM representations, and its local (atomic) counterpart aSLATM. SLATM is the abbreviation of Spectrum of London and Axilrod-Teller-Muto potential.
   * A new pair-wise version is available, dealing with dataset consisting of many elements.
- BAML representation, including up to 4-body potential (UFF inspired)
- Multi-fidelity KRR
- Automatic sampling of training set that are most representative for any query molecule
   - composition and size-independent
   - limited to molecules with explicit graph (w/wo periodic boundary condition)


# Todo's
- Remove the dependency on `ase` and `oechem`
- force prediction using SLATM
   * geometry optimization, molecular dynamics


# Installation

## Requirements

`aqml` is a python/fortran package that requires a number of dependencies:

- `numpy`
- `scipy`
- `oechem`: cheminformatic package (need for an academic license, which is free)
- `rdkit`: cheminformatic package 
- `networkx` a Python package for the creation, manipulation, and study of the structure, dynamics, and functions of complex networks. [https://networkx.github.io/documentation/stable/install.html]
- `ase`: Atomic Simulation Environment [https://wiki.fysik.dtu.dk/ase/install.html]


optional:
- `dftd3`: A dispersion correction for density functionals and other methods [https://www.chemie.uni-bonn.de/pctc/mulliken-center/software/dft-d3/get-the-current-version-of-dft-d3]
- `imolecule`: draw mol interactively in jupyter-notebook (recommended)
- `indigo`: cheminformatic package https://lifescience.opensource.epam.com/indigo/index.html#download-and-install
- `openbabel`: cheminformatic package http://openbabel.org/wiki/Category:Installation
- `cairosvg`: convert svg to png, pdf, etc.
- `deepdish`: hdf5 python library 

I recommend using `conda` (for Python 3+) to install all dependencies


## Build & Install 

Steps

- miniconda or anaconda (go to https://docs.conda.io/projects/conda/en/latest/user-guide/install/ and follow the instructions there. Note that the version Python 3.7 is preferred!)

- oechem (first apply for an academic license from https://www.eyesopen.com/academic-licensing) and then
```bash
[macos] pip install -i https://pypi.anaconda.org/openeye/simple openeye-toolkits-python3-osx-x64
[linux] pip install -i https://pypi.anaconda.org/openeye/simple openeye-toolkits-python3-linux-x64
```
Afterwards, install the license file
```bash
echo "export OE_LICENSE=/path/to/oe_license.txt" >>~/.bashrc
```

- rdkit
```bash
conda install -y -c rdkit rdkit 
```

- other dependencies
```bash
pip install ase imolecule networkx 
```

- clone the repository

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
echo "export PYTHONPATH=$PWD:$PYTHONPATH" >>~/.bashrc
echo "export PATH=$PWD/bin:$PATH >>~/.bashrc
source ~/.bashrc
```

Now you are ready to go!

# Usage

## command line

### aqml
```bash
usage: aqml [-h] [-nprocs [NPROCS]] [-bj] [-abc] [-p [P]] [-z [Z]] [-fp [FP]]
            [-scu [SCU]] [-wd [W0]] [-rcut [RCUT]] [-c [COEFFS [COEFFS ...]]]
            [-k [KERNEL]] [-cab] [-reusek] [-savek] [-fk [FK]] [-savex]
            [-reusex] [-fx1 [FX1]] [-fx2 [FX2]] [-train [TRAIN [TRAIN ...]]]
            [-exclude [_EXCLUDE [_EXCLUDE ...]]] [-keep [_KEEP [_KEEP ...]]]
            [-l [LAMBDAS]] [-test [TEST [TEST ...]]] [-n2 [N2]] [-i1by1]
            [-idx1 [_IDX1 [_IDX1 ...]]] [-idx2 [_IDX2 [_IDX2 ...]]]
            [-iaml [IAML]] [-i2 [I2]] [-add [ADD [ADD ...]]] [-dmxby [DMXBY]]
            [-ref [REF]] [-iprta] [-ieaq] [-debug]

optional arguments:
  -h, --help            show this help message and exit
  -nprocs [NPROCS], --nprocs [NPROCS]
                        Number of threads to be used
  -bj                   manually add dft-d3 correction to energy?
  -abc                  manually add ATM correction to energy?
  -p [P], --property [P]
                        property to be trained/test
  -z [Z]                if `p is atomic property, -z [val] must be specified!
  -fp [FP]              property file. If specified, properties would not be
                        read from xyz file
  -scu [SCU]            property to be trained/test
  -wd [W0]              current working directory, default is "./"
  -rcut [RCUT], --rcut [RCUT]
                        SLATM cutoff radius, default is 4.8 Ang
  -c [COEFFS [COEFFS ...]], -coeffs [COEFFS [COEFFS ...]]
                        scaling factor for `dmax
  -k [KERNEL], --kernel [KERNEL]
                        gaussian or lapalacian
  -cab
  -reusek, --reusek
  -savek, --savek
  -fk [FK], --fk [FK]   filename of kernel
  -savex, --savex
  -reusex, --reusex
  -fx1 [FX1], --fx1 [FX1]
                        filename of x1
  -fx2 [FX2], --fx2 [FX2]
                        filename of x2
  -train [TRAIN [TRAIN ...]], --train [TRAIN [TRAIN ...]]
                        Name of the folder(s) containing all training mols
  -exclude [_EXCLUDE [_EXCLUDE ...]], -remove [_EXCLUDE [_EXCLUDE ...]], --exclude [_EXCLUDE [_EXCLUDE ...]], --remove [_EXCLUDE [_EXCLUDE ...]]
                        molecular idxs (for all mols, i.e., including mols
                        from all training folders) to be excluded for training
  -keep [_KEEP [_KEEP ...]], --keep [_KEEP [_KEEP ...]]
                        molecular idxs (in the j-th training folder, where `j
                        must be specified as a negative integer) to be kept
                        for training, skip the rest. E.g., "-keep -1 35 40-49
                        -2 23-25" would keep mols with idx 35, 50-59 in
                        ag.train[-1] and mols 23-25 in ag.train[-2]!
  -l [LAMBDAS], -lambda [LAMBDAS], -lambdas [LAMBDAS]
                        llambdas= 10^{-ls}
  -test [TEST [TEST ...]], --test [TEST [TEST ...]]
                        Name of the folder(s) containing all test molecules
  -n2 [N2], --n2 [N2]   Number of test molecules; must be specified when no
                        test folder ia avail
  -i1by1                is training/test to be done 1 by 1?
  -idx1 [_IDX1 [_IDX1 ...]]
                        specify training set by idxs of mols
  -idx2 [_IDX2 [_IDX2 ...]]
                        specify test set by idxs of mols
  -iaml [IAML]          use AML?
  -i2 [I2]              Target mol idx
  -add [ADD [ADD ...]]  Idx of mols to be added for training, default is []
  -dmxby [DMXBY]        calc `dmax using amons/target/all?
  -ref [REF]            folder containing a set of mols for regression of
                        atomic reference energy
  -iprta
  -ieaq                 display energy of atoms in query mol(s)?
  -debug
```


### genamons
```bash
usage: genamon [-h] [-iprt [IPRT]] [-i3d [I3D]] [-iwarn [IWARN]]
               [-fixgeom [FIXGEOM]] [-imap [IMAP]] [-wg [WG]] [-ra [RA]]
               [-iextl [IEXTL]] [-debug [DEBUG]] [-nocrowd [NOCROWD]]
               [-ioc [IOC]] [-iocn [IOCN]] [-icrl2o [ICRL2O]] [-igchk [IGCHK]]
               [-icpchk [ICPCHK]] [-irddtout [IRDDTOUT]] [-ivdw [IVDW]]
               [-ivao [IVAO]] [-keepHalogen [KEEPHALOGEN]]
               [-icc4Rsp3out [ICC4RSP3OUT]] [-iasp2arout [IASP2AROUT]]
               [-nogc [NOGC]] [-noextra [NOEXTRA]] [-k [K]] [-k2 [K2]]
               [-opr [OPR]] [-ff [FF]] [-gopt [GOPT]] [-label [LABEL]]
               [-np [NPROCS]] [-nmaxcomb [NMAXCOMB]] [-thresh [THRESH]]
               [ipts [ipts ...]]

positional arguments:
  ipts

optional arguments:
  -h, --help            show this help message and exit
  -iprt [IPRT]
  -i3d [I3D]
  -iwarn [IWARN]
  -fixgeom [FIXGEOM]
  -imap [IMAP]
  -wg [WG]
  -ra [RA]
  -iextl [IEXTL]
  -debug [DEBUG]
  -nocrowd [NOCROWD]
  -ioc [IOC]
  -iocn [IOCN]
  -icrl2o [ICRL2O]
  -igchk [IGCHK]
  -icpchk [ICPCHK]
  -irddtout [IRDDTOUT]
  -ivdw [IVDW]
  -ivao [IVAO]
  -keepHalogen [KEEPHALOGEN]
  -icc4Rsp3out [ICC4RSP3OUT]
  -iasp2arout [IASP2AROUT]
  -nogc [NOGC]
  -noextra [NOEXTRA]
  -k [K]
  -k2 [K2]
  -opr [OPR]
  -ff [FF]
  -gopt [GOPT]
  -label [LABEL]
  -np [NPROCS], -nproc [NPROCS], -nprocs [NPROCS]
  -nmaxcomb [NMAXCOMB]
  -thresh [THRESH]
```
- generate SMILES only without 3d coordinates
```bash
genamon -k 6 -i3d F -verbose 0 "Cc1ccccc1"
```
produces 7 unique canonical SMILES (oechem standard)
```bash
cans= ['C', 'C=C', 'C=CC=C', 'CC=C', 'CC(=C)C=C', 'CC=CC=C', 'c1ccccc1']
```


## python functions

### amons generataion
```bash
import cheminfo.oechem.amon as coa

obj = coa.ParentMols(fs)
a = obj.generate_amons()
a.cans
```

# Publications

If you have used `aqml` in your research, please consider citing these papers:

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



