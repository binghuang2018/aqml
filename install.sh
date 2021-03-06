#!/bin/bash


touch __init__.py

#export AQML_ROOT=$PWD
#export CHEMPACK=RDKIT # or OECHEM

# Build core ML library (mainly fortran code for
# time-consuming part of compuation)
cd $AQML_ROOT/coreml
python setup.py install



# dependencies
sysname=`uname`
if [[ "$CHEMPACK" == "OECHEM" ]]
then
  if [[ "$sysname" == "Linux" ]]; then
    #pip install -i https://pypi.anaconda.org/openeye/simple openeye-toolkits-python3-linux-x64
    pip install -i https://pypi.anaconda.org/OpenEye/simple OpenEye-toolkits
  elif [[ "$sysname" == "Darwin" ]]; then
    #pip install -i https://pypi.anaconda.org/openeye/simple openeye-toolkits-python3-osx-x64
    pip install -i https://pypi.anaconda.org/OpenEye/simple OpenEye-toolkits
  else
    echo "system not supported"
    exit 1
  fi
fi


# install numpy, scipy and others
pip install numpy scipy ase networkx imolecule

# install rdkit (version>=2019)
conda install -y -c rdkit rdkit


# at last,
echo "export AQML_ROOT=$AQML_ROOT" >>~/.bashrc
echo "export PYTHONPATH=\$AQML_ROOT/../:\$PYTHONPATH" >>~/.bashrc
echo "export PATH=\$AQML_ROOT/bin:\$PATH" >>~/.bashrc

#source ~/.bashrc

