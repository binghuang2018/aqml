#!/bin/bash

## This script installs python codes only.

touch __init__.py

echo "export PYTHONPATH=$PWD:\$PYTHONPATH" >>~/.bashrc
echo "export PATH=$PWD/bin:\$PATH" >>~/.bashrc

source ~/.bashrc


