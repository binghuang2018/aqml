

from rdkit import Chem
import numpy as np

class ecfp(object):

    def __init__(self, smi):

        m = Chem.MolFromSmiles(smi)
        na = m.GetNumHeavyAtoms()
        nb = m.GetNumBonds()

        for i in range(na):

