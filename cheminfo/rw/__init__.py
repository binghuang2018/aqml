# -*- coding: utf-8 -*-


from rdkit import Chem
from .pdb import *
from .ctab import *

T, F = True, False

class rawmol(object):

    """
      read/write RDKit mol
    """

    def __init__(self,  zs, chgs, bom, coords):
        na = len(zs)
        self.na = na
        self.zs = zs
        self.chgs = chgs
        self.bom = bom
        self.coords = coords

    def build(self, sort_atom=F, sanitize=True, removeHs=False):
        """ create a RDKit molecule object from scratch info """
        if self.na > 999:
            ctab = write_pdb( (self.zs, self.chgs, self.bom, self.coords), sort_atom=sort_atom)
            m = Chem.MolFromPDBBlock(ctab, sanitize=sanitize, removeHs=removeHs)
        else:
            ctab = write_ctab(self.zs, self.chgs, self.bom, self.coords, sort_atom=sort_atom)
            #print('ctab=',ctab)
            m = Chem.MolFromMolBlock(ctab, sanitize=sanitize, removeHs=removeHs)
        return m

