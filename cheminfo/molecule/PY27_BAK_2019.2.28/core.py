#!/usr/bin/env python

import os, sys, re, copy
import numpy as np
import indigo
from cheminfo.base import *

__all__ = [ 'rawmol' ]

T,F = True,False

class rawmol(object):

    """
    build a Indigo mol object from scratch
    """

    def __init__(self, obj):
        assert isinstance(obj,(list,tuple)), "#ERROR: obj not a list/tuple?"
        assert len(obj)==4, "#ERROR: `obj should .eq. (zs,chgs,bom,coords)"
        zs, coords, chgs, bom = obj
        # unfortunately, Indigo cannot load file formats other than sdf/mol
        # so we manually construct molecule here
        na = len(zs)
        ias = np.arange(na).astype(np.int)
        newobj = indigo.Indigo()
        m = newobj.createMolecule()
        ats = []
        for ia in range(na):
            ai = m.addAtom(chemical_symbols[zs[ia]])
            ai.setXYZ( coords[ia,0],coords[ia,1],coords[ia,2] )
            ai.setCharge(chgs[ia])
            ats.append(ai)
        for ia in range(na):
            ai = ats[ia]
            jas = ias[ np.logical_and(bom[ia]>0, ias>ia) ]
            for ja in jas:
                bi = ai.addBond(ats[ja], bom[ia,ja])
        self.m = m

    def tocan(self, nostereo=True, aromatize=True):
        """
        one may often encounter cases that different SMILES
        correspond to the same graph. By setting aromatize=True
        aliviates such problem
        E.g., C1(C)=C(C)C=CC=C1 .eqv. C1=C(C)C(C)=CC=C1
        """
        m2 = self.m.clone()
        if nostereo: m2.clearStereocenters()
        if aromatize: m2.aromatize()
        can = m2.canonicalSmiles()
        return can

