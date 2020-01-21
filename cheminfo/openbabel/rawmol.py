#!/usr/bin/env python

import numpy as np
import openbabel as ob
import pybel as pb
import copy
from cheminfo import *
import cheminfo.openbabel.amon_f as cioaf
import cheminfo.openbabel.amon as cioa
import cheminfo.openbabel.obabel as cib
#import networkx as nx


def is_connected(g):
    nv = g.shape[0]
    ne = (g>0).sum()/2
    return ne - nv + 1 >= 0

class RawMol(object):

    def __init__(self, zs, coords, allow_radical=False, allow_charge=False):
        """
        special cases:
           SMILES          xyzfile (initial)     xyzfile updated
        1) C[NH+]=CC(=O)O  CN[C]C([O])O          CN[C]C(=O)O
        2) CC(=O)[O-]      CC([O])([O])          CC(=O)(=O)

        """
        self.zs = zs
        self.coords = coords
        iok = True
        if not allow_radical:
            if sum(zs)%2 != 0: iok = False; print ' **'
        g = cib.perceive_g(zs, coords)
        self.g = g

        can = None; m = None

        if iok:
          ic = is_connected(g)
          if not ic:
            iok = False
            print ' ** dissociated'
          else:
            cns = g.sum(axis=0)
            na = len(zs)
            chgs = np.zeros(na,np.int)
            tvs = -999*np.ones(na,np.int)
            dic = {1:1, 4:2,  5:3, 6:4, 7:3, 8:2, 9:1, \
                    13:3, 14:4, 32:4, 34:2, 50:4 }
            zsf = dic.keys()
            ias = np.arange(na).astype(np.int)
            ias_visited = []
            for ia in range(na):
                #print ' ia, tvs = ', ia, tvs
                zi = zs[ia]
                nb = (g[ia] > 0).sum()
                if zi in zsf:
                    if tvs[ia] < 0:
                        tvs[ia] = dic[zi]
                    if zi == 6 and cns[ia] == 1:
                        # e.g., -N$C (or -[N+]#[C-]
                        chgs[ia] = -1; ibs = ias[ g[ia] > 0 ]
                        ib = ibs[0]
                        assert ibs.shape[0] == 1 and zs[ib] == 7
                        #if ib not in ias_visited:
                        #ias_visited.append(ib)
                        tvs[ib] = 5; chgs[ib] = 1
                    elif zi == 7:
                        if cns[ia] == 4:
                            if not allow_charge:
                                iok = False; print ' *** '
                        elif cns[ia] == 3:
                            ibs = ias[ g[ia] > 0 ]
                            for ib in ibs:
                                if zs[ib] in [8,] and (g[ib]>0).sum() == 1:
                                    # R-N(=O)=O or R-N(=CR)=O
                                    tvs[ia] = 5; break
                        elif cns[ia] == 1:
                            #print '+++++'
                            ibs = ias[ g[ia] > 0 ]
                            ib = ibs[0]
                            if len(ibs) == 1 and zs[ib] == 7:
                                # C=N#N,
                                chgs[ia] = -1;
                                #if ib not in ias_visited:
                                chgs[ib] = 1; tvs[ib] = 5
                                #ias_visited.append( ib )
                                #print 'tvs = ', tvs
                elif zi in [15,33,51]:
                    assert nb in [3,4,5]
                    tvs[ia] = {3:3, 4:5, 5:5}[nb]
                elif zi in [16,34,52]:
                    assert nb in [2,3,4]
                    tvs[ia] = {2:2, 3:4, 4:6}[nb]
                elif zi in [17,35,53]:
                    tvs[ia] = 1 if nb == 1 else 7
                else:
                    raise '#ERROR: new element?'


            #print ' --   zs = ', zs
            #print ' --  tvs = ', tvs
            #print ' -- chgs = ', chgs
            #print 'iok = ', iok
            if iok:
                bosr = []
                cmg = cioaf.MG(bosr, zs, chgs, tvs, g, coords, use_bosr=False)
                #cmg = cioa.MG(bosr, zs, chgs, tvs, g, coords, use_bosr=False)
                cans, ms = cmg.update_m() #debug=True, icon=True)
                nm = len(cans)
                if nm == 0:
                    iok = False
                    print ' ++ found radical or charged species'
                elif nm == 1:
                    can = cans[0]; m = ms[0]
                else:
                    iok = False
                    print ' ++ more than one possiblities: ', cans

        self.iok = iok
        self.can = can
        self.m = m


if __name__ == '__main__':

    import os, sys
    import cheminfo.rw.xyz as cix

    # Note that file in `fs represent
    fs = sys.argv[1:]
    for f in fs:
        symbols, positions = cix.read_xyz(f)[0]
        numbers = [ atomic_numbers[si] for si in symbols ]
        obj = RawMol( np.array(numbers), np.array(positions) )
        if obj.iok:
            print f[:-4], obj.can
        else:
            print f[:-4], ' ** FAILED'

