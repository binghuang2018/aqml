
"""
write mol, sdf files

format
=====================================================================
 cyclobutane
     RDKit          3D

  4  4  0  0  0  0  0  0  0  0999 V2000
   -0.8321    0.5405   -0.1981 C   0  0  0  0  0  0  0  0  0  0  0  0
   -0.3467   -0.8825   -0.2651 C   0  0  0  0  0  0  0  0  0  0  0  0
    0.7190   -0.5613    0.7314 C   0  0  0  0  0  0  0  0  0  0  0  0
    0.4599    0.9032    0.5020 C   0  0  0  0  0  0  0  0  0  0  0  0
  1  2  1  0
  2  3  1  0
  3  4  1  0
  4  1  1  0
M  END
=====================================================================
"""

import numpy as np
from cheminfo.core import *

def write_ctab(zs, chgs, bom, coords=None, isotopes=[], prop={}, \
               sdf=None, sort_atom=True):
    """
    vars
    =================
    sort_atom: if set to T, H atoms will be pushed to the end
               of atom list
    """
    na = zs.shape[0]
    ias = np.arange(na)
    ias_heav = ias[zs > 1]
    iash = ias[zs==1]
    nheav = len(ias_heav)
    nb = (np.array(bom) > 0).ravel().sum()/2
    if coords is None:
        coords = np.zeros((na,3))

    ctab = 'none\n     RDKit          3D\n\n'
    fmt1 = '%3d'*6 + '  0  0  0  0999 V2000\n'
    ctab += fmt1%( na, nb, 0,0,0,0)

    fmt1 = '%10.4f'*3 + ' %-3s'
    fmt2 = '%2d' + '%3d'*11 + '\n'
    str2 = fmt2%(tuple([0,]+ [0,]*11))
    fmt = fmt1 + str2
    iasU = np.concatenate((ias_heav,iash))
    for i in range(na):
        ia = iasU[i]
        px, py, pz = coords[ia]
        zi = zs[ia]
        ctab += fmt%(px, py, pz, chemical_symbols[zi])

    # write bonds between heav atoms
    bcnt = 0
    if nheav > 1:
        for i in range(nheav):
            for j in range(i+1, nheav):
                iu, ju = ias_heav[i], ias_heav[j]
                boij = bom[iu,ju]
                if boij > 0:
                    ctab += '%3d%3d%3d%3d\n'%(i+1,j+1,boij,0)
                    bcnt += 1

    # write bonds between heav and H
    for i in range(nheav):
        for j in range(nheav, na):
            iu, ju = ias_heav[i], iash[j-nheav]
            boij = bom[iu,ju]
            if boij > 0:
                ctab += '%3d%3d%3d%3d\n'%(i+1,j+1,boij,0)
                bcnt += 1
    #print('nb, bcnt=', nb, bcnt)
    assert bcnt == nb, '#ERROR: not all bonds are written??'

    # write isotopes
    if len(isotopes) > 0:
        naiso = len(isotopes)
        ctab += 'M  ISO%3d'%naiso
        #print isotopes
        for ia in isotopes:
            ctab += ' %3d %3d'%(ia+1,2) # assume [2H], i.e., D in [H,D,T]
        ctab += '\n'

    chgs_u = np.array(chgs,np.int)[iasU]
    ias = np.arange(na) + 1
    # write charges
    iasc = ias[ np.array(chgs_u) != 0 ]
    nac = iasc.shape[0]
    if nac > 0:
        ctab += 'M  CHG%3d'%nac
        for iac in iasc:
            ctab += ' %3d %3d'%(iac, chgs_u[iac-1])
        ctab += '\n'

    ctab += 'M  END'
    if len(prop.keys()) > 0:
        # write properties after "M  END"
        # an example
        # =======================
        #>  <HF>  (1)
        #-100.0
        #
        #>  <smiles_indigo>  (1)
        #C1=CC=C1N=O
        #
        #$$$$
        # =======================
        ctab += '\n'
        for key in prop.keys():
            ctab += '>  <%s>  (1)\n%s\n\n'%(key, prop[key])
        ctab += '$$$$'

    if sdf != None:
        with open(sdf,'w') as f: f.write(ctab)
        return
    else:
        return ctab

