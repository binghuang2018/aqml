# -*- coding: utf-8 -*-

import itertools as itl
import numpy as np
import os, sys, re, copy
import aqml.cheminfo as co
import aqml.cheminfo.core as cc
import aqml.cheminfo.molecule.core as cmc
import aqml.cheminfo.rdkit.core as coc
from rdkit import Chem


# reference coordination number
#cnsr = {1:1, 3:1, 4:2, 5:3, 6:4, 7:3, 8:2, 9:1, 13:3, 14:4, 15:3, 16:2, 17:1, 35:1, 53:1}

T,F = True,False

dic_bonds = { Chem.BondType.SINGLE:1.0,
              Chem.BondType.DOUBLE:2.0,
              Chem.BondType.TRIPLE:3.0,
              Chem.BondType.AROMATIC:1.5,
              Chem.BondType.UNSPECIFIED:0.0}

def sdf2oem(f, sanitize=T, removeHs=F):
    return Chem.MolFromMolFile(f, sanitize=sanitize, removeHs=removeHs)

def pdb2oem(f, sanitize=T, removeHs=F):
    return Chem.MolFromPDBFile(f, sanitize=sanitize, removeHs=removeHs)

def oem2can(m, isomericSmiles=F, kekuleSmiles=F, canonical=T, allHsExplicit=F):
    return Chem.MolToSmiles(m, isomericSmiles=isomericSmiles, \
             kekuleSmiles=kekuleSmiles, canonical=canonical, \
             allHsExplicit=allHsExplicit)


def smi2oem(smi, addh=F, sanitize=T):
    iok = T
    try:
        m = Chem.MolFromSmiles(smi, sanitize=sanitize)
    except:
        iok = F
    if addh:
        m = Chem.AddHs(m)
    return iok, m


def get_bom(m0, kekulize=False):
    # somehow it's problematic to use `Chem.GetAdjacencyMatrix(m, useBO=True)
    m = copy.deepcopy(m0)
    if kekulize:
        Chem.Kekulize(m)
    na = m.GetNumAtoms()
    bom = np.zeros((na, na))
    for bi in m.GetBonds():
        i, j = bi.GetBeginAtomIdx(), bi.GetEndAtomIdx()
        bt = bi.GetBondType(); #print ' ***** bt = ', bt
        bom[i,j] = bom[j,i] = dic_bonds[ bt ]
    return bom.astype(int)


def get_coords(m):
    na = m.GetNumAtoms()
    zs = np.array([ ai.GetAtomicNum() for ai in m.GetAtoms() ])
    #assert m.GetNumConformers() == 1, '#ERROR: more than 1 Conformer exist?'
    try:
        c0 = m.GetConformer(-1)
        coords = []
        for i in range(na):
            coords_i = c0.GetAtomPosition(i)
            coords.append([ coords_i.x, coords_i.y, coords_i.z ])
    except:
        print(' ** No coords found')
        coords = np.zeros((na,3))
    return np.array(coords)


def prepare_protein(f):
    raise Exception('Not implemented using RDKit yet! Please use the relevant func in ../oechem/oechem.py')


class ConnMol(cmc.RawMol):
    """
    Mol with connectivity only
    No bond order perceived
    """
    def __init__(self, obj, ivdw=False, scale=1.0, iprt=F):
        cmc.RawMol.__init__(self, obj, ivdw=ivdw, scale=scale, iprt=iprt)

    def get_mol(self):
        return coc.newmol(self.zs, np.zeros(self.na), self.g, self.coords).mol

    @property
    def mol(self):
        if not hasattr(self, '_mol'):
            self._mol = self.get_mol()
        return self._mol

    @property
    def is_mcplx(self): # is mol complex?
        if not hasattr(self, '_imcplx'):
            self._imcplx = (not self.is_connected)
        return self._imcplx



class StringM(coc.newmol):

    """
    build molecule object with a string (SMILES or sdf file) as input
    """

    def __init__(self, obj, stereo=F, isotope=F, woH=F, ds=None,\
                 pls=None, scale_vdw=1.2, resort=F, simple=F, debug=F, \
                 nprocs=1):

        self.debug = debug
        istat = T
        if isinstance(obj, str):
            string = obj
            if os.path.exists(string):
                if string.endswith( ('sdf','mol') ):
                    m = sdf2oem(string)
                elif string.endswith( ('pdb') ):
                    m = pdb2oem(string)
                else:
                    raise Exception('#ERROR: file type not supported')
                if woH:
                    m = Chem.RemoveHs(m)
            else: # presumably a SMILES string
                #print('------------------------')
                if ('@' in string) and (not stereo):
                    # now remove stereo and isotopes, otherwise we may
                    # encounter error " idx out of bound when calling get_bom()
                    istat, _m = smi2oem(string, addh=F)
                    _s = oem2can(_m)
                else:
                    _s = string
                istat, m = smi2oem(_s)
                if istat and (not woH):
                    m = Chem.AddHs(m)
        elif isinstance(obj, Chem.rdchem.Mol):
            m = obj
        else:
            raise Exception('#ERROR: input `string not supported')

        self.istat = istat
        if not istat:
            raise Exception('istat is False??')

        _zs = []; chgs = []
        for ai in m.GetAtoms():
            _zs.append( ai.GetAtomicNum() )
            chgs.append( ai.GetFormalCharge() )
        chgs = np.array(chgs,dtype=int)
        zs = np.array(_zs,dtype=int)
        na = len(zs)
        ias = np.arange(na)
        bom = get_bom(m, kekulize=T) # must set kekulize=T, otherwise BO may be 1.5
        coords = get_coords(m)
        if resort and (1 in _zs):
            ih1 = _zs.index(1)
            if np.any(zs[ih1:] > 1):
                print(' ***** hydrogens were pushed to the end')
                _ias = np.concatenate((ias[zs>1], ias[zs==1]))
                zs = zs[_ias]
                chgs = chgs[_ias]
                coords = coords[_ias]
                _bom = bom.copy()
                bom = _bom[_ias][:,_ias]
        self._zs = zs
        self._bom = bom
        self._coords = coords
        self._chgs = chgs

        coc.newmol.__init__(self, zs, chgs, bom, coords=coords, ds=ds, pls=pls, \
                         scale_vdw=scale_vdw, debug=debug, nprocs=nprocs)



class smiles_db(object):

    def __init__(self, obj, sort=T):
        if isinstance(obj,(tuple,list)):
            ss1 = obj
        elif isinstance(obj,str):
            assert os.path.exists(obj), '#ERROR: file does not exist!'
            ss1 = [ si.strip().split()[0] for si in open(obj).readlines() ]
        else:
            raise Exception('unknown input object!')
        nm = len(ss1)
        ims = np.arange(nm).astype(int)
        self.ims = ims
        nas = []
        ss2 = []
        iN5s = []
        for si in ss1:
             om = StringM(si, suppressH=F, simple=T)
             iasN5 = om.ias[ np.logical_and(om.zs==7, om.tvs==5) ]
             iN5 = F
             for iaN in iasN5:
                 _bosi = om.bom[iaN]
                 bosi = _bosi[_bosi>0]
                 bosi.sort()
                 bosi = bosi[::-1]
                 sbo = ''.join([ '%d'%bo for bo in bosi ])
                 if not (sbo=='32' or ((om.zs[om.bom[iaN]>0] == 8).sum() == 2)):
                     iN5 = T
                     break
             iN5s.append(iN5)
             ss2.append( oem2can(om.oem) ) # oechem can strings
             nas.append(om.nheav)
        nas = np.array(nas,dtype=int)
        iN5s = np.array(iN5s, dtype=bool)
        ns = np.unique(nas) #dtype=int)
        if sort:
            ss3 = []
            nms = []
            nsheav = []
            imap = []
            iN5s_new = []
            for ni in ns:
                idx = ims[ni==nas]
                ssi = np.array(ss2)[idx]
                irs = np.argsort(ssi)
                idx2 = idx[irs] #.sort()
                iN5s_new += list(iN5s[idx2])
                nmi = len(ssi)
                nms.append(nmi)
                nsheav += [ni]*nmi
                ss3 += list(ssi[irs])
                imap += list(idx2)
            print(' ** smiles were sorted by num_heav_atom and then name')
            imap = np.array(imap, dtype=int)
            iN5s = np.array(iN5s_new, dtype=bool)
        else:
            ss3 = ss2
            nsheav = nas
            nms = None
            imap = None
        self.smiles = ss3
        self.nsheav = nsheav
        self.imap = imap
        self.iN5s = iN5s
        self.nms = nms


