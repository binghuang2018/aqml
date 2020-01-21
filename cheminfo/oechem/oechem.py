# -*- coding: utf-8 -*-

from openeye import *
from openeye.oechem import *
import networkx as nx
import itertools as itl
import scipy.spatial.distance as ssd
import multiprocessing

import numpy as np
import ase.io as aio
import ase.data as ad
import ase, os, sys, re, copy

import cheminfo as co
import cheminfo.core as cc

import cheminfo.math as cim
from cheminfo.molecule.elements import Elements
import cheminfo.molecule.core as cmc
import cheminfo.molecule.geometry as GM
import cheminfo.molecule.nbody as MB
import cheminfo.rdkit.resonance as crr
import cheminfo.oechem.core as coc

import cheminfo.rw.ctab as crc
from rdkit import Chem


# reference coordination number
#cnsr = {1:1, 3:1, 4:2, 5:3, 6:4, 7:3, 8:2, 9:1, 13:3, 14:4, 15:3, 16:2, 17:1, 35:1, 53:1}

T,F = True,False


#class Match(object):
#    def __init__(self, mols_q, mol_t, smiles=None):
#        if smiles is None:
#            smiles =
#        patt = smi2patt(smiles)
#
#    def filter(self,thresh):
#        matches = []
#        return icsr

dic_fmt = {'sdf': oechem.OEFormat_SDF, 'pdb': oechem.OEFormat_PDB, \
            'mol': oechem.OEFormat_MDL, 'xyz': oechem.OEFormat_XYZ}

def sdf2oem(sdf):
    ifs = oemolistream()
    assert ifs.SetFormat( dic_fmt[ sdf[-3:] ] )
    assert ifs.open(sdf)
    for m in ifs.GetOEGraphMols(): #.next()
        break
    return m

def pdb2oem(f):
    return sdf2oem(f)

def oem2can(oem, ImpHCount=F, rebuild=F):
    # ISOMERIC identical to Isotopes | AtomStereo | BondStereo | Canonical | AtomMaps | RGroups
    if ImpHCount:
        flavor = OESMILESFlag_ImpHCount | OESMILESFlag_Canonical
    else:
        flavor = OESMILESFlag_Canonical
    m = rebuild_m(oem) if rebuild else oem
    return OECreateSmiString(m, flavor)

def oem2smi(oem, ImpHCount=F, rebuild=F):
    return oem2can(oem, ImpHCount=ImpHCount, rebuild=rebuild)

def smi2oem(smi, addh=False):
    m = OEGraphMol()
    iok = OESmilesToMol(m,smi)
    #assert iok, '#ERROR: parsing SMILES failed!'
    if addh:
        OEAddExplicitHydrogens(m)
    else:
        OESuppressHydrogens(m,F,F,F)
    return iok, m

def rebuild_mol(m):
    mu = OEGraphMol()
    atoms = {}; icnt = 0
    for ai in m.GetAtoms():
        ia = ai.GetIdx()
        zi = ai.GetAtomicNum()
        if zi > 1:
            aiu = mu.NewAtom( zi )
            aiu.SetHyb( OEGetHybridization(ai) )
            aiu.SetImplicitHCount( ai.GetImplicitHCount() )
            atoms[ icnt ] = aiu
            icnt += 1
    for bi in m.GetBonds():
        p, q = bi.GetBgnIdx(), bi.GetEndIdx()
        biu = mu.NewBond( atoms[p], atoms[q], bi.GetOrder() )
    OEFindRingAtomsAndBonds(mu)
    OEAssignAromaticFlags(mu, OEAroModel_OpenEye)
    OECanonicalOrderAtoms(mu)
    OECanonicalOrderBonds(mu)
    return mu

def get_bom(m):
    na = m.NumAtoms()
    bom = np.zeros((na,na), dtype=int)
    for bi in m.GetBonds():
        i,j,boij = bi.GetBgnIdx(), bi.GetEndIdx(), bi.GetOrder()
        bom[i,j] = bom[j,i] = boij
    return bom

def get_coords(m):
    coords = []
    for ai in m.GetAtoms():
        coords_ai = m.GetCoords(ai)
        coords.append( coords_ai )
    return np.array(coords,dtype=float)

def vang(u,v):
    cost = np.dot(u,v)/(np.linalg.norm(u) * np.linalg.norm(v))
# sometimes, `cost might be 1.00000000002, then np.arccos(cost)
# does not exist!
    u = cost if abs(cost) <= 1 else 1.0
    return np.arccos( u )*180/np.pi


def prepare_protein(f):
    m = pdb2oem(f)
    OESuppressHydrogens(m, F, F, F)

    lig = oechem.OEGraphMol()
    prot = oechem.OEGraphMol()
    wat = oechem.OEGraphMol()
    other = oechem.OEGraphMol()
    assert oechem.OESplitMolComplex(lig, prot, wat, other, m)

    OERemoveFormalCharge(prot) # note that for quaternary amines (>[N+]<), charges retain
    OEAddExplicitHydrogens(prot)
    OESet3DHydrogenGeom(prot)
    f2 = f[:-4]+'-new.pdb'
    write(prot,f2)
    write(prot,f[:-4]+'-new.xyz')
    obj = StringM(f2)
    obj.check_valence_states()
    #obj.check_interatomic_distance()
    #assert iok


class XYZMol(object):
    """
    perceive BO from geom by OEChem
    """
    def __init__(self, fx):
        ifs = oemolistream()
        assert ifs.SetFormat( dic_fmt['xyz'] )
        assert ifs.open(fx)
        for mol in ifs.GetOEGraphMols(): #.next()
            break
        oechem.OEDetermineConnectivity(mol)
        oechem.OEFindRingAtomsAndBonds(mol)
        oechem.OEPerceiveBondOrders(mol)
        oechem.OEAssignImplicitHydrogens(mol)
        oechem.OEAssignFormalCharges(mol)
        self.mol = mol

    @property
    def g(self):
        return


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

    def get_strained(self):
        return np.any( [ [ OEAtomIsInRingSize(ai, n) for n in [3,4,5,7] ] for ai in self.mol.GetAtoms() ] )

    @property
    def strained(self):
        if not hasattr(self, '_strained'):
            self._strained = self.get_strained()
        return self._strained

    @property
    def is_mcplx(self): # is mol complex?
        if not hasattr(self, '_imcplx'):
            self._imcplx = (not self.is_connected)
        return self._imcplx


class StringM(coc.newmol):

    """
    build molecule object with a string (SMILES or sdf file) as input
    """

    def __init__(self, obj, stereo=F, isotope=F, suppressH=F, ImpHCount=F, ds=None,\
                 pls=None, scale_vdw=1.2, resort=F, simple=F, debug=F, nprocs=1):

        self.suppressH = suppressH
        self.ImpHCount = ImpHCount
        self.debug = debug
        istat = T
        if isinstance(obj, str):
            string = obj
            if os.path.exists(string):
                if string.endswith( ('sdf','mol','pdb') ):
                    m = sdf2oem(string)
                    #print('######################')
                    if suppressH:
                        OESuppressHydrogens(m, F, F, F)
                else:
                    raise Exception('#ERROR: file type not supported')
                if suppressH:
                    OESuppressHydrogens(m, False, False, False)
            else: # presumably a SMILES string
                #print('------------------------')
                if ('@' in string) and (not stereo):
                    # now remove stereo and isotopes, otherwise we may
                    # encounter error " idx out of bound when calling get_bom()
                    istat, _m = smi2oem(string)
                    _s = OECreateSmiString(_m, OESMILESFlag_Canonical)
                else:
                    _s = string
                istat, m = smi2oem(_s)
                if istat and (not suppressH):
                    iok = OEAddExplicitHydrogens(m)
        elif isinstance(obj, oechem.OEGraphMol):
            m = obj
        else:
            raise Exception('#ERROR: input `string not supported')

        self.istat = istat
        if not istat:
            raise Exception('istat is False??')

        na = m.NumAtoms()
        _zs = []; chgs = []
        for ai in m.GetAtoms():
            _zs.append( ai.GetAtomicNum() )
            chgs.append( ai.GetFormalCharge() )
        chgs = np.array(chgs,dtype=int)
        zs = np.array(_zs,dtype=int)
        ias = np.arange(na)
        bom = get_bom(m)
        coords = get_coords(m)
        if resort and (1 in _zs):
            ih1 = _zs.index(1)
            if np.any(zs[ih1:] > 1):
                print(' ***** hydrogens were pushed to the end')
                newm = OEGraphMol()
                _ias = np.concatenate((ias[zs>1], ias[zs==1]))
                zs = zs[_ias]
                chgs = chgs[_ias]
                coords = coords[_ias]
                _bom = bom.copy()
                bom = _bom[_ias][:,_ias]
        self.zs = zs
        self.bom = bom
        self.coords = coords
        self.chgs = chgs

        coc.newmol.__init__(self, zs, chgs, bom, coords=coords, ds=ds, pls=pls, \
                         scale_vdw=scale_vdw, debug=debug, nprocs=nprocs)

    #@property
    #def newm(self):
    #    if not hasattr(self, '_newm'):
    #        self._newm = coc.newmol(self.zs, self.chgs, self.bom, coords=self.coords)
    #    return self._newm


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


def find_neighbors(g, ias):
    """
    get the neighbors of a list of atoms `ias
    """
    neibs = []
    na = g.shape[0]
    ias0 = np.arange(na)
    for ia in ias:
        for ja in ias0[ g[ia] > 0 ]:
            if ja not in ias:
                neibs.append( ja )
    return neibs


