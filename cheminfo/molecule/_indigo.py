
import os,sys,re
import numpy as np
import indigo
from cheminfo.core import *
from cheminfo.molecule.core import *
from cheminfo.molecule.molecule import *
try:
  import cheminfo.fortran.famon as cf
except:
  import cheminfo.fortran.famon_mac as cf

#__all__ = [ 'get_bo_matrix', 'states', '_indigo' ]

T,F = True,False


def get_bo_matrix(m):
    na = m.countAtoms()
    bom = np.zeros((na, na)).astype(np.int)
    nb = m.countBonds()
    for ib in range(nb):
        bi = m.getBond(ib)
        i, j = bi.source().index(), bi.destination().index()
        bo = bi.bondOrder()
        bom[i,j] = bom[j,i] = bo
    return bom


class states(object):
    def __init__(self, parse_status, hadd_status, kekulize_status):
        self.parse_status = parse_status
        self.hadd_status = hadd_status
        self.kekulize_status = kekulize_status

def amend_smiles(smi,ioks):
    """ in case Indigo cannot parse a SMILES, amend this by
    calling
    """
    obj = indigo.Indigo()
    # some failure is due to stereo chem, now remove it
    if not ioks.parse_status:
        _pat = re.compile("\@")
        smi = re.sub("", s1)
        iok = T
        try:
            _m = obj.loadMolecule(smi)
        except:
            print(' ** reload mol failed after removing stereochemistry')
            iok = F
        ioks.parse_status = iok
    else:
        # failure largely due to aromization problem
        m = obj.loadMolecule(smi)
        _bom = get_bo_matrix(m)

    #_m.unfolderHydrogens()
    na = _m.countAtoms()
    _bom = get_bo_matrix(_m)
    chgs = np.array([_m.getAtom(i).charge() for i in range(na)],np.int)
    zs = np.array([m.getAtom(i).atomicNumber() for i in range(na)],np.int)

    tvsi = []
    nrmax = 0
    nbmax = 0
    iok, bom = cf.update_bom(nrmax,nbmax,zs,tvsi,_bom,F)

    blk = (zs,chgs,bom,coords)
    rawm = rawmol_indigo(blk)
    return rawm.m


class _indigo(object):

    def __init__(self, _obj, addh=True, kekulize=False, amend=False):
        hadd_status = T
        parse_status = T
        kekulize_status = T
        if isinstance(_obj, indigo.IndigoObject):
            m = _obj
        elif isinstance(_obj, str):
            obj = indigo.Indigo()
            if os.path.exists(_obj):
                m = obj.loadMoleculeFromFile(_obj)
            else:
                self.smi = _obj
                try:
                    m = obj.loadMolecule(_obj)
                except:
                    print(' ** failed to parse SMILES')
                    parse_status = F
            if parse_status:
                if addh:
                    try:
                        m.unfoldHydrogens()
                    except:
                        hadd_status = F
                        print(' ** failed to add H')
        elif isinstance(_obj, (tuple,list)):
            # unfortunately, Indigo cannot load file formats other than sdf/mol
            # so we manually construct molecule here
            rawm = rawmol_indigo(_obj)
            m = rawm.m
        else:
            raise '#unknown instance'
        if parse_status and hadd_status:
            if kekulize:
                try:
                    kekulize_status = m.dearomatize()
                except:
                    kekulize_status = F
                if not kekulize_status: print(' ** failed to kekulize')
        ioks = states(parse_status, hadd_status, kekulize_status)
        self.ioks = ioks
        self.status = parse_status and hadd_status and kekulize_status
        if self.status:
            self.m = m
        else:
            print('  # parse_status,hadd_status,kekulize_status=',parse_status,hadd_status,kekulize_status)
            if amend:
                self.m = amend_smiles(self.smi, ioks)

    def get_basic(self):
        self.na = self.m.countAtoms()
        self.zs = np.array([self.m.getAtom(i).atomicNumber() for i in range(self.na)],np.int)
        self.ias = np.arange(self.na)
        self.ias_heav = self.ias[self.zs>1]
        self.nheav = len(self.ias_heav)

    def get_coords(self):
        coords = []; zs = []
        for i in range(self.na):
            ai = self.m.getAtom(i)
            zs.append( ai.atomicNumber() )
            coords_i = ai.xyz()
            coords.append( coords_i )
        return np.array(coords)

    def kekulize(self):
        iar = self.m.dearomatize()
        self.iar = iar

    def get_bom(self):
        #m2 = self.copy()
        #iar = m2.dearomatize()
        return get_bo_matrix(self.m)

    def get_states(self):
        """ valency states, charges and positions of atoms """
        self.coords = self.get_coords()
        self.bom = self.get_bom()
        self.g = (self.bom>0).astype(np.int)
        self.cns = (self.bom>0).sum(axis=0)
        self.cns_heav = (self.bom[self.ias_heav][:,self.ias_heav]>0).sum(axis=0)
        self.nhs = self.cns[self.ias_heav]-self.cns_heav
        self.chgs = self.get_charges()
        self.vs = self.get_valences()

    def update_states(self,coords,bom,chgs,vs):
        self.coords = coords
        self.bom = bom
        self.g = (bom>0).astype(np.int)
        self.cns = (self.bom>0).sum(axis=0)
        self.cns_heav = (self.bom[self.ias_heav][:,self.ias_heav]>0).sum(axis=0)
        self.nhs = self.cns[self.ias_heav]-self.cns_heav
        self.chgs = chgs
        self.vs = vs

    def check_states(self,bom,chgs,tvsr,cnsr):
        tvs = bom.sum(axis=0) - chgs
        cns = (bom>0).sum(axis=0)
        return np.all([ (tvs[i] in tvsr[zi]) and (cns[i]<=cnsr[zi]) for i,zi in enumerate(self.zs) ])

    def check_detailed_states(self,bom,chgs,tvsr,cnsr):
        tvs = bom.sum(axis=0) - chgs
        cns = (bom>0).sum(axis=0)
        for i,zi in enumerate(self.zs):
            if not ( (tvs[i] in tvsr[zi]) and (cns[i]<=cnsr[zi]) ):
                print('i,zi,vi,vr, cn,cnr = ', i,zi,tvs[i],tvsr[zi],cns[i],cnsr[zi])

    def copy(self):
        return self.m.clone()

    def get_charges(self):
        return np.array([self.m.getAtom(i).charge() for i in range(self.na)],np.int)

    def get_raw_valences(self):
        """ original valences associated to SMILES """
        return np.array([self.m.getAtom(i).valence() for i in range(self.na)],np.int)

    def get_valences(self):
        """
        indigo gives rather different valences compared to rdkit (desired)

        [NH+]#[C-]     4,4 -> 4,3
        N=[NH+][NH-] 3,4,3 -> 3,4,2
        O=[NH]=O     2,4,2 -> 2,5,2
        """
        vs = []
        for i in range(self.na):
            vi = self.m.getAtom(i).valence()
            ci = self.chgs[i]
            if ci == 0:
                if self.zs[i]==7 and self.bom[i].sum()==5:
                    vi = 5 # Indigo somehow give a valence of 4 for N_2 in N=[NH]=N
            elif ci < 0:
                vi += ci
            vs.append(vi)
        return np.array(vs,np.int)

    def remove_isotope(self):
        for i in range(self.na):
            self.m.getAtom(i).setIsotope(0)

    def has_isotope(self):
        return np.any([self.m.getAtom(i).isotope() for i in range(self.na)])

    def has_standalone_charge(self):
        hsc = False
        iasc = self.ias[self.chgs!=0]
        nac = len(iasc)
        if nac > 0:
            chgsc = self.chgs[iasc]
            gc = self.g[iasc][:,iasc]
            cliques = find_cliques(gc)
            for csi in cliques:
                if np.sum(chgsc[csi])!=0:
                    hsc = True
                    break
        return hsc


    def is_radical(self):
        return np.any([self.m.getAtom(i).radical() for i in range(self.na)])

    def tocan(self, nostereo=True, aromatize=True):
        """
        one may often encounter cases that different SMILES
        correspond to the same graph. By setting aromatize=True
        aliviates such problem
        E.g., C1(C)=C(C)C=CC=C1 .eqv. C1=C(C)C(C)=CC=C1
        """
        m2 = self.copy()
        if nostereo: m2.clearStereocenters()
        if aromatize: m2.aromatize()
        can = m2.canonicalSmiles()
        return can

