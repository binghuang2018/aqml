# -*- coding: utf-8 -*-

from rdkit import Chem
from rdkit.Chem import AllChem, rdMolTransforms, rdMolDescriptors
from rdkit.Chem import rdDepictor
from rdkit.Chem import ChemicalForceFields as cff
import rdkit.Chem.rdForceFieldHelpers as rcr
from rdkit.Geometry.rdGeometry import Point3D
import os, sys, re
import numpy as np
import ase.io as aio
import io2, ase, copy
import tempfile as tpf
#import cheminfo.openbabel.obabel as cib
import cheminfo.math as cim
import cheminfo.graph as cg
import cheminfo.core as cc
from cheminfo.molecule.elements import Elements
from cheminfo.rw.ctab import *
import scipy.spatial.distance as ssd
import itertools as itl
import tempfile as tpf
import multiprocessing
#import copy_reg
#import types as _TYPES

global cnsDic, bt2bo, bo2bt, h2kc, c2j, dsHX, _hyb
cnsDic = {5:3, 6:4, 7:3, 8:2, 9:1, 14:4, 17:1}

bt2bo = { Chem.BondType.SINGLE:1.0,
          Chem.BondType.DOUBLE:2.0,
          Chem.BondType.TRIPLE:3.0,
          Chem.BondType.AROMATIC:1.5,
          Chem.BondType.UNSPECIFIED:0.0}

bo2bt = { '1.0': Chem.BondType.SINGLE,
          '2.0': Chem.BondType.DOUBLE,
          '3.0': Chem.BondType.TRIPLE,
          '1.5': Chem.BondType.AROMATIC,
          '0.0': Chem.BondType.UNSPECIFIED}

uo = io2.Units()
h2kc = uo.h2kc
c2j = uo.c2j

dsHX = {5:1.20, 6:1.10, 7:1.00, 8:0.98, 9:0.92, 14:1.48, 15:1.42, 16:1.34, 17:1.27}

_hyb = { Chem.rdchem.HybridizationType.SP3: 3, \
            Chem.rdchem.HybridizationType.SP2: 2, \
            Chem.rdchem.HybridizationType.SP: 1, \
            Chem.rdchem.HybridizationType.UNSPECIFIED: 0}

T,F = True, False

## register instance method
## otherwise, the script will stop with error:
## ``TypeError: can't pickle instancemethod objects
#def _reduce_method(m):
#    if m.im_self is None:
#        return getattr, (m.im_class, m.im_func.func_name)
#    else:
#        return getattr, (m.im_self, m.im_func.func_name)
#copy_reg.pickle(_TYPES.MethodType, _reduce_method)


import importlib as iml
def reload(module):
    iml.reload(module)

def smi2patt(smi):
    """ convert SMILES to coordnation number retained pattern
    C=CC=C --> [#6;X3]~[#6;X3]~[#6;X3]~[#6;X3]
    =============
    Note this is necessary as RDKit will not find any substructure
    like C=CC=C in a target molecule c1ccccc1C=C!

    =============
    RDKit cannot find C=[NH]=C substructure in a query mol C=CC=[NH]=C,
    so one has to give up on this; OpenEye can do this, but license has
    to be renewed anually!
    """
    # By default, RDKit won't recognize "C=[NH]=C",
    # setting sanitize=F recitify this
    _m = Chem.MolFromSmiles(smi) #, sanitize=False)
    zs = []; cns = []; repls = []; atypes = []
    for ai in _m.GetAtoms():
        zi = ai.GetAtomicNum()
        cni = ai.GetTotalDegree()
        # here we use '<' & '>' instead of '[' & ']'
        # is due to the fact that we need to sequentially
        # replace the content within [] by `repl
        repls.append( '<#%d;X%d>'%(zi,cni) )
        atypes.append( '%02d;X%d'%(zi,cni) )
        zs.append( zi )
        cns.append( cni )
    zs = np.array(zs,dtype=int)
    na = len(zs)
    assert np.all(zs>1), '#ERROR: found H?'
    for bi in _m.GetBonds():
        bi.SetBondType(bo2bt['1.0'])
        # The line below is necessary! If not, we may end up with
        # smarts like '[#6]12:[#6]:[#6]:[#6]=1:[#6]:[#6]:2', originating
        # from input SMILES 'C12C=CC=1C=C2'
        bi.SetIsAromatic(False)
    sma = Chem.MolToSmarts(_m)
    #print ' repls = ', repls
    for i in range(na):
        sma = re.sub('\[.*?\]', repls[i], sma, count=1)
    #print ' sma = ', sma
    patts = [ '<', '>', '-'] #'-\[', '\]-' ]
    repls = [ '[', ']', '~'] #'~[', ']~' ]
    n = len(patts)
    for i in range(n):
        sma = re.sub(patts[i], repls[i], sma)
    return atypes, sma

def _get_angles_csp2(m):
    """ get all angles with the central atom being sp2-hybridized
    This is necessary when optimizing geometry (due to a bug of rdkit):
    it may happen that the initial geometry of C=CC=C is totally
    insane but a local minima (determined by forcefield), i.e., angle(H-C-C) = 90.
    By setting a reasonable range of these angles can circumvent this issue.
    """
    sma = '[$([cX3](:*):*),$([cX2+](:*):*),$([CX3]=*),$([CX2+]=*)]' # corresponds to any C-sp2
    q = Chem.MolFromSmarts(sma)
    matches = m.GetSubstructMatches(q)
    iass3 = []
    for match in matches:
        j = match[0]
        ma = m.GetAtomWithIdx(j)
        neighbors = ma.GetNeighbors()
        nneib = len(neighbors)
        if nneib > 1:
            for i0 in range(nneib):
                for k0 in range(i0+1,nneib):
                    i, k = [ neighbors[i0].GetIdx(), neighbors[k0].GetIdx() ];
                    ias = [i,j,k]
                    if (ias not in iass3) or (ias[::-1] not in iass3):
                        iass3.append(ias)
    return iass3


def _get_ring_nodes(m, namin=3, namax=9, remove_redudant=T):
    """
    get nodes of `namin- to `namax-membered ring

    We focus on those nodes which constitute the
    `extended smallest set of small unbreakable fragments,
    including aromatic rings, 3- and 4-membered rings
    (accompanied with high strain typically)
    """
    # first search for rings
    sets = []
    for i in range(namin, namax+1):
        #if i in [3,4,5]:
        pat_i = '*~1' + '~*'*(i-2) + '~*1'
        #else:
        #    pat_i = '*:1' + ':*'*(i-2) + ':*1'
        Qi = Chem.MolFromSmarts( pat_i )
        for tsi in m.GetSubstructMatches(Qi):
            set_i = set(tsi)
            if set_i not in sets:
                sets.append( set(tsi) )
    if remove_redudant:
        # now remove those rings that are union of smaller rings
        n = len(sets)
        sets_remove = []
        ijs = itl.combinations( list(range(n)), 2 )
        sets_u = []
        for i,j in ijs:
            set_ij = sets[i].union( sets[j] )
            if (set_ij in sets) and (set_ij not in sets_remove):
                sets_remove.append( set_ij )
        sets_u = cim.get_compl(sets, sets_remove)
    else:
        sets_u = sets
    return sets_u



def merge_conformers(mols):
    """ merge serveral molecules that are conformers into one """
    new = Chem.Mol(mols[0])
    new.RemoveAllConformers()
    nc = len(mols)
    for i in range(nc):
        ci = mols[i].GetConformer(-1)
        new.AddConformer(ci, assignId=True)
    return new


class molecules(object):
    """
    a list of molecules
    """
    def __init__(self, objs, props=[]):
        nas = []
        zs = []
        coords = []
        nsheav = []
        ys = []
        iprop = T if len(props)>0 else F
        for obj in objs:
            assert isinstance(obj,str)
            assert os.path.isfile(obj)
            fmt = obj[-3:]
            if fmt in ['sdf','pdb']:
                objc = RDMol(obj)
                _zs = objc.zs
                _na = objc.na
                _coords = objc.coords
                _nheav = objc.nheav
                _ydic = objc.prop
            else:
                print(' FORMAT not supported')
                raise #'#ERROR: '
            zs += list(_zs); coords += list(_coords)
            nas.append(_na); nsheav.append(_nheav)
            ysi = []
            if len(props)>0:
                assert len(list(_ydic.keys()))>0, "#ERROR: xyz file format incorrect! E.g., `-100.0 #E`"
                ysi = [ _ydic[p] for p in props ]
            ys.append(ysi)
        self.nas = np.array(nas,np.int)
        self.zs = np.array(zs,np.int)
        self.coords = np.array(coords)
        self.nsheav = np.array(nsheav,np.int)
        self.ys = np.array(ys)
        #self.ias2 = np.cumsum(self.nas)
        #self.ias1 =




class SMILES(object):

    def __init__(self, smi):
        self.normal = T
        mchs = list( set( re.findall('\[[nN]H(\d?)\]', smi) ) )
        ndic = {'':1, '1':1, '2':2, '3':3}
        self.i_remove_isotope = F
        if len(mchs) > 0:
            self.i_remove_isotope = T
            for patt in mchs:
                repl = 'N' + '([2H])'*ndic[patt]
                smi = re.sub('\[[nN]H%s\]'%patt, repl, smi)
        self.smiles = smi
        self.m = Chem.MolFromSmiles(self.smiles, sanitize=F)
        try:
            Chem.SanitizeMol(self.m)
        except:
            self.fixN5()

    @property
    def ias(self):
        if not hasattr(self, '_ias'):
            self._ias = np.arange( self.m.GetNumAtoms() )
        return self._ias

    @property
    def atoms(self):
        if not hasattr(self, '_atoms'):
            self._atoms = [ ai for ai in self.m.GetAtoms() ]
        return self._atoms

    @property
    def pls(self):
        """ get minimal path length matrix """
        if not hasattr(self, '_pls'):
            self._pls = Chem.GetDistanceMatrix(self.m)
        return self._pls

    def _clone(self):
        return copy.deepcopy(self.m)

    @property
    def chgs(self):
        if not hasattr(self, '_chgs'):
            self._chgs = np.array([ ai.GetFormalCharge() for ai in self.atoms ], dtype=int)
        return self._chgs

    @property
    def zs(self):
        if not hasattr(self, '_zs'):
            self._zs = np.array([ ai.GetAtomicNum() for ai in self.atoms ], dtype=int)
        return self._zs

    @property
    def bom(self):
        if not hasattr(self, '_bom'):
            self._bom = get_bom(self.m)
        return self._bom

    @property
    def iasN5(self):
        if not hasattr(self, '_iasN5'):
            self._iasN5 = self.ias[ np.logical_and(self.bom.sum(axis=0)>3, self.zs==7) ]
        return self._iasN5

    def fixN5(self):
        """ fix N atom with 5 total valencies (produced
        by amons generation algorithm based on OEChem),
        E.g., convert say A=N(R)=B to [A+][N+](R)=B

        Note that the use of N5 is due to the unique nature
        of SMILES string like A~C=N=C~B without the need to distinguish
        between A-[C+]-[NH+]=C~B and A~C=[NH+][C-]~B """
        smi = None
        _bom = self.bom.copy()
        _chgs = self.chgs.copy()
        for ia in self.iasN5:
            ai = self.atoms[ia]
            for bi in ai.GetBonds():
                ia1, ja1 = bi.GetBeginAtomIdx(), bi.GetEndAtomIdx()
                ja = ia1 if ja1 == ia else ja1
                aj = self.atoms[ja]
                assert ja != ia
                if _bom[ia,ja] == 2: # re-assign BO to 1 for the first double bond found
                    _bom[ia,ja] = _bom[ja,ia] = 1
                    _chgs[ia] = 1
                    _chgs[ja] = -1
                    bi.SetBondType( bo2bt['1.0'] )
                    ai.SetFormalCharge(1)
                    aj.SetFormalCharge(-1)
                    break
        self._bom = _bom
        self._chgs = _chgs
        if self.i_remove_isotope:
            self.remove_isotope()
        try:
            Chem.SanitizeMol(self.m)
            smi = Chem.MolToSmiles(self.m, canonical=T)
        except:
            raise Exception(':: fixN5() failed??')
        self.smiles = smi

    def remove_isotope(self):
        for ai in self.atoms:
            ai.SetIsotope(0)

    def nbrs(self, ia):
        _nbrs = []
        for ai in self.atoms[j].GetNeighbors():
            _nbrs.append(ai.GetIdx())
        return _nbrs

    def fix_aromaticN(self):
        m = self._clone()
        try:
            m.UpdatePropertyCache(False)
            cp = Chem.Mol(m.ToBinary())
            Chem.SanitizeMol(cp)
            m = cp
            iok = True
        except ValueError:
            nm = Normalize.AdjustAromaticNs(m)
            iok = False
            if nm is not None:
                Chem.SanitizeMol(nm)
                #print 'fixed:',Chem.MolToSmiles(nm)
                smi = Chem.MolToSmiles(nm)
                iok = True
            else:
                print('still broken:',smi)
        self.smi = smi
        self.iok = iok

    @staticmethod
    def _FragIndicesToMol(oMol,indices):
        em = Chem.EditableMol(Chem.Mol())
        newIndices={}
        for i,idx in enumerate(indices):
            em.AddAtom(oMol.GetAtomWithIdx(idx))
            newIndices[idx]=i

        for i,idx in enumerate(indices):
            at = oMol.GetAtomWithIdx(idx)
            for bond in at.GetBonds():
                if bond.GetBeginAtomIdx()==idx:
                    oidx = bond.GetEndAtomIdx()
                else:
                    oidx = bond.GetBeginAtomIdx()
                # make sure every bond only gets added once:
                if oidx<idx:
                    continue
                em.AddBond(newIndices[idx],newIndices[oidx],bond.GetBondType())
        res = em.GetMol()
        res.ClearComputedProps()
        Chem.GetSymmSSSR(res)
        res.UpdatePropertyCache(False)
        res._idxMap=newIndices
        return res

    @staticmethod
    def _recursivelyModifyNs(mol,matches,indices=None):
        if indices is None:
            indices=[]
        res=None
        while len(matches) and res is None:
            tIndices=indices[:]
            nextIdx = matches.pop(0)
            tIndices.append(nextIdx)
            nm = Chem.Mol(mol.ToBinary())
            nm.GetAtomWithIdx(nextIdx).SetNoImplicit(True)
            nm.GetAtomWithIdx(nextIdx).SetNumExplicitHs(1)
            cp = Chem.Mol(nm.ToBinary())
            try:
                Chem.SanitizeMol(cp)
            except ValueError:
                res,indices = Normalize._recursivelyModifyNs(nm,matches,indices=tIndices)
            else:
                indices=tIndices
                res=cp
        return res,indices

    def AdjustAromaticNs(m, Npatt='[n&D2&H0;r5,r6]'):
        """
           default nitrogen pattern matches Ns in 5 rings and 6 rings in order to be able
           to fix: O=c1ccncc1
        """
        Chem.GetSymmSSSR(m)
        m.UpdatePropertyCache(False)

        # break non-ring bonds linking rings:
        em = Chem.EditableMol(m)
        linkers = m.GetSubstructMatches(Chem.MolFromSmarts('[r]!@[r]'))
        plsFix=set()
        for a,b in linkers:
            em.RemoveBond(a,b)
            plsFix.add(a)
            plsFix.add(b)
        nm = em.GetMol()
        for at in plsFix:
            at=nm.GetAtomWithIdx(at)
            if at.GetIsAromatic() and at.GetAtomicNum()==7:
                at.SetNumExplicitHs(1)
                at.SetNoImplicit(True)

        # build molecules from the fragments:
        fragLists = Chem.GetMolFrags(nm)
        frags = [Normalize._FragIndicesToMol(nm,x) for x in fragLists]

        # loop through the fragments in turn and try to aromatize them:
        ok=True
        for i,frag in enumerate(frags):
            cp = Chem.Mol(frag.ToBinary())
            try:
                Chem.SanitizeMol(cp)
            except ValueError:
                matches = [x[0] for x in frag.GetSubstructMatches(Chem.MolFromSmarts(Npatt))]
                lres,indices = Normalize._recursivelyModifyNs(frag,matches)
                if not lres:
                    #print 'frag %d failed (%s)'%(i,str(fragLists[i]))
                    ok=False
                    break
                else:
                    revMap={}
                    for k,v in frag._idxMap.items():
                        revMap[v]=k
                    for idx in indices:
                        oatom = m.GetAtomWithIdx(revMap[idx])
                        oatom.SetNoImplicit(True)
                        oatom.SetNumExplicitHs(1)
        if not ok:
            return None
        return m


    def neutralise(self):
        """
        neutralise the SMILES of a molecule

        Attention: the order of atoms is not retained!!
        """
        smi = self.smiles

        patts = [
            # Imidazoles
            ('[n+;H]','n'),
            # Amines
            ('[N+;!H0]','N'),
            # Carboxylic acids and alcohols
            ('[$([O-]);!$([O-][#7])]','O'),
            # Thiols
            ('[S-;X1]','S'),
            # Sulfonamides
            ('[$([N-;X2]S(=O)=O)]','N'),
            # Enamines
            ('[$([N-;X2][C,N]=C)]','N'),
            # Tetrazoles
            ('[n-]','[nH]'),
            # Sulfoxides
            ('[$([S-]=O)]','S'),
            # Amides
            ('[$([N-]C=O)]','N')  ]

        reactions = [(Chem.MolFromSmarts(x),Chem.MolFromSmiles(y,False)) for x,y in patts]

        m = Chem.MolFromSmiles(smi)
        for i,(reactant, product) in enumerate(reactions):
            while m.HasSubstructMatch(reactant):
                rms = AllChem.ReplaceSubstructs(m, reactant, product)
                m = rms[0]

        # it doesn't matter is u choose to output a canonical smiles as the
        # sequence of atoms is changed calling `AllChem.ReplaceSubstructs
        self.smiles = Chem.MolToSmiles(m, isomericSmiles=False) #, canonical=False)



class NBody(object):
    """
    This class is slightly different compared to <cheminfo.molecule.nbody>
    """
    def __init__(self, m, wH=T, key='z'):
        self.m = m
        self.zs = [ ai.GetAtomicNum() for ai in m.GetAtoms() ]
        self._zs = np.array(self.zs,dtype=int)
        self.wH = wH
        self.key = key

    @property
    def clone(self):
        return copy.deepcopy(self.m)

    def get_atypes(self):
        """ atomic type
        of format: '%02d;X%d'%(Z,CN) """
        atypes = {}
        for ai in self.m.GetAtoms():
            ia = ai.GetIdx()
            zi = ai.GetAtomicNum()
            assert zi < 100
            cni = ai.GetTotalDegree()
            atypes[ia] = '%02d;X%d'%(zi,cni)
        self.atypes = atypes

    @property
    def dangs(self):
        if not hasattr(self, '_dangs'):
            self._dangs = self.get_dihedral_angles()
        return self._dangs

    def get_dihedral_angles(self):
        """ get torsion types & dihedral angles
        returns a dictionary {'1-6-6-8':[100.], ...}
        Attention:
        ====================================================
        Only torsions made up of heavy atoms are considered
        even if wH is set to T (fix this in the future)

        vars
        ====================================================
        key:  'z',
              'ia', idx of atom as key
        """
        mol = self.m
        c1 = mol.GetConformer(-1)
        torsma = '[!$(*#*)&!D1]~[!$(*#*)&!D1]'
        q = Chem.MolFromSmarts(torsma)
        matches = mol.GetSubstructMatches(q)
        nmat = len(matches)
        dic = {}
        for match in matches:
            j = match[0]
            k = match[1]
            bond = mol.GetBondBetweenAtoms(j, k)
            aj = mol.GetAtomWithIdx(j)
            ak = mol.GetAtomWithIdx(k)
            hj, hk = [ _hyb[_a.GetHybridization()] for _a in [aj,ak] ]
            iok1 = ( hj not in [2,3] )
            iok2 = ( hk not in [2,3] )
            if iok1 or iok2: continue
            for b1 in aj.GetBonds():
                if (b1.GetIdx() == bond.GetIdx()):
                    continue
                i = b1.GetOtherAtomIdx(j)
                for b2 in ak.GetBonds():
                    if (b2.GetIdx() == bond.GetIdx()) or (b2.GetIdx() == b1.GetIdx()):
                        continue
                    l = b2.GetOtherAtomIdx(k)
                    # skip 3-membered rings
                    if (l == i):
                        continue
                    _dang = rdMolTransforms.GetDihedralDeg(c1, i,j,k,l)
                    dang = abs(_dang)
                    assert dang <= 180.0
                    ias4 = (i,j,k,l)
                    if not self.wH:
                        if np.any([ self.zs[iaa]==1 for iaa in ias4 ]):
                            continue
                    if self.key in ['z']:
                        #print('atsi=',ias4, 'zsi=', [_zs[iaa] for iaa in ias4])
                        zi,zj,zk,zl = [ self.zs[iaa] for iaa in ias4 ]
                        if (zj==zk and zi>zl) or (zj>zk):
                            ias4 = (l,k,j,i)
                        #torsions.append(ias4)
                        #_zi,_zj,_zk,_zl = [ zs[_] for _ in ias4 ]
                        #typez = '%d-%d-%d-%d'%(_zi,_zj,_zk,_zl)
                        type4 = tuple([self.zs[iaa] for iaa in ias4])
                        if type4 in list(dic.keys()):
                            dic[type4] += [dang]
                        else:
                            dic[type4] = [dang]
                    elif self.key in ['ia','i']:
                        type4 = ias4
                        dic[type4] = dang
                    else:
                        raise Exception('#unknown key')
        return dic

    @property
    def angs(self):
        if not hasattr(self, '_angs'):
            self._angs = self.get_angles()
        return self._angs

    def get_angles(self):
        m2 = self.m
        c1 = m2.GetConformer(-1)
        zs = [ ai.GetAtomicNum() for ai in m2.GetAtoms() ]
        _zs = np.array(zs,dtype=int)
        _atoms = m2.GetAtoms()
        na = len(_atoms)
        dic = {}; iass3 = []
        for j in range(na):
            neighbors = _atoms[j].GetNeighbors()
            nneib = len(neighbors)
            if nneib > 1:
                for i0 in range(nneib):
                    for k0 in range(i0+1,nneib):
                        i, k = [ neighbors[i0].GetIdx(), neighbors[k0].GetIdx() ];
                        ias = (i,j,k)
                        if not self.wH:
                            if np.any(self._zs[[i,j,k]] == 1):
                                continue
                        if (ias in iass3) or (ias[::-1] in iass3): continue
                        iass3.append(ias)
                        _ang = rdMolTransforms.GetAngleDeg(c1, i,j,k)
                        ang = abs(_ang)
                        assert ang <= 180.0
                        if self.key in ['i','ia']:
                            dic[ias] = ang
                        else:
                            raise Exception('#not implemented')
        return dic


class RDMol(object):
    """
    enhanced RDKit molecule class with extended functionality
    """
    def __init__(self, obj, isortH=F, forcefield='mmff94', doff=F, \
                 steps=500, ih=T, kekulize=F, sanitize=T, wH4mb=F, \
                 debug=F):
        self.forcefield = forcefield
        self.ih = ih
        self.steps = steps
        self.kekulize = kekulize
        self.debug = debug

        ismi = False
        hasCoord = True
        prop = {}
        self.keep_parent = F
        if type(obj) is str:
            if os.path.exists(obj):
                fmt = obj[-3:]
                assert fmt in ['sdf','mol','pdb']
                # default:  `strictParsing=True
                # then error like this shows up
                # [06:01:51] CTAB version string invalid at line 4
                if fmt in ['sdf','mol']:
                    #m1 = Chem.MolFromMolFile(obj,removeHs=False, \
                    #                         strictParsing=False)
                    _ = Chem.SDMolSupplier(obj,removeHs=False)
                    m1 = next(_)
                    prop = m1.GetPropsAsDict()
                elif fmt in ['pdb',]:
                    # some pdb file may not contain H
                    m1 = Chem.MolFromPDBFile(obj,removeHs=False)
                else:
                    print(' #ERROR: format not recognized')
                    raise
                mu = m1
            else:
                ismi = True
                _sobj = SMILES(obj)
                #m = Chem.MolFromSmiles(obj,sanitize=sanitize)
                m = _sobj.m
                #m2 = Chem.RemoveHs(m)
# H's always appear after heavy atoms
                mu = Chem.AddHs(m) if ih else m
                #if doff:
                if AllChem.EmbedMolecule(mu):
                    print('RDKit failed to embed, use openbabel instead!')
                    import cheminfo.openbabel.obabel as cib
                    s2 = cib.Mol(obj, make3d=True)
                    self.keep_parent = T # in this case, keep the parent geom as the only conformer
                    mu = s2.to_RDKit()
                m1 = mu
                #hasCoord = False
        elif obj.__class__.__name__ == 'Mol': # rdkit.Chem.rdchem.Mol
            mu = obj
            m1 = mu
        else:
            print('#ERROR: non-supported type')
            raise

        self.prop = prop

        zs = [ ai.GetAtomicNum() for ai in m1.GetAtoms() ]
        if 1 in  zs:
            ih1 = zs.index(1)
            ihok = np.all( zs[ih1:]==1 ) # do all H's appear at the end?
            if isortH:
# if `ihok is not True, otherwise wierd molecules result when generating amons
                if not ihok:
                    #mu = self.sort()
                    print('#ERROR: H not sorted')
                    raise

        self.bom = get_bom(mu, kekulize=kekulize)
        self.cns = (self.bom > 0).sum(axis=0)
        self.na = mu.GetNumAtoms()
        self.nb = mu.GetNumBonds()
        self.ias = np.arange(self.na).astype(np.int)
        self.zs = np.array([ ai.GetAtomicNum() for ai in mu.GetAtoms() ], np.int)
        self.ias_heav = self.ias[ self.zs > 1 ]
        self.nheav = len(self.ias_heav)

        obj = NBody(mu, wH=wH4mb, key='ia')
        self.dangs0 = obj.dangs

        self.iFFOpt = False # geom optimized by FF ?
        if doff:
            # sometimes u encounter error messages like
            # """ ValueError: Bad Conformer Id """
            # if u use RDKit
            # Here we use openbabel to get initial geometry
            hasCoord = True
            if ismi:
                self.m = mu
                self.optg()

        self.m0 = copy.deepcopy(mu)
        if not ih:
            mu = Chem.RemoveHs(mu)
        self.m = mu
        self.coords = get_coords(mu)
        # get formal charges
        self.chgs = np.array([ ai.GetFormalCharge() for ai in self.m0.GetAtoms() ])

    @property
    def ds(self):
        if not hasattr(self, '_ds'):
            self._ds = ssd.squareform( ssd.pdist(self.coords) )
        return self._ds

    def clone(self):
        return copy.deepcopy(self.m)

    def get_subm(self, nodes):
        """ useful for diagnose when genearting amons """
        na1 = len(nodes)
        bonds = []
        for i in range(na1):
            for j in range(i+1,na1):
                if self.bom[i,j] > 0:
                    bonds.append([i,j])
        return nodes,bonds

    def update_bom(self):
        """update bom based on `chgs
        e.g., C=N#N, bond orders = [2,3],
        Considering that `chgs = [0,+1,-1],
        bond orders has to be changed to [2,2]"""
        bom2 = copy.copy(self.bom)
        ias1 = self.ias[self.chgs == 1]
        vs2 = copy.copy(self.vs)
        for i in ias1:
            iasc = self.ias[ np.logical_and(self.chgs==-1, self.bom[i]>0) ]
            nac = len(iasc)
            if nac > 0:
                #assert nac == 1
                j = iasc[0]
                bij = self.bom[i,j] - 1
                bom2[i,j] = bij
                bom2[j,i] = bij
                vs2[i] = vs2[i]+1; vs2[j] = vs2[j]+1
        self.bom = bom2
        self.vs = vs2

    def get_properties(self):
        """
        get various useful properties of atoms in molecule
        for amon geneartion
        """
        assert self.kekulize, '#ERROR: u need to get explicit BOs for amon generation'
        self.vs = np.array([ ai.GetTotalValence() for ai in self.m0.GetAtoms() ], np.int)
        #self.update_bom()
        self.ias_heav = self.ias[ self.zs > 1 ]
        bom_heav = self.bom[ self.ias_heav, : ][ :, self.ias_heav ]
        self.vs_heav = bom_heav.sum(axis=0)
        self.cns_heav = ( bom_heav > 0 ).sum(axis=0)
        self.nhs = self.vs[:self.nheav] - self.vs_heav - self.chgs[:self.nheav]
        self.dvs = self.vs_heav - self.cns_heav
        self.hybs = np.array([ _hyb[ai.GetHybridization()] for ai in self.m.GetAtoms() ])

    def get_ab(self):
        """
        get atoms and bonds info
        a2b: bond idxs associated to each atom
        b2a: atom idxs associated to each bond
        """
        # it's not necessary to exclude H's here as H's apprear at the end
        b2a = [] #np.zeros((self.nb,2), np.int)
        ibs = []
        for bi in self.m.GetBonds():
            i, j = bi.GetBeginAtomIdx(), bi.GetEndAtomIdx()
            if self.zs[i] > 1 and self.zs[j] > 1:
                ib_heav = bi.GetIdx()
                b2a.append( [i,j] )
        assert len(b2a) == ib_heav+1, '#ERROR: not all H apprear at the end?'
        b2a = np.array(b2a, np.int)

        a2b = -np.ones((self.nheav, 6), np.int)
        for ia in self.ias_heav:
            ai = self.m.GetAtomWithIdx(ia)
            icnt = 0
            for bi in ai.GetBonds():
                ib = bi.GetIdx()
                if ib <= ib_heav: #np.all( self.zs[b2a[ib]] > 1 ):
                    a2b[ia, icnt] = ib
                    icnt += 1
        return a2b, b2a

    def get_ds(self):
        # get interatomic distance matrix
        self.ds = cdist(self.coords)

    def get_atoms_within_radius(self, rcut, centers):
        self.get_ds()
        ds = self.ds
        iast = np.arange(self.na)
        for i,center in enumerate(centers):
            ias = iast[ ds[i] <= rcut ]
        # to be continued
        return

    def get_ring_nodes(self, namin=3, namax=9, remove_redudant=T):
        return _get_ring_nodes(self.m, namin=namin, namax=namax, remove_redundant=remove_redundant)

    def get_angles_csp2(self):
        return _get_angles_csp2(self.m)

    def get_conjugated_nodes(self):
        """
        get sets of conjugated nodes
        """
        sets = []
        self.get_backbone()
        m = self.mbb
        for bi in m.GetBonds():
            #print ' -- idx = ', bi.GetIdx()
            n = len(sets)
            iconj = bi.GetIsConjugated()
            ins = ( bt2bo[ bi.GetBondType() ] > 1 ) # is non-single bond?
            if iconj or ins:
                ia1, ia2 = bi.GetBeginAtomIdx(), bi.GetEndAtomIdx()
                set_i = set([ia1, ia2])
                if n == 0:
                    sets.append( set_i )
                else:
                    for j, set_j in enumerate(sets):
                        if set_i.intersection( set_j ) > set([]):
                            sets[j].update( set_i )
                        else:
                            if set_i not in sets: sets.append( set_i )
            #print '-- sets = ', sets
        sets_u = cim.merge_sets(sets)
        return sets_u

    def get_ring_smiles(self, namin=3, namax=9):
        sets = self.get_rings(namin=namin, namax=namax)
        for set_i in sets:
            # now get the adjacent nodes that u cannot
            # detach from the ring structure, i.e., atoms
            # that are doubly bonded to any node of the ring
            set_j = list(set_i)
        return

    def get_rigid_nodes(self):
        """
        rigid nodes include two types of nodes:
        1) ring with size in range [3,6], note that nodes in any
           such ring are considered to be rigid while the nodes
           attached to such ring are not rigid unless the ring is
           aromatic
        2) conjugate functional groups, e.g.,
            -C(=O)N-
            -C=C-C=C-
            -C=N#N
            ...
        """
        # conjugate nodes
        #
        sets_c = self.get_conjugated_nodes() # sets_c

        # Part (1)

        # Now we keep only non-aromatic rings as nodes in aromatic
        # rings are all included in `sets_c
        sets_r0 = self.get_ring_nodes(namin=3, namax=6)
        sets_r1 = []
        for set_i in sets_r0:
            # some rings may be a subset of set in sets_c, e.g., ring c1ccccc1 is
            # a subgraph of the conj structure c1ccccc1C=C. In this case, skip
            if np.any( [ set_i <= set_j for set_j in sets_c ] ):
                continue
            sets_r1.append( set_i )
        # Note that ring such as C1C=CCC1 will be present in `sets_r
        # in spite of the fact that it may share two common nodes
        # as c1cccccc1 in molecule c1c2(CCC2)cccc1. This won't be a
        # problem as later we'll merge those nodes again in Part (3).
        sets_r = cim.merge_sets( sets_r1 )


        # Part (2)

        # Now add rigid neighboring atoms
        # `sets_c2 is of the same size as `sets_c. Each entry of `sets_c2
        # corresponds to a set of nodes that are neighbors (half-rigid,
        # i.e., only that node atom is rigid because it's bonded to a set
        # of rigid atoms, while the atoms attached to that atom is
        # non-rigid if 1) they are H's; or 2) they are heavy atom and
        # not in `set_i which shares the same idx as the aforementioned
        # entry of `sets_c2.
        sets_c2 = []
        filt1 = ( self.zs > 1 )
        filt2 = ( self.zs == 1 )
        ias = np.arange(self.na)
        for i, set_i in enumerate(sets_c):
            ias1 = list(set_i)
            set_j = set([])
            for ia in ias1:
                filt3 = ( self.bom[ia,:] > 0 )
                ias_heav = ias[ np.logical_and(filt1, filt3) ]
                ias_H = ias[ np.logical_and(filt2, filt3) ]
                # neighboring heavy atoms and hydrogens are treated
                # differently as more non-rigid atoms can be bonded
                # to the heavy atoms
                sets_c[i].update( list(ias_H) )
                set_j.update( set(ias_heav).difference( set_i ) )
            if set_j not in sets_c2: sets_c2.append( set_j)

        # Part (3)

        # update sets_c & sets_c2
        # some rings
        for set_i in sets_r:
            iupd = False
            for j, set_j in enumerate(sets_c):
                if set_i.intersection(set_j) > set([]):
                    sets_c[i].update( set_j ); iupd = True
            if not iupd:
                sets_c.append( set_i )
                sets_c2.append( set([]) )
        return sets_c, sets_c2, sets_r


    def get_many_bodies(self, PLmax, wH=True):
        """
        get the idxs for all 2- and 3-body terms
        To be used for SLATM generation

        var's
        =================
        PLmax -- the cutoff radius
        #nmax  -- the maximal number of neighbors, used
        #         for zero padding if there is less than
        #         `nmax neighbors
        """
        if wH and (not self.ih):
            m = Chem.AddHs(self.m)
        PLs = get_PLs(m)
        iast = np.arange(self.na)

        # sets_c, sets_c2, sets_r
        sets1, sets2, sets3 = self.get_rigid_nodes() #

        n1 = len(sets1)
        neibs2 = [] # for 2-body terms
        neibs3 = [] # for 3-body terms
        for i in range(n1):
            set_i = list( cim.union( [sets1[i], sets2[i]] ) )
            for j in set_i:
                js = np.setdiff1d(set_i, [j])
                js_u = js[ PLs[j, js] <= PLmax ]
                for ju in js_u:
                    jsi = [j,ju]
                    if jsi not in neibs2:
                        neibs2.append(jsi)
                for ks in itl.combinations(js_u, 2):
                    k, l = ks
                    for tsi in [ [j,k,l], [j,l,k] ]:
                        if tsi not in neibs3:
                            neibs3.append( tsi )

        # get 2- and 3-body terms for the remaining nodes & check
        # if some 3-body terms are missing for the rigid nodes
        hrnodes = list( cim.union( sets2 + sets3 ) )
        # now consider nodes that are flexible
        fnodes = np.setdiff1d( np.arange(self.na), cim.union( sets1 + sets2 + sets3 ) )
        nodes = hrnodes + list( fnodes )
        for i in range(self.na): #nodes:
            PLsi = PLs[i]
            js = iast[ np.logical_and(PLsi <= 2, PLsi > 0) ]
            for j in js:
                jsi = [i,j]
                if jsi not in neibs2:
                    neibs2.append(jsi)
            for ks in itl.combinations(js, 2):
                j,k = ks
                PLsi = np.array([ PLs[i,j], PLs[i,k], PLs[j,k] ])
                PLsi.sort()
                if np.all(np.array(PLsi) == np.array([1,1,2])):
                    for tsi in [ [i,j,k], [i,k,j] ]:
                        if tsi not in neibs3:
                            neibs3.append( tsi )
        neibs2.sort()
        neibs3.sort()
        return neibs2, neibs3

    def get_building_blocks(self, level=1, debug=False):
        """
        The building blocks are the smallest fragments

        var's
        ==============================
        level -- 0 : break standalone single bonds only, that
                     is, don't touch aromatic bonds and bonds
                     in rings (3-, 4-, ..., 9-membered).
                 1 : break all single bonds except armoatic ones
                 2 : break all possible bonds as long as
                     the hybs of atoms in the fragments
                     can be retained.
        """

        def get_aidx_star(dic, ias, kas):
            iat, jat = kas
            if dic[iat] == 0:
                return iat, jat
            elif dic[jat] == 0:
                return jat, iat
            else:
                print('#ERROR:?')
                raise

        def get_aidxs_patt(m0, patt, ias0):
            Qi = Chem.MolFromSmarts( patt )
            zs_i = [ ai.GetAtomicNum() for ai in Qi.GetAtoms() ]
            iass_i = m0.GetSubstructMatches(Qi)
            ias0.sort()
            #print '                   **         ias0 = ', ias0
            iok = False
            for ias in iass_i:
                #print '                   ** matched ias = ', ias
                if set(ias) == set(ias0):
                    iok = True; break
            assert iok
            dic = dict(list(zip(ias, zs_i)))
            return ias, dic

        assert (not self.ih), '#ERROR: pls set `ih=False to get building blocks'
        m1 = copy.deepcopy( self.m )
        Chem.RemoveStereochemistry(m1)

        iars = []
        for ai in m1.GetAtoms():
            iars.append( ai.GetIsAromatic() )

        # first update BO for groups such as amide (-N-C(=O), -O-C(=O), ...
        # that is, we consider that the single bonds in these groups can
        # not be broken. This has to be imposed for predicting mp/bp.
        bom = copy.deepcopy( self.bom )
        # as no single bond in any of ['[N-]=[N+]=C', '[N+]#[C-]', '[N-]=[N+]=N']
        # we skip them here
        for pat_i in [ '[O-][N+](=O)',  ]: #  'NC(=O)', 'OC(=O)'
            Qi = Chem.MolFromSmarts( pat_i )
            for tsi in m1.GetSubstructMatches(Qi):
                i,j,k = tsi
                bij = bom[i,j] + 100 ##
                bjk = bom[j,k] + 100
                bom[i,j] = bom[j,i] = bij
                bom[k,j] = bom[j,k] = bjk

        obsolete = """
        # don't break any ring, as such rigid structure has a dramtic effect
        # on mp prediction, so keep them as much as possible for selection
        # of molecules for training
        nodes_r = self.get_ring_nodes(3,6)
        for nodes_i0 in nodes_r:
            nodes_i = list( nodes_i0 )
            nai = len(nodes_i)
            for i in range(nai-1):
                for j in range(i+1,nai):
                    boij = bom[i,j]
                    if boij > 0:
                        bom[i,j] = bom[j,i] = boij + 0.15
        """

        ## RDKit somehow cannot correctly process '[*;!H]' as a heavy
        ## atom; instead '[*;!#1]' works. A bug??
        heav_smarts = '*;!#1'

        m = Chem.AddHs(m1)
        m.UpdatePropertyCache(False)

        # get bond idxs that can be broken
        # We assume aromatic bonds can be broken; otherwise
        # very few amons can be found for molecules consisting
        # of aromatic atoms
        bom2 = np.triu( bom )
        #ias1, ias2 = np.where( bom2 > 0 ) #
        ias1, ias2 = np.where( np.logical_and( bom2 <= 3, bom2 > 0 ) )
        nb = len(ias1)
        bidxs = []
        for i in range(nb):
            ia1, ia2 = ias1[i], ias2[i]
            bi = m.GetBondBetweenAtoms(ia1, ia2)
            bidx = bi.GetIdx()
            bidxs.append( bidx )
        nb = len(bidxs)
        if nb == 0:
            # no bonds can be broken, i.e., a big aromatic system
            return Chem.MolToSmiles(m)

        bidxs.sort()
        #print ' -- bonds = '
        #print np.array([ias1,ias2]); sys.exit(2)
        self.bidxs = bidxs

        # now get fragments

        # break all bonds with bo = 1
        m2 = Chem.FragmentOnBonds(m, bidxs)
        ts = Chem.MolToSmiles(m2).split('.')

        # vital step
        # if this is not done, a fragment like C([*])([*])([*])
        # will also match >CH-, >CH2, -CH3, which we hope not to happen
        # This is inevitable if we don't substitute "*" by "*;!H"
        # ( H's are present in `m)
        tsU = []
        for ti in ts:
            tsU.append( re.sub('\*', heav_smarts, ti) )
        ts = tsU
        tsU = list( set( ts ) )
        #print ' -- tsU = ', tsU

        if level == 1:
            return tsU
        else:
            iass = []
            mqs = []
            dics = []
            tss = []
            cnodes = []
            for tsi in tsU:
                Qi = Chem.MolFromSmarts( tsi )
                zs_i = []; degrees_i = []
                for ai in Qi.GetAtoms():
                    zs_i.append( ai.GetAtomicNum() )
                    degrees_i.append( ai.GetDegree() )
                naQ = len(zs_i); iasQ = np.arange(naQ)
                dgrmax = max(degrees_i)
                zs_i = np.array(zs_i)
                degrees_i = np.array(degrees_i)
                ics = iasQ[ np.logical_and(degrees_i == dgrmax, zs_i > 1) ]
                if debug: print((' ics, tsi = ', ics, tsi))
                assert len(ics) == 1, '#ERROR: there should be only one heavy atom with maxiaml degree!'
                #ic = ics[0]
                iass_i = m.GetSubstructMatches(Qi)
                for ias in iass_i:
                    #ias = np.array(ias)
                    mqs.append( Qi )
                    tss.append( tsi )
                    dics.append( dict(list(zip(ias, zs_i))) )
                    iass.append( list(ias) )
                    cnodes.append( ias[ics[0]] ) # [ias[ic] for ic in ics] )

            ng = len(iass)
            ts2 = []
            if level == 1.5:
                for i in range(ng-1):
                    ias = iass[i]
                    mi = mqs[i]
                    na1 = len(ias)
                    dic_i = dics[i]
                    for j in range(i+1,ng):
                        mj = mqs[j]
                        jas = iass[j]
                        dic_j = dics[j]
                        kas = list( set(ias).intersection( set(jas) ) )
                        if len(kas) == 2:
                            # get idx of atom in `m corresponding to [*] in `mi and `mj

                            if bom[kas[0],kas[1]] == 0:
                                # C1C=CC(=O)C=C1
                                # 0 1 23  4 5 6 -- atomic index
                                # mi = '[*]C=C[*]', ias = [0,1,2,3]
                                # mj = '[*]C=C[*]', jas = [3,5,6,0]
                                # kas = [3,0] but bom[0,3] = 0, i.e., these two frags cannot bind!
                                continue
                            try:
                                iat, jat = get_aidx_star(dic_i, ias, kas)
                            except:
                                # e.g., [*]O is a frag of [*][N+](=O)[O-]
                                # [*][N+](=O)[O-] [*]O [25, 26]
                                # [24, 25, 27, 26] [25, 26]
                                # {24: 0, 25: 7, 26: 8, 27: 8} {25: 0, 26: 8}
                                continue
                            ia = ias.index(iat); ja = jas.index(jat)
                            mij = Chem.CombineMols(mi,mj)
                            mc = Chem.EditableMol(mij)

                            # reconnect the bond first
                            ia2 = ias.index(jat); ja2 = jas.index(iat)
                            print(('ia2,ja2 = ', ia2,ja2))
                            bij = m.GetBondBetweenAtoms(iat,jat)
                            mc.AddBond(ia2, ja2+na1, bij.GetBondType() ) #rdkit.Chem.rdchem.BondType.SINGLE)

                            # delete the atom in mij
                            ia = ias.index(iat);
                            ldxs = [ia, ja+na1]

                            for l in range(2):
                                mc.RemoveAtom(ldxs[l]-l)

                            mcU = mc.GetMol()
                            #mcU2 = Chem.RemoveHs(mcU)
                            smi = Chem.MolToSmiles( mcU)
                            if '.' in smi:
                                # e.g., [*]C[*] [*]C[*] [19, 21]
                                #       [18, 19, 21] [20, 19, 21]
                                #       {18: 0, 19: 6, 21: 0} {19: 6, 20: 0, 21: 0}
                                #       [*].[*]C[*]
                                continue
                            #if '[*]' not in smi:
                            #    print '\n', tss[i], tss[j], kas
                            #    print ias, jas
                            #    print dic_i, dic_j
                            #    print smi
                            if smi not in ts2: ts2.append(smi)
            elif level == 2:
                # account for all neighbors of any env in
                ifs = list(range(ng))
                for i in ifs:

                    ias = iass[i]
                    mi = mqs[i]; #mic = mqs[i]
                    na1 = len(ias); #na1c = len(ias)
                    dic_i = dics[i]; #dic_ic = dics[i]
                    jfs = list( set(ifs)^set([i]) )

                    if debug: print(('i, mi, ias = ', i, tss[i], ias))
                    #print ' -- i = ', i

                    icnt = 0
                    cnode = cnodes[i]
                    for j in jfs:
                        #print '    icnt = ', icnt
                        mj = mqs[j]
                        jas = iass[j]
                        if debug:
                            print(('   j, mj, jas = ', j, tss[j], jas))
                            if icnt > 0:
                                print('      mi, ias = ', '', patt, ias)
                                print('      dic_i = ', dic_i)
                            else:
                                print('      _mi, ias = ', '', tss[i], ias)
                        dic_j = dics[j]
                        kas = list( set(ias).intersection( set(jas) ) )
                        #print '  -- cnode, kas = ', cnode, kas
                        if ( len(kas) == 2 ) and ( cnode in set(kas) ):
                            if debug:
                                print('   -- kas = ', kas)
                            if bom[kas[0],kas[1]] == 0:
                                # C1C=CC(=O)C=C1
                                # 0 1 23  4 5 6 -- atomic index
                                # mi = '[*]C=C[*]', ias = [0,1,2,3]
                                # mj = '[*]C=C[*]', jas = [3,5,6,0]
                                # kas = [3,0] but bom[0,3] = 0, i.e., these two frags cannot bind!
                                continue

                            las = list( set(ias) | set(jas) ); las.sort()
                            try:
                                # get idx of atom in `m corresponding to [*] in `mi and `mj
                                iat, jat = get_aidx_star(dic_i, ias, kas)
                            except:
                                # e.g., [*]O is a frag of [*][N+](=O)[O-]
                                # [*][N+](=O)[O-] [*]O [25, 26]
                                # [24, 25, 27, 26] [25, 26]
                                # {24: 0, 25: 7, 26: 8, 27: 8} {25: 0, 26: 8}
                                continue


                            mij = Chem.CombineMols(mi,mj)
                            #print '       combined smi = ', Chem.MolToSmiles(mij,canonical=False)
                            mc = Chem.EditableMol(mij)

                            # reconnect the bond first
                            ia2 = ias.index(jat); ja2 = jas.index(iat)
                            #print '     __ ia2, ja2 = ', ia2, ja2+na1
                            bij = m.GetBondBetweenAtoms(iat,jat)
                            mc.AddBond(ia2, ja2+na1, bij.GetBondType() ) #rdkit.Chem.rdchem.BondType.SINGLE)

                            # delete the atom in mij
                            ia = ias.index(iat); ja = jas.index(jat)
                            #print '     __ ia2, ja2, ia, ja = ', ia2, ja2, ia, ja
                            ldxs = [ia, ja+na1]; #print '     __ ldxs = ', ldxs
                            for l in range(2):
                                mc.RemoveAtom(ldxs[l]-l)

                            # update `mi
                            #try:
                            mi2 = mc.GetMol()
                            patt = Chem.MolToSmiles( mi2, canonical=False )
                            mi3 = Chem.MolFromSmarts(patt)
                            patt = re.sub('\-', '', patt)
                            patt = re.sub('\*', heav_smarts, patt)
                            if debug:
                                print('     -- patt = ', patt)

                            if '.' in patt:
                                # e.g., [*]C[*] [*]C[*] [19, 21]
                                #       [18, 19, 21] [20, 19, 21]
                                #       {18: 0, 19: 6, 21: 0} {19: 6, 20: 0, 21: 0}
                                #       [*].[*]C[*]
                                continue
                            else:
                                # update `ias
                                ias, dic_i = get_aidxs_patt(m, patt, las)
                                mi = mi3
                                if debug:
                                    print('     -- ias = ', ias)
                                na1 = len(ias)

                            icnt += 1
                    try:
                        smi = Chem.MolToSmiles( Chem.MolFromSmarts(patt), canonical=True )
                        smi = re.sub('\-', '', smi)
                        smi = re.sub('\*', heav_smarts, smi)

                        if smi not in ts2: ts2.append(smi)
                    except:
                        pass
                        print('   icnt = ', icnt)
                        print('   j, mj, jas = ', j, tss[j], jas)
                        print('   i, mi, ias = ', i, tss[i], ias)
                return ts2
            else:
                print('#ERROR: not implemented')
                raise

    def check_abnormal_local_geometry(self):
        qs = [ '[*]C(=O)O[*]', '[*]C(=O)N[*]', ]
        return


    def to_can(self):
        ctab = Chem.MolToMolBlock( self.m )
        mu = Chem.MolFromMolBlock( ctab, removeHs=True )
        return Chem.MolToSmiles( mu )

    def get_energy(self):
        return get_forcefield_energy(self.m, self.forcefield)

    def optg2(self, label=None, algo='BFGS', meth='PM7', iffopt=F):
        # use MOPAC/PM7 to further optimize
        assert iffopt or self.iFFOpt, '#ERROR: Plz call `optg() first to get some coarse geometry'
        na = self.na
        s = '%s PRECISE %s'%(meth,algo) # EF BFGS'
        s += '\nTitle: ASE\n\n'
        # Write coordinates:
        for ia in range(self.na):
            symbol = chemical_symbols[ self.zs[ia] ]
            xyz = self.coords[ia]
            s += ' {0:2} {1} 1 {2} 1 {3} 1\n'.format(symbol, *xyz)
        if label is None:
            label = tpf.NamedTemporaryFile(dir='/tmp').name
        try:
            exe = os.environ['MOPAC_EXE']
        except:
            raise Exception('#ERROR: you may need `export MOPAC=/path/to/MOPAC/executable')
        if not os.path.exists(exe):
            raise Exception('#ERROR: cannot locate file %s'%exe)
        with open(label+'.mop','w') as fid: fid.write(s)
        opf = label+'.out'
        #print ' - label=',label
        iok = os.system( '%s %s.mop 2>/dev/null'%(exe, label) )
        if iok > 0:
            raise Exception('#ERROR: MOPAC failed !!')
        else:
            if io2.cmdout2("grep 'EXCESS NUMBER OF OPTIMIZATION CYCLES' %s"%opf) != '':
                raise Exception('#ERROR: excess optg cycles, try bfgs optimizer!')
        cmd = "sed -n '/                             CARTESIAN COORDINATES/,/Empirical Formula:/p' %s"%opf
        conts = os.popen(cmd).read().strip().split('\n')[2:-3]
        # get energy
        cmd = "grep 'FINAL HEAT' %s | tail -n 1 | awk '{print $6}'"%opf
        #print('cmd=',cmd)
        e = eval( io2.cmdout2(cmd) ) # Heat of formation [kcal/mol]
        self.prop['PM7'] = e
        self.e = e
        #if not os.path.exists('../trash'): os.system('mkdir ../trash')
        #iok = os.system('mv %s.arc %s.mop %s.out ../trash/'%(label,label,label))
        iok = os.system('rm %s.arc %s.mop %s.out'%(label,label,label))
        symbs = []; coords = []
        for k in range(na):
            _, symb, px, py, pz = conts[k].strip().split()
            symbs.append(symb)
            coords_i = np.array([px,py,pz]).astype(np.float)
            coords.append( coords_i )
        #print('symbs=',symbs)
        self.atoms = cc.atoms(symbs, coords, self.prop)

        ## check if graph has changed!! ## Todo
        ##
        ##
        self.coords = np.array(coords)
        c1 = self.m.GetConformer(-1)
        for i in range(na):
            pi = Point3D()
            pi.x, pi.y, pi.z = coords[i]
            c1.SetAtomPosition(i, pi)
            #self.m.GetConformer(-1)

    def is_overcrowded(self):
        ds = ssd.squareform( ssd.pdist(self.coords) )
        non_bonds = np.where( np.logical_and(self.bom==0, ds!=0.) )
        rcs = Elements().rcs[ self.zs ]
        dsmin = rcs[..., np.newaxis] + [rcs]
        return np.any(ds[non_bonds]<dsmin[non_bonds])

    def get_backbone(self):
        # first check if `mbb is ready (molecular backbone, i.e.,
        # a molecule with all H's removed
        if not hasattr(self,'mbb'):
            m1 = copy.deepcopy( self.m0 )
            m2 = Chem.RemoveHs(m1)
            self.mbb = m2

    def estimate_nc(self):
        """ a coarse estimation of max number of conformers is 3^n
        where n is the number of quantified rotatable bonds
        A better method uses bond orders.
        Note that not all rotatable bonds are quantified, e.g.,
        for H3C-CRR'R'', H2N-CRR'R'', H2C=CRR'
        R1       R3
          \     /
           C = C      C=C contributes 2
          /     \
        R2       R4
        """
        mol = self.m
        torsma = '[!$(*#*)&!D1]~[!$(*#*)&!D1]'
        q = Chem.MolFromSmarts(torsma)
        matches = mol.GetSubstructMatches(q)
        nmat = len(matches)
        #torsions = []

        # since mostly the molecules concerned here are amons
        # with N_I <=7, we care about 3- to 7-membered rings
        atsr = _get_ring_nodes(mol,3,7,F)
        #print ' -- atsr = ', atsr
        inrs = np.zeros(self.na, dtype=int) # [this atom is] in [how many] number of rings
        for ia in self.ias_heav:
            _sets = []
            for _ats in atsr:
                if ia in _ats:
                    _sets.append(_ats)
            #print ' -- ia, _sets = ', ia, _sets
            inr = find_number_of_unique_set(_sets)
            inrs[ia] = inr
        #print ' -- inrs = ', inrs
        if nmat == 0:
            ns = [1]
            if self.debug: print('    |__ ns = ', ns)
            nc = 1
            self.nc = nc
        else:
            ns = []; patts = []
            scale = 0
            for match in matches:
                j = match[0]
                k = match[1]
                cb = set([j,k])
                bond = mol.GetBondBetweenAtoms(j, k)
                aj = mol.GetAtomWithIdx(j)
                ak = mol.GetAtomWithIdx(k)
                hj, hk = [ _hyb[_a.GetHybridization()] for _a in [aj,ak] ]
                iok1 = (hj != 2); iok2 = (hj != 3)
                iok3 = (hk != 2); iok4 = (hk != 3)
                if (iok1 and iok2) or (iok3 and iok4): continue

                # do not allow internal rotation about two adjacent sp2 atoms are in a ring
                if inrs[j] and inrs[k] and hj==2 and hk==2: continue

                pjk = []
                jk = [j,k]
                hsjk = [hj,hk]
                for _ in range(2):
                    ia1 = jk[_]
                    ia2 = j if ia1==k else k
                    hyb = hsjk[_]
                    nbrs = np.setdiff1d(self.ias[self.bom[ia1]>0], [ia2])
                    ihs = (self.zs[nbrs]==1)
                    if np.all(ihs):  # case 'a', e.g., 'a1','a2','a3'
                        # check ~X-CH3, ~X-NH2, ...
                        nh = len(ihs)
                        if hyb==3:
                            # for rotor X-C in ~X-CH3, one torsion is allowed
                            sn = {1:'a3', 2:'a2', 3:'a1'}[nh]
                        else: # hyb==2
                            sn = {1:'a2', 2:'a1', 3:'a1'}[nh]
                    else: # case 'b', e.g., 'b1','b2','b3'
                        inr = inrs[ia1]
                        if self.cns[ia1]==2 and inr: # e.g., O<, S<, Se<,
                            sn = 1
                        else:
                            if hyb==3:
                                sn = 2 if inr <= 1 else 1 # {0:'b3', 1:'b3', 2:'b2', 3:'b1', 4:'b1'}[inr]
                            else: # hyb==2:
                                sn = 'b2' if inr == 0 else 'b1'
                                #sn = {0:'b2', 1:'b1', 2:'b1', 3:'b1'}[inr]
                    _patt = '%d%s'%(hyb,sn)
                    pjk.append(_patt)
                #print 'j,k = ', j,k, ', pjk = ', pjk
                nci = min([ int(patt[-1]) for patt in pjk ]) # ndic[patt]; sci = scdic[patt]
                if nci > 1:
                    ns.append( nci )
                    if not np.any([inrs[j],inrs[k]]):
                        scale += 1
            if scale == 0: scale = 1
            nc = np.int(np.floor(np.product(ns))) * scale #* 2
            self.nc = nc if nc > 99 else 99
            if self.debug: print('    |__ ns = ', ns)
            if self.debug: print('    |__ scale = %d, nc = %d'%(scale, nc))
            self.ns = np.array(ns, np.int)

    def optg_c1(self,maxIters=60):
        """
        relax H's only
        """
        mu = self.clone()
        if self.forcefield in ['mmff94',]:
            mp = AllChem.MMFFGetMoleculeProperties(mu)
            ff = AllChem.MMFFGetMoleculeForceField(mu, mp, \
                        ignoreInterfragInteractions=False)
        else:
            ff = AllChem.UFFGetMoleculeForceField(mu, \
                        ignoreInterfragInteractions=False)
        v1,v2=0.,0.
        for i in range(self.na):
            if self.zs[i] > 1: # and self.zs[j] > 1:
                if self.forcefield in ['mmff94',]:
                    ff.MMFFAddPositionConstraint(i,0.,9999)
                else:
                    ff.UFFAddPositionConstraint(i,0.,9999)
        ff.Minimize(maxIts=maxIters)
        self.m = mu
        coords_u = get_coords( mu )
        self.coords = coords_u

    def optg_c(self,dev4=1.0,maxIters=90,wH=False,its=None, \
               hack=True):
        """
        constrained optimization
        ================================
        used for generating amons with local chemical
        enviroments as close to that in parent molecule
        as possible

        Here, we fix all dihedral angles.

        vars
        =================================
        hack -- default to True. Offers a workaround to deal
                with wierd geometries produced from rdkit optg.
                This happens when either two H atoms are too close
                in the input geometry or some torsion angles are
                exactly 0/180 degrees (planar envs). Two steps
                constitute the workaround: 1) relax H's only
                (freezing all atoms);
                2) retain any tor if its dang .eq. 0/180.
        dev  -- deviation to the present dang
        wH   -- if H is considered to be part of the strain?
        """

        if hack:
            self.optg_c1(60)

        mu = self.clone()

        iass4 = list(self.dangs0.keys())
        #iass3 = list(self.angs0.keys())

        #c = mu.GetConformer()
        if self.forcefield in ['mmff94',]:
            mp = AllChem.MMFFGetMoleculeProperties(mu)
            ff = AllChem.MMFFGetMoleculeForceField(mu, mp, \
                        ignoreInterfragInteractions=False)
        else:
            ff = AllChem.UFFGetMoleculeForceField(mu, \
                        ignoreInterfragInteractions=False)
        #for ias_i in iass3: #[:1]:
        #    i,j,k = ias_i
        #    if self.forcefield in ['mmff94',]:
        #        ff.MMFFAddAngleConstraint(i,j,k,True,-dev3,dev3,9999)
        #    else:
        #        ff.UFFAddAngleConstraint(i,j,k,True,-dev3,dev3,9999)
        nt = len(iass4)
        if its is None:
            its = list(range(nt))
        #print ' -- overall %s torsions'%nt
        for it in its:
            i,j,k,l = iass4[it]
            if hack:
                dang = self.dangs0[ iass4[it] ]; #print iass4[it], dang
                if abs(dang)<=6.0 or abs(dang-180.0)<=6.0:
                    v1,v2 = 0., 0.; #print( '##')
                else:
                    v1,v2 = -dev4,dev4
            else:
                v1,v2 = -dev4, dev4
            if self.forcefield in ['mmff94',]:
                ff.MMFFAddTorsionConstraint(i,j,k,l,True,v1,v2,9999)
            else:
                ff.UFFAddTorsionConstraint(i,j,k,l,True,v1,v2,9999)
        ff.Minimize(maxIts=maxIters)
        self.m = mu
        coords_u = get_coords( mu )
        self.coords = coords_u
        self._ds = ssd.squareform( ssd.pdist(self.coords) )

        #self.update_coords(coords_u)

    def optg(self,maxIters=900):
        """
        full relaxation of geometries using MMFF94
        """
        mu = self.clone()
        optimizer = {'uff': AllChem.UFFOptimizeMolecule, \
                'mmff94':AllChem.MMFFOptimizeMolecule }[ self.forcefield.lower() ]
        if optimizer(mu, maxIters=maxIters, ignoreInterfragInteractions=False):
            print('FF OPTG failed')
        #c = mu.GetConformer()
        #if self.forcefield in ['mmff94',]:
        #    mp = AllChem.MMFFGetMoleculeProperties(mu)
        #    ff = AllChem.MMFFGetMoleculeForceField(mu, mp, \
        #                ignoreInterfragInteractions=False)
        #else:
        #    ff = AllChem.UFFGetMoleculeForceField(mu, \
        #                ignoreInterfragInteractions=False)
        #ff.Minimize(maxIts=maxIters)
        coords_u = get_coords( mu )
        self.coords = coords_u
        self.update_coords(coords_u)
        #self.energy = ff.CalcEnergy()
        self.m = mu
        self.atoms = cc.atoms(self.zs, coords_u)
        self.iFFOpt = True
        self._ds = ssd.squareform( ssd.pdist(self.coords) )

        obj = NBody(mu, wH=F, key='ia')
        dangs = obj.dangs
        #angs = obj.angs
        iokg = True
        #if not hasattr(self, 'dangs0'):
        #    raise Exception('you need to call optg_c() first??')
        for k in dangs:
            if abs(self.dangs0[k] - dangs[k]) > 60.:
                iokg = False
                break
        self.iokg = iokg


    def optg_xtb(self, acc='normal', verbose=0, nproc=1):
        import xtb, ase
        from xtb import GFN2
        from ase.units import Hartree
        from ase.optimize.precon import Exp, PreconFIRE

        """ use xtb to optg """
        for k in ['OMP_NUM_THREADS', 'MKL_NUM_THREADS']:
            os.environ[k] = '%d'%nproc

        uc = io2.Units()
        const = uc.h2e #/uc.b2a
        fmax = {'normal':1e-3, 'tight':8.e-4, 'vtight':2e-4}[acc] * const
        # 0.025 eV/A (ase force unit) ~ 0.001 Hartree/A (xtb force unit)
        print('acc = ', acc, 'fmax=', fmax)
        m1 = ase.Atoms(self.zs, self.coords)
        calc = GFN2(print_level=verbose) # setting to 1 or 2 (the default) is ok, not so for 3
        m1.set_calculator(calc)
        e = m1.get_potential_energy()/Hartree
        print("Initial energy: eV, Eh", e, e/Hartree)
        relax = PreconFIRE(m1, precon=None, trajectory=None) # = 'tmp.traj')
        relax.run(fmax = fmax)
        coords = m1.positions
        self.prop['energy'] = e
        self.e = e # in Ha
        symbs = m1.get_chemical_symbols()
        self.atoms = cc.atoms(symbs, coords, self.prop)

        ## now check if the graph has changed!! ## Todo
        ##
        ##

        self.coords = coords
        self._ds = ssd.squareform( ssd.pdist(self.coords) )
        mc = self.clone()
        c1 = mc.GetConformer(-1)
        for i in range(self.na):
            pt = Point3D()
            pt.x, pt.y, pt.z = coords[i]
            c1.SetAtomPosition(i, pt)
            #self.m.GetConformer(-1)

        self.m = mc

        obj = NBody(mc, wH=F, key='ia')
        dangs = obj.dangs
        #angs = obj.angs
        iokg = True
        #if not hasattr(self, 'dangs0'):
        #    raise Exception('you need to call optg_c() first??')
        for k in dangs:
            if abs(self.dangs0[k] - dangs[k]) > 60.:
                iokg = False
                break
        self.iokg = iokg



    def get_oemol(self):
        import cheminfo.oechem.oechem as coo
        oemol = coo.newmol(self.zs, self.chgs, self.bom, self.coords)
        return oemol

    @property
    def oemol(self):
        if not hasattr(self, '_oemol'):
            self._oemol = self.get_oemol()
        return self._oemol

    def update_coords(self, coords):
        """
        assign coordinates to RDKit molecule object
        """
        Mol = copy.deepcopy(self.m)
        c1 = Mol.GetConformer(-1)
        for i in range(self.na):
            coords_i = Point3D()
            coords_i.x, coords_i.y, coords_i.z = coords[i]
            c1.SetAtomPosition(i, coords_i)
        self.m = Mol

    def write_all(self, sdf):
        write_ctab(self.zs, self.chgs, self.bom, self.coords, sdf=sdf, prop=self.prop)

    def write_sdf(self, sdf=None):
        if sdf is None:
            sdf = tpf.NamedTemporaryFile(dir='/tmp').name+'.sdf'
        Chem.MolToMolFile(self.m, sdf)
        return sdf

    def write_pdb(self, pdb):
        # using `flavor=4 and `Chem.MolToPDBBlock is the safest way
        open(pdb,'w').write( Chem.MolToPDBBlock(self.m, flavor=4) )

    def write_xyz(self, f):
        coords = get_coords(self.m)
        symbs = [ chemical_symbols[zi] for zi in self.zs ]
        write_xyz_simple(f, (symbs,coords), self.prop)

    def get_stablest_conformer(self, optc=False, nconfs=None, nthread=1):
        """
        generate a series of conformers and choose the lowest energy conformer

        ETKDG is the default conformation generation method and due to its robustness,
        there should be no need to use a minimisation step to clean up the structures.
        """
        m1 = self.clone()
        if nconfs is None:
            self.estimate_nc()
            nconfs = self.nc
        if self.keep_parent:
            print(' *** warning: rdkit would fail to find any conformer! Keep parent geom as the only conformer!' )
            self.optg()
            return

        if self.nheav >= 3:
            #seeds = [1, 4, 7, 10, ]
            #emin = 999999.
            #for seed in seeds: # different seed results in different initial ref geometry
            cids = AllChem.EmbedMultipleConfs(m1, nconfs) #, numThreads=nthread, \
                      #pruneRmsThresh=1.0, randomSeed=seed) #, \
                      # useBasicKnowledge=True, useExpTorsionAnglePrefs=True)
            if optc:
                for cid in cids:
                    _ = AllChem.MMFFOptimizeMolecule(m1, confId=cid, maxIters=500)
            props = AllChem.MMFFGetMoleculeProperties(m1) #, mmffVariant='MMFF94')
            es = []; ics = []
            for cid in cids:
                try:
                    ff = AllChem.MMFFGetMoleculeForceField(m1, props, confId=cid)
                    e = ff.CalcEnergy()
                    es.append( e ); ics.append(cid)
                except:
                    pass #print(' ***** warning: no such conformer id %d'%cid)
            if len(es) > 0:
                e1 = min(es)
                cid_u = ics[ es.index(e1) ]
                conf = m1.GetConformer(cid_u)
                #m2 = Chem.Mol(m1, conf.GetId() )
                #self.m = mU
                coords_u = [] # conf.GetPositions()
                for i in range(self.na):
                    oi = conf.GetAtomPosition(i)
                    coords_u.append([ oi.x, oi.y, oi.z ])
                coords_u = np.array(coords_u)
                self.atoms = cc.atoms(self.zs, coords_u)
                self.update_coords(coords_u)
            # At last, run ff optg
            self.optg()

    def get_charged_pairs(self):
        """
        get pairs of atoms with opposite charges
        """
        charges = [ ai.GetFormalCharge() for ai in self.m.GetAtoms() ]
        # search for the pairs of atoms with smarts like '[N+](=O)[O-]'
        patt = '[+1]~[-1]'
        q = Chem.MolFromSmarts(patt)
        cpairs = np.array( self.m.GetSubstructMatches(q) ).astype(np.int)
        self.charges = charges
        self.cpairs = cpairs


    def neutralise_raw(self):
        """
        neutralize a molecule, typically a protein by rebuilding
        the molecule from scratch (i.e., zs, bom and coords)

        Less recommended as it's slower than `netralise()
        """
        # kekulization has to be done, otherwise u will encounter
        # issues when assigning bond types later
        Chem.Kekulize(self.m)

        # get pairs of charged atoms
        self.get_charged_pairs()

        # eliminate the charges by rebuilding the molecule
        m = Chem.Mol()
        mc = Chem.EditableMol(m)
        for i, az in enumerate(self.zs):
            ai = Chem.Atom( az )
            ci = self.charges[i]
            if ci != 0:
                if ci == 1:
                    filt = (self.cpairs[:,0] == i)
                    if np.any(filt):
                        ai.SetFormalCharge( 1 )
                elif ci == -1:
                    filt = (self.cpairs[:,1] == i)
                    if np.any(filt): ai.SetFormalCharge( -1 )
                else:
                    print((' -- i, charges[i] = ', i, self.charges[i]))
                    print(' #ERROR: abs(charge) > 1??')
                    raise
            mc.AddAtom( ai )

        ijs = np.array( np.where( np.triu(self.bom) > 0 ) ).astype(np.int)
        nb = ijs.shape[1]
        for i in range(nb):
            i, j = ijs[:,i]
            mc.AddBond( i, j, bo2bt[ '%.1f'%self.bom[i,j] ] )

        m = mc.GetMol()
        m2 = assign_coords(m, self.coords)
        self.m = m2


    def neutralise(self):
        """
        a simple version to neutralise a molecule, typically a protein
        """
        m = self.m

        # Regenerates computed properties like implicit
        # valence and ring information.
        m.UpdatePropertyCache(strict=False)
        numHs = []; tvs = []
        for ai in m.GetAtoms():
            numHs.append( ai.GetNumExplicitHs() + ai.GetNumImplicitHs() )
            tvs.append( ai.GetTotalValence() )

        self.get_charged_pairs()

        for i in range(self.na):
            ai = m.GetAtomWithIdx(i)
            ci = self.charges[i]
            if ci != 0:
                if i not in self.cpairs.ravel():
                    msg = ' zi = %d, tvi = %d, ci = %d, neib = %d'%(self.zs[i], tvs[i], ci, cnsDic[zs[i]])
                    assert tvs[i] - ci == cnsDic[zs[i]], msg
                    if numHs[i] == 0 and ci > 0:
                        # in the case of >[N+]<, i.e., N with CoordNum = 4
                        # we don't have to do anything
                        continue
                    ai.SetFormalCharge( 0 )
                    ai.SetNoImplicit(True)
                    ai.SetNumExplicitHs( numHs[i]-ci )
                    print('i, zi, ci, nH = ', self.zs[i], ci, numHs[i])
        self.m = m


    def get_atypes(self):
        """
        get the type of each atom in a molecule
        """
        self.atypes = []
        self.hybs = []
        #self.zs = []
        for ai in self.m.GetAtoms():
            hybi = str( ai.GetHybridization() )
            self.hybs.append( hybi )
            zi = ai.GetAtomicNum()
            #self.zs.append( zi )
            si = ai.GetSymbol()
            if hybi == 'SP2':
                ar = ai.GetIsAromatic()
                ar_suffix = '_R' if ar else '_2'
                ap = si + ar_suffix # atomic_pattern
            elif hybi == 'SP3':
                if zi == 16 and ai.GetExplicitValence() == 6:
                    ap = si + 'o3'
                elif zi in [9, 17, 35, 53]:
                    ap = si
                else:
                    ap = si + '_3'
            elif hybi == 'SP':
                ap = si + '_1'
            elif hybi in ['S', ]: #'UNSPECIFIED']:
                ap = si
            else:
                print((' unknown atom type: `%s`'%hybi))
                raise
            self.atypes.append( ap )

    def get_atom_contrib(self, groupBy='m'):
        """
        1-body part in BAML repr
        """
        self.es1 = -0.5 * np.array(self.zs)**2.4 * h2kc
        self.types1 = {'m': self.atypes, 'n': self.zs}[groupBy]

    def get_bond_contrib(self, dic, pot='Morse', groupBy='m'):
        """
        the 1st part of 2-body interactions in BAML repr

        Two-body potential could be 'Morse' or 'Harmonic'.
        `dic must be obtained by calling
            dic = BondEnergies().dic
        """
        nb = self.m.GetNumBonds()
        self.nb = nb
        self.bos = []
        ds = []
        self.ias2 = []
        e2 = 0.0
        es2 = []
        types2_z = []
        types2_m = []

        for ib in range(nb):
            bi = self.m.GetBondWithIdx(ib)
            boi = bt2bo[ bi.GetBondType() ]
            self.bos.append( boi )
            ia1,ia2 = bi.GetBeginAtomIdx(), bi.GetEndAtomIdx()
            self.ias2.append( [ia1,ia2] )

            zsi = [ self.zs[ia1], self.zs[ia2] ]; zsi.sort()
            types2_z.append( '-'.join([ '%d'%zi for zi in zsi ]) )

            aps = [ self.atypes[ia] for ia in [ia1,ia2] ]
            aps.sort()
            bpi = '-'.join( aps + ['%.2f'%boi ] )
            types2_m.append( bpi )

            kb, re = rcr.GetUFFBondStretchParams(self.m, ia1, ia2)
            r = AllChem.GetBondLength(self.m.GetConformer(), ia1, ia2)
            ds.append(r)
            De = dic[ bpi ]
            spot = pot.lower()
            Dr_harmonic = 0.5 * kb * (r - re)**2
            if spot == 'morse':
                a = (kb/(2.0*De))**0.5
                Dr = De*( (np.exp(-a*(r-re)) - 1.0)**2 - 1.0 )
            else: #
                Dr = Dr_harmonic
            es2.append(Dr); e2 += Dr
        self.e2 = e2
        self.e2_harmonic = Dr_harmonic
        self.es2 = es2
        self.n2 = len(es2)
        self.types2 = {'m': types2_m, 'n': types2_z}[groupBy]
        #return e2, n2, types2, es2


    def get_vdw_contrib(self, rcut=8.0, groupBy='m'):
        """
        the 2nd part of 2-body interactions in BAML repr
        """
        e2v = 0.0
        ias2v = []
        es2v = [];
        types2v_z = []
        types2v_m = []
        for i in range(self.na):
            for j in range(i+1,self.na):
                bond = self.m.GetBondBetweenAtoms(i, j)
                #print ' -- i,j,bond = ', i,j,bond
                if bond is None:
                    neibs_i = set( [ ai.GetIdx() for ai in self.m.GetAtomWithIdx(i).GetNeighbors() ] )
                    neibs_j = set( [ ai.GetIdx() for ai in self.m.GetAtomWithIdx(j).GetNeighbors() ] )
                    if neibs_j.intersection(neibs_i) == set():
                        eij = 0.0
                        ias = [ i,j ]
                        ias2v.append(ias)
                        zs = [ self.zs[ia] for ia in ias]
                        zs.sort()
                        aps = [ self.atypes[ia] for ia in ias ]
                        aps.sort()
                        r0, D0 = rcr.GetUFFVdWParams(self.m, i,j)
                        rij = rdMolTransforms.GetBondLength(self.m.GetConformer(), i, j)
                        if rij <= rcut:
                            ratio = r0/rij
                            r6 = ratio**6
                            r12 = r6*r6
                            eij = D0*(r12 - 2.0*r6)
                            #print 'i,j,rij, r0,D0, evdw = %2d %2d %5.2f %5.2f %8.4f %8.4f'%( i,j,rij, r0,D0, evdw )
                            e2v += eij
                            es2v.append(eij)
                            types2v_z.append('-'.join([ '%d'%zi for zi in zs ]))
                            types2v_m.append( '-'.join( aps ) )
        self.e2v = e2v
        self.es2v = es2v
        self.n2v = len(es2v)
        self.types2v = {'m':types2v_m, 'n':types2v_z}[groupBy]
        #print ' -- types2v = ', self.types2v
        #return e2v, n2v, types2v, es2v


    def get_angle_contrib(self, groupBy='m'):
        """
        3-body parts in BAML representation
        """
        ias3 = []
        types3_z = []
        types3_m = []

        e3 = 0.0
        es3 = []
        for aj in self.m.GetAtoms():
            j = aj.GetIdx()
            zj = self.zs[j]
            neibs = aj.GetNeighbors()
            nneib = len(neibs)
            if zj > 1 and nneib > 1:
                  for i0 in range(nneib):
                    for k0 in range(i0+1,nneib):
                        eijk = 0.0
                        i, k = neibs[i0].GetIdx(), neibs[k0].GetIdx()
                        ias = [i,j,k]
                        ias3.append(ias)

                        ap1, ap2, ap3 = [ self.atypes[ia] for ia in ias ]
                        if ap1 > ap3:
                            tv = ap1; ap1 = ap3; ap3 = tv # tv: temperay value
                        types3_m.append( '-'.join( [ap1, ap2, ap3] ) )

                        z1, z2, z3 = [ self.zs[ia] for ia in ias ]
                        if z1 > z3:
                            tv = z1; z1 = z3; z3 = tv
                        types3_z.append( '-'.join(['%d'%zi for zi in [z1,z2,z3] ]) )

                        theta = rdMolTransforms.GetAngleRad(self.m.GetConformer(), i, j, k)
                        cosT = np.cos(theta)
                        ka, theta0 = rcr.GetUFFAngleBendParams(self.m, i, j, k)
                        theta0 = theta0*np.pi/180.0
                        cosT0 = np.cos(theta0); sinT0 = np.sin(theta0)

                        #print ' -- atypes = ', self.atypes
                        hybj = self.hybs[j]
                        if hybj == 'SP':
                            eijk = ka*(1.0 + np.cos(theta))
                        elif hybj == 'SP2':
                            # energy expression from Openbabel's src file "forcefielduff.cpp',
                            # different from that of Rappe's bad formula,
                            eijk = (ka/4.5)*(1.0 + (1.0 + cosT)*(4.0*cosT))
                        elif hybj == 'SP3':
                            c2 = 1.0 / (4.0 * sinT0 * sinT0)
                            c1 = -4.0 * c2 * cosT0;
                            c0 = c2*(2.0*cosT0*cosT0 + 1.0);
                            eijk = ka*(c0 + c1*cosT + c2*(2.0*cosT*cosT - 1.0))
                        else:
                            print('not supported atomic type: %s'%apj)
                            assert 0

                        e3 += eijk
                        es3.append(eijk)
        self.e3 = e3
        self.es3 = es3
        self.n3 = len(es3)
        self.types3 = {'m':types3_m, 'n':types3_z}[groupBy]
        #return e3, n3, types3, es3


    def get_torsion_contrib(self, groupBy='m'):
        """
        4-body parts in BAML representation
        """
        e4 = 0.0
        es4 = []
        iass4 = []

        types4_z = []
        types4_m = []

        zs8 = [8, 16, 34] # zs of group 8 elements
        set_hybs = set(['SP2','SP3'])

        for ib in range(self.nb):
            j, k = self.ias2[ib]
            if self.zs[j] > self.zs[k]:
                tv = j; k = j; j = tv
            neibs1 = self.m.GetAtomWithIdx(j).GetNeighbors(); n1 = len(neibs1);
            neibs2 = self.m.GetAtomWithIdx(k).GetNeighbors(); n2 = len(neibs2);
            for i0 in range(n1):
                for l0 in range(n2):
                    i = neibs1[i0].GetIdx(); l = neibs2[l0].GetIdx()
                    if len(set([i,j,k,l])) == 4:
                        eijkl = 0.0
                        ias = [ i,j,k,l ]; iass4.append(ias)
                        zsi = [ self.zs[ia] for ia in ias ]
                        types4_z.append( '-'.join([ '%d'%zi for zi in zsi ]) )
                        types4_m.append( '-'.join([ self.atypes[ia] for ia in ias ]) )
                        V = rcr.GetUFFTorsionParams(self.m, i, j, k, l)
                        tor = rdMolTransforms.GetDihedralRad(self.m.GetConformer(), i,j,k,l)
                        hyb2 = self.hybs[j]
                        hyb3 = self.hybs[k]
                        if (hyb2 == 'SP3') and (hyb3 == 'SP3'):
                            order = 3; cosNPhi0 = -1 # Phi0 = 60 degree
                            if self.bos[ib] == 1 and set([self.zs[j],self.zs[k]]) <= set(zs8):
                                orde = 2; cosNPhi0 = -1
                            eijkl = 0.5*V*( 1.0 - cosNPhi0*np.cos(tor*order) )
                        elif (hyb2 == 'SP2') and (hyb3 == 'SP2'):
                            order = 2; cosNPhi0 = 1.0 # phi0 = 180
                            eijkl = 0.5*V*( 1.0 - cosNPhi0*np.cos(tor*order) )
                        elif set([hyb2,hyb3]) == set_hybs:
                            # SP2 - SP3,  this is, by default, independent of atom type in UFF
                            order = 6; cosNPhi0 = 1.0 # phi0 = 0
                            if self.bos[ib] == 1.0:
                                # special case between group 6 sp3 and non-group 6 sp2:
                                #if hyb2 == 'SP3' and hyb3 == 'SP3' and set([zs[j],zs[k]]) <= zs8:
                                #    order = 2; cosNPhi0 = -1 # phi0 = 90
                                if ((self.zs[j] in zs8) and (self.hybs[k] == 'SP2')) or \
                                    ((self.zs[k] in zs8) and (self.hybs[j] == 'SP2')):
                                    order = 2; cosNPhi0 = -1 # phi0 = 90
                            eijkl = 0.5*V*( 1.0 - cosNPhi0*np.cos(tor*order) )
                        #else:
                        #    raise '#ERROR: unknown senario?'
                        #print '[i,j,k,l] = [%d,%d,%d,%d], eijkl = %.4f'%(i,j,k,l, eijkl )
                        #print V, order, cosNPhi0, tor, eijkl
                        es4.append(eijkl)
                        e4 += eijkl
        self.e4 = e4
        self.es4 = es4
        self.n4 = len(es4)
        self.types4 = {'m': types4_m, 'n': types4_z}[groupBy]
        #return e4, n4, types4, es4

    def get_baml_energy(self):
        return self.e2_harmonic + self.e2v + self.e3 + self.e4 # unit: kcal/mol


def get_coords(m):
    na = m.GetNumAtoms()
    zs = np.array([ ai.GetAtomicNum() for ai in m.GetAtoms() ])
    #assert m.GetNumConformers() == 1, '#ERROR: more than 1 Conformer exist?'
    c0 = m.GetConformer(-1)
    coords = []
    for i in range(na):
        coords_i = c0.GetAtomPosition(i)
        coords.append([ coords_i.x, coords_i.y, coords_i.z ])
    return np.array(coords)


def assign_coords(m, coords):
    """
    assign coordinates to RDKit molecule object
    """
    Mol = copy.deepcopy(m)
    na = Mol.GetNumAtoms()
    if not m.GetNumConformers():
        rdDepictor.Compute2DCoords(Mol)
    c1 = Mol.GetConformer(-1)
    for i in range(na):
        coords_i = Point3D()
        coords_i.x, coords_i.y, coords_i.z = coords[i]
        c1.SetAtomPosition(i, coords_i)
    return Mol


def get_bom(_m, kekulize=False):
    # somehow it's problematic to use `Chem.GetAdjacencyMatrix(m, useBO=True)
    m = copy.deepcopy(_m)
    if kekulize: Chem.Kekulize(m)
    na = m.GetNumAtoms()
    bom = np.zeros((na, na))
    for bi in m.GetBonds():
        i, j = bi.GetBeginAtomIdx(), bi.GetEndAtomIdx()
        bt = bi.GetBondType()
        bom[i,j] = bom[j,i] = bt2bo[ bt ]
    return bom



def fix_COO_and_NH(pdb):
    """
    fix the valence states of -C(O)O- in protein, i.e.,
    -C(O)[O] --> -C(=O)O
    """
    M = Molecule(pdb)
    ds = M.ds
    g = M.g
    m = Chem.MolFromPDBFile(pdb, removeHs=True) # don't use True, otherwise Error
    mc = Chem.EditableMol(m)
    q = Chem.MolFromSmarts( 'C(~[O])~[O]' )
    matches = m.GetSubstructMatches(q)
    #bom = Chem.GetAdjacencyMatrix(m,useBO=True)
    for (i,j,k) in matches:
        d1, d2 = ds[i,j], ds[i,k]
        b1 = m.GetBondBetweenAtoms(i,j)
        b2 = m.GetBondBetweenAtoms(i,k)
        if d1 < d2:
            bij = 2.0; bik = 1.0
        else:
            bij = 1.0; bik = 2.0
        mc.RemoveBond(i,j)
        mc.RemoveBond(i,k)
        mc.AddBond(i,j,bo2bt['%.1f'%bij])
        mc.AddBond(i,k,bo2bt['%.1f'%bik])
    mu = mc.GetMol()
    return mu
    # NX4, e.g., >[N+]<, >[NH+]-, >[NH2+], -[NH3+]
    #q = Chem.MolFromSmarts( '[NX4]' )

    # [*]~NX2, e.g., >C=[NH2+]

    # [*]~NX, e.g., >C=[NH+]-

def cdist(coords):
    return ssd.squareform( ssd.pdist(coords) )

def get_bom(m0, kekulize=False):
    # somehow it's problematic to use `Chem.GetAdjacencyMatrix(m, useBO=True)
    m = copy.deepcopy(m0)
    if kekulize: Chem.Kekulize(m)
    na = m.GetNumAtoms()
    bom = np.zeros((na, na))
    for bi in m.GetBonds():
        i, j = bi.GetBeginAtomIdx(), bi.GetEndAtomIdx()
        bt = bi.GetBondType(); #print ' ***** bt = ', bt
        bom[i,j] = bom[j,i] = bt2bo[ bt ]
    return bom


def get_forcefield_energy(m, forcefield='mmff94'):
    ffl = forcefield.lower()
    if ffl == 'mmff94':
        mp = AllChem.MMFFGetMoleculeProperties(m)
        hdl = AllChem.MMFFGetMoleculeForceField(m, mp, \
                ignoreInterfragInteractions=False)
    elif ffl == 'uff':
        hdl = AllChem.UFFGetMoleculeForceField(m, \
                ignoreInterfragInteractions=False)
    return hdl.CalcEnergy()



def calc_rmsd(mol, mode='RMSD'):
    """
    calculate conformer-conformer RMSD.
    """
    if mode == "TFD":
        ds = TorsionFingerprints.GetTFDMatrix(mol)
    else:
        nc = mol.GetNumConformers()
        ds = np.zeros((nc,nc),dtype=float)
        cs = mol.GetConformers()
        for i, ci in enumerate(cs):
            ic = ci.GetId()
            for j, cj in enumerate(cs):
                if i >= j: continue
                jc = cj.GetId()
                ds[i,j] = ds[j,i] = AllChem.GetBestRMS(mol, mol, ic, jc)
    return ds

def reset(_m):
    m = copy.deepcopy(_m)
    na = m.GetNumAtoms()
    for bi in m.GetBonds():
        i, j = bi.GetBeginAtomIdx(), bi.GetEndAtomIdx()
        bi.SetBondType( bo2bt['1.0'] ) #Chem.BondType.SINGLE )
    return m

def unset(_m, bom):
    m = copy.deepcopy(_m)
    for bi in m.GetBonds():
        i, j = bi.GetBeginAtomIdx(), bi.GetEndAtomIdx()
        bi.SetBondType( bo2bt['%.1f'%bom[i,j]] )
    return m

def find_number_of_unique_set(s):
    """
    E.g., for a list of sets
      s = [ {1,2,3}, {2,3,4}, {2,3,4,5}, {1,2,3,4}]
    The corresponding unique set is [ {1,2,3}, {2,3,4} ]
    """
    ns = [ len(si) for si in s ]
    n = len(s)
    iss = np.arange(n)
    idx = np.argsort(ns)
    so = []
    for i in idx:
        si = s[i]
        if i==0:
            so.append(si)
        else:
            _idx = np.setdiff1d(iss,[i])
            iadd = T
            for j in _idx:
                if si.issuperset(s[j]):
                    iadd = F
                    break
            if iadd: so.append(si)
    return len(so)


class EmbedMol(RDMol):

    def __init__(self, mol):
        self.mol = mol
        RDMol.__init__(mol)

        ias = np.arange(self.na)
        self.ias = ias
        self.ias_heav = ias[self.zs>1]
        self.optg0 = F # is ff optg done?

    def gen_conformers(self, nc=None, nthread=1, maxiter=1200, pruneRmsThresh=-1):
        """ generate conformers """
        # trick rdkit by setting a bond order of 1 for all bonds
        mol = reset(self.mol)
        if nc is None:
            self.estimate_nc()
            nc = self.nc
            #print('    |__ estimated num of conformers: ', nc)

        # ETKDG method becomes the default since version RDKit_2018
        # (currently we r working with RDKit_2017_09_1, ETKDG has to be manually turned on)
        params = AllChem.ETKDG()
        #params = AllChem.EmbedParameters() # default in RDKit version <= 2017, no ETKDG is used
        #
        #params.maxIterations = maxiter
        params.useBasicKnowledge = T #F
        #params.numThreads = nthread
        #params.pruneRmsThresh = pruneRmsThresh # -1
        params.useExpTorsionAnglePrefs = F #T
        params.onlyHeavyAtomsForRMS = F
        cids = AllChem.EmbedMultipleConfs(mol, nc, params)
        #cids = AllChem.EmbedMultipleConfs(mol, numConfs=nc, maxAttempts=1000, \
        #             pruneRmsThresh=pruneRmsThresh, useExpTorsionAnglePrefs=T, \
        #             useBasicKnowledge=T, enforceChirality=T, numThreads=nthread)
        self.cids = np.array(list(cids),np.int)
        istat = T
        if len(self.cids) == 0:
            # For smi="C1OC2CCC21", no conformer was found when i=0; while it has only 1 conformer, ...
            #print '#ERROR: RDKit failed to find any conformer!'
            istat = F
        self.istat = istat
        # now restore
        self.mol = unset(mol, self.bom)
        self.nconf = len(self.cids)

    @staticmethod
    def get_rmsd(mode='RMSD'):
        """
        calculate conformer-conformer RMSD.
        """
        return calc_rmsd(self.mol, mode=mode)

    def optg(self, ff='mmff94', n=1000):
        """ optimize geometries of all conformers """
        immff = F
        if ff in ['mmff94']:
            props = AllChem.MMFFGetMoleculeProperties(self.mol)
            immff = T
        angs = _get_angles_csp2(self.mol)
        self.es = []
        if not hasattr(self,'cids'):
            cids = [-1]
        else:
            cids = self.cids
        for cid in cids:
            for cycle in [0,1]:
                """
                minization is split into 2 parts
                a) The first part tries to correct some ill local geometries in conformers,
                   realized through constraints in angles and will be iterated for maximally
                   200 steps;
                b) Normal geometry minization without constraints, number of iterations: `n-200
                """
                if immff:
                    ff = AllChem.MMFFGetMoleculeForceField(self.mol, props, confId=cid)
                else:
                    ff = AllChem.UFFGetMoleculeForceField(self.mol, confId=cid)
                ff.Initialize()
                if cycle == 0:
                    _n = 200
                    ## The two lines below are essential to obtain reasonable conformer geometries
                    ## If not present, then conformer with some angle centered on sp2-C may be ~90
                    ## degrees
                    for i,j,k in angs:
                        ff.MMFFAddAngleConstraint(i,j,k, F, 95, 145, 9999.) # relative=False
                    ## Here, in essense, we manually constrain such angles to be within the range
                    ## of [95,145] degree
                else:
                    _n = n - 200
                if n > 0:
                    converged = ff.Minimize(maxIts=_n, forceTol=0.0001, energyTol=1e-05)
                #RETURNS: 0 if the optimization converged, 1 if more iterations are required.
            self.es.append( ff.CalcEnergy() )
        #res = AllChem.MMFFOptimizeMoleculeConfs(self.mol, numThreads=1, maxIters=n)
        self.optg0 = T

    def optg_xtb(self, acc='normal', nproc=1):
        """ use xtb to optg """
        for k in ['OMP_NUM_THREADS', 'MKL_NUM_THREADS']:
            os.environ[k] = '%d'%nproc
        uc = io2.Units()
        const = uc.h2e #/uc.b2a
        fmax = {'normal':1e-3, 'tight':8.e-4, 'vtight':2e-4}[acc] * const
        # 0.025 eV/A (ase force unit) ~ 0.001 Hartree/A (xtb force unit)
        print('acc = ', acc, 'fmax=', fmax)
        for cid in range(self.nconf):
            print('cid=', cid)
            c1 = self.mol.GetConformer(cid)
            zs = [ ai.GetAtomicNum() for ai in c1.GetAtoms() ]
            coords = []
            for ia in range(self.na):
                o = c1.GetAtomPosition(ia)
                coords.append( [o.x, o.y, o.z] )
            m1 = ase.Atoms(zs, coords)
            # create the calculator for GFN0-xTB under periodic boundary conditions
            calc = GFN2(print_level=2) # setting to 1 or 2 (the default) is ok, not so for 3
            m1.set_calculator(calc)
            e = m1.get_potential_energy()
            print("Initial energy: eV, Eh", e, e/Hartree)
            relax = PreconFIRE(m1, precon=None, trajectory=None) # = 'tmp.traj')
            relax.run(fmax = fmax)
            for i in range(self.na):
                pt = Point3D()
                pt.x, pt.y, pt.z = coordsU[i]
                c1.SetAtomPosition(i, pt)


    def optg2(self, label=None):
        # use MOPAC/PM7 for further optimization
        es = []
        assert self.optg0, '#ERROR: Plz call `optg() first'
        for cid in range(self.nconf):
            print('cid=', cid)
            c1 = self.mol.GetConformer(cid)
            m1 = self.get_atoms([cid])[0]
            s = 'PM7 PRECISE' # 'BFGS'
            s += '\nTitle: ASE\n\n'
            # Write coordinates:
            for ia in range(self.na):
                symbol = m1.symbols[ia]
                xyz = m1.coords[ia]
                s += ' {0:2} {1} 1 {2} 1 {3} 1\n'.format(symbol, *xyz)
            if label is None:
                label = tpf.NamedTemporaryFile(dir='/tmp').name
            with open(label+'.mop','w') as fid: fid.write(s)
            try:
                exe = os.environ['MOPAC_EXE']
            except:
                raise Exception('#ERROR: Plz do `export MOPAC=/path/to/MOPAC/executable')
            iok = os.system( '%s %s.mop 2>/dev/null'%(exe, label) )
            if iok > 0:
                raise Exception('#ERROR: MOPAC failed !!')

            # get energy
            opf = label+'.out'
            if self.debug: print('    |__ opf = ', opf)
            cmd = "grep 'FINAL HEAT' %s | tail -n 1 | awk '{print $6}'"%opf
            e = eval( io2.cmdout2(cmd) ) # Heat of formation [kcal/mol]
            if self.debug: print('    |____ e = ', e); es.append( e )

            # get coords
            cmd = "sed -n '/                             CARTESIAN COORDINATES/,/Empirical Formula:/p' %s"%opf
            conts = io2.cmdout2(cmd).split('\n')[2:-3]
            _coordsU = []
            for k in range(self.na):
                tag, symb, px, py, pz = conts[k].strip().split()
                _coordsU.append( [px,py,pz] )
            coordsU = np.array(_coordsU, dtype=float)
            for i in range(self.na):
                pi = Point3D()
                pi.x, pi.y, pi.z = coordsU[i]
                c1.SetAtomPosition(i, pi)

            #if not os.path.exists('../trash'): os.system('mkdir ../trash')
            #iok = os.system('mv %s.arc %s.mop %s.out ../trash/'%(label,label,label))
        self.es = es


    def prune_conformers(self, param={'M':'cml1', 'rp':1.0,'thresh':0.25,'wz':F,'sort':T}):
        """
        get the chosen conformer ids, i.e., the unique conformers

        vars
        ======================
        param['M']: 'cm','cml1'

        Notes
        ======================
        a) deficiency of 'rmsd': cannot distinguish two distinct conformers of OC=O,

            O                  O
            ||                 ||
            C     H            C
           / \  /             / \
          H   O              H   O
                                 |
                                 H
                RMSD = 0.003 (UFF optimized geom)
          While with 'cml1+sort': dcm = 0.163 if wz=F else 1.311
        b) caveat: use of 'cm' may fail to distinguish conformers of CH3-CH3
        """
        if param['M'] in ['rmsd']:
            ds = self.get_rmsd()
        elif param['M'] in ['cm','cml1']:
            ds = self.get_dcm(param)
        else:
            raise '#ERROR: unknow rep'
        #print ' ++ ds = ', ds
        #print '   |__ es = ', np.array(self.es)
        seq = np.argsort(self.es)  # sort by increasing energy
        ccids = []
        for i in seq:
          # always keep lowest-energy conformer
          if len(ccids) == 0:
            ccids.append(i)
            continue

          # discard conformers within the RMSD threshold
          if np.all(ds[i][ccids] >= thresh):
            ccids.append(i)
        self.nconf = len(ccids)
        # creat a new mol object with unique conformers
        new = Chem.Mol(self.mol)
        new.RemoveAllConformers()
        for i in ccids:
            ci = self.mol.GetConformer(i)
            new.AddConformer(ci, assignId=True)
        self.mol = new

    def write_conformers(self, filename): # ccids):
        """ write conformers to sdf files """
        cnt = 0
        for confId in range(self.nconf): #ccids:
            w = Chem.SDWriter('%s_c%03d.sdf'%(filename,cnt+1))
            w.write(self.mol, confId=confId)
            w.flush()
            w.close()
            cnt += 1

    def write(self, f):
        Chem.MolToMolFile(self.mol, f)

    def get_atoms(self, cids=None):
        na = self.na
        zs = self.zs
        if cids == None:
            cids = list(range(self.nconf))
        ms = []
        for cid in cids:
            ps = []
            c = self.mol.GetConformer(cid)
            for ia in range(na):
                psi = c.GetAtomPosition(ia)
                ps.append( [psi.x, psi.y, psi.z] )
            ms.append( atoms(zs,ps) )
        return ms

    def get_dcm(self, param):
        objs = self.get_atoms()
        return cdist(objs, param) #[-1]

#if __name__ == "__main__":

#   smi =  'N#CC1=CC=C(ON2P(OC3=CC=C(C=C3)C#N)N=P(OC3=CC=C(C=C3)C#N)(OC3=CC=C(C=C3)C#N)N(OC3=CC=C(C=C3)C#N)P2OC2=CC=C(C=C2)C#N)C=C1'
#   o = cir.RDMol( smi, ih=False, doff=False)
#   level = 1
#   print(' building blocks, level 1\n', o.get_building_blocks(level=level))

if __name__ == "__main__":
    """
    generate conformers for a input molecule

    Attention: most frequently, the input are sdf files of AMONs !!!!!!!!!!
    """
    import stropr as so

    _args = sys.argv[1:]
    if ('-h' in _args) or (len(_args) < 3):
        print("Usage: ")
        print("   genconf [-rms 0.1] [-thresh 0.1] [-nthread 1] [-fd folder_name] [q1.sdf q2.sdf ...]")
        print("   genconf [-rms 0.1] [-thresh 0.1] [-nthread 1] [-fd folder_name]  01.smi")
        sys.exit()

    print(' \n Now executing ')
    print('         genconf ' + ' '.join(sys.argv[1:]) + '\n')

    idx = 0
    keys=['-fd','-ofd']; hask,fd,idx = so.parser(_args,keys,'amons/',idx)
    if not os.path.exists(fd): os.system('mkdir %s'%fd)

    keys=['-rms']; hask,_rms,idx = so.parser(_args,keys,'-1',idx,F); rms=eval(_rms)
    keys=['-thresh']; hask,_thresh,idx = so.parser(_args,keys,'0.25',idx,F); thresh=eval(_thresh)
    keys=['-nthread']; hask,_nthread,idx = so.parser(_args,keys,'1',idx,F); nthread=int(_nthread)
    keys=['-nstep']; hask,_nstep,idx = so.parser(_args,keys,'999',idx,F); nstep=int(_nstep)
    keys=['-ow','-overwrite']; ow,idx = so.haskey(_args,keys,idx)
    keys=['-allow_smi']; allow_smi,idx = so.haskey(_args,keys,idx)
    keys=['-optg2','-pm7','-mopac']; optg2,idx = so.haskey(_args,keys,idx)
    keys=['-nc','-nconf']; hask,snc,idx = so.parser(_args,keys,'None',idx,F); nc = eval(snc)
    keys=['-ff',]; hask,ff,idx = so.parser(_args,keys,'mmff94',idx,F)

    keys=['-nj','-njob']; hask,snj,idx = so.parser(_args,keys,'1',idx,F); nj = int(snj)
    if nj > 1:
        keys=['-i','-id']; hask,sid,idx = so.parser(_args,keys,None,idx)
        assert hask, '#ERROR: need to set -i [INT_VAL]'
        id = int(sid)
    else:
        hasID,idx = so.haskey(_args,keys,idx)
        assert not hasID

    args = _args[idx:]
    narg = len(args)
    ms = []; fs = []; lbs = []; smiles=[]
    isdf = T; fs0 = []
    for arg in args:
        assert os.path.exists(arg), '#ERROR: input should be file, of format *.smi or *.sdf'
        if arg[-3:] in ['smi','can']:
            isdf = F
            if allow_smi:
                assert narg == 1
                for _si in file(arg).readlines():
                    si = _si.strip().split()[0]; #print ' +++++ si = ', si
                    if si != '':
                        #print '    ** smi = ', si
                        _mi = Chem.MolFromSmiles(si)
                        mi = Chem.AddHs(_mi)
                        ms.append(mi); lbs.append(si); smiles.append(si)
                nm = len(ms)
                fmt = '%s/frag_%%0%dd'%(fd, len(str(nm)) )
                fs = [ fmt%i for i in range(1,nm+1) ]
            else:
                print('''
 #############################################################
 RDKit has problem processing some molecule with high strain
 (i.e., multiple small-membered rings). For 100% success of
 conformer generation, plz use as input SDF files produced
 by openbabel (from SMILES strings).

    obabel amons.smi -O frag_.sdf -m --gen3d slowest'
    /usr/bin/rename -- 's/(\w+)_(\d+)/sprintf("%%s_%%03d",$1,$2)/e' frag_*sdf
    for f in frag_*sdf
    do
        base=${f%%.sdf}
        obminimize -ff uff -cg -c 1e-4 $f >$base.pdb
    done
    obabel frag_*sdf -osdf -m

 Or you can use an existing shell script called `obgenc1 and
 the usage is:
    obgenc1 amons.smi


 At last, if you insist on using SMILES file as input, turn on -allow_smi

    genconf -thresh 0.1 -fd amons -allow_smi amons.smi

#############################################################
''')
                sys.exit(2)
        elif os.path.isdir(arg): #arg[-3:] in ['sdf','mol']:
            _fs = io2.Folder(arg,'sdf').fs # assume ground state geometries of amons are provided as sdf files
            if len(_fs) == 0:
                print('#ERROR: no sdf file (for ground state geometries of amons) was found!')
                print('        Try to run `optg -ff mmff99 input.smi` to generate these sdf files')
                sys.exit(2)
            fs0 += _fs
            for f in _fs:
                fs.append(fd+'/'+f[:-4].split('/')[-1])
                m1 = Chem.MolFromMolFile(f,removeHs=F)
                m1c = Chem.MolFromMolFile(f)
                smiles.append( Chem.MolToSmiles(m1c) )
                ms.append(m1); lbs.append('')
        else:
            raise '#ERROR: input format not allowed'

    nm = len(ms)
    if nj > 1:
        nav = nm/nj + 1 if nm%nj > 0 else nm/nj
        i1 = id*nav; i2 = (id+1)*nav
        if i2 > nm: i2 = nm
    else:
        i1 = 0; i2 = nm

    for i in range(i1,i2):
        _fn = fs[i]
        _lb = lbs[i]
        m = ms[i]
        _smi = smiles[i]
        print(" #Molecule %30s %20s %s"%(_fn,_lb,_smi))
        nheav = m.GetNumHeavyAtoms()
        #s1,s2 = _fn.split('_')
        fn = _fn #'%sNI%d_%s'%(s1,nheav,s2)
        _ofs = []
        if os.path.exists('%s_c001.sdf'%fn):
            _ofs = io2.cmdout('ls %s_c???.sdf'%fn)
        if len(_ofs) > 0 and (not ow): continue
        obj = EmbedMol(m)
        obj.gen_conformers(nc=nc,pruneRmsThresh=rms,nthread=nthread)
        if not obj.istat:
            assert isdf, '#ERROR: __only__ sdf is allowed in this case'
            print('#ERROR: when RDKit failed to generate any conformer for %s %s.sdf %s'%(_smi,fs0[i]))
            iok = os.system('cp %s %s_c001.sdf'%(fs0[i], _fn))
            continue
        obj.optg(n=nstep, ff=ff) # somehow, setting ff to 'mmff94' may result in 1 conformer for many mols, e.g., CCC=N
        param = {'M':'cml1','rp':1,'wz':F,'thresh':thresh}
        obj.prune_conformers(param=param)
        print("    |__ %4d conformers generated"%( obj.nconf ))
        if optg2: # further optg by MOPAC
            obj.optg2()
            obj.prune_conformers(param=param)
            print("    |__ %4d refined conformers (by MOPAC)"%( obj.nconf ))
        obj.write_conformers(fn) # ccids)
        #else:
        #    assert len(smiles)>0, '#ERROR: how comes?'
        #    obm = cob.Mol(_smiles[i],addh=T,make3d=T,ff='uff',steps=900)
        #    obm.write(fn+'_c001.sdf')
        #    print "    |__ OpenBabael was used to generate structure!"


