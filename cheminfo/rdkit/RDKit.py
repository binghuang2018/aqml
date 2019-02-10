
import rdkit
from rdkit import Chem
from rdkit.Chem import AllChem, rdMolTransforms, rdMolDescriptors
from rdkit.Chem import rdDepictor
from rdkit.Chem import ChemicalForceFields as cff
import rdkit.Chem.rdForceFieldHelpers as rcr
from rdkit.Geometry.rdGeometry import Point3D
import os, sys, re
import numpy as np
import ase.data as ad
import ase.units as au
import ase.io as aio
import ase, copy
import tempfile as tpf
import cheminfo.openbabel.obabel as cib
import cheminfo.math as cim
import cheminfo.graph as cg
import scipy.spatial.distance as ssd
import itertools as itl
import multiprocessing
import copy_reg
import types as _TYPES

global cnsDic, dic_bonds, dic_atypes, h2kc, c2j, dsHX, dic_hyb
cnsDic = {5:3, 6:4, 7:3, 8:2, 9:1, 14:4, 17:1}
dic_bonds = { Chem.BondType.SINGLE:1.0,
              Chem.BondType.DOUBLE:2.0,
              Chem.BondType.TRIPLE:3.0,
              Chem.BondType.AROMATIC:1.5,
              Chem.BondType.UNSPECIFIED:0.0}
_btypes = {1.0: Chem.BondType.SINGLE,
           2.0: Chem.BondType.DOUBLE,
           3.0: Chem.BondType.TRIPLE,
           1.5: Chem.BondType.AROMATIC,
           0.0: Chem.BondType.UNSPECIFIED }

dic_atypes =  { 'H': ['H',], \
                'F': ['F',], \
                'Cl': ['Cl',],\
                'O': ['O_3',  'O_2',  'O_R',], \
                'S': ['S_3', 'S_2',  'So3',  'S_R', ], \
                'N': ['N_3', 'N_2', 'N_1', 'N_R', ], \
                'C': ['C_3', 'C_2', 'C_1',  'C_R', ]}
h2kc = au.Hartree * au.eV/(au.kcal/au.mol)
c2j = au.kcal/au.kJ

dsHX = {5:1.20, 6:1.10, 7:1.00, 8:0.98, 9:0.92, 14:1.48, 15:1.42, 16:1.34, 17:1.27}

dic_hyb = { Chem.rdchem.HybridizationType.SP3: 3, \
            Chem.rdchem.HybridizationType.SP2: 2, \
            Chem.rdchem.HybridizationType.SP: 1, \
            Chem.rdchem.HybridizationType.UNSPECIFIED: 0}

## register instance method
## otherwise, the script will stop with error:
## ``TypeError: can't pickle instancemethod objects
def _reduce_method(m):
    if m.im_self is None:
        return getattr, (m.im_class, m.im_func.func_name)
    else:
        return getattr, (m.im_self, m.im_func.func_name)
copy_reg.pickle(_TYPES.MethodType, _reduce_method)


class Molecule(object):

    def __init__(self, f, threshR=0.6):
        #assert os.path.exists(f)
        self.threshR = threshR
        atoms = aio.read(f)
        self.na = len(atoms)
        self.zs = atoms.numbers
        self.coords = atoms.positions
        og = cg.graph(self.zs, self.coords)
        og.perceive_connectivity()
        self.g = og.g
        self.ds = og.ds #ssd.squareform( self.dsU )

    def check_sanity(self):
        """
        check if the coord_num is rational ?
        """
        filt = np.logical_and(self.ds > 0, self.ds < self.threshR)
        if np.any(filt):
            print np.where( filt )

        nnbs = self.g.sum(axis=0)
        dic = { 1:[1], 6:[1,2,3,4], 7:[1,2,3,5], 8:[1,2], 9:[1], \
               14:[1,2,3,4], 16:[1,2,3,4], 17:[1] }
        for i in range(self.na):
            zi = self.zs[i]
            if nnbs[i] not in dic[zi]:
                print 'i, zi, coord_num = ', i, self.zs[i], nnbs[i]


class RDMols(object):
    def __init__(self, strings, nproc=2, forcefield='mmff94', doff=False, \
                 ih=True, kekulize=False):
        pool = multiprocessing.Pool(processes=nproc)
        ipts = [ [string,ih] for string in strings ]
        self.objs = pool.map(self.processInput, ipts)

    def processInput(self, ipt):
        string, ih = ipt
        obj = RDMol(string, ih=ih)
        return obj


class RDMol(object):
    """
    enhanced RDKit molecule class with extended functionality
    """
    def __init__(self, obj, isortH=False, forcefield='mmff94',\
                 doff=False, steps=500, ih=True, kekulize=False, \
                 sanitize=True, use_ob=True):
        self.forcefield = forcefield
        self.ih = ih
        self.steps = steps
        self.kekulize = kekulize

        ismi = False
        hasCoord = True
        if type(obj) is str:
            if os.path.exists(obj):
                fmt = obj[-3:]
                assert fmt in ['sdf','mol','pdb']
                # default:  `strictParsing=True
                # then error like this shows up
                # [06:01:51] CTAB version string invalid at line 4
                if fmt in ['sdf','mol']:
                    m1 = Chem.MolFromMolFile(obj,removeHs=False,strictParsing=False)
                elif fmt in ['pdb',]:
                    # some pdb file may not contain H
                    m1 = Chem.MolFromPDBFile(obj,removeHs=False)
                else:
                    raise ' #ERROR: format not recognized'
                zs = [ ai.GetAtomicNum() for ai in m1.GetAtoms() ]
                assert np.all(zs[zs.index(1)+1:] == 1), '#ERROR: not all H atoms at the end?'

                # in case H's appear before the last heavy atom,
                # fix this
                if isortH and (1 in zs):
                  if np.all(zs[zs.index(1)+1:] > 1):
                    m1_skeleton = Chem.MolFromMolFile(obj,removeHs=True,strictParsing=False)
                    m2 = Chem.AddHs(m1_skeleton)
                    self.m = m2
                    self.na = m2.GetNumAtoms()
                    coords = get_coords(m1)
                    coords_heav = []; coords_Hs = []
                    for i, zi in enumerate(zs):
                      if zi == 1:
                        coords_Hs.append( coords[i] )
                      else:
                        coords_heav.append( coords[i] )
                    coords_u = coords_heav + coords_H
                    mu = self.update_coords(coords_u)
                else:
                  mu = m1
            else:
                ismi = True
                m = Chem.MolFromSmiles(obj,sanitize=sanitize); m2 = Chem.RemoveHs(m)
# H's always appear after heavy atoms
                mu = Chem.AddHs(m)
                hasCoord = False
        elif obj.__class__.__name__ == 'Mol': # rdkit.Chem.rdchem.Mol
            mu = obj
        else:
            raise '#ERROR: non-supported type'

        if doff:
            # sometimes u encounter error messages like
            # ValueError: Bad Conformer Id
            # if u use RDKit
            #self.get_coarse_geometry()
            # Here we use openbabel to get initial geometry
            hasCoord = True
            if ismi:
                if use_ob:
                    s2 = cib.Mol( obj, make3d=True, steps=steps )
                    mu = s2.to_RDKit()
                else:
                    mu = self.get_coarse_geometry(mu)

        self.m0 = copy.deepcopy(mu)
        if not ih:
            mu = Chem.RemoveHs(mu)

        self.m = mu
        self.bom = get_bom(mu, kekulize=kekulize)
        self.na = mu.GetNumAtoms()
        self.nb = mu.GetNumBonds()
        self.ias = np.arange(self.na).astype(np.int)
        self.zs = np.array([ ai.GetAtomicNum() for ai in mu.GetAtoms() ], np.int)

# assertion should hold; otherwise wierd molecules result when generating amons
#        assert np.all(zs[list(zs).index(1)+1:] > 1)

        self.ias_heav = self.ias[ self.zs > 1 ]
        self.nheav = len(self.ias_heav)
        #if hasCoord:
        #    self.coords = get_coords(mu)
        self.iFFOpt = False # geom optimized by FF ?

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
        # get formal charges
        self.chgs = np.array([ ai.GetFormalCharge() for ai in self.m0.GetAtoms() ])
        self.vs = np.array([ ai.GetTotalValence() for ai in self.m0.GetAtoms() ], np.int)
        #self.update_bom()
        self.ias_heav = self.ias[ self.zs > 1 ]
        bom_heav = self.bom[ self.ias_heav, : ][ :, self.ias_heav ]
        self.vs_heav = bom_heav.sum(axis=0)
        self.cns_heav = ( bom_heav > 0 ).sum(axis=0)

        self.cns = ( self.bom > 0).sum(axis=0)
        self.nhs = self.vs[:self.nheav] - self.vs_heav - self.chgs[:self.nheav]
        self.dvs = self.vs_heav - self.cns_heav
        self.hybs = np.array([ dic_hyb[ai.GetHybridization()] for ai in self.m.GetAtoms() ])

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

    def get_ring_nodes(self, namin=3, namax=9):

        """
        get nodes of `namin- to `namax-membered ring

        We focus on those nodes which constitute the
        `extended smallest set of small unbreakable fragments,
        including aromatic rings, 3- and 4-membered rings
        (accompanied with high strain typically)
        """


        self.get_backbone()
        m = self.mbb
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

        # now remove those rings that are union of smaller rings
        n = len(sets)
        sets_remove = []
        ijs = itl.combinations( range(n), 2 )
        sets_u = []
        for i,j in ijs:
            set_ij = sets[i].union( sets[j] )
            if set_ij in sets and (set_ij not in sets_remove):
                sets_remove.append( set_ij )
        sets_u = cim.get_compl(sets, sets_remove)
        return sets_u

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
            ins = ( dic_bonds[ bi.GetBondType() ] > 1 ) # is non-single bond?
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
                raise '#ERROR:?'

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
            dic = dict(zip(ias, zs_i))
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
                if debug: print ' ics, tsi = ', ics, tsi
                assert len(ics) == 1, '#ERROR: there should be only one heavy atom with maxiaml degree!'
                #ic = ics[0]
                iass_i = m.GetSubstructMatches(Qi)
                for ias in iass_i:
                    #ias = np.array(ias)
                    mqs.append( Qi )
                    tss.append( tsi )
                    dics.append( dict(zip(ias, zs_i)) )
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
                            print 'ia2,ja2 = ', ia2,ja2
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
                ifs = range(ng)
                for i in ifs:

                    ias = iass[i]
                    mi = mqs[i]; #mic = mqs[i]
                    na1 = len(ias); #na1c = len(ias)
                    dic_i = dics[i]; #dic_ic = dics[i]
                    jfs = list( set(ifs)^set([i]) )

                    if debug: print 'i, mi, ias = ', i, tss[i], ias
                    #print ' -- i = ', i

                    icnt = 0
                    cnode = cnodes[i]
                    for j in jfs:
                        #print '    icnt = ', icnt
                        mj = mqs[j]
                        jas = iass[j]
                        if debug:
                            print '   j, mj, jas = ', j, tss[j], jas
                            if icnt > 0:
                                print '      mi, ias = ', '', patt, ias
                                print '      dic_i = ', dic_i
                            else:
                                print '      _mi, ias = ', '', tss[i], ias
                        dic_j = dics[j]
                        kas = list( set(ias).intersection( set(jas) ) )
                        #print '  -- cnode, kas = ', cnode, kas
                        if ( len(kas) == 2 ) and ( cnode in set(kas) ):
                            if debug:
                                print '   -- kas = ', kas
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
                                print '     -- patt = ', patt

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
                                    print '     -- ias = ', ias
                                na1 = len(ias)

                            icnt += 1
                    try:
                        smi = Chem.MolToSmiles( Chem.MolFromSmarts(patt), canonical=True )
                        smi = re.sub('\-', '', smi)
                        smi = re.sub('\*', heav_smarts, smi)

                        if smi not in ts2: ts2.append(smi)
                    except:
                        pass
                        print '   icnt = ', icnt
                        print '   j, mj, jas = ', j, tss[j], jas
                        print '   i, mi, ias = ', i, tss[i], ias
                return ts2
            else:
                raise '#ERROR: not implemented'

    def check_abnormal_local_geometry(self):
        qs = [ '[*]C(=O)O[*]', '[*]C(=O)N[*]', ]
        return


    def to_can(self):
        ctab = Chem.MolToMolBlock( self.m )
        mu = Chem.MolFromMolBlock( ctab, removeHs=True )
        return Chem.MolToSmiles( mu )

    def get_energy(self):
        return get_forcefield_energy(self.m, self.forcefield)

    def get_coarse_geometry(self, m):
        # default case: use MMFF94 to optimize geometry
        AllChem.EmbedMolecule(m, AllChem.ETKDG())
                              #, useExpTorsionAnglePrefs=True, \
                              #useBasicKnowledge=True)
        ff = {'uff': AllChem.UFFOptimizeMolecule, \
               'mmff94': AllChem.MMFFOptimizeMolecule }[ self.forcefield.lower() ]
        ff(m, maxIters=self.steps) #, confId=0)
        self.iFFOpt = True
        return m

    def get_descent_geometry(self, label=None):
        # use MOPAC/PM7 to further optimize
        assert self.iFFOpt, '#ERROR: Plz call `get_coarse_geometry() first'
        self.to_Atoms()
        na = self.na
        s = 'PM7 BFGS'
        s += '\nTitle: ASE\n\n'
        # Write coordinates:
        for ai in self.atoms:
            symbol = ad.chemical_symbols[ ai.number ]
            xyz = ai.position
            s += ' {0:2} {1} 1 {2} 1 {3} 1\n'.format(symbol, *xyz)
        if label is None:
            label = tpf.NamedTemporaryFile(dir='/tmp').name
        try:
            exe = os.environ['MOPAC']
        except:
            raise '#ERROR: Plz do `export MOPAC=/path/to/MOPAC/executable'
        iok = os.system( '%s %s.mop 2>/dev/null'%(exe, label) )
        if iok > 0:
            raise '#ERROR: MOPAC failed !!'
        cmd = "sed -n '/                             CARTESIAN COORDINATES/,/Empirical Formula:/p' %s"%opf
        conts = os.popen(cmd).read().strip().split('\n')[2:-3]
        if not os.path.exists('../trash'): os.system('mkdir ../trash')
        iok = os.system('mv %s.arc %s.mop %s.out ../trash/'%(label,label,label))
        symbs = []; coords = []
        atomsU = ase.Atoms([], cell=[1,1,1])
        for k in range(na):
            tag, symb, px, py, pz = conts[k].strip().split()
            coords_i = np.array([px,py,pz]).astype(np.float)
            coords.append( coords_i )
            atomsU.append(ase.Atom(symb, coords_i))
        self.atoms = atomsU
        c1 = self.m.GetConformer(-1)
        for i in range(na):
            pi = Point3D()
            pi.x, pi.y, pi.z = coords[i]
            c1.SetAtomPosition(i, pi)
            #self.m.GetConformer(-1)

    def get_backbone(self):
        # first check if `mbb is ready (molecular backbone, i.e.,
        # a molecule with all H's removed
        if not hasattr(self,'mbb'):
            m1 = copy.deepcopy( self.m0 )
            m2 = Chem.RemoveHs(m1)
            self.mbb = m2

    def enum_torsions(self,wH=False):
        """
        enumerate Torsions in a molecule
        """
        mr = self.m
        if self.ih and (not iH):
            self.get_backbone()
            mr = self.mbb
        torsionSmarts = '[!$(*#*)&!D1]~[!$(*#*)&!D1]'
        torsionQuery = Chem.MolFromSmarts(torsionSmarts)
        matches = mr.GetSubstructMatches(torsionQuery)
        iass4 = [] #torsion List
        for match in matches:
            idx2 = match[0]
            idx3 = match[1]
            bond = mr.GetBondBetweenAtoms(idx2, idx3)
            jAtom = mr.GetAtomWithIdx(idx2)
            kAtom = mr.GetAtomWithIdx(idx3)

            iok1 = (jAtom.GetHybridization() != Chem.HybridizationType.SP2)
            iok2 = (jAtom.GetHybridization() != Chem.HybridizationType.SP3)
            iok3 = (kAtom.GetHybridization() != Chem.HybridizationType.SP2)
            iok4 = (kAtom.GetHybridization() != Chem.HybridizationType.SP3)
            if (( iok1 and iok2 ) or ( iok3 and iok4 )):
                continue
            for b1 in jAtom.GetBonds():
                if (b1.GetIdx() == bond.GetIdx()):
                    continue
                idx1 = b1.GetOtherAtomIdx(idx2)
                for b2 in kAtom.GetBonds():
                    if ((b2.GetIdx() == bond.GetIdx()) or \
                            (b2.GetIdx() == b1.GetIdx())):
                        continue
                    idx4 = b2.GetOtherAtomIdx(idx3)
                    # skip 3-membered rings
                    if (idx4 == idx1):
                        continue
                    iass4.append( [idx1, idx2, idx3, idx4] )
        self.iass4 = iass4

    def enum_angles(self,wH=False):
        mr = self.m
        if self.ih and (not iH):
            self.get_backbone()
            mr = self.mbb

        if not iH: m2 = self.get_backbone()
        atoms = m2.GetAtoms()
        na = len(atoms)
        iass3 = []
        for j in range(na):
            neighbors = atoms[j].GetNeighbors()
            nneib = len(neighbors)
            if nneib > 1:
                for i0 in range(nneib):
                    for k0 in range(i0+1,nneib):
                        i, k = [ neighbors[i0].GetIdx(), neighbors[k0].GetIdx() ];
                        ias = [i,j,k]
                        if (ias not in iass3) or (ias[::-1] not in iass3):
                            iass3.append(ias)
        self.iass3 = iass3

    def fixTorsionOpt(self,maxIters=200,iH=False):
        """
        fix all dihedral angles and optimize geometry,
        used for generating amons with local chemical
        enviroments as close to that in parent molecule
        as possible

        iH -- if H is considered to be part of the strain?
        """

        mu = self.m
        self.enum_torsions(iH=iH)
        #c = mu.GetConformer()
        mp = AllChem.MMFFGetMoleculeProperties(mu)
        ff = AllChem.MMFFGetMoleculeForceField(mu, mp, \
                ignoreInterfragInteractions=False)
        for ias_i in self.iass4: #[:1]:
            i, j, k, l = ias_i
            ff.MMFFAddTorsionConstraint(i, j, k, l, True, -0.0, 0.0, 9999)
        ff.Minimize(maxIts=maxIters)
        self.m = mu
        coords_u = get_coords( mu )
        self.coords = coords_u
        self.update_coords(coords_u)

    def Opt(self,maxIters=200):
        """
        full relaxation of geometries using MMFF94
        """
        mu = self.m
        c = mu.GetConformer()
        mp = AllChem.MMFFGetMoleculeProperties(mu)
        ff = AllChem.MMFFGetMoleculeForceField(mu, mp, \
                ignoreInterfragInteractions=False)
        ff.Minimize(maxIts=maxIters)
        coords_u = get_coords( mu )
        self.coords = coords_u
        self.update_coords(coords_u)
        self.energy = ff.CalcEnergy()

    def to_Atoms(self):
        """
        rdMol object to ase.Atoms object
        """
        #assert self.m.GetNumConformers() == 1
        coords = get_coords(self.m)
        self.atoms = ase.Atoms(self.zs, coords)

    def Atoms(self):
        self.to_Atoms()
        return self.atoms

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

    def write_sdf(self, sdf):
        Chem.MolToMolFile(self.m, sdf)

    def write_pdb(self, pdb):
        # using `flavor=4 and `Chem.MolToPDBBlock is the safest way
        open(pdb,'w').write( Chem.MolToPDBBlock(self.m, flavor=4) )

    def write_xyz(self, f):
        self.to_Atoms()
        aio.write(f, self.atoms)

    def get_stablest_conformer(self, nconfs=50, nthread=1):

        """
        generate a series of conformers and
        choose the lowest energy conformer
        """

        ctab = Chem.MolToMolBlock( self.m )

        if self.nheav >= 3:
            seeds = [1, 4, 7, 10, ]
            emin = 99999.
            for seed in seeds: # different seed results in different initial ref geometry
                cids = AllChem.EmbedMultipleConfs(self.m, nconfs, numThreads=nthread, \
                          pruneRmsThresh=1.0, randomSeed=seed) #, \
                          # useBasicKnowledge=True, useExpTorsionAnglePrefs=True)
                for cid in cids:
                    _ = AllChem.MMFFOptimizeMolecule(self.m, confId=cid, maxIters=500)
                props = AllChem.MMFFGetMoleculeProperties(self.m, mmffVariant='MMFF94')
                es = []
                for i in range(nconfs):
                    try:
                        ff = AllChem.MMFFGetMoleculeForceField(self.m, props, confId=i)
                        e = ff.CalcEnergy()
                        es.append( e )
                    except:
                        pass # such conformer Id does not exist
                if len(es) > 1:
                    emin_ = min(es)
                    if emin_ <= emin:
                        emin = emin_
                        cid_u = es.index( emin_ )
                        conf = self.m.GetConformer(cid_u)
                        #mU = Chem.Mol(self.m, cid_u) # conf.GetId() )
                        #self.m = mU
                        coords_u = conf.GetPositions()
                        self.atoms = ase.Atoms(self.zs, coords_u)
                        self.update_coords(coords_u)
                else:
                    self.m = Chem.MolFromMolBlock( ctab, removeHs=False )
                    break #print ' -- ooooops, no conformer found'


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
                    print ' -- i, charges[i] = ', i, self.charges[i]
                    raise ' #ERROR: abs(charge) > 1??'
            mc.AddAtom( ai )

        ijs = np.array( np.where( np.triu(self.bom) > 0 ) ).astype(np.int)
        nb = ijs.shape[1]
        for i in range(nb):
            i, j = ijs[:,i]
            mc.AddBond( i, j, _btypes[ self.bom[i,j] ] )

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
                    print 'i, zi, ci, nH = ', self.zs[i], ci, numHs[i]
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
                raise ' unknown atom type: `%s`'%hybi
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
            boi = dic_bonds[ bi.GetBondType() ]
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
                            print 'not supported atomic type: %s'%apj
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
    for i in xrange(na):
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


class SMILES(object):

    def __init__(self, smi, canonical=True):
        self.canonical = canonical
        self.smi = smi

    def fix_aromaticN(self):
        m = Chem.MolFromSmiles(self.smi,False)
        try:
            m.UpdatePropertyCache(False)
            cp = Chem.Mol(m.ToBinary())
            Chem.SanitizeMol(cp)
            m = cp
            #print 'fine:',Chem.MolToSmiles(m)
            iok = True
        except ValueError:
            nm = AdjustAromaticNs(m)
            iok = False
            if nm is not None:
                Chem.SanitizeMol(nm)
                #print 'fixed:',Chem.MolToSmiles(nm)
                smi = Chem.MolToSmiles(nm)
                iok = True
            else:
                print 'still broken:',smi
        self.smi = smi
        self.iok = iok

    def fix_charges(self):
        if self.canonical:
            smiU = neutralise_smiles(self.smi)
            self.smi = smiU
        else:
            # you can also retain the order?
            # use similar codes as in `neutralise()
            pass


#====================================================================================================
""" sanifix4.py
  Contribution from James Davidson
"""

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
            res,indices = _recursivelyModifyNs(nm,matches,indices=tIndices)
        else:
            indices=tIndices
            res=cp
    return res,indices

def AdjustAromaticNs(m,nitrogenPattern='[n&D2&H0;r5,r6]'):
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
    frags = [_FragIndicesToMol(nm,x) for x in fragLists]

    # loop through the fragments in turn and try to aromatize them:
    ok=True
    for i,frag in enumerate(frags):
        cp = Chem.Mol(frag.ToBinary())
        try:
            Chem.SanitizeMol(cp)
        except ValueError:
            matches = [x[0] for x in frag.GetSubstructMatches(Chem.MolFromSmarts(nitrogenPattern))]
            lres,indices=_recursivelyModifyNs(frag,matches)
            if not lres:
                #print 'frag %d failed (%s)'%(i,str(fragLists[i]))
                ok=False
                break
            else:
                revMap={}
                for k,v in frag._idxMap.iteritems():
                    revMap[v]=k
                for idx in indices:
                    oatom = m.GetAtomWithIdx(revMap[idx])
                    oatom.SetNoImplicit(True)
                    oatom.SetNumExplicitHs(1)
    if not ok:
        return None
    return m
#====================================================================================================

def neutralise_smiles(smi):
    """
    neutralise the SMILES of a molecule

    Attention: the order of atoms is not retained!!
    """
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
    smi = Chem.MolToSmiles(m, isomericSmiles=False) #, canonical=False)
    return smi

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
        mc.AddBond(i,j,_btypes[bij])
        mc.AddBond(i,k,_btypes[bik])
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
        bom[i,j] = bom[j,i] = dic_bonds[ bt ]
    return bom

def get_PLs(m):
    """
    get path length matrix
    """
    return Chem.GetDistanceMatrix(m)

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


if __name__ == "__main__":

    smi =  'N#CC1=CC=C(ON2P(OC3=CC=C(C=C3)C#N)N=P(OC3=CC=C(C=C3)C#N)(OC3=CC=C(C=C3)C#N)N(OC3=CC=C(C=C3)C#N)P2OC2=CC=C(C=C2)C#N)C=C1'
    o = cir.RDMol( smi, ih=False, doff=False)
    level = 1
    print ' building blocks, level 1\n', o.get_building_blocks(level=level)

