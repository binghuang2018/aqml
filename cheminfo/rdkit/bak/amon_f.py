#!/usr/bin/env python

"""
Enumerate subgraphs & get amons
"""

from cheminfo import *
import cheminfo.math as cim
import cheminfo.graph as cg
from cheminfo.rw.ctab import write_ctab
from cheminfo.rw.pdb import write_pdb
import networkx as nx
from itertools import chain, product
import numpy as np
import os, re, copy, time
from rdkit import Chem
import pybel as pb

global dic_bonds
dic_bonds = { Chem.BondType.SINGLE:1.0,
              Chem.BondType.DOUBLE:2.0,
              Chem.BondType.TRIPLE:3.0,
              Chem.BondType.AROMATIC:1.5,
              Chem.BondType.UNSPECIFIED:0.0}
#dic_hyb = { Chem.rdchem.HybridizationType.SP3: 3, \
#            Chem.rdchem.HybridizationType.SP2: 2, \
#            Chem.rdchem.HybridizationType.SP: 1, \
#            Chem.rdchem.HybridizationType.S: 0, \
#            Chem.rdchem.HybridizationType.UNSPECIFIED: 0}

ves = {1:[1], 3:[1], 4:[2], 5:[3], 6:[4], 7:[3], 8:[2], 9:[1], \
       11:[1], 12:[2], 13:[3], 14:[4], 15:[3,5], 16:[2,4,6], 17:[1]}
vsmax = { 3:1, 4:2, 5:3, 6:4,7:3,8:2,9:1, 14:4,15:5,16:6,17:1, 32:4,33:3,34:2,35:1, 50:4,51:3,52:2,53:1 }

def to_can_obabel_rdkit(blk, fmt):
    """
    there is a bug in rdkit_2017_09_1,
    i.e., it cannot convert sdf file for a SMILES like 'OP(=S)=P(=PC)C' to SMILES,
    with corresponding error messages:
    ======================================
        RuntimeError: Pre-condition Violation
            bad bond type
            Violation occurred on line 29 in file Code/GraphMol/Canon.cpp
            Failed Expression: bond->getBondType() == Bond::SINGLE || bond->getIsAromatic()
            RDKIT: 2017.09.1
            BOOST: 1_58
    We can circumvent this through the use of openbabel
    """
    _mb = pb.readstring(fmt,blk)
    _sb = _mb.write('smi').split('\t')[0]; print ' * smile_ob = ', _sb
    _mr = Chem.MolFromSmiles(_sb)
    return Chem.MolToSmiles(_mr)


def find_cliques(g1):
    """
    the defintion of `clique here is not the same
    as that in graph theory, which states that
    ``a clique is a subset of vertices of an
    undirected graph such that every two distinct
    vertices in the clique are adjacent; that is,
    its induced subgraph is complete.''
    However, in our case, it's simply a connected
    subgraph, or a fragment of molecule. This is useful
    only for identifying the conjugated subset of
    atoms connected all by double bonds or bonds of
    pattern `double-single-double-single...`
    """
    n = g1.shape[0]
    G = nx.Graph(g1)
    if nx.is_connected(G):
        cliques = [ range(n), ]
    else:
        cliques = []
        sub_graphs = nx.connected_component_subgraphs(G)
        for i, sg in enumerate(sub_graphs):
            cliques.append( sg.nodes() )
    return cliques


class RawMol(object):
    """
    molecule object with only `zs & `coords
    """
    def __init__(self, zs, coords):
        self.zs = zs
        self.coords = coords

    def generate_coulomb_matrix(self):
        """ Coulomb matrix"""
        na = len(self.zs)
        mat = np.zeros((na,na))
        ds = ssd.squareform( ssd.pdist(self.coords) )
        np.fill_diagonal(ds, 1.0)
        X, Y = np.meshgrid(self.zs, self.zs)
        mat = X*Y/ds
        np.fill_diagonal(mat, -np.array(self.zs)**2.4 )
        L1s = np.linalg.norm(mat, ord=1, axis=0)
        ias = np.argsort(L1s)
        self.cm = mat[ias,:][:,ias].ravel()


def accommodate_chgs_0(zs, chgs, bom):
    """update bom based on `chgs
    e.g., C=N#N, bond orders = [2,3],
    Considering that `chgs = [0,+1,-1],
    bond orders has to be changed to [2,2]"""
    bom2 = copy.copy(bom)
    na = len(chgs); nheav = (zs>1).sum()
    ias = np.arange(na)
    achgs = np.abs(chgs)
    _ias1 = ias[achgs == 1]
    na1 = len(_ias1)
    assert na1 % 2 == 0
    g1 = (bom[_ias1][:,_ias1] >0).astype(np.int)
    cns = g1.sum(axis=0)
    assert np.all(cns > 0)

    ias1 = ias[chgs == 1] # we care only about +1 charged atoms; -1 charged atom is paired
    filt = np.logical_and(cns == 1, chgs[_ias1] == 1)
    if np.any(filt):
        iasg_ar = _ias1[ filt ]; iasg = list(iasg_ar)
    else:
        iasg = ias1[0]; iasg_ar = np.array(iasg,np.int)

    chgs2 = copy.deepcopy(chgs)

    print ' \nnew mol'
    print ' chgs = ', chgs
    print ' zs = ', zs
    print ' bom = ', bom[:nheav][:,:nheav]

    visited = []
    while len(visited) < na1:
      for i in ias1:
        iasc = ias[ np.logical_and(chgs==-1, bom[i]>0) ]
        if len(iasc) == 0: continue
        #assert len(iasc) == 1 # in the case of "C[P+]1(O)[N-][P+](O)(O)[N-][P+](O)(O)[N-]1", len(iasc) = 2
        j = iasc[0]
        #print 'i,j,boij=', i,j, bom[i,j]
        if (i in visited) or (j in visited): continue
        visited += [i,j]
        print ' now visiting (',i,j,')', ', visited=',visited

        icon = False # proceed for update?
        conn = bom[i]
        if zs[i] == 7: # e.g., RN$C, RN(=O)=O, atom N has a valence >3
            if conn.sum() > 3: icon = True
        else:
            # set chg to 0
            print ' ---------- reset chg to 0 for ', i,j
            chgs2[i] = 0; chgs2[j] = 0
            chgs = chgs2

        if icon:
            print ' ---------------- i = ', i
            bij = bom[i,j] - 1
            bom2[i,j] = bij
            bom2[j,i] = bij
    return bom2, chgs2


def accommodate_chgs(zs, chgs, bom, allow_bb=False):

    """
    update bom based on `chgs
    e.g., C=N#N, bond orders = [2,3],
    Considering that `chgs = [0,+1,-1],
    bond orders has to be changed to [2,2]

    vars
    ================
    allow_bb: allow bond breaking? Default to False;
              It's set to "True" only for debugging
    """

    iok = True

    vs = bom.sum(axis=0)
    achgs = np.abs(chgs)

    bom2 = copy.copy(bom)
    chgs2 = copy.copy(chgs)

    na = len(chgs)
    ias = np.arange(na)

    _ias1 = ias[achgs == 1]
    if len(_ias1) == 0:
        return iok, bom2,chgs2
    g1= bom[_ias1][:,_ias1]
    if np.all(g1 == 0):
        iasr_g1 = [ [ja] for ja in range(len(_ias1)) ]
    else:
        iasr_g1 = find_cliques(g1)
    for iasr_sg in iasr_g1:
        iasg = list(_ias1[iasr_sg])
        nag = len(iasr_sg)
        assert chgs[iasg].sum() == 0 and nag%2 == 0 # must be in pairs and neutral
        _sg = ( bom[iasg][:,iasg] > 0 ).astype(np.int)
        neibs = _sg.sum(axis=0)
        pairs = []
        if nag == 2:
            pairs = [ iasg ]
        else:
            visited = set([])
            while len(visited) < nag:
                iasg_ar = np.array(iasg)
                jas = iasg_ar[neibs==1]
#                print ' -- jas = ', jas
                jas2 = iasg_ar[neibs==2]
                if len(jas) > 0:
                    _j = 0
                    ja = jas[_j]
                    j = iasg.index(ja) ## attention here!
                    #print ' _sg = ', _sg
                    kas = iasg_ar[_sg[j]==1]
#                    print '** ', iasg, visited
#                    print '** ', kas
#                    print '** ', ja,kas[0],chgs[ja],chgs[kas[0]]
                    assert len(kas) == 1 and chgs[ja]+chgs[kas[0]]==0
                    pairs.append( [ja,kas[0]] )
                    visited.update( [ja,kas[0]] )
                else:
                    ja = jas2[0]
                    ka = iasg_ar[_sg[0]==1][0]
                    assert chgs[ja]+chgs[ka] == 0
                    visited.update( [ja,ka] )
                    pairs.append( [ja,ka] )

                iasg = list( set(iasg).difference(visited) )
                _sg = ( bom[iasg][:,iasg] > 0).astype(np.int)
                neibs = _sg.sum(axis=0)

        for pair in pairs:
            i,j = pair
            cbo = False # change bo?
            iaN = None
            zi = zs[i]; zj = zs[j]
            if 7 in [zi,zj]:
                if zi == 7 and chgs[i] == 1:
                    iat = i; vmax = 3; try_cbo = True
                elif zj == 7 and chgs[j] == 1:
                    iat = j; vmax = 3; try_cbo = True
                else:
                    try_cbo = False # -[P+](R)(R)(R)[N-]C
            elif 8 in [zi,zj]:
                if zi == 8 and chgs[i] == 1:
                    iat = i; vmax = 2; try_cbo = True
                elif zj == 8 and chgs[j] == 1:
                    iat = j; vmax = 2; try_cbo = True
                else:
                    try_cbo = False # e.g., -[P+](R)(R)(R)[O-]
            else:
                try_cbo = False

            if try_cbo:
                if vs[iat] > vmax:
                    #print ' -- cbo '
                    boij = bom[i,j] - 1
                    if boij == 0 and (not allow_bb):
                        # e.g., query: "O=C[CH-][NH2+]C=C", a possible fragment
                        # is "C=[CH-].[NH2+]=C" if allow_bb=True, which is apparently
                        # an invalid amon
                        iok = False
                        break
                    bom2[i,j] = boij
                    bom2[j,i] = boij
                    cbo = True
            if not cbo:
                chgs2[i] = 0; chgs2[j] = 0
                #print ' -- cchg'

        if (not iok) and (not allow_bb): break

    return iok, bom2, chgs2



class vars(object):
    def __init__(self, bosr, zs, chgs, tvs, g, coords):
        self.bosr = bosr
        self.zs = zs
        self.chgs = chgs
        self.tvs = tvs
        self.g = g
        self.coords = coords


class MG(vars):

    def __init__(self, bosr, zs, chgs, tvs, g, coords, use_bosr=True):
        """
        use_bosr: set to True for generating amons, i.e., we need the
                  bond orders between the atom_i and all its neighbors,
                  where `i runs through 1 to N_A;
                  It must be set to False when inferring the BO's between
                  atoms given only the xyz file, i.e., with graph being
                  the only input
        """
        vars.__init__(self, bosr, zs, chgs, tvs, g, coords)
        self.use_bosr = use_bosr


    def update_m(self, once=True, debug=False, icon=False):

        try:
            import cheminfo.fortran.famon as cf
        except:
            import cheminfo.fortran.famon_mac as cf

        g = self.g
        chgs = self.chgs
        vs = g.sum(axis=0).astype(np.int)
        tvs = self.tvs # `tvs has been modified according to `chgs
        zs = self.zs
        bosr = self.bosr
        na = len(zs)
        ias = np.arange(na)

        #icon = True
        if icon:
            print ' zs = ', zs
            print 'tvs = ', tvs
            print 'dvs = ', tvs - vs

        #print 'g = ', g
        #t1 = time.time()
        #print ' ## e1'
        nrmax = na/2
        nbmax = (g>0).sum()/2
        iok, bom = cf.update_bom(nrmax,nbmax,zs,tvs,g,icon)
        if icon: print '     +++ Passed with `iok = ', iok


        #t2 = time.time()
        #print '      update_m: ', t2-t1
        #print ' ** iok = ',iok
        #print ' ** bom = ', bom
        if not iok:
            #print ' zs = ', zs
            #print ' vs = ', vs
            #print 'tvs = ', tvs
            #print ''
            return [],[]

        boms = [bom]
        cans = []; ms = []
        iok = True
        for bom in boms:

            # note that the order of calling `get_bos() and `accommodate_chgs()
            #  matters as `bosr was obtained based on modified `bom, i.e., all
            # pairs of positive & negative charges (the relevant two atoms are
            # bonded) were eliminated
            bos = get_bos(bom)

            # now restore charges for case, e.g., NN bond in C=N#N, or -N(=O)=O
            iok_U, bom_U, chgs_U = accommodate_chgs(zs, chgs, bom, allow_bb=False)
            if not iok_U: continue
            vs = bom_U.sum(axis=0)
            #print ' ** vs = ', vs


            # for query molecule like -C=CC#CC=C-, one possible amon
            # is >C-C-C-C< with dvs = [1,2,2,1] ==> >C=C=C=C<, but
            # apparently this is not acceptable!! We use `obsr to
            # kick out these fragments if `use_bosr is set to .true.
            #ipass = True
            #print ' __ bos[zs>1] = ', bos[zs>1]
            #print ' __ bosr      = ', bosr
            if self.use_bosr:
                #print ' -- bos = ', bos
                if np.any(bos[zs>1] != bosr):
                    #print ' bosr = ', bosr, ', bos = ', bos[zs>1]
                    #ipass = False
                    continue

            t1 = time.time()

            isotopes = []
            obsolete = """# handle multivalent cases
            #    struct                obabel_amons
            # 1) R-N(=O)=O,            O=[SH2]=O
            # 2) R1-P(=O)(R2)(R3)
            # 3) R-S(=O)-R,
            # 4) R-S(=O)(=O)-R
            # 5) R-Cl(=O)(=O)(=O), one possible amon is
            # "O=[SH2]=O", however,
            # openbabel cannot succeed to add 2 extra H's. We can circumvent this
            # by using isotopes of H's
            zsmv = [7,15,16,17]
            vsn = [3,3,2,1]
            zsc = np.intersect1d(zs, zsmv)
            if zsc.shape[0] > 0:
                nheav = (zs > 1).sum()
                ias = np.arange(len(zs))
                for ia in range(nheav):
                    if (zs[ia] in zsmv) and (vs[ia]>vsn[ zsmv.index(zs[ia]) ]):
                        jas = ias[bom_U[ia] > 0]
                        for ja in jas:
                            if zs[ja] == 1:
                                isotopes.append(ja)"""
            ###### the above lines are commented out since
            ###### rdkit can handle those cases well

            ### annoying bug of RDKit (__version__ = '2017.09.1')
            ### It cannot convert `blk to `smiles for molecules containing Al
            ### e.g.,
            sample = """
none
     RDKit          3D

  7  6  0  0  0  0  0  0  0  0999 V2000
    0.0000    0.0000    0.0000 C   0  0  0  0  0  0  0  0  0  0  0  0
    0.0000    0.0000    0.0000 Al  0  0  0  0  0  0  0  0  0  0  0  0
    0.0000    0.0000    0.0000 H   0  0  0  0  0  0  0  0  0  0  0  0
    0.0000    0.0000    0.0000 H   0  0  0  0  0  0  0  0  0  0  0  0
    0.0000    0.0000    0.0000 H   0  0  0  0  0  0  0  0  0  0  0  0
    0.0000    0.0000    0.0000 H   0  0  0  0  0  0  0  0  0  0  0  0
    0.0000    0.0000    0.0000 H   0  0  0  0  0  0  0  0  0  0  0  0
  1  2  1  0
  1  3  1  0
  1  4  1  0
  1  5  1  0
  2  6  1  0
  2  7  1  0
M  END
"""
            ### the converted SMILES is 'C[AlH5]' instead of 'C[AlH2]'
            ### Due to this bug, we have to exclude any molecule containing Al
            ### see `check_elements()


            if na < 100:
                blk = write_ctab(zs, chgs_U, bom_U, self.coords, isotopes=isotopes, sdf=None)
                #print blk
                m = Chem.MolFromMolBlock(blk)
                fmt = 'sdf'
            else:
                blk = write_pdb( (zs,self.coords,chgs_U,bom_U) )
                m = Chem.MolFromPDBBlock(blk)
                fmt = 'pdb'

            try:
                can_i = Chem.MolToSmiles( m ); #print ' ** ', can_i
            except:
                # to handle some anomalies, e.g., 'COP(=P(=PC)C)=S'
                # (see function below for detail)
                can_i = to_can_obabel_rdkit(blk, fmt)

            # now obsolete due to the same reason as above
            obsolete = """# remove isotopes
            sp = r"\[[1-3]H\]"
            sr = "[H]"
            _atom_name_pat = re.compile(sp)
            can_i = _atom_name_pat.sub(sr, can_i)"""

            #print ' ++ zs, can, isotopes = ', zs, can_i, isotopes
            #t3 = time.time()
            #print '                |_ dt2 = ', t3-t2
            #print '             __ can = ', can_i
            if can_i not in cans:
                cans.append(can_i)
                ms.append(m)
        #if 'CC(C)C' in cans: print ' Alert!!!'
        return cans, ms


def get_coords(m):
    na = m.GetNumAtoms()
    assert m.GetNumConformers() == 1, '#ERROR: more than 1 Conformer'
    c0 = m.GetConformer(-1)
    coords = []
    for i in xrange(na):
        coords_i = c0.GetAtomPosition(i)
        coords.append([ coords_i.x, coords_i.y, coords_i.z ])
    return np.array(coords)


def get_bom(m):
    # somehow it's problematic to use `Chem.GetAdjacencyMatrix(m, useBO=True)
    na = m.GetNumAtoms()
    bom = np.zeros((na, na))
    for bi in m.GetBonds():
        i, j = bi.GetBeginAtomIdx(), bi.GetEndAtomIdx()
        bt = bi.GetBondType(); #print ' ***** bt = ', bt
        bom[i,j] = bom[j,i] = dic_bonds[ bt ]
    return bom


def check_hydrogens(m0):
    m0u = copy.deepcopy(m0)
    m0uh = Chem.AddHs(m0u)
    return m0.GetNumAtoms() == m0uh.GetNumAtoms()


def get_bos(bom):
    na = bom.shape[0]
    bosr = []
    for i in range(na):
        bosi = bom[i]
        t = bosi[ bosi > 0 ]; t.sort()
        n = len(t)
        v = 0
        for j in range(n):
            v += t[j]*10**j
        bosr.append( v )
    return np.array(bosr,np.int)


class mol(object):

    def __init__(self, m0, kekulize=True):
        if kekulize: Chem.Kekulize(m0)
        na = m0.GetNumAtoms()
        self.na = na
        m1 = copy.deepcopy(m0); m = Chem.RemoveHs(m1)
        self.m0 = m0
        self.m = m1

        # test
        vs = []
        for ai in m0.GetAtoms(): vs.append( ai.GetTotalValence() )
        #print ' ++++ vs = ', vs

        zs = [ ai.GetAtomicNum() for ai in m0.GetAtoms() ]
        self.zs = np.array(zs)
        self.bom = get_bom(m0)
        self.nheav =  (self.zs > 1).sum()
        ihyd = False
        if 1 in zs:
            ihyd = True; idxh = zs.index( 1 )
        self.ias = np.arange( na )
        self.ias_heav = self.ias[ self.zs > 1 ]
        try:
            self.coords = get_coords(m0)
            self.has_coord = True
        except:
            self.coords = np.zeros((na,3))
            self.has_coord = False

        self.chgs = np.array([ ai.GetFormalCharge() for ai in m0.GetAtoms() ])
        if ihyd:
            if np.any(self.zs[idxh+1:] > 1):
                # not all H apprear appear at the end, u have to sort it
                #print '++ Now sort'
                self.sort()

        # m0 may have been changed (if `sort is called)
        self.vs = np.array([ ai.GetTotalValence() for ai in self.m0.GetAtoms() ], np.int)

        if np.any(self.chgs != 0):
            self.eliminate_charges()
        # with rdkit, chgs are automatically right for cases
        # like R-N(=O)=O, C=N#N, no need to call `recover_charges()

#       print ' -- chgs = ', self.chgs

        bom_heav = self.bom[ self.ias_heav, : ][ :, self.ias_heav ]
#       print 'bom_heav = ', bom_heav
        self.vs_heav = bom_heav.sum(axis=0)
        self.cns_heav = ( bom_heav > 0 ).sum(axis=0)
        # get formal charges
        self.cns = ( self.bom > 0).sum(axis=0)
        self.nhs = self.vs[:self.nheav] - self.vs_heav #- self.chgs[:self.nheav]
        self.dvs = self.vs_heav - self.cns_heav

        #self.hybs = np.array([ dic_hyb[ai.GetHybridization()] for ai in m0.GetAtoms() ])

        # get bosr, i.e., bond order (reference data) array
        # concatenated into a integer
        self.bosr = get_bos(self.bom)
        self.dbnsr = (self.bom==2).sum(axis=0)
        #print ' -- bosr = ', self.bosr


    def sort(self):
        """ sort atoms so that H's appear at the end
        """
        nheav = self.nheav
        ias_heav = list(self.ias_heav)
        g = np.zeros((nheav, nheav))
        xhs = [] # X-H bonds
        ih = nheav
        coords = []; coords_H = []
        chgs = []; chgs_H = []
        dic = dict( zip(ias_heav, range(nheav)) )
#       print ' *** dic = ', dic
        for i, ia in enumerate( ias_heav ):
            coords.append( self.coords[ia] )
            chgs.append( self.chgs[ia] )
            jas = self.ias[ self.bom[ia,:] > 0 ]
            for ja in jas:
                if self.zs[ja] == 1:
                    coords_H.append( self.coords[ja] )
                    chgs_H.append( self.chgs[ja] )
                    xhs.append([i,ih]); ih += 1
                else:
                    g[i,dic[ja]] = g[dic[ja],i] = self.bom[ia,ja]
        coords_U = np.concatenate( (coords, coords_H) )
        self.coords = coords_U
        chgs_U = np.concatenate( (chgs, chgs_H) )
        self.chgs = chgs_U
        g2 = np.zeros((ih,ih))
        g2[:nheav, :nheav] = g
        for xh in xhs:
            i,j = xh
            g2[i,j] = g2[j,i] = 1
        self.bom = g2
        nh = ih - nheav
        zsU = np.array( list(self.zs[ias_heav]) + [1,]*nh )
        self.zs = zsU
        self.ias_heav = self.ias[ self.zs > 1 ]
        if self.na < 100:
            blk = write_ctab(zsU, chgs_U, g2, coords_U, sdf=None)
            #with open('t.sdf','w') as fid: fid.write(blk)
            m0 = Chem.MolFromMolBlock(blk, removeHs=False)
        else:
            blk = write_pdb( (zsU,coords_U,chgs_U,g2) )
            m0 = Chem.MolFromPDBBlock(blk)
        m1 = copy.deepcopy(m0)
        #print ' ********** na, smi = ', self.na, Chem.MolToSmiles(m1)
        m = Chem.RemoveHs(m1)
        self.m0 = m0; self.m = m


    def eliminate_charges(self):
        """update bom based on `chgs
        e.g., bom of C=[N+]=[N-] will be converted to bom of C=N#N
        based on `chgs = [0,+1,-1]
        Note that only bom and the resulting `vs will be updated, no
        changes regarding the SMILES string (i.e., we still prefer
        a SMILES string like C=[N+]=[N-] instead of C=N#N

        ## Attention!!
        =========================================
        There are some very special cases, e.g., R-[P+](O)(O)[N-][P+](O)(O)[N-]-R
                                                    1   2  3  4   5   6  7  8
        If the indices of atoms is [1,4,5,8], then everything is fine (i.e., the resulting
        bond patter is -P(O)(O)=N-P(O)(O)=N-); otherwise, we may end up with
        R-[P+](O)(O)-N=P(O)(O)-[N-]-R, if atomic order is [4,5,1,8], not we desired.
        Thus, we need to select those atoms with only one neighbor with opposite charge first,
        then update BO.
        Meanwhile check if valences are all saturated!
        """

        bom2 = copy.copy(self.bom)
        vs2 = self.vs
        chgs = self.chgs
        achgs = np.abs(chgs)
        if np.any(achgs >= 2):
            raise '#ERROR: some atom has a charge >= 2?'
        _ias1 = self.ias[achgs == 1]
        g1= self.bom[_ias1][:,_ias1]
        iasr_g1 = find_cliques(g1)
        for iasr_sg in iasr_g1:
            #print ' ---------------- ', iasr_sg
            iasg = list(_ias1[iasr_sg])
            nag = len(iasr_sg)
            assert chgs[iasg].sum() == 0 and nag%2 == 0 # must be in pairs and neutral
            _sg = ( self.bom[iasg][:,iasg] > 0 ).astype(np.int)
            neibs = _sg.sum(axis=0)
            pairs = []
            if nag == 2:
                pairs = [ iasg ]
            else:
                visited = set([])
                while len(visited) < nag:
                    iasg_ar = np.array(iasg)
                    jas = iasg_ar[neibs==1]
                    #print ' -- jas = ', jas
                    jas2 = iasg_ar[neibs==2]
                    if len(jas) > 0:
                        _j = 0
                        ja = jas[_j]
                        j = iasg.index(ja) ## attention here!
                        #print ' _sg = ', _sg
                        kas = iasg_ar[_sg[j]==1]
                        #print '** ', iasg, visited
                        #print '** ', kas
                        #print '** ', ja,kas[0],chgs[ja],chgs[kas[0]]
                        assert len(kas) == 1 and chgs[ja]+chgs[kas[0]]==0
                        pairs.append( [ja,kas[0]] )
                        visited.update( [ja,kas[0]] )
                    else:
                        ja = jas2[0]
                        ka = iasg_ar[_sg[0]==1][0]
                        assert chgs[ja]+chgs[ka] == 0
                        visited.update( [ja,ka] )
                        pairs.append( [ja,ka] )

                    iasg = list( set(iasg).difference(visited) )
                    _sg = ( self.bom[iasg][:,iasg] > 0).astype(np.int)
                    neibs = _sg.sum(axis=0)

            for pair in pairs:
                i,j = pair
                bij = self.bom[i,j] + 1
                bom2[i,j] = bij
                bom2[j,i] = bij
                vs2[i] = vs2[i]+1; vs2[j] = vs2[j]+1
        self.bom = bom2
        #print ' __ bom2 = ', bom2
        self.vs = vs2 #bom2.sum(axis=0)  #vs2


    def recover_charges(self):
        """figure out the charges of N atoms contraining that
        all have a valence of 3. E.g., for "CC=CC=N#N", the final
        charges of atoms is [0,0,0,0,1,-1], corresponding to the
        SMILES string of "CC=CC=[N+]=[N-]". It's similar for "CCN(=O)=O".
        """
        bom2 = copy.copy(self.bom)
        vs2 = self.vs
        ias1 = self.ias[ np.logical_and(vs2 == 5, self.zs == 7) ]
        chgs = self.chgs
        for ia in ias1:
            bom_ia = bom2[ia]
            jas = self.ias[ bom_ia >=2 ]
            bosj = bom_ia[ bom_ia >= 2 ]
            if len(jas) == 2:
                zsj = self.zs[ jas ]
                if set(bosj) == set([2])         or set(bosj) == set([2,3]):
                    # e.g., O=N(=C)C=C, O=N(=O)C        CC=CC=N#N
                    for ja in jas:
                        if (bom2[ja] > 0).sum() == 1:
                            chgs[ia] = 1; chgs[ja] = -1
                            break
                else:
                    raise '#ERROR: wierd case!'
        self.chgs = chgs


    def get_ab(self):
        """
        For heav atoms only

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


def remove_standalone_charge_0(m):
    """
    m  -- RDKit mol without H
    m0 -- RDKit with H added

    So far graph only; no coordinates associated
    obabel molecule as input
    return the canonical SMILES with charges removed
    """
    #bom = get_bom(m0)

    dic = { 4:2, 5:3, 6:4,7:3,8:2,9:1, 14:4,15:3,16:2,17:1, 32:4,33:3,34:2,35:1, 50:4,51:3,52:2,53:1 }
    iok = True # assume the mol's charge can be removed
    visited = set([])
    for ai in m.GetAtoms():
        idx = ai.GetIdx()
        if idx in visited: continue
        zi = ai.GetAtomicNum()
        vi = ai.GetTotalValence()
        chgi = ai.GetFormalCharge(); #print idx, zi, chgi

        # Note that with RDKit, R-Cl(=O)(=O)(=O) cannot be recognized
        # It has to be transformed to R-[Cl+3]([O-])([O-])([O-]) to be
        # processed successfully by RDKit
        if abs(chgi) > 0: # in [1,-1]:
            chgs = [chgi, ]
            jdxs = [idx, ]
            atmsj = [ai, ]
            for aj in ai.GetNeighbors():
                jdx = aj.GetIdx()
                chgj = aj.GetFormalCharge()
                if chgj != 0:
                    chgs.append( chgj )
                    jdxs.append( jdx )
                    atmsj.append( aj )
            visited.update( jdxs )
            nac = len(chgs)
            achgs = np.abs(chgs)
            _tchg = np.sum(chgs) # total charge
            if _tchg != 0:
                # "chgs = []" rules out cases like -[N+](=O)[O-],
                # R-[Cl3+]([O-])([O-])([O-]) (the canonical form of
                # which in rdkit is R-Cl(=O)(=O)(=O) )

# it's possible that _tchg > 0 while the subgroup is still charged neutral
# e.g., [O-][Cl3+]([O-])([O-])R
# in such case, we have to do the following
                if nac > 1:
                    iasc = np.arange(nac).astype(np.int)
                    cmax = np.max(achgs)
                    nuniq = len(np.unique(achgs))
                    if cmax > 1 and nuniq > 1: # e.g., achgs=[1,2]
                        _jdx = iasc[ cmax == achgs ][0]
                        #
                        print ' -- atmsj = ', atmsj, ' _jdx = ', _jdx
                        print ' -- chgi = ', chgi
                        aj = atmsj[_jdx]
                        for ak in aj.GetNeighbors():
                            kdx = ak.GetIdx()
                            chgk = ak.GetFormalCharge()
                            if kdx not in jdxs and chgk != 0:
                                visited.update( [kdx] )
                                chgs.append( chgk )
                        _tchg = np.sum(chgs)

            if _tchg != 0:
                nh = ai.GetNumExplicitHs()
                ai.SetFormalCharge( 0 )  # fix e.g., -C(=O)[O-], -[C+]<
                #print ' ** 1'
                if nh > 0:
                    if dic[zi] == vi - chgi:
                        # e.g., -[NH3+], -[CH2-], -[O-]
################# added lately to deal with cases like "C[ClH+2]CCC"
                        if nh - chgi < 0:
                            iok = False
                            break
                        else:
################# added lately
                            ai.SetNumExplicitHs( nh - chgi )
                    else:
                        # e.g., >[BH-]-
                        iok = False
                        break
                else:
                    #print ' ** 2'
                    if dic[zi] != vi - chgi:
                        # valence requirement not met!
                        # e.g., -[CH3+], -O[Al+]C
                        iok = False
                        break
                    else:
                        if chgi > 0:
                            # e.g., C=[N+](C)C, >[N+]<
                            iok = False
                            break #return None
                        else:
                            # e.g., "C1=NN=C[N-]1"
                            # if no explicit H is assigned to [N-] (to remove charge),
                            # the final SMILES string would be "c1ncnn1", which cannot
                            # be kekulized; the acutal SMILES is "c1nnc[nH]1"
                            ai.SetNumExplicitHs( -chgi )

                # to continue, you need to remove one H atom
                # and reassign the values of atom indices;
                # Alternatively, simply return an updated SMILES
                #if chgi == 1:
                #    # remove one hydrogen atom from, say [NH3+]
    if iok:
        su = Chem.MolToSmiles(m)
    else:
        su = None
    #print ' ++ ', su
    return su


def remove_standalone_charge(m):

    dic = { 1:1, 3:1, 4:2, 5:3, 6:4,7:[3,5],8:2,9:1, 14:4,15:[3,5],16:[2,4,6],17:1, \
            32:4,33:3,34:2,35:1, 50:4,51:3,52:2,53:1 }
    dic_vmax = {}
    for key in dic.keys():
        val = dic[key]
        if isinstance(val,int):
            dic_vmax[key] = val
        else:
            dic_vmax[key] = max(val)

    changed = False
    m0 = Chem.AddHs(m)
    chgs = np.array([ ai.GetFormalCharge() for ai in m0.GetAtoms() ])
    zs = np.array([ ai.GetAtomicNum() for ai in m0.GetAtoms() ], np.int)
    achgs = np.abs(chgs)
    na = len(chgs)
    ias = np.arange(na)
    bom = get_bom(m0)
    g =  ( get_bom(m0) > 0 ).astype(np.int)

    iasc = ias[ chgs != 0 ]
    g1 = g[iasc][:,iasc]
    cnsc = g1.sum(axis=0)

    #print ' FFFFFFFFFFFFFFFFFFFF'

    # exclude SMILES with atom charged more than 1, e.g., ROCl(=O)(=O)=O,
    # Note that interestingly, RDKit can recognize OCl(=O)(=O)=O instead of CCl(=O)(=O)=O
    # The latter has to be changed to C[Cl+3]([O-])([O-])[O-] to be parsed with success
    if np.any(achgs > 1):
        print '  ** some atom has an absolute charge >1?'
        return True, None

    # exclude SMILES with any atom satisfying
    vs = bom.sum(axis=0).astype(np.int) - chgs
    vsmax = np.array([ dic_vmax[zi] for zi in zs ], np.int)
    if np.any(vs > vsmax): # [B-]X4, [P-]X6
        print ' ** valency larger than allowed'
        return True, None

    if (np.sum(chgs) != 0) or np.any(cnsc == 0):
        # ================================================
        #         now remove standalone charge
        # ================================================
        ## np.any(cnsc==0) means there is some atom with
        ## standalone charged atom

        bom2 = copy.copy(bom)
        iasr_g1 = find_cliques(g1)
        iok = True
        for _iasr in iasr_g1:
            iasr_sg = list( _iasr )
            iasg = list(iasc[iasr_sg])
            chgsi = chgs[iasg]
            if chgsi.sum() != 0:
                changed = True
                # there must be standalone charged atom(s)
                nag = len(iasg)
                assert np.all(chgsi[chgsi < 0] == -1), '#ERROR: negative charge with magnitutde >1?'
                _sg = ( bom[iasg][:,iasg] > 0 ).astype(np.int)
                if nag > 1:
                    # case such as -[NH+]([O-])[O-] is sort of elusive,
                    # it could be either -[NH]([O-])=O or -[NH-](=O)=O and after
                    # saturating with removing/adding H, we end up with
                    # -[NH](O)=O and -[N](=O)=O, but which one should we choose is an issue.
                    # so we skip such cases.
                    iok = False
                    print ' -- now exit for loop'
                    break
                    elusive = """for _iar in range(nag):
                        if chgsi[_iar] == -1:
                            _ia = iasg[_iar]
                            iasg.remove(_ia)
                            _jas = iasg[ _sg[_iar] == 1 ] # it's possible that the num of charged neib >1
                            assert len(_jas) == 1 # we skip the case that n_neib > 1
                            _ja = _jas[0]
                            bom2[_ia,_ja] = bom[_ia,_ja]+1
                            bom2[_ja,_ia] = bom[_ja,_ia]+1 """

                ia = iasg[0]
                ai = m.GetAtomWithIdx(ia)
                chgi = chgs[ia]
                zi = ai.GetAtomicNum()

                vs = bom2.sum(axis=0).astype(np.int)
                vib = vs[ia] # valency_ia_bom
                vic = vib - chgi # charge corrected valency of `ia-th atom
                if zi in [7,15,16]:
                    vsr = np.array(dic[zi], np.int)
                    _t = vsr[vsr >= vic]
                    if len(_t) == 0: # e.g., the second P in "N[P+](N)(N)C[PH-]X4"
                        iok = False
                        break
                    vir = _t[0]
                else:
                    vir = dic[zi]

                nh = ai.GetNumExplicitHs()
                #print ' -------- nh, vir, vib, chgi = ', nh,vir,vib,chgi
                if vir == vic:
                    nh_add = nh - chgi
                    if nh_add < 0: # e.g., "C[ClH+2]CC", ">[N+]<", "C=[N+](C)C",
                        iok = False
                        #print ' -- now exit for loop 2'
                        break
                    else: # e.g., -[NH3+], -[CH2-], -[O-], -C(=O)[O-]
                        #print ' -- num_explicit_H = ', nh_add
                        ai.SetNumExplicitHs( nh_add )
                        ai.SetFormalCharge( 0 )
                else:
                    # vir > vic, e.g., -[CH2], -[CH3+],-[CH2+],  -O[NH2+],-O[NH+]
                    # vir < vic, e.g., "R[N-](=O)=O"
                    iok = False
                    #print ' -- now exit for loop 3'
                    break #return None
        if iok:
            # Note that this is an intermediate SMILES. It's safer to set
            # kekuleSmiles=True due to the reason that aromatic atoms may
            # have been changed after removing/adding hydrogens. 
            # E.g., "[NH2+]=C1[CH-]C(=O)C=CN1" --> su = "N=c1cc(=O)cc[nH]1" 
            # (if kekuleSmiles=False) which cannot be kekulized; By setting 
            # kekuleSmiles=True fixes this problem.
            su = Chem.MolToSmiles(m,kekuleSmiles=True) 
            print ' -- su = ', su
        else:
            su = None
        return changed, su

    else:
        return [False]


def check_elements(zs):
    # metals are all excluded, including
    # Li, Na,Mg,Al, K,Ca,Rb,Ba,Sr,Cs,Ra and
    # Sc-Zn
    # Y-Cd
    # La-Lu, Hf-Hg
    #zsa = [3, 11,12,13, 19,20, 37,38, 55,56] + \
    #     range(21,31) + \
    #     range(39,49) + \
    #     range(57,81) + \
    #     range(89,113) # Ac-Lr, Rf-Cn

    # consider only these elements
    symbs = [ 'H','Be','B',\
               'C', 'N', 'O', 'F',\
              'Si', 'P', 'S','Cl',\
              'Ge','As','Se','Br',\
              'Sn','Sb','Te', 'I']
    #return np.all([ zi not in zsa for zi in zs ])
    zsa = [ atomic_numbers[si] for si in symbs ]
    return np.all([ zi in zsa for zi in zs ])


class amon(object):

    """
    use RDKit only
    """

    def __init__(self, s, k, k2=None, wg=False, ligand=None, \
                 fixGeom=False, ikeepRing=True, \
                 allow_isotope=False, allow_charge=False, \
                 allow_radical=False):
        """
        ligand -- defaulted to None; otherwise a canonical SMILES
        has to be specified

        vars
        ===============
        s -- input string, be it either a SMILES string or sdf file
        k -- limitation imposed on the number of heav atoms in amon
        """

        if k2 is None: k2 = k
        self.k = k
        self.k2 = k2
        self.wg = wg
        self.fixGeom = fixGeom
        self.ikeepRing = ikeepRing

        iok = True
        if os.path.exists(s):
            m0 = Chem.MolFromMolFile(s, removeHs=False)

            # set isotope to 0
            # otherwise, we'll encounter SMILES like 'C[2H]',
            # and error correspondently.
            # In deciding which atoms should be have spin multiplicity
            # assigned, hydrogen atoms which have an isotope specification
            # (D,T or even 1H) do not count. So SMILES N[2H] is NH2D (spin
            # multiplicity left at 0, so with a full content of implicit
            # hydrogens), whereas N[H] is NH (spin multiplicity=3). A
            # deuterated radical like NHD is represented by [NH][2H].
            na = m0.NumAtoms()
            if not allow_isotope:
                for ai in m0.GetAtoms():
                    ai.SetIsotope(0)

            # add lines below to tell if HX bond appears before some heav atom bonds
            # _____________
            #
            #
            assert check_hydrogens(m0), '#ERROR: some hydrogens are missing'
            coords0 = get_coords(m0)
            charge = sum( [ ai.GetFormalCharge() for ai in m0.GetAtoms() ] )
            nesr = np.array([ ai.GetNumRadicalElectrons() for ai in m0.GetAtoms() ])
            spin = np.any(nesr > 0)
            if charge != 0 and (not allow_charge): iok = False
            if spin_polarised and (not allow_radical): iok = False

            m1 = copy.deepcopy(m0); m = Chem.RemoveHs(m1)
        else:
            if not allow_isotope:
                # remove isotopes
                _pat = r"\[([0-9][0-9]*(\w+))\]"
                # e.g., C1=C(C(=O)NC(=O)N1[C@H]2[C@H]([C@@H]([C@H](O2)CO)O)F)[124I]
                #       [3H]C
                #       CN([11CH3])CC1=CC=CC=C1SC2=C(C=C(C=C2)C#N)N
                _splits = re.findall(_pat, s)
                for _split in _splits:
                    sp, sr = _split
                    idx = s.index(sp)
                    s2 = s[:idx] + sr + s[idx+len(sp):]
                    s = s2
                #print ' ++ s = ', s

            try:
                m = Chem.MolFromSmiles(s)
                Chem.Kekulize(m)
                m0 = Chem.AddHs(m)
            except:
                print '   ** failed to parse smiles/kekulize/addh [1]'
                iok = False
            if iok:
                zs = [ ai.GetAtomicNum() for ai in m.GetAtoms() ]
                if not check_elements(zs):
                    iok = False
                ## newly added, to exclude H2
                if not np.any([ _z > 1 for _z in zs ]):
                    print '   ** H2 molecule found!'
                    iok = False

            if iok:
                nesr = np.array([ ai.GetNumRadicalElectrons() for ai in m0.GetAtoms() ])
                spin = np.any(nesr > 0)
                if spin and (not allow_radical):
                    print '   ** radical!'
                    iok = False

            if iok:
                if not allow_charge:
                    # now remove standalone charge
                    ts = remove_standalone_charge(m)
                    if ts[0]:
                        su = ts[1]
                        if su is None:
                            print '   ** failed to remove standalone charge'
                            iok = False
                        else:
                            #print su
                            try:
                                m = Chem.MolFromSmiles(su)
                                m0 = Chem.AddHs(m)
                            except:
                                # it may fail, due to valency larger than allowed
                                iok = False
                                print '   ** failed to parse SMILES & addh [2]'

                # to see if kekulization is successful
                if iok:
                    m1 = copy.deepcopy(m0)
                    m = Chem.RemoveHs(m1)

        self.iok = iok
        if iok:
            # the last check
            # are the chemistries wierd (uncommon)?
            q = mol(m0)
            obsolete = """bom = q.bom
            vs = bom.sum(axis=0)
            zs = q.zs[:q.nheav]
            for i,z in enumerate(zs):
                if vs[i] > vsmax[z]:
                    print ' ++ exotic chemistry'
                    iok = False
                    break
        self.iok = iok
        if iok:"""
            self.objQ = q


    def get_subm(self, las, lbs, sg):
        """
        add hydrogens & retrieve coords
        """
        #sets = [ set(self.objQ.bs[ib]) for ib in lbs ] # bond sets for this frag
        nheav = len(las)
        dic = dict( zip(las, range(nheav)) )
        ih = nheav;
        xhs = [] # X-H bonds
        if self.wg:
            coords = []; coords_H = []
            for i,ia in enumerate(las):
                coords.append( self.objQ.coords[ia] )
                jas = self.objQ.ias[ self.objQ.bom[ia,:] > 0 ]
                for ja in jas:
                    if self.objQ.zs[ja] == 1:
                        coords_H.append( self.objQ.coords[ja] )
                        xhs.append([i,ih]); ih += 1
                    else:
                        #if (ja not in las) or ( (ja in las) and (set(ia,ja) not in sets) ):
                        if (ja not in las) or ( (ja in las) and (sg[i,dic[ja]] > 0) ):
                            v = self.objQ.coords[ja] - coords_i
                            coords_H.append( coord + dsHX[z] * v/np.linalg.norm(v) )
                            xhs.append([i,ih]); ih += 1
            coords_U = np.concatenate( (coords, coords_H) )
        else:
            for i,ia in enumerate(las):
                jas = self.objQ.ias[ self.objQ.bom[ia,:] > 0 ]
                for ja in jas:
                    if self.objQ.zs[ja] == 1:
                        xhs.append([i,ih]); ih += 1
                    else:
                        if (ja not in las) or ( (ja in las) and (sg[i,dic[ja]] == 0) ):
                            xhs.append([i,ih]); ih += 1
            coords_U = np.zeros((ih,3))
        sg_U = np.zeros((ih,ih))
        sg_U[:nheav, :nheav] = sg
        for xh in xhs:
            i,j = xh
            sg_U[i,j] = sg_U[j,i] = 1

        nh = ih - nheav
        bosr1 = self.objQ.bosr[las] # for heav atoms only
        zs1 = np.array( list(self.objQ.zs[las]) + [1,]*nh )
        chgs1 = np.array( list(self.objQ.chgs[las]) + [0,]*nh )
        tvs1 = np.array( list(self.objQ.vs[las]) + [1,]*nh )
        vars1 = vars(bosr1, zs1, chgs1, tvs1, sg_U, coords_U)
        self.vars = vars1


    def get_amons(self):
        """
        tell if a given frag is a valid amon
        """

        objQ = self.objQ

        amons = []
        smiles = []

        # get amon with N_I = 2 to k
        g0 = ( objQ.bom > 0 ).astype(np.int)

        amons = []
        cans = []; ms = []; ns = []
        if objQ.nheav == 1:
            for _zi in objQ.zs:
                if _zi > 1: break
            _t = '[%sH%d]'%(chemical_symbols[_zi], objQ.na-1)
            _m = Chem.MolFromSmiles(_t)
            _can = Chem.MolToSmiles(_m)
            cans = [ _can ]
            ns = [1]
            return cans, ns

        a2b, b2a = objQ.get_ab()
        bs = [ set(jas) for jas in b2a ]
        for seed in generate_subgraphs(b2a, a2b, self.k):
            # lasi (lbsi) -- the i-th list of atoms (bonds)
            lasi, lbsi = list(seed.atoms), list(seed.bonds)
            _lasi = np.array(lasi).astype(np.int)
            #can = Chem.MolFragmentToSmiles(objQ.m, atomsToUse=lasi, kekuleSmiles=False, \
            #                               bondsToUse=lbsi, canonical=True)

            iprt = False
            bs = []
            for ibx in lbsi:
                bs.append( set(b2a[ibx]) )
                #if iprt:
                #    print '  -- ibx, ias2 = ', ibx, tuple(b2a[ibx])

            na = len(lasi)
            if na == 1:
                ia = lasi[0]; zi = objQ.zs[ ia ]
                iok1 = (zi in [9, 17, 35, 53])
                iok2 = ( np.any(objQ.bom[ia] >= 2) ) # -S(=O)-, -P(=O)(O)-, -S(=O)(=O)- and #N
                if np.any([iok1, iok2]):
                    continue
                can = chemical_symbols[ zi ]
                if can not in cans:
                    cans.append( can ); ns.append(1)
                    if wg:
                        if not self.fixGeom:
                            ms.append( ms0[can] )
                        else:
                            raise '#ERROR: not implemented yet'
                else:
                #    if wg and self.fixGeom:
                    ns[ cans.index(can) ] += 1
                #if wg:
                #    self.get_subm(lasi, lbsi, np.zeros((1,1)))
                continue

            sg0_heav = g0[lasi,:][:,lasi]
            nr0 = cg.get_number_of_rings(sg0_heav)

            # property of atom in the query mol
            nhs_sg0 = objQ.nhs[lasi]
#           print ' cns_heav = ', objQ.cns_heav
            cns_sg0_heav = objQ.cns_heav[lasi]

            zs_sg = objQ.zs[ lasi ]
            sg_heav = np.zeros((na,na))
            for i in range(na-1):
                for j in range(i+1,na):
                    bij = set([ lasi[i], lasi[j] ])
                    if bij in bs:
                        sg_heav[i,j] = sg_heav[j,i] = 1
            nr = cg.get_number_of_rings(sg_heav)
            ir = True
            if self.ikeepRing:
                if nr != nr0:
                    ir = False

            cns_sg_heav = sg_heav.sum(axis=0)

#           if iprt:
#               print '  -- cns_sg0_heav, cns_sg_heav = ', cns_sg0_heav, cns_sg_heav
#               print '  -- dvs_sg_heavy = ', objQ.dvs[lasi]
#               print '  -- nhs = ', objQ.nhs[lasi]

#           print zs_sg, cns_sg0_heav, cns_sg_heav #
            dcns = cns_sg0_heav - cns_sg_heav # difference in coordination numbers
            assert np.all( dcns >= 0 )
            num_h_add = dcns.sum()
#           if iprt: print '  -- dcns = ', dcns, ' nhs_sg0 = ', nhs_sg0
            ztot = num_h_add + nhs_sg0.sum() + zs_sg.sum()
#           if iprt: print '  -- ztot = ', ztot
            chg0 = objQ.chgs[lasi].sum()

            # test
            #_cns2 = list(objQ.cns[lasi]); _cns2.sort()
            icon = False
            #if na == 7 and np.all(np.unique(zs_sg)==np.array([6,16])) and \
            #       np.all(np.array(_cns2) == np.array([2,2,2,2,3,3,4])):
            #    icon = True; print ' ***** '

            if ir and ztot%2 == 0 and chg0 == 0:
                # ztot%2 == 1 implies a radical, not a valid amon for neutral query
                # this requirement kills a lot of fragments
                # e.g., CH3[N+](=O)[O-] --> CH3[N+](=O)H & CH3[N+](H)[O-] are not valid
                #       CH3C(=O)O (CCC#N) --> CH3C(H)O (CCC(H)) won't survive either
                #   while for C=C[N+](=O)[O-], with ztot%2 == 0, [CH2][N+](=O) may survive,
                #       by imposing chg0 = 0 solve the problem!

                zsi = zs_sg
                ias = np.arange(na)


                ## 0) check if envs like '>S=O', '-S(=O)(=O)-', '-P(=O)<',
                ## '-[N+](=O)[O-]' (it's already converted to '-N(=O)(=O)', so `ndb=2)
                ## 'R-S(=S(=O)=O)(=S(=O)(=O))-R', '-C=[N+]=[N-]' or '-N=[N+]=[N-]'
                ## ( however, '-Cl(=O)(=O)(=O)' cannot be
                ## recognized by rdkit )
                ## are retained if they are part of the query molecule
                bom0_heav = objQ.bom[lasi,:][:,lasi]
                dbnsi = (bom0_heav==2).sum(axis=0) #np.array([ (bom0_heav[i]==2).sum() for i in range(na) ], np.int)

                ##### lines below are not necessary as `bosr will be used to assess
                ##### if the local envs have been kept!
                ## actually, the role of the few lines below is indispensible.
                ## E.g., for a mol c1ccccc1-S(=O)(=O)C, an amon like C=[SH2]=O
                ## has bos='2211', exactly the same as the S atom in query. But
                ## it's not a valid amon here as it's very different compared
                ## to O=[SH2]=O...
                ## Another example is C=CS(=O)(=O)S(=O)(=O)C=C, an amon like
                ## [SH2](=O)=[SH2](=O) has bos='2211' for both S atoms, but are
                ## not valid amons

                ## ============================================================
                ## To rule these cases out, we need to compare the value of `dbn
                ## of multivalent atom in the subm to that of the same atom
                ## in the query molecule: If they are different, then
                ## kick this subm out; otherwise, keep it.
                ##
                ## Here comes the detailed steps:
                ## 1) identify multivalent atoms
                ## 2) compare `dbnsi and allowed num of doulbe bonds in `_dbns
                ##    E.g., for a query C=CS(=O)(=O)C, a subg is [CH]S(=O)[H]C,
                ##          which is an invalid amon as bosr for the S atom is 
                ##          2111, different compared to 2211 in the query
                ## ============================================================

                tvs1  = [   4,      6,   5,   5 ]
                zs1   = [  16,     16,  15,   7]
                _dbns = [ [1], [2, 3], [1], [2] ] # number of double bonds
                #               |  |
                #               |  |___  'R-S(=S(=O)=O)(=S(=O)(=O))-R',
                #               |
                #               |___ "R-S(=O)(=O)-R"

                istop = False
                tvsi0 = objQ.vs[lasi] # for N in '-[N+](=O)[O-]', tvi=4 (rdkit)
                # now gather all atomic indices need to be compared
                jas = np.array([], np.int)
                # step 1), identify multivalent atoms
                for j,tvj in enumerate(tvs1):
                    filt = np.logical_and(tvsi0 == tvj, zsi == zs1[j])
                    _jas = ias[filt].astype(np.int)
                    jas = np.concatenate( (jas,_jas) )
                # now compare the num_double_bonds
                if len(jas) > 0:
                    dbnsj = dbnsi[jas]
                    dbnsrj = objQ.dbnsr[ _lasi[jas] ]
                    if np.any(dbnsj != dbnsrj):
                        istop = True; continue #break
                        #print 'tvj, zs1[j], dbnsj, dbns1[j] = ', tvj, zs1[j], dbnsj, dbns1[j]
                        #print ' __ zsi = ', zsi, ', istop = ', istop
                #if istop: continue #"""

                self.get_subm(lasi, lbsi, sg_heav)
                vr = self.vars

                so = ''
                for i in range(na):
                    for j in range(i+1,na):
                        if vr.g[i,j] > 0: so += '[%d,%d],'%(i+1,j+1)
                #print vr.zs, '\n', vr.tvs, '\n', so, '\n'
                #if ','.join(['%d'%_zi for _zi in vr.zs]) == '13,6,8,6,13':
                #    icon=True; print ' ***** '

                #print 'tvs=',vr.tvs
                cmg = MG( vr.bosr, vr.zs, vr.chgs, vr.tvs, vr.g, vr.coords )

                # for diagnosis
                gr = []
                nat = len(vr.zs); ic = 0
                for i in range(nat-1):
                    for j in range(i+1,nat):
                        gr.append( vr.g[i,j] ); ic += 1
                test = """
                s = ' ########## %d'%nat
                for i in range(nat): s += ' %d'%vr.zs[i]
                for i in range(nat): s += ' %d'%vr.tvs[i]
                for i in range(ic): s += ' %d'%gr[i]
                print s
                #"""

                #if len(objQ.zs[lasi])==3:
                #    if np.all(objQ.zs[lasi] == np.array([7,7,7])): print '## we r here'

                cans_i = []
                cans_i, ms_i = cmg.update_m(debug=True,icon=icon)
                #if icon: print ' -- cans = ', cans_i
                for can_i in cans_i:
                    if can_i not in cans:
                        cans.append( can_i ); ns.append( 1 )
                    else:
                        ns[ cans.index(can_i) ] += 1
                #if icon: print ''
                if icon:
                    print '###############\n', cans_i, '############\n'
        return cans, ns


"""
For an explanation of the algorithm see
  http://dalkescientific.com/writings/diary/archive/2011/01/10/subgraph_enumeration.html
"""

#######

class Subgraph(object):
    def __init__(self, atoms, bonds):
        self.atoms = atoms
        self.bonds = bonds

def get_nbr(ia, b):
    ia1, ia2 = b
    if ia == ia1:
        return ia2
    else:
        return ia1

def find_extensions(considered, new_atoms, b2a, a2b):
    # Find the extensions from the atoms in 'new_atoms'.
    # There are two types of extensions:
    #
    #  1. an "internal extension" is a bond which is not in 'considered'
    # which links two atoms in 'new_atoms'.
    #
    #  2. an "external extension" is a (bond, to_atom) pair where the
    # bond is not in 'considered' and it connects one of the atoms in
    # 'new_atoms' to the atom 'to_atom'.
    #
    # Return the internal extensions as a list of bonds and
    # return the external extensions as a list of (bond, to_atom) 2-ples.
    internal_extensions = set()
    external_extensions = []
    #print 'type, val = ', type(new_atoms), new_atoms
    for atom in new_atoms: # atom is atom_idx
        ibsc = a2b[atom] # idxs of bond candidates
        for outgoing_bond in ibsc[ ibsc >= 0 ]: #atom.GetBonds():
            if outgoing_bond in considered:
                continue
            other_atom = get_nbr(atom, b2a[outgoing_bond]) #outgoing_bond.GetNbr(atom)
            if other_atom in new_atoms:
                # This this is an unconsidered bond going to
                # another atom in the same subgraph. This will
                # come up twice, so prevent duplicates.
                internal_extensions.add(outgoing_bond)
            else:
                external_extensions.append( (outgoing_bond, other_atom) )

    return list(internal_extensions), external_extensions



def all_combinations(container):
    "Generate all 2**len(container) combinations of elements in the container"
    # This just sets up the underlying call
    return _all_combinations(container, len(container)-1, 0)

def _all_combinations(container, last, i):
    # This does the hard work recursively
    if i == last:
        yield []
        yield [container[i]]
    else:
        for subcombinations in _all_combinations(container, last, i+1):
            yield subcombinations
            yield [container[i]] + subcombinations

## I had an optimization that if limit >= len(external_extensions) then
## use this instead of the limited_external_combinations, but my timings
## suggest the result was slower, so I went for the simpler code.

#def all_external_combinations(container):
#    "Generate all 2**len(container) combinations of external extensions"
#    for external_combination in all_combinations(container):
#        # For each combination yield 2-ples containing
#        #   {the set of atoms in the combination}, [list of external extensions]
#        yield set((ext[1] for ext in external_combination)), external_combination

def limited_external_combinations(container, limit):
    "Generate all 2**len(container) combinations which do not have more than 'limit' atoms"
    return _limited_combinations(container, len(container)-1, 0, limit)

def _limited_combinations(container, last, i, limit):
    # Keep track of the set of current atoms as well as the list of extensions.
    # (An external extension doesn't always add an atom. Think of
    #   C1CC1 where the first "CC" adds two edges, both to the same atom.)
    if i == last:
        yield set(), []
        if limit >= 1:
            ext = container[i]
            yield set([ext[1]]), [ext]
    else:
        for subatoms, subcombinations in _limited_combinations(container, last, i+1, limit):
            assert len(subatoms) <= limit
            yield subatoms, subcombinations
            new_subatoms = subatoms.copy()
            ext = container[i]
            new_subatoms.add(ext[1])
            if len(new_subatoms) <= limit:
                yield new_subatoms, [ext] + subcombinations


def all_subgraph_extensions(subgraph, internal_extensions, external_extensions, k):
    # Generate the set of all subgraphs which can extend the input subgraph and
    # which have no more than 'k' atoms.
    assert len(subgraph.atoms) <= k

    if not external_extensions:
        # Only internal extensions (test case: "C1C2CCC2C1")
        it = all_combinations(internal_extensions)
        it.next()
        for internal_ext in it:
            # Make the new subgraphs
            bonds = frozenset(chain(subgraph.bonds, internal_ext))
            yield set(), Subgraph(subgraph.atoms, bonds)
        return

    limit = k - len(subgraph.atoms)

    if not internal_extensions:
        # Only external extensions
        # If we're at the limit then it's not possible to extend
        if limit == 0:
            return
        # We can extend by at least one atom.
        it = limited_external_combinations(external_extensions, limit)
        it.next()
        for new_atoms, external_ext in it:
            # Make the new subgraphs
            atoms = frozenset(chain(subgraph.atoms, new_atoms))
            bonds = frozenset(chain(subgraph.bonds, (ext[0] for ext in external_ext)))
            yield new_atoms, Subgraph(atoms, bonds)
        return

    # Mixture of internal and external (test case: "C1C2CCC2C1")
    external_it = limited_external_combinations(external_extensions, limit)
    it = product(all_combinations(internal_extensions), external_it)
    it.next()
    for (internal_ext, external) in it:
        # Make the new subgraphs
        new_atoms = external[0]
        atoms = frozenset(chain(subgraph.atoms, new_atoms))
        bonds = frozenset(chain(subgraph.bonds, internal_ext,
                                (ext[0] for ext in external[1])))
        yield new_atoms, Subgraph(atoms, bonds)
    return

def generate_subgraphs(b2a, a2b, k=5):
    if k < 0:
        raise ValueError("k must be non-negative")

    # If you want nothing, you'll get nothing
    if k < 1:
        return

    # Generate all the subgraphs of size 1
    na = len(a2b)
    for atom in range(na): #mol.GetAtoms():
        yield Subgraph(frozenset([atom]), frozenset())

    # If that's all you want then that's all you'll get
    if k == 1:
        return

    # Generate the intial seeds. Seed_i starts with bond_i and knows
    # that bond_0 .. bond_i will not need to be considered during any
    # growth of of the seed.
    # For each seed I also keep track of the possible ways to extend the seed.
    seeds = []
    considered = set()
    nb = len(b2a)
    for bond in range(nb): #mol.GetBonds():
        considered.add(bond)
        subgraph = Subgraph(frozenset(b2a[bond]), #[bond.GetBgn(), bond.GetEnd()]),
                            frozenset([bond]))
        yield subgraph
        internal_extensions, external_extensions = find_extensions(considered,
                                                   subgraph.atoms, b2a, a2b)
        # If it can't be extended then there's no reason to keep track of it
        if internal_extensions or external_extensions:
            seeds.append( (considered.copy(), subgraph,
                           internal_extensions, external_extensions) )

    # No need to search any further
    if k == 2:
        return

    # seeds = [(considered, subgraph, internal, external), ...]
    while seeds:
        considered, subgraph, internal_extensions, external_extensions = seeds.pop()

        # I'm going to handle all 2**n-1 ways to expand using these
        # sets of bonds, so there's no need to consider them during
        # any of the future expansions.
        new_considered = considered.copy()
        new_considered.update(internal_extensions)
        new_considered.update(ext[0] for ext in external_extensions)

        for new_atoms, new_subgraph in all_subgraph_extensions(
            subgraph, internal_extensions, external_extensions, k):

            assert len(new_subgraph.atoms) <= k
            yield new_subgraph

            # If no new atoms were added, and I've already examined
            # all of the ways to expand from the old atoms, then
            # there's no other way to expand and I'm done.
            if not new_atoms:
                continue

            # Start from the new atoms to find possible extensions
            # for the next iteration.
            new_internal, new_external = find_extensions(new_considered, new_atoms, b2a, a2b)
            if new_internal or new_external:
                seeds.append( (new_considered, new_subgraph, new_internal, new_external) )


## test!

if __name__ == "__main__":
    import time, sys, gzip

    args = sys.argv[1:]
    nargs = len(args)
    if nargs == 0:
        ss = ["C=C[N+]#[C-]", "[NH3+]CC(=O)[O-]", "CC[N+]([O-])=O", \
             "C=C=C=CC=[N+]=[N-]", "CCS(=O)(=O)[O-]", \
             "C#CS(C)(=C=C)=C=C", "C1=CS(=S(=O)=O)(=S(=O)=O)C=C1",\
             "C#P=PP(#P)P(#P)P=P#P", \
             "c1ccccc1S(=O)(=O)S(=O)(=N)S(=O)(=O)c2ccccc2"] # test molecules
        ss += ['[NH3+]CC1=N[N-]N=N1', 'OC=[O+][CH2-]']
        k = 7
    elif nargs == 1:
        ss = args[0:1]
        k = 7
    elif nargs == 2:
        ss = args[1:2]
        k = int(args[0])
    else:
        raise SystemExit("""Usage: dfa_subgraph_enumeration.py <smiles> [<k>]
List all subgraphs of the given SMILES up to size k atoms (default k=5)
""")

    for s in ss:
        print ' \n## %s'%s
        if not os.path.exists(s):
            _k = k
            if s in ["C1=CS(=S(=O)=O)(=S(=O)=O)C=C1",]:
                _k = 9 # setting k = 9 can tell if some amons are missing
                print '     ** k = 9'
            t1 = time.time()
            obj = amon(s, _k)
            if obj.iok:
                cans = obj.get_amons()
                for can in cans:
                    print can
                print ' time elapsed: ', time.time() - t1
            else:
                print '  ++ radical or charged species'
        else:
            assert s[-3:] == 'smi'

            fn = s[:-4]
            ts = file(s).readlines()

            icnt = 0
            ids = []
            for i,t in enumerate(ts):
                si = t.strip()
                print i+1, icnt+1, si
                if '.' in si: continue
                obj = ciao.amon(si, k)
                if not obj.iok: print ' ** radical **'; continue
                print '  ++ '
                cansi = obj.get_amons()
                print '  +++ '
                nci = len(cansi)
                map_i = []
                for ci in cansi:
                    if ci not in cs:
                        cs.append(ci); map_i += [idxc]; idxc += 1
                    else:
                        jdxc = cs.index(ci)
                        if jdxc not in map_i: map_i += [jdxc]
                print 'nci = ', nci
                map_i += [-1,]*(nmaxc-nci)
                maps.append( map_i )
                #nmaxc = max(nmaxc, nci)
                ids.append( i+1 )
                icnt += 1

            with open(fn+'_all.smi','w') as fo: fo.write('\n'.join(cs))
            cs = np.array(cs)

            maps = np.array(maps,np.int)
            ids = np.array(ids,np.int)
            dd.io.save(fn+'.h5', {'ids':ids, 'cans':cs, 'maps':maps})

