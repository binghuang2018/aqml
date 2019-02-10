
from itertools import chain, product
import os, sys, re, copy, ase
import ase.data as ad
from openeye.oechem import *
import numpy as np
import networkx as nx
import networkx.algorithms.isomorphism  as iso
import cheminfo.OEChem as oe
import cheminfo.RDKit as cir
from rdkit import Chem
import scipy.spatial.distance as ssd
import cheminfo.obabel as cib
import multiprocessing
import cheminfo.math as cim
import deepdish as dd
import itertools as itl
import tempfile as tpf
#tsdf = tpf.NamedTemporaryFile(dir=tdir)
import cheminfo.fortran.famoneib as fa

global dsHX
dsHX = {5:1.20, 6:1.10, 7:1.00, 8:0.98, 9:0.92, 14:1.48, 15:1.42, 16:1.34, 17:1.27}


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


class Parameters(object):

    def __init__(self, wg, fixGeom, k, k2, ivdw, dminVDW, \
                 forcefield, thresh, do_ob_ff, \
                 do_rk_ff, idiff, iters):
        self.wg = wg
        self.fixGeom = fixGeom
        self.ff = forcefield
        self.k = k
        self.k2 = k2
        self.ivdw = ivdw
        self.dminVDW = dminVDW
#       self.threshDE = threshDE
        self.thresh = thresh
        self.do_ob_ff = do_ob_ff
        self.do_rk_ff = do_rk_ff
        self.iters = iters
        self.idiff = idiff

def merge(Ms): #Mli1, Mli2):
    """merge two or more `ctab"""
    nas = []
    zs = []; coords = []; charges = []; boms = []
    for M in Ms:
        zs1, coords1, bom1, charges1 = M
        zs.append( zs1)
        na1 = len(zs1); nas.append(na1)
        coords.append( coords1)
        charges.append( charges1)
        boms.append(bom1)
    zs = np.concatenate( zs )
    coords = np.concatenate(coords, axis=0)
    charges = np.concatenate(charges)
    na = sum(nas); nm = len(nas)
    bom = np.zeros((na,na), np.int)
    ias2 = np.cumsum(nas)
    ias1 = np.array([0] + list(ias2[:-1]))
    for i in range(nm):
        ia1 = ias1[i]; ia2 = ias2[i]
        bom[ia1:ia2,ia1:ia2] = boms[i]
    return zs, coords, bom, charges


class Sets(object):

    def __init__(self, param):

        self.cans = [] #cans
        self.ms = [] #ms
        self.rms = [] #rms
        self.es = [] #es
        self.nhas = [] #nhas
        self.ms0 = [] #ms0
        self.maps = [] #maps
        self.cms = [] # coulomb matrix
        self.param = param

    def check_eigval(self):
        """ check if the new kernel (after adding one molecule) has
        some very small eigenvalue, i.e., if it's true, it means that
        there are very similar molecules to the new comer, thus it won't
        be included as a new amon"""
        iok = True
        thresh = self.param.thresh

    def update(self, ir, can, Mli):
        """
        update `Sets

        var's
        ==============
        Mli  -- Molecule info represented as a list
                i.e., [zs, coords, bom, charges]
        """
        zs, coords, bom, charges = Mli
        rmol = RawMol(zs, coords)
        if self.param.idiff == 1: rmol.generate_coulomb_matrix()
        nha = (zs > 1).sum()
        self.ncan = len(self.cans)
        if can in self.cans:
            ican = self.cans.index( can )
            # for molecule with .LE. 3 heavy atoms, no conformers
            if (not self.param.fixGeom) and nha <= 3:
                # but u still need to tell if it belongs to the
                # `ir-th query molecule (so, the amon `m0 might
                # have appeared as an amon of another query molecule
                # considered previously.
                # Note that we use a 3-integer list for labeling the
                # generated amons, i.e., [ir,ican,iconfonmer].
                amon_idx = [ir, ican, 0]
                if amon_idx not in self.maps:
                    self.maps.append( amon_idx )
            else:
                m0, m, ei = self.Opt(Mli)
                ms_i = self.ms[ ican ] # stores the updated geom
                rms_i = self.rms[ ican ]
                ms0_i = self.ms0[ ican ] # stores the original geom
                nci = len(ms_i)
                es_i = self.es[ ican ]

                inew = True
                if self.param.idiff == 0: # use difference of energy as citeria
                    dEs = np.abs( np.array(es_i) - ei )
                    if np.any( dEs <= self.param.thresh ): inew = False
                elif self.param.idiff == 1:
                    xs = np.array([ rmol.cm, ] )
                    ys = np.array([ ma.cm for ma in self.rms[ican] ])
                    #print ' -- ', xs.shape, ys.shape, can
                    drps = ssd.cdist(xs, ys, 'cityblock')[0]
                    if np.any( drps <= self.param.thresh ): inew = False
                elif self.param.idiff == 2:
                    if not self.check_eigval():
                        inew = False
                else:
                    raise '#ERROR: not supported `idiff'

                if inew:
                    self.ms[ ican ] = ms_i + [m, ]
                    self.rms[ ican ] = rms_i + [ rmol, ]
                    self.ms0[ ican ] = ms0_i + [m0, ]
                    self.es[ ican ] = es_i + [ei, ]
                    self.maps.append( [ir, ican, nci] )
        else:
            m0, m, ei = self.Opt(Mli)
            self.maps.append( [ir, self.ncan, 0] )
            self.cans.append( can )
            self.nhas.append( nha )
            self.ms.append( [m, ] )
            self.rms.append( [rmol, ] )
            self.ms0.append( [m0, ] )
            self.es.append( [ei, ] )
            self.ncan += 1

    def update2(self, ir, can, Mli):
        """
        update mol set if we need SMILES only
        """
        self.ncan = len(self.cans)
        if can not in self.cans:
            print '++', can #, '\n\n'
            self.maps.append( [ir, self.ncan, 0] )
            self.cans.append( can )
            self.nhas.append( nha )
            self.ncan += 1
        else:
            ican = self.cans.index( can )
            entry = [ir, ican, 0]
            if entry not in self.maps:
                self.maps.append( entry )

    def Opt(self, Mli):
        """
        post process molecular fragement retrieved
        from parent molecule by RDKit
        """

        #import io2.mopac as im
        import tempfile as tpf

        zs, coords, bom, charges = Mli
        ctab = oe.write_sdf_raw(zs, coords, bom, charges)

        # get RDKit Mol first
        m0 = Chem.MolFromMolBlock( ctab, removeHs=False ) # plz keep H's
        m0_copy = copy.deepcopy(m0)
        rd = cir.RDMol( m0_copy, forcefield=self.param.ff )

        if self.param.wg:
# the default case, use openbabel to do constrained optimization
            if self.param.do_ob_ff:
                ob1 = cib.Mol( ctab, fmt='sdf' )
                ob1.fixTorsionOpt(iconstraint=3, ff="MMFF94", \
                           optimizer='cg', steps=[30,90], ic=True)
                rd = cir.RDMol( ob1.to_RDKit(), forcefield=self.param.ff )
# if u prefer to use rdkit to do FF optimization
# This seems to be a bad choice as it's likely that
# some bugs exist in RDKit code regarding FF opt
# with dihedral constrains in my system. Test it
# yourself for your own system.
            elif self.param.do_rk_ff:
                rd.fixTorsionOpt(maxIters=self.param.iters[0]) #200) #20)
                rd.Opt(maxIters=self.param.iters[1]) #500) # 25)
# if u prefer to do a partial optimization using PM7 in MOPAC
# for those H atoms and their neighboring heavy atoms
            else:
                pass # no ff opt
        if hasattr(rd, 'energy'):
            e = rd.energy
        else:
            e = rd.get_energy()
        m = rd.m
        return m0, m, e

    def _sort(self):
        """ sort Mlis """
        maps = np.array(self.maps)
        ncan = len(self.cans)
        seqs = np.arange(ncan)
        nhas = np.array(self.nhas)
        ncs = [ len(ms_i) for ms_i in self.ms ]
        cans = np.array(self.cans)
        nhas_u = []
        ncs_u = []
        seqs_u = []
        cans_u = []
        ms_u = []; ms0_u = []

        # now sort the amons by the number of heavy atoms
        for i in range(1, self.param.k2+1):
            seqs_i = seqs[ i == nhas ]
            cans_i = cans[ seqs_i ]
            seqs_j = seqs_i[ np.argsort(cans_i) ]
            seqs_u += list( seqs_j )
            for j in seqs_j:
                cans_u.append( cans[j] )
                ms_j = self.ms[j]; ms0_j = self.ms0[j]
                ncj = len(ms_j)
                ncs_u.append( ncj )
                nhas_u.append( nhas[j] )
                ms_u.append( ms_j ); ms0_u.append( ms0_j )

        seqs_u = np.array(seqs_u)

        # now get the starting idxs of conformers for each amon
        ias2 = np.cumsum(ncs_u)
        ias1 = np.concatenate( ([0,],ias2[:-1]) )

        # now get the maximal num of amons one molecule can possess
        nt = 1+maps[-1,0]; namons = []
        for i in range(nt):
            namon = (maps[:,0] == i).sum()
            namons.append(namon)
        namon_max = max(namons)

        # `maps_u stores the amon idx for each target molecule
        # (Note: any conformer is an amon)
        maps_u = np.zeros((nt, namon_max))
        for i in range(nt):
            filt_i = (maps[:,0] == i)
            maps_i = maps[filt_i, 1:]
            jcnt = 0
            for j in range(namons[i]):
                jcan, jc = maps_i[j,:] # `jcan: the old idx of can
                jcan_u = seqs[ seqs_u == jcan ] # new idx of can
                maps_u[i, jcnt] = ias1[jcan_u] + jc
                jcnt += 1
        self.ms = ms_u
        self.ms0 = ms0_u
        self.cans = cans_u
        self.nhas = nhas_u
        self.ncs = ncs_u
        self.maps = maps_u


    def _sort2(self):
        """ sort Mlis for wg = False"""
        maps = np.array(self.maps)
        ncan = len(self.cans)
        seqs = np.arange(ncan)
        nhas = np.array(self.nhas)
        cans = np.array(self.cans)

        nhas_u = []
        seqs_u = []
        cans_u = []
        # now sort the amons by the number of heavy atoms
        for i in range(1, self.param.k2+1):
            seqs_i = seqs[ i == nhas ]
            cans_i = cans[ seqs_i ]
            seqs_j = seqs_i[ np.argsort(cans_i) ]
            seqs_u += list( seqs_j )
            for j in seqs_j:
                cans_u.append( cans[j] )
                nhas_u.append( nhas[j] )

        seqs_u = np.array(seqs_u)

        # now get the maximal num of amons one molecule can possess
        nt = maps[-1,0]; namons = []
        for i in range(nt):
            namon = (maps[:,0] == i).sum()
            namons.append(namon)
        namon_max = max(namons)

        # `maps_u stores the amon idx for each target molecule
        # (Note: any conformer is an amon)
        maps_u = np.zeros((nt, namon_max))
        for i in range(nt):
            filt_i = (maps[:,0] == i)
            maps_i = maps[filt, 1:]
            jcnt = 0
            for j in range(namons[i]):
                jcan = maps_i[j,1] # `jcan: the old idx of can
                jcan_u = seq[ seqs_u == jcan ] # new idx of can
                maps_u[i, jcnt] = jcan_u
                jcnt += 1
        self.cans = cans_u
        self.nhas = nhas_u
        self.maps = maps_u
        self.ncs = np.ones(ncan).astype(np.int)


class Subgraph(object):
    def __init__(self, atoms, bonds):
        self.atoms = atoms
        self.bonds = bonds

def find_extensions(considered, new_atoms):
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
    for atom in new_atoms:
        for outgoing_bond in atom.GetBonds():
            if outgoing_bond in considered:
                continue
            other_atom = outgoing_bond.GetNbr(atom)
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

def generate_subgraphs(mol, k):

    # Generate all the subgraphs of size 1
    for atom in mol.GetAtoms():
        yield Subgraph(frozenset([atom]), frozenset())

    # Generate the intial seeds. Seed_i starts with bond_i and knows
    # that bond_0 .. bond_i will not need to be considered during any
    # growth of of the seed.
    # For each seed I also keep track of the possible ways to extend the seed.
    seeds = []
    considered = set()
    for bond in mol.GetBonds():
        considered.add(bond)
        subgraph = Subgraph(frozenset([bond.GetBgn(), bond.GetEnd()]),
                            frozenset([bond]))
        yield subgraph
        internal_extensions, external_extensions = find_extensions(considered,
                                                                   subgraph.atoms)
        # If it can't be extended then there's no reason to keep track of it
        if internal_extensions or external_extensions:
            seeds.append( (considered.copy(), subgraph,
                           internal_extensions, external_extensions) )

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
            new_internal, new_external = find_extensions(new_considered, new_atoms)
            if new_internal or new_external:
                seeds.append( (new_considered, new_subgraph, new_internal, new_external) )

def is_subset(a, b):
    """
    a = [1,2], b = [[2,4], [2,1], [3,9,10], ]
    is `a a subset of `b? Yes
    Order of elements in a list DOES NOT matter
    """
    iok = False
    for si in b:
        if set(si) == set(a):
            iok = True
            break
    return iok

def get_number_of_rings(g):
    gu = (g > 0); v = gu.shape[0];
    # in some case, u might have graph with weighted nodes
    # the safe way is  to first assign diagonal elements to 0
    for i in range(v): gu[i,i] = 0
    e = gu.ravel().sum()/2
    r = 2**(e - v + 1) - 1
    return r


class atom_db(object):
    def __init__(self, symbol):
        wd = 'data/atoms/'
        symbs = ['B','C','N','O','Si','P','S', 'F','Cl','Br']
        assert symb in symbs, '#ERROR: no such atomic data?'
        self.oem = oe.StringM( 'data/%s.sdf'%symb ).oem


class ParentMol(object):

    def __init__(self, string, isort=False, iat=None, wg=True, k=7, \
                 k2=7, opr='.le.', fixGeom=False, keepHalogen=False, \
                 ivdw=False, dminVDW=1.2, inmr=False, \
                 covPLmin=5, debug=False):

        self.covPLmin = covPLmin

        self.k = k
        self.k2 = k2
        self.fixGeom = fixGeom
        self.iat = iat
        self.keepHalogen = keepHalogen
        self.debug = debug
        self.vsa = {'.le.': [-1,0], '.eq.': [0, ]}[opr] # valences accepted
        self.wg = wg
        self.ivdw = ivdw
        self.dminVDW = dminVDW
        self.s2cnr = {'H':1, 'B':3, 'C':4, 'N':3, 'O':2, 'F':1, \
                'Si':4, 'P':3, 'S':2, 'Cl':1, 'Br':1, 'I':1}
        self.z2cnr = {1:1, 5:3, 6:4, 7:3, 8:2, 9:1, 14:4, 15:3, 16:2, 17:1, 35:1, 53:1}

        # subg must be a subm (i.e., hybs are all retained)
        # and rings cannot be broken!
        self.FORCE_RING_CLOSED = True

        # ready the molecule
        M = oe.StringM(string, debug=debug)

        # the block below is not necessary. It's only useful
        # for test purpose to check if some fragments are missing.
        # Plus, OEChem has some limitation on num_atoms for
        # subgraph match when the subgraph is actually the whole mol
#       if not isort:
#           # protein is too huge, the program complains that
#           # the default number of matches limit reached in
#           # substructure search if you do `M.sort_atoms()
#           M.sort_atoms()

        self.M = M
        m = M.oem
        self.oem = m
        self.na = M.na
        self.g0 = ( M.bom > 0 ).astype(np.int)
        np.fill_diagonal(self.g0, 0)
        self.bom0 = M.bom

        smi = M.can
        zs0 = np.array(M.zs)
        self.zs0 = zs0
        self.coords = M.coords

        # get CNs of all heavy atoms
        cns0 = self.g0.sum(axis=0)
        self.cns0 = cns0
        cnrs0 = np.array( [ self.z2cnr[zi] for zi in zs0 ] )

        # reference net charge of each atom
        self.charges = M.charges

        # reference total valences
        self.tvs0 = self.bom0.sum(axis=0) + np.abs( self.charges )

        # get reference aromaticity of atoms
        # to be genuinely aromatic, the atom has to be unsaturated
        # Note that the so-called `genuinely aromatic means that the
        # corresponding molecular fragment cannot be described by a
        # unique SMILES string.
        ars0_1 = ( self.tvs0 - cns0 == 1 ); #print ' -- ars0_1 = ', ars0_1
        ars0_2 = [ ai.IsAromatic() for ai in m.GetAtoms() ]
        self.ars0 = np.logical_and(ars0_1, ars0_2) # ars0_1 #
        #print ' iPause, smi = ', smi
        # get envs that corresponds to multiple SMILES
        self.envs = self.get_elusive_envs()
        # get envs like 'C=C=C', 'C=C=N', 'C=N#N', etc
        self.envsC = self.get_envsC()

        if not inmr:
            ncbs = []
            if self.wg and self.ivdw:
                ncbs = M.perceive_non_covalent_bonds(dminVDW=self.dminVDW, \
                                  covPLmin=self.covPLmin)
            self.ncbs = ncbs

    def get_atoms_within_cutoff(self, qa=None, za=None, cutoff=3.6):
        """
        For now, for prediction of NMR only

        retrieve atoms around atom `ia-th H atom within a radius of
        `cutoff.

        This function will be used when dealing with large molecules
        like proteins where long-range interactions are significant.
        Ther related properties include NMR shifts.
        """

        m = self.oem
        self.m = m

        ias = np.arange(self.na)
        ias_hvy = ias[ self.zs0 > 1]
        if za is None:
            ias_za = ias
        else:
            ias_za = ias[ self.zs0 == za ]
        self.ias_za = ias_za

        self.ias_hvy = ias_hvy
        #ds = self.ds
        ds = ssd.squareform( ssd.pdist(self.coords) )
        self.ds = ds

        atoms = [ ai for ai in m.GetAtoms() ]
        self.atoms = atoms

        # get degree of heavy atom
        degrees = []
        for i in range(self.na):
            csi = self.bom0[i]
            degree_i = np.sum( np.logical_and( csi > 0, self.zs0 > 1 ) )
            degrees.append( degree_i )
        self.degrees = degrees

        if qa is None:
            qsa = ias_za
        else:
            qsa = [ias_za[qa], ]
        self.get_rigid_and_flexible_nodes()

        msf = []
        self.sets = []
        boms = []
        mappings = []
        for ia in qsa:
            filt = ( ds[ia] <= cutoff )
            jas = list( ias[filt] )

            # get heavy atoms
            jas_hvy = []
            for j,ja in enumerate(jas):
                zja = self.zs0[ja]
                if zja == 1:
                    nbr = ias[self.g0[ja] == 1][0]
                    #if self.zs0[nbr] in [7,8,9,15,16,17]:
                        # these electronegative atoms will induce electrostatic effects (long-ranged)
                    jas_hvy.append( nbr )
                else:
                    jas_hvy.append( ja )
            #print ' -- jas_hvy = ', jas_hvy
            # get neighbors of those heavy atoms
            sg = self.g0[jas_hvy,:][:,jas_hvy]
            #istop = self.extend_heavy_nodes(jas_hvy, sg)
            self.extend_heavy_nodes(jas_hvy, sg)

            jas_u = self.jas_u
            #print ' -- jas_u = ', jas_u
            bom_u, mapping, mf = self.build_m(jas_u)
            boms.append(bom_u)
            mappings.append( mapping )
            msf.append(mf)

        # the extracted molecular fragments (i.e., `mf) may contain
        # several disconnected components, now add some atoms
        #  re-connecting these standalone entities
        msf_u = []
        self.sets = [] # update !! Vital!!
        for i in range(len(msf)):
            mf_i = msf[i]
            bom_i = boms[i]
            mapping_i = mappings[i]
            mapping_i_reverse = {}
            nodes_i = [] # heavy nodes of `mf
            for keyi in mapping_i.keys():
                val_i = mapping_i[keyi]; nodes_i.append( keyi )
                mapping_i_reverse[val_i] = keyi
            if self.debug: print ' --         nodes = ', nodes_i
            dic_i = mf_i.GetCoords()
            coords_i = []
            for j in range(mf_i.NumAtoms()): coords_i.append( dic_i[j] )
            zsi = np.array([ aj.GetAtomicNum() for aj in mf_i.GetAtoms() ])
            dsi = ssd.squareform( ssd.pdist(coords_i) )

            nodes_new = self.get_nodes_bridge(zsi, mf_i, bom_i, dsi, mapping_i_reverse, nodes_i)
            if self.debug: print ' --     new nodes = ', nodes_new
            jas_hvy = nodes_i + nodes_new
            sg = self.g0[jas_hvy,:][:,jas_hvy]
            #istop = self.extend_heavy_nodes(jas_hvy, sg)
            self.extend_heavy_nodes(jas_hvy, sg)
            jas_u = self.jas_u
            if self.debug: print ' -- jas_u = ', jas_u, ' [updated]'
            mf_u = self.build_m( jas_u )[-1]
            msf_u.append( mf_u )
        msf = msf_u

        # Finally remove any fragment that are part of some larger fragment
        sets = self.sets
        nmf = len(msf)
        nas = np.array( [ len(set_i) for set_i in sets ] )
        seq = np.argsort( nas )[::-1]
        #print ' -- nas = ', nas
        #print ' --seq = ', seq
        sets1 = []
        msf1 = []
        qsa1 = []
        for i in seq:
            sets1.append( sets[i ] )
            msf1.append( msf[i ] )
            qsa1.append( qsa[i ] )

        sets_u = [sets1[0], ]
        msf_u = [msf1[0], ]
        qsa_u = [ qsa1[0], ]
        for i in range( 1, nmf ):
            #print ' -- sets_u = ', sets_u
            ioks2 = [ sets1[i] <= set_j for set_j in sets_u ]
            if not np.any(ioks2): # now remove the `set_j in `sets
                sets_u.append( sets1[i] )
                msf_u.append( msf1[i] )
                qsa_u.append( qsa1[i] )

        self.sets = sets_u
        self.qsa = qsa_u
        self.msf = msf_u


    def get_cutout(self, ias0, cutoff=8.0):
        """
        retrieve the union of local structure within a radius
        of `cutoff of atom in `ias0
        """

        m = self.oem
        self.m = m

        ias = np.arange(self.na)
        ias_hvy = ias[ self.zs0 > 1]

        self.ias_hvy = ias_hvy
        #ds = self.ds
        ds = ssd.squareform( ssd.pdist(self.coords) )
        self.ds = ds

        atoms = [ ai for ai in m.GetAtoms() ]
        self.atoms = atoms

        # get degree of heavy atom
        degrees = []
        for i in range(self.na):
            csi = self.bom0[i,:]
            degree_i = np.sum( np.logical_and( csi > 0, self.zs0 > 1 ) )
            degrees.append( degree_i )
        self.degrees = degrees

        self.get_rigid_and_flexible_nodes()

        msf = []
        self.sets = []
        boms = []
        mappings = []

        jas_u = set()
        icnt = 0
        for ia in ias0:
            filt = ( ds[ia] <= cutoff )
            jas = list( ias[filt] )

            # get heavy atoms
            jas_hvy = []
            for j,ja in enumerate(jas):
                zja = self.zs0[ja]
                if zja == 1:
                    nbr = ias[self.g0[ja,:] == 1][0]
                    #if self.zs0[nbr] in [7,8,9,15,16,17]:
                        # these electronegative atoms will induce electrostatic effects (long-ranged)
                    jas_hvy.append( nbr )
                else:
                    jas_hvy.append( ja )
            #print ' -- jas_hvy = ', jas_hvy
            # get neighbors of those heavy atoms
            sg = self.g0[jas_hvy,:][:,jas_hvy]
            #istop = self.extend_heavy_nodes(jas_hvy, sg)
            self.extend_heavy_nodes(jas_hvy, sg)
#           if 1339 in self.jas_u:
#               if icnt == 0: print self.jas_u
#               icnt += 1
            jas_u.update( self.jas_u )

        bom_u, mapping, mf = self.build_m( list(jas_u) )

        # the extracted molecular fragments (i.e., `mf) may contain
        # several disconnected components, now add some atoms
        #  re-connecting these standalone entities
        mf_i = mf
        bom_i = bom_u
        mapping_i = mapping

        mapping_i_reverse = {}
        nodes_i = [] # heavy nodes of `mf
        for keyi in mapping_i.keys():
            val_i = mapping_i[keyi]; nodes_i.append( keyi )
            mapping_i_reverse[val_i] = keyi
        if self.debug: print ' --         nodes = ', nodes_i
        dic_i = mf_i.GetCoords()
        coords_i = []
        for j in range(mf_i.NumAtoms()): coords_i.append( dic_i[j] )
        zsi = np.array([ aj.GetAtomicNum() for aj in mf_i.GetAtoms() ])
        dsi = ssd.squareform( ssd.pdist(coords_i) )

        nodes_new = self.get_nodes_bridge(zsi, mf_i, bom_i, dsi, mapping_i_reverse, nodes_i)

        if self.debug: print ' --     new nodes = ', nodes_new
        jas_hvy = list( set(nodes_i + nodes_new) )
        sg = self.g0[jas_hvy,:][:,jas_hvy]
        #istop = self.extend_heavy_nodes(jas_hvy, sg)
        #if 1339 in jas_hvy:
        #    idx = jas_hvy.index(1339)
        #    iasU = np.arange(sg.shape[0])
        #    print iasU[ sg[idx,:] > 0 ]

        self.extend_heavy_nodes(jas_hvy, sg)
        jas_u = self.jas_u

        if self.debug: print ' -- jas_u = ', jas_u, ' [updated]'
        mf_u = self.build_m( list(set(jas_u)) )[-1]
        return mf_u


    def get_nodes_bridge(self, zsi, mf_i, bom_i, dsi, mapping_i_reverse, nodes_i):
        """
        get nodes connecting two or more standalone parts in a molecule/fragment
        """
        na_i = len(zsi)
        ias_i = np.arange(na_i)
        iasH = ias_i[ zsi == 1 ]
        nH = len(iasH)
        # get all pairs of H's that are not connected to the same heavy atom
        nodes_new = []
        for jh in range(nH):
            for kh in range(jh+1,nH):
                jh_u = iasH[jh]; kh_u = iasH[kh]
                h_j = mf_i.GetAtom( OEHasAtomIdx(jh_u) )
                h_k = mf_i.GetAtom( OEHasAtomIdx(kh_u) )
                nbr_jh = ias_i[ bom_i[jh_u] == 1 ][0]
                nbr_kh = ias_i[ bom_i[kh_u] == 1 ][0]
                if nbr_jh != nbr_kh:
                    dHH = dsi[kh_u,jh_u]
                    if dHH > 0 and dHH <= 1.6: # a thresh of 1.6 \AA --> ~2 heavy atoms in the shortest path will be added
                        nbr_jh_old = mapping_i_reverse[nbr_jh]
                        nbr_kh_old = mapping_i_reverse[nbr_kh]
                        a1 = self.m.GetAtom( OEHasAtomIdx(nbr_jh_old) )
                        a2 = self.m.GetAtom( OEHasAtomIdx(nbr_kh_old) )
                        #print ' nbr_jh_old, nbr_kh_old = ', nbr_jh_old, nbr_kh_old
                        for a3 in OEShortestPath(a1,a2):
                            ia3 = a3.GetIdx()
                            if ia3 not in nodes_i:
                                nodes_new.append( ia3 )
        return nodes_new

    def extend_heavy_nodes(self, jas_hvy, sg):

        degrees = self.degrees
        sets = self.sets
        ds = self.ds

        set_i = set()
        # get neighbors of those heavy atoms
        for j,ja in enumerate(jas_hvy):
            degree0_j = degrees[ja]
            degree_j = sg[j,:].sum()
            #if ja == 1339: print 'Yeah', degree_j, degree0_j
            if degree_j < degree0_j:
                if ja in self.flexible_nodes: # saturated node
                    set_i.update( [ja,] )
                else:
                    #if ja == 36: print ' Gotha 3 !'
                    for nodes_i in self.rigid_nodes:
                        if ja in nodes_i:
                            set_i.update( nodes_i ); #print ' -- ja, nodes_i = ', ja, nodes_i
            else:
                set_i.update( [ja, ] )

        jas_u = list(set_i) # nodes_of_heavy_atoms
        sets.append( set_i )
        self.sets = sets
        self.jas_u = jas_u
        #return istop


    def build_m(self, nodes_to_add):
        """
        nodes_to_add -- atomic indices to be added to build `mf
        """
        atoms = self.atoms # parent molecule

        mf = OEGraphMol()
        mapping = {}
        atoms_sg = [];

        # step 1, add heavy atoms to `mf
        icnt = 0
        for ja in nodes_to_add:
            aj = atoms[ja]; zj = self.zs0[ja]
            aj2 = mf.NewAtom( zj )
            atoms_sg.append( aj2 ); mapping[ja] = icnt; icnt += 1
            aj2.SetHyb( OEGetHybridization(aj) )
            mf.SetCoords(aj2, self.coords[ja])
        # step 2, add H's and XH bond
        bonds = []
        #print ' -- nodes_to_add = ', nodes_to_add
        for j,ja in enumerate(nodes_to_add):
            aj = atoms[ja]
            zj = self.zs0[ja]
            aj2 = atoms_sg[j]
            for ak in aj.GetAtoms():
                ka = ak.GetIdx()
                zk = ak.GetAtomicNum()
                if zk == 1:
                    ak2 = mf.NewAtom( 1 )
                    b2 = mf.NewBond( aj2, ak2, 1)
                    mf.SetCoords(ak2, self.coords[ka]); #print ' - ka, ', self.coords[ka]
                    bonds.append( [icnt,j,1] ); icnt += 1
                else:
                    # __don't__ add atom `ak to `mf as `ak may be added to `mf later
                    # in the for loop ``for ja in nodes_to_add`` later!!
                    if ka not in nodes_to_add:
                        # add H
                        v1 = self.coords[ka] - self.coords[ja]
                        dHX = dsHX[zj];
                        coords_k = self.coords[ja] + dHX*v1/np.linalg.norm(v1)
                        ak2 = mf.NewAtom( 1 )
                        mf.SetCoords(ak2, coords_k); #print ' --- ka, ', coords_k
                        b2 = mf.NewBond( aj2, ak2, 1)
                        bonds.append( [icnt,j,1] ); icnt += 1

        nadd = len(nodes_to_add)
        #print ' __ nodes_to_add = ', nodes_to_add
        for j in range(nadd):
            for k in range(j+1,nadd):
                #print ' j,k = ', j,k
                ja = nodes_to_add[j]; ka = nodes_to_add[k]
                ja2 = mapping[ja]; ka2 = mapping[ka]
                bo = self.bom0[ja,ka]
                if bo > 0:
                    aj2 = atoms_sg[ja2]; ak2 = atoms_sg[ka2]
                    bonds.append( [j,k,bo] )
                    b2 = mf.NewBond( aj2, ak2, bo )
                    #print ' (ja,ka,bo) = (%d,%d,%d), '%(ja, ka, bo), '(ja2,ka2,bo) = (%d,%d,%d)'%(ja2,ka2,bo)
        assert mf.NumAtoms() == icnt
        bom_u = np.zeros((icnt,icnt), np.int)
        for bond_i in bonds:
            bgn,end,bo_i = bond_i
            bom_u[bgn,end] = bom_u[end,bgn] = bo_i

        return bom_u, mapping, mf


    def get_rigid_and_flexible_nodes(self):
        """
        NMR only

        (1) rigid nodes
            extended smallest set of small unbreakable fragments,
            including aromatic rings, 3- and 4-membered rings
            (accompanied with high strain, not easy to cover these
            interactions in amons) and -C(=O)N- fragments

            These nodes is output as a list of lists, with each
            containing the atom indices for a unbreakable ring
            with size ranging from 3 to 9, or -C(=O)N-
        (2) flexible nodes
            a list of saturated atom indices
        """

        def update_sets(set_i, sets):
            if np.any([ set_i <= set_j for set_j in sets ]):
                return sets
            intersected = [ set_i.intersection(set_j) for set_j in sets ]
            istats = np.array([ si != set() for si in intersected ])
            nset = len(sets); idxs = np.arange( nset )
            if np.any( istats ):
                #assert istats.astype(np.int).sum() == 1
                for iset in idxs:
                    if istats[iset]:
                        sets[iset] = set_i.union( sets[iset] )
            else:
                sets.append( set_i )
            return sets

        m = self.oem
        nodes_hvy = list( np.arange(self.na)[ self.zs0 > 1 ] )

        # first search for rings
        namin = 3
        namax = 10
        sets = []
        for i in range(namin, namax+1):
            if i in [3,4,]:
                pat_i = '*~1' + '~*'*(i-2) + '~*1'
            else:
                pat_i = '*:1' + ':*'*(i-2) + ':*1'
            ss = OESubSearch(pat_i)
            iok = OEPrepareSearch(m, ss)
            for match in ss.Match(m):
                set_i = set()
                for ma in match.GetAtoms():
                    set_i.update( [ma.target.GetIdx()] )
                if set_i not in sets: sets.append( set_i )
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
        sets = sets_u

        # then find atoms with hyb .le. 2, e.g., -C(=O)N-, -C(=O)O-,
        # -[N+](=O)[O-], -C#N, etc
        iasc = []
        for ai in m.GetAtoms():
            hyb = OEGetHybridization(ai)
            if hyb < 3 and hyb > 0:
                iasc.append( ai.GetIdx() )
        sg = self.g0[iasc,:][:,iasc]
        na_sg = len(iasc)
        dic_sg = dict(zip(range(na_sg), iasc))
        for sgi in oe.find_cliques( sg ):
            set_i = set([ dic_sg[ii] for ii in sgi ])
            sets = update_sets(set_i, sets)

        for pat_i in ['[CX3](=O)[O,N]', '[#7,#8,#9;!a][a]', ]:
            ss = OESubSearch(pat_i)
            iok = OEPrepareSearch(m, ss)
            for match in ss.Match(m):
                set_i = set()
                for ma in match.GetAtoms():
                    set_i.update( [ma.target.GetIdx()] )
                sets = update_sets(set_i, sets)
        rigid_nodes = [ list(si) for si in sets ]
        rigid_nodes_ravel = []
        for nodes_i in rigid_nodes: rigid_nodes_ravel += nodes_i
        self.rigid_nodes = rigid_nodes

        # now find flexible nodes, i.e., saturated nodes with breakable connected bonds
        flexible_nodes = list( set(nodes_hvy)^set(rigid_nodes_ravel) )

        obsolete = """
        flexible_nodes = set()
        for pat_i in ['[CX4,SiX4,PX3,F,Cl,Br,I]', ]:
            ss = OESubSearch(pat_i)
            iok = OEPrepareSearch(m, ss)
            for match in ss.Match(m):
                for ma in match.GetAtoms():
                    flexible_nodes.update( [ma.target.GetIdx()] )

        for pat_i in [ '[NX3;!a]', '[OX2;!a]', '[SX2;!a]', ]:
            ss = OESubSearch(pat_i)
            iok = OEPrepareSearch(m, ss)
            for match in ss.Match(m):
                for ma in match.GetAtoms():
                    ia = ma.target.GetIdx()
                    if ia not in rigid_nodes_ravel:
                        flexible_nodes.update( [ia] )"""

        self.flexible_nodes = list(flexible_nodes)

    def get_elusive_envs(self):
        """
        check if the bond linking two atoms in `ias is eligbile
        for breaking by inspecting if these two atoms are in a
        elusive environment, i.e., a genuinely aromatic env
        e.g., Cc1c(C)cccc1 (two SMILES exist!); exceptions:
        o1cccc1, since it has only one possible SMILES string
        """
        filt = np.array( self.ars0 ) #.astype(np.int32)
        if filt.astype(np.int).sum() == 0: return set([])
        ias0 = np.arange( len(self.ars0) )
        ias1 = ias0[filt]
        g2 = self.bom0[filt, :][:, filt]
        iok = False
        envs = set([])
        for cliques_i in oe.find_cliques(g2):
            if len(cliques_i) > 2:
                gci = g2[cliques_i, :][:, cliques_i]

                # set `irad to False to ensure that no atom is unsaturated in valence
                ess, nrss = oe.edges_standalone_updated(gci, irad=False)
                #print ' __ ess = ', ess
                #print ' __ nrss = ', nrss

                # note that for genuinely aromatic env, `nrss should contain
                # no more than 1 unique list
                nes = len(ess)
                if nes > 0:
                    n1 = len(ess[0])
                    nrs1 = set( nrss[0] )
                    for nrs in nrss[1:]:
                        if set(nrs) != nrs1:
                            raise '#ERROR: more than 2 different sets in nodes_residual??'
                #    # get the common list in `ess, then remove it
                #    # e.g., a C=C attached to a benzene ring is an
                #    # explicit env, inly the benzene ring is an implicite
                #    # env as there are more than 1 corresponding SMIELS string
                #    comms = []
                #    for i in range(n1):
                #        comm = ess[0][i]
                #        if np.all( [ comm in ess[j] for j in range(1,nes) ] ):
                #            comms.append( comm )
                #
                #    envs.update( set( oe.comba( cim.get_compl_u(ess[0],comms) ) ) )
                    envs.update( set( oe.comba( ess[0]) ) )
        envs_u = set( [ ias1[k] for k in list(envs) ] )
        return envs_u

    def get_envsC(self):
        """
        get conjugated environments containing adjacent double bonds,
        e.g., C=C=C, C=N#N
        """
        qs = ['[*]=[*]#[*]', '[*]=[*]=[*]']
        ts = []
        for q in qs:
            ots = oe.is_subg(self.oem, q, iop = 1)
            if ots[0]:
                for tsi in ots[1]:
                    tsi_u = set(tsi)
                    if len(ts) == 0:
                        ts.append( tsi_u )
                    else:
                        iexist = False
                        for j, tsj in enumerate(ts):
                            if tsi_u.intersection(tsj):
                                iexist = True
                                tsj.update( tsi_u )
                                ts[j] = tsj
                                break
                        if not iexist: ts.append( tsi_u )
        return ts

    def check_strains(self, strains_sg0, m, nmax_no_match=0):
        ## remove molecule with inconsistent strains
        strains_sg = []
        for aj in m.GetAtoms():
            istrain = False
            if OEAtomIsInRingSize(aj, 3) or OEAtomIsInRingSize(aj, 4):
                istrain = True
            strains_sg.append(istrain)
        strains_sg = np.array(strains_sg)
        iok = True
        #if sum(strains_sg) != sum(strains_sg0):
        n_no_match = ( strains_sg != strains_sg0 ).sum()
        if n_no_match > nmax_no_match:
            iok = False
        return iok


    def generate_amons(self): #
        """
        generate all canonial SMARTS of the fragments (up to size `k)
        of any given molecule
        """

        nat = self.na
        m0 = self.oem
        OESuppressHydrogens(m0, False, False, False)
        iast = np.arange(nat)
        cans_u = []
        for seed in generate_subgraphs(m0, self.k2):
            #print ' \n '
            # Make a new molecule based on the substructure
            subm = OEGraphMol()
            new_atoms = {}
            atoms_U = []
            nha = 0; icnt = 0;
            ias0 = []; dic = {}
            ars = []
            for atom in seed.atoms:
                zi = atom.GetAtomicNum()
                if zi > 1:
                    nha += 1
                    ia0 = atom.GetIdx()
                    ars.append( self.ars0[ia0] )
                    ias0.append(ia0)
                    dic[ia0] = icnt
                    new_atom = subm.NewAtom( zi )
                    new_atoms[icnt] = new_atom
                    atoms_U.append( new_atom )
                    icnt += 1
            # constraint on number of heavy atoms
            if cmp(nha, self.k2) not in self.vsa:
                continue

            iasP = ias0
            for i in ias0:
                iasP += list( iast[ np.logical_and(self.zs0[i]==1, self.bom0[i]==1) ])

            if self.iat is not None:
                # if `iat is not a hydrogen atom
                jja = self.iat - 1
                zjj = self.zs0[jja]
                #print ' -- jja, zjj = ', jja, zjj
                if zjj > 1:
                    jok = ( jja in ias0 ) # self.iat is an idx starting from 1
                # otherwise, check if this H atom is connected to any heavy atom in `ias0
                else:
                    jok = False
                    for iia in ias0:
                        if self.bom0[iia, jja] > 0:
                            jok = True
                            break
                if not jok:
                    continue

            if self.debug:
                ias0_u = [ iaU for iaU in ias0 ]
                #ias0_uu.sort()
                print '    -- ias0 = ', ias0_u
                print '    -- zs0  = ', [ self.zs0[iaU] for iaU in ias0 ]

            iPause = False
            #if set(ias0) == set([0,1,3,4,5,6,7]): iPause = True

            zs_sg0 = self.zs0[ias0]
            sg0 = self.bom0[ias0,:][:,ias0]
            # get number of rings
            nr0 = get_number_of_rings(sg0)
            cns_sg0 = self.cns0[ias0]
            tvs_sg0 = self.tvs0[ias0]
            # get the coords_sg0 and cns_sg0 before matching SMARTS to target
            coords_sg0 = self.coords[ias0]

            sg = np.zeros((nha,nha), np.int)
            nncb = 0
            for bond in seed.bonds:
                ap, aq = bond.GetBgn(), bond.GetEnd()
                z1, z2 = ap.GetAtomicNum(), aq.GetAtomicNum()
                idx1, idx2 = bond.GetBgnIdx(), bond.GetEndIdx()
                ias12 = [idx1, idx2]; ias12.sort()
                if z1 > 1 and z2 > 1:
                    ia1, ia2 = dic[idx1], dic[idx2]
                    # automatically neglect the hydrogen atoms since later they would
                    # be added on demand; again this occurs for composite systems
                    # involving hydrogen bonds
                    a1, a2 = new_atoms[ia1], new_atoms[ia2]
                    b12 = bond.GetOrder()

                    # with new algorithm, lines below won't be used
#                   if is_subset(ias12, self.ncbs):
#                       nncb += 1 # do not add such a bond to the fragment!
#                   else:

                    new_bond = subm.NewBond(a1, a2, b12)
                    sg[ia1,ia2] = sg[ia2,ia1] = 1
                    # GetOrder() sets the Kekule form, but I want the aromatic
                    # form.  If I don't do this I end up with terms like "c=c"
                    # in my results. Setting it to a single bond works because
                    # the output will always have "" as the bond (as in "CC",
                    # "cC" or "cc"). The only way to get an incorrect result
                    # is if the structure should be "c-c", which is rare. But
                    # since the "" connection means "single or aromatic", then
                    # the SMARTS won't miss it, so we're okay. As long as I'm
                    # consistent.
                    if bond.IsAromatic():
                        new_bond.SetOrder(1)

            # check if the fragment contains `nhbmin HB's
            #if len(self.hbs) > 0:
            #    if nhb < self.nhbmin:
            #        continue

            nr = get_number_of_rings(sg); delta_nr = nr0 - nr
            # delta_nr == 0 to ensure it's a complete ring
            if delta_nr > 0 and self.FORCE_RING_CLOSED:
                continue

            cns_sg = sg.sum(axis=0)
            nihs = cns_sg0 - cns_sg

            # tell if the molecule is gonna be a radical
            dvs_sg = tvs_sg0 - cns_sg0
            istat = False
            #if set(ias0) == set([0,1,2,3,]):
            #    istat = True
#                print ' dvs_sg = ', dvs_sg; print ' ias0 = ', ias0
            if nha == 1:
                if dvs_sg[0] != 0:
                    continue
            #print ' -- tvs_sg0 = ', tvs_sg0
            #print ' -- cns_sg0 = ', cns_sg0
            #print ' -- dvs_sg = ', dvs_sg

# first set the entries in `dvs_sg corresponding to nodes involved in
# local envs like "C=C=C", "C=C=N", "C=N#N", etc,
# i.e., we don't have to amend the BO's related to these atoms as later
# the correction of BO's can be done by a robust function `perceive_bond_order()
            for ii,iia in enumerate(ias0):
                if np.any([ iia in list(tsi) for tsi in self.envsC ]):
                    dvs_sg[ii] = 0
            #print '     ****** dvs_sg = ', dvs_sg

            filt = (np.array(dvs_sg) == 1)
            g3 = sg[filt, :][:, filt]
            na3 = g3.shape[0]
            if self.debug: print ' ________na3 = ', na3
            if na3 == 1:
                continue
            elif na3 > 1:
                is_rad = False
                #if np.all(g3 == 0): # e.g., [CH2]N[CH2]
                #    is_rad = True
                #else:
                for cliques_i in oe.find_cliques(g3):
                    nnode_i = len(cliques_i)
                    if nnode_i == 1:
                        is_rad = True; break
                    elif len(cliques_i) > 2:
                        #print ' -- cliques_i = ', cliques_i
                        gci = g3[cliques_i, :][:, cliques_i]

                        # simply set `irad to True so as to get all possible
                        # combinations of double bonds
                        ess, nrss = oe.edges_standalone_updated(gci, irad=True)
                        if self.debug: print ' _______ess, nrss = ', ess, nrss
                        if np.all([ len(nrs) > 0 for nrs in nrss ]):
                            is_rad = True; break
                if is_rad:
                    continue

            #print ' iPause #2, ', self.ars0, self.envs
            if istat:
                print ' ##1'
                self.debug = True
            else:
                self.debug = False

            istop = False
            atoms_U = [ ]; icnt = 0
            #for ak in subm.GetAtoms():
            #    ak.SetImplicitHCount( nihs[icnt] )
            #    if self.wg:
            #        subm.SetCoords( ak, coords_sg0[icnt] )
            #    icnt += 1
            atoms_U = [ aq for aq in subm.GetAtoms() ]
            if self.debug:
                print ' -- sg0 = '
                for kk,zk in enumerate(zs_sg0):
                    x1,y1,z1 = coords_sg0[kk]
                    print '%2d %8.4f %8.4f %8.4f'%(zk,x1,y1,z1)

            if istat: print ' ____ atoms_U = ', atoms_U
# don't use `subm.GetAtoms(), as it will be updated later,
# and the size of it constantly changes.
            for ai in atoms_U:
                ia = ai.GetIdx()
                coords_i = coords_sg0[ia]
                subm.SetCoords(ai, coords_i)
                ia0 = ias0[ia]
                zi = self.zs0[ia0]
                jas0 = iast[ self.bom0[ia0] > 0 ]

                for ja0 in jas0:
                    zj = self.zs0[ja0]
                    if ja0 not in ias0:
                        #print ' --- ia0, ja0 = ', ia0, ja0
                        boij = self.bom0[ia0, ja0]
                        #if istat:

                        # don't break any triple bond in, say, N#N=R
                        yes1 = ( boij > 2 ) #

                        if self.fixGeom:
# don't break standalone double bonds (i.e.,BO=2 and not in aromatic env)
                            yes2 = ( boij == 2 and \
                                     (not (set([ia0,ja0]) < self.envs ) ) )
                            #if set(ias0) == set([3,4,5,6]):
                            #    print ' -- yes2, envs = ', yes2, self.envs
                            #    print '     ia0,ja0,boij = ', ia0,ja0,boij
                        else:
## if the amons are allowed to fully relax, it's possible to break a double bond
## as long the hybs are attained. e.g., for c1cccc1C=O, C=CC(=C)C=C is also a valid amon
                            yes2 = ( boij == 2 and \
                                     ( oe.is_subsub(set([ia0,ja0]), self.envsC) ) )
                        if yes1 or yes2:
                            istop = True
                            break
                        else:
                            # since H's were not added before
                            newa = subm.NewAtom( 1 )
                            newb = subm.NewBond(ai, newa, 1)
                            if self.wg:
                                if zj == 1:
                                    subm.SetCoords(newa, self.coords[ja0])
                                else:
                                    v1 = self.coords[ja0] - coords_i
                                    dHX = dsHX[zi]
                                    coords_j = coords_i + dHX*v1/np.linalg.norm(v1)
                                    subm.SetCoords(newa, coords_j)
                if istop: break
            if istop: continue

            #if iPause: print ' iPause #3'
            if istat: print ' ##2', cans

            OEFindRingAtomsAndBonds(subm)
            smarts = OECreateSmiString(subm, OESMILESFlag_Canonical)
            #print '    ** smarts = ', smarts
            if '.' not in smarts:
                if nha > self.k:
                    continue

            if iPause: print ' iPause #4'

            # at most 2 standalone molecules in one Amon
            if len( smarts.split('.') ) > 2: continue

            #print ' -- nha, na, smarts = ', nha, subm.NumAtoms(), smarts # 1111
            if self.debug:
                print ' -- na, smarts = ', subm.NumAtoms(), smarts
                print ' -- na = ', subm.NumAtoms()
                print ' -- g = ',
                print np.array( oe.oem2g(subm) )

            if nha == 1:
                z1 = zs_sg0[0]
                symb1 = ad.chemical_symbols[z1]
                if not self.keepHalogen:
                    if symb1 in ['F','Cl','Br','I',]:
                        continue
            M = oe.Mol(subm)

            # set `irad to False to ensure that all unsaturated atoms are
            # involved in double bonds
            M.perceive_bond_order(once=False, user_cns0=tvs_sg0, irad=False)
            zs_i = M.zs # it's the same for all configs sharing the same `can
            coords_i = M.coords

            # check if `coords changed
            dd = np.abs( coords_i[np.array(zs_i)>1, :] - coords_sg0 )
            if np.any(dd > 1e-4): raise '#ERROR: coords changed! [#1]'

            Mols = M.oem
            boms = M.bom
            charges = M.charges
            cans = M.can

            cans_u = []
            for i, can in enumerate(cans):
                mu = Mols[i]
                bom_i = boms[i]
                charges_i = charges[i]

#========================================================
                ## added lately
                if self.fixGeom:
                    ds_i = ssd.squareform( ssd.pdist(coords_i) )
                    gnb = (bom_i == 0) # non-bonded graph
                    np.fill_diagonal(gnb, False)
                    if np.any(ds_i[gnb] <= self.dminVDW): continue
#========================================================

                Mli = [ zs_i, coords_i, bom_i, charges_i ]
                if can in cans_u:
                    if (nha <= 3) and (not self.fixGeom):
                        continue
                else:
                    cans_u.append( can )
                yield Mli, iasP, can #, mu, can, nha


class ParentMols(object):

    def __init__(self, strings, fixGeom, iat=None, wg=True, k=7,\
                 nmaxcomb=3,icc=None, substring=None, rc=6.4, \
                 isort=False, k2=7, opr='.le.', wsmi=True, irc=True, \
                 iters=[30,90], dminVDW= 1.2, \
                 idiff=0, thresh=0.2, \
                 keepHalogen=False, debug=False, ncore=1, \
                 forcefield='mmff94', do_ob_ff=True, do_rk_ff=False, \
                 ivdw=False, covPLmin=5, prefix=''):
#                 do_pm7=False, relaxHHV=False, \
        """
        prefix -- a string added to the beginning of the name of a
                  folder, where all sdf files will be written to.
                  It should be ended with '_' if it's not empty
        irc    -- T/F: relax w/wo dihedral constraints

        substring -- SMILES of a ligand.
                  Typically in a protein-ligand complex, we need
                  to identify the ligand first and then retrieve
                  all the local atoms that bind to the ligand via
                  vdW interaction as amons for training in ML. The
                  thus obtained fragment is dubbed `centre.

                  If `substring is assigned a string,
                  we will generated only amons that are
                  a) molecular complex; b) any atom in the centre
                  must be involved.
        rc     -- cutoff radius centered on each atom of the central
                  component. It's used when `icc is not None.
        """

        def check_ncbs(a, b, c):
            iok = False
            for si in itl.product(a,b):
                if set(si) in c:
                    iok = True; break
            return iok

        param = Parameters(wg, fixGeom, k, k2, ivdw, dminVDW, \
                           forcefield, thresh, do_ob_ff, \
                           do_rk_ff, idiff, iters)

        # at most 1 True can be set
        ioks = [ do_ob_ff, do_rk_ff, ]
        assert np.array(ioks).astype(np.int).sum() <= 1

        ncpu = multiprocessing.cpu_count()
        if ncore > ncpu:
            ncore = ncpu

        # temparary folder
        tdirs = ['/scratch', '/tmp']
        for tdir in tdirs:
            if os.path.exists(tdir):
                break

        # num_molecule_total
        assert type(strings) is list, '#ERROR: `strings must be a list'
        nmt = len(strings)
        if iat != None:
            assert nmt == 1, '#ERROR: if u wanna specify the atomic idx, 1 input molecule at most is allowed'

        cans = []; nhas = []; es = []; maps = []
        ms = []; ms0 = []

        # initialize `Sets
        seta = Sets(param)
        for ir in range(nmt):
            print ' -- Mid %d'%(ir+1)
            string = strings[ir]
            obj = ParentMol(string, isort=isort, iat=iat, wg=wg, k=k, k2=k2, \
                            opr=opr, fixGeom=fixGeom, covPLmin=covPLmin, \
                            ivdw=ivdw, dminVDW=dminVDW, \
                            keepHalogen=keepHalogen, debug=debug)
            ncbs = obj.ncbs
            Mlis, iass, cans = [], [], []
            # we needs all fragments in the first place; later we'll
            # remove redundencies when merging molecules to obtain
            # valid vdw complexes
            nas = []; nasv = []; pss = []
            iass = []; iassU = []
            for Mli, ias, can in obj.generate_amons():
                iasU = ias + [-1,]*(k-len(ias)); nasv.append( len(ias) )
                Mlis.append( Mli ); iass.append( ias ); cans.append( can )
                iassU.append( iasU ); pss += list(Mli[1])
                nas.append( len(Mli[0]) )
            nmi = len(cans)
            print ' -- nmi = ', nmi

            nas = np.array(nas, np.int)
            nasv = np.array(nasv, np.int)
            pss = np.array(pss)
            iassU = np.array(iassU, np.int)
            ncbsU = np.array(ncbs, np.int)

            # now combine amons to get amons complex to account for
            # long-ranged interaction
            if wg and ivdw:
                if substring != None:
                    cliques_c = set( oe.is_subg(obj.oem, substring, iop=1)[1][0] )
                    #print ' -- cliques_c = ', cliques_c
                    cliques = oe.find_cliques(obj.g0)
                    Mlis_centre = []; iass_centre = []; cans_centre = []
                    Mlis_others = []; iass_others = []; cans_others = []
                    for i in range(nmi):
                        #print ' %d/%d done'%(i+1, nmi)
                        if set(iass[i]) <= cliques_c:
                            Mlis_centre.append( Mlis[i] )
                            iass_centre.append( iass[i] )
                            cans_centre.append( cans[i] )
                        else:
                            Mlis_others.append( Mlis[i] )
                            iass_others.append( iass[i] )
                            cans_others.append( cans[i] )
                    nmi_c = len(Mlis_centre)
                    nmi_o = nmi - nmi_c
                    print ' -- nmi_centre, nmi_others = ', nmi_c, nmi_o
                    Mlis_U = []; cans_U = []
                    for i0 in range(nmi_c):
                        ias1 = iass_centre[i0]
                        t1 = Mlis_centre[i0]; nha1 = (np.array(t1[0]) > 1).sum()
                        for j0 in range(nmi_o):
                            ias2 = iass_others[j0]
                            t2 = Mlis_others[j0]; nha2 = np.array((t2[0]) > 1).sum()
                            if nha1 + nha2 <= k2 and check_ncbs(ias1, ias2, ncbs):
                                dmin = ssd.cdist(t1[1], t2[1]).min()
                                if dmin >= dminVDW:
                                    cansij = [cans_centre[i0], cans_others[j0]]
                                    cansij.sort()
                                    cans_U.append( '.'.join(cansij) )
                                    Mlis_U.append( merge(t1, t2) )
                    Mlis = Mlis_U; cans = cans_U
                    print ' -- nmi_U = ', len(Mlis)
                else:
                    obsolete = """
                    for i0 in range(nmi-1):
                        ias1 =  iass[i0]
                        t1 = Mlis[i0]; nha1 = (np.array(t1[0]) > 1).sum()
                        for j0 in range(i0+1,nmi):
                            ias2 = iass[j0]
                            t2 = Mlis[j0]; nha2 = np.array((t2[0]) > 1).sum()
                            if nha1 + nha2 <= k2 and check_ncbs(ias1, ias2, obj.ncbs):
                                dmin = ssd.cdist(t1[1], t2[1]).min()
                                if dmin >= dminVDW:
                                    cansij = [cans[i0], cans[j0]]
                                    cansij.sort()
                                    cans.append( '.'.join(cansij) )
                                    Mlis.append( merge(t1, t2) )"""
                    obsolete = """print 'nas.shape = ', nas.shape
                    print 'nasv.shape = ', nasv.shape
                    print 'iassU.shape = ', iassU.shape
                    print 'pss.shape = ', pss.shape
                    print 'ncbsU.shape = ', ncbsU.shape"""
                    print 'dminVDW = ', dminVDW
                    gv,gc = fa.get_amon_adjacency(k2,nas,nasv,iassU.T,pss.T,ncbsU.T,dminVDW)
                    print 'amon connectivity done'
                    #print 'gv=',gv # 'np.any(gv > 0) = ', np.any(gv > 0)
                    ims = np.arange(nmi)
                    combs = []
                    for im in range(nmi):
                        nv1 = nasv[im]
                        jms = ims[ gv[im] > 0 ]
                        nj = len(jms)
                        if nj == 1:
                            # in this case, nmaxcomb = 2
                            jm = jms[0]
                            if nmaxcomb == 2:
                                # setting `nmaxcomb = 2 means to include
                                # all possible combinations consisting of
                                # two standalone molecules
                                comb = [im,jms[0]]; comb.sort()
                                if comb not in combs:
                                    combs += [comb]
                            else:
                                # if we are not imposed with `nmaxcomb = 2,
                                # we remove any complex corresponding to 2) below
                                #
                                # 1)    1 --- 2  (no other frag is connected to `1 or `2)
                                #
                                # 2)    1 --- 2
                                #              \
                                #               \
                                #                3
                                if len(gv[jm]) == 1:
                                    comb = [im,jm]; comb.sort()
                                    if comb not in combs:
                                        combs += [comb]
                        else:
                            if nmaxcomb == 2:
                                for jm in jms:
                                    comb = [im,jm]; comb.sort()
                                    if comb not in combs:
                                        combs += [comb]
                            elif nmaxcomb == 3:
                                for jm in jms:
                                    comb = [im,jm]; comb.sort()
                                    if comb not in combs:
                                        combs += [comb]

                                # this is the default choice and is more reasonable
                                # as only the most relevant local frags are included.
                                # Here we don't consider frags like [im,p],[im,q] as
                                # 1) the local envs are covered by [im,p,q]; 2) it's less
                                # relevant to [im,p,q]
                                for (p,q) in itl.combinations(jms,2):
                                    nv2 = nasv[p]; nv3 = nasv[q]
                                    if nv1+nv2+nv3 <= k2 and gc[p,q] == 0:
                                        comb = [im,p,q]; comb.sort()
                                        if comb not in combs:
                                            combs += [comb]
                    print 'atom indices of all amons done'
                    for comb in combs:
                        #print comb
                        cans_i = [ cans[ic] for ic in comb ]; cans_i.sort()
                        cans.append('.'.join(cans_i))
                        ts_i = [ Mlis[ic] for ic in comb ]
                        Mlis.append( merge(ts_i) )
                    print 'amons now ready for filtering'

            ncan = len(cans)
            # now remove redundancy
            if wg:
                #print ' cans = ', cans
                for i in range(ncan):
                    #print '** ', cans[i], (np.array(Mlis[i][0]) > 1).sum(),\
                    #                       len(Mlis[i][0]), Mlis[i][0]
                    seta.update(ir, cans[i], Mlis[i])
                seta._sort()
            else:
                for i in range(ncan):
                    seta.update2(ir, cans[i], Mlis[i])
                seta._sor2()
            print 'amons are sorted and regrouped'

        cans = seta.cans; ncs = seta.ncs; nhas = seta.nhas

        ncan = len(cans)
        self.cans = cans
        if not wsmi: return
        nd = len(str(ncan))

        s1 = 'EQ' if opr == '.eq.' else ''
        svdw = '_vdw%d'%k2 if ivdw else ''
        scomb = '_comb2' if nmaxcomb == 2 else ''
        sthresh = '_dE%.2f'%thresh if thresh > 0 else ''
        if prefix == '':
            fdn = 'g%s%d%s%s_covL%d%s'%(s1,k,svdw,sthresh,covPLmin,scomb)
        else:
            fdn = prefix

        if not os.path.exists(fdn): os.system('mkdir -p %s'%fdn)
        self.fd = fdn

        if iat is not None:
            fdn += '_iat%d'%iat # absolute idx
        if wg and (not os.path.exists(fdn+'/raw')): os.system('mkdir -p %s/raw'%fdn)
        with open(fdn + '/' + fdn+'.smi', 'w') as fid:
            fid.write('\n'.join( [ '%s %d'%(cans[i],ncs[i]) for i in range(ncan) ] ) )
        dd.io.save('%s/maps.h5'%fdn, {'maps': maps} )

        if wg:
            ms = seta.ms; ms0 = seta.ms0;
            for i in range(ncan):
                ms_i = ms[i]; ms0_i = ms0[i]
                nci = ncs[i]
                labi = '0'*(nd - len(str(i+1))) + str(i+1)
                print ' ++ %d %06d/%06d %60s %3d'%(nhas[i], i+1, ncan, cans[i], nci)
                for j in range(nci):
                    f_j = fdn + '/frag_%s_c%05d'%(labi, j+1) + '.sdf'
                    f0_j = fdn + '/raw/frag_%s_c%05d_raw'%(labi, j+1) + '.sdf'
                    m_j = ms_i[j]; m0_j = ms0_i[j]
                    Chem.MolToMolFile(m_j, f_j)
                    Chem.MolToMolFile(m0_j, f0_j)
            print ' -- nmi_u = ', sum(ncs)
            print ' -- ncan = ', len(np.unique(cans))
        else:
            if wsmi:
                with open(fdn + '/' + fdn+'.smi', 'w') as fid:
                    fid.write('\n'.join( [ '%s'%(cans[i]) for i in range(ncan) ] ) )

