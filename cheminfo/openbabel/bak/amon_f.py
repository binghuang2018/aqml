#!/usr/bin/env python

"""
Enumerate subgraphs & get amons
"""

import cheminfo as ci
import cheminfo.math as cim
import cheminfo.graph as cg
import networkx as nx
from itertools import chain, product
import numpy as np
import os, re, copy, time
#from rdkit import Chem
import openbabel as ob
import pybel as pb
import cheminfo.fortran.famon as cf

global dic_smiles
dic_smiles = {6:'C', 7:'N', 8:'O', 14:'Si', 15:'P', 16:'S'}


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


def accommodate_chgs(chgs, bom):
    """update bom based on `chgs
    e.g., C=N#N, bond orders = [2,3],
    Considering that `chgs = [0,+1,-1],
    bond orders has to be changed to [2,2]"""
    bom2 = copy.copy(bom)
    na = len(chgs)
    ias = np.arange(na)
    ias1 = ias[chgs == 1]
    for i in ias1:
        iasc = ias[ np.logical_and(chgs==-1, bom[i]>0) ]
        nac = len(iasc)
        if nac > 0:
            #assert nac == 1
            j = iasc[0]
            bij = bom[i,j] - 1
            bom2[i,j] = bij
            bom2[j,i] = bij
    return bom2





class vars(object):
    def __init__(self, zs, chgs, tvs, g, coords):
        self.zs = zs
        self.chgs = chgs
        self.tvs = tvs
        self.g = g
        self.coords = coords


class MG(vars):

    def __init__(self, zs, chgs, tvs, g, coords):
        vars.__init__(self, zs, chgs, tvs, g, coords)

    def update_bom(self, once=True, debug=False):
        g = self.g
        chgs = self.chgs
        vs = g.sum(axis=0).astype(np.int)
        tvs = self.tvs # `tvs has been modified according to `chgs
        zs = self.zs
        na = len(zs)
        ias = np.arange(na)

#       print ' zs = ', zs
#       print 'tvs = ', tvs
#       print 'dvs = ', tvs - vs
        #t1 = time.time()
        iok, bom = cf.update_bom(zs,tvs,g)
        #t2 = time.time()
        #print '      update_bom: ', t2-t1
#       print ' ** iok = ',iok
#       print ' ** bom = ', bom
        if not iok: return [],[]

        boms = [bom]
        cans = []; ms = []
        for bom in boms:
            # now restore charges for case, e.g., NN bond in C=N#N
            bom_U = accommodate_chgs(chgs, bom)

            t1 = time.time()
            blk = write_ctab(zs, chgs, bom_U, self.coords, sdf=None)
            m = obconv(blk)
#           t2 = time.time()
#           print '                |_ dt = ', t2-t1
            can_i = pb.Molecule(m).write('can').split('\t')[0]
#           t3 = time.time()
#           print '                |_ dt = ', t3-t2
#           print '             __ can = ', can_i
            if can_i not in cans:
                cans.append(can_i)
                ms.append(m)
        #if 'CC(C)C' in cans: print ' Alert!!!'
        return cans, ms

def get_coords(m):
    coords = [] # np.array([ ai.coords for ai in pb.Molecule(m).atoms ])
    na = m.NumAtoms()
    for i in range(na):
        ai = m.GetAtomById(i)
        coords.append( [ ai.GetX(), ai.GetY(), ai.GetZ() ] )
    return np.array(coords)

def get_bom(m):
    """
    get connectivity table
    """
    na = m.NumAtoms()
    bom = np.zeros((na,na), np.int)
    for i in range(na):
        ai = m.GetAtomById(i)
        for bond in ob.OBAtomBondIter(ai):
            ia1 = bond.GetBeginAtomIdx()-1; ia2 = bond.GetEndAtomIdx()-1
            bo = bond.GetBO()
            bom[ia1,ia2] = bo; bom[ia2,ia1] = bo
    return bom

def clone(m):
    m2 = pb.Molecule(m).clone
    return m2.OBMol

def check_hydrogens(m):
    mu = pb.Molecule(m).clone # a copy
    mu.addh()
    m2 = mu.OBMol
    return m.NumAtoms() == m2.NumAtoms()

def obconv(s,fmt='sdf'):
    """ convert string(s) to molecule given a format
    e.g, 'CCO','smi'
         or sdf_file_content,'sdf' """
    conv = ob.OBConversion()
    m = ob.OBMol()
    #assert type(s) is str
    conv.SetInFormat(fmt)
    conv.ReadString(m,s)
    return m


class mol(object):

    def __init__(self, m0):
        na = m0.NumAtoms()
        m1 = clone(m0); m1.DeleteHydrogens()
        self.m0 = m0
        #print 'self.m = ', m1
        self.m = m1
        chgs = []; zs = []
        for i in range(na):
            ai = m0.GetAtomById(i)
            zi = ai.GetAtomicNum(); zs.append( zi )
            chgi = ai.GetFormalCharge(); chgs.append( chgi )

        self.zs = np.array(zs)
        self.bom = get_bom(m0)
        self.nheav =  (self.zs > 1).sum()
        idxh = zs.index( 1 )
        self.ias = np.arange( len(self.zs) )
        self.ias_heav = self.ias[ self.zs > 1 ]
        try:
            self.coords = get_coords(m0)
        except:
            self.coords = np.zeros((na,3))

        self.chgs = np.array(chgs, np.int)
        if np.any(self.zs[idxh+1:] != 1):
            # not all H apprear appear at the end, u have to sort it
            #print ' ** molecule sorted'
            self.sort()

        self.vs = self.bom.sum(axis=0)
        if np.any(self.chgs != 0):
            #print ' ** update bom due to charges'
            self.update()
#       print ' -- chgs = ', self.chgs

        bom_heav = self.bom[ self.ias_heav, : ][ :, self.ias_heav ]
#       print 'bom_heav = ', bom_heav
        self.vs_heav = bom_heav.sum(axis=0)
        self.cns_heav = ( bom_heav > 0 ).sum(axis=0)
        # get formal charges
        self.cns = ( self.bom > 0).sum(axis=0)
        self.nhs = self.vs[:self.nheav] - self.vs_heav #- self.chgs[:self.nheav]
        self.dvs = self.vs_heav - self.cns_heav

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
        blk = write_ctab(zsU, chgs_U, g2, coords_U, sdf=None)
        m0 = obconv(blk)
        m1 = clone(m0)
#       print ' *** ', Chem.MolToSmiles(m1)
        m1.DeleteHydrogens()
        self.m0 = m0; self.m = m1

    def update(self):
        """update bom based on `chgs
        e.g., C=N#N, bond orders = [2,3],
        Considering that `chgs = [0,+1,-1],
        bond orders has to be changed to [2,2]"""
        bom2 = copy.copy(self.bom)
        vs2 = self.vs
        ias1 = self.ias[self.chgs == 1]
        for i in ias1:
            iasc = self.ias[ np.logical_and(self.chgs==-1, self.bom[i]>0) ]
            nac = len(iasc)
            if nac > 0:
                #assert nac == 1
                j = iasc[0]
                bij = self.bom[i,j] + 1
                bom2[i,j] = bij
                bom2[j,i] = bij
                vs2[i] = vs2[i]+1; vs2[j] = vs2[j]+1
        self.bom = bom2
        self.vs = vs2

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
        nb = self.m.NumBonds()
        for ib in range(nb):
            bi = self.m.GetBondById(ib)
            i, j = bi.GetBeginAtomIdx()-1, bi.GetEndAtomIdx()-1
            if self.zs[i] > 1 and self.zs[j] > 1:
                ib_heav = bi.GetIdx()
                b2a.append( [i,j] )
        #assert len(b2a) == ib_heav+1, '#ERROR: not all H apprear at the end?'
        b2a = np.array(b2a, np.int)

        a2b = -np.ones((self.nheav, 6), np.int) # -1 means no corresponding bond
        for ia in self.ias_heav:
            ai = self.m.GetAtomById(ia)
            icnt = 0
            for bi in ob.OBAtomBondIter(ai):
                ib = bi.GetId()
                if ib <= ib_heav: #np.all( self.zs[b2a[ib]] > 1 ):
                    a2b[ia, icnt] = ib
                    icnt += 1
        return a2b, b2a


class amon(object):

    """
    use RDKit only
    """

    def __init__(self, s, k, k2=None, wg=False, ligand=None, \
                 fixGeom=False, ikeepRing=True):
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

        if os.path.exists(s):
            m0 = obconv(s,s[-3:])
            assert check_hydrogens(m0), '#ERROR: some hydrogens are missing'
            coords0 = get_coords(m0)
            #m1 = copy.deepcopy(m0); m = Chem.RemoveHs(m1)
        else:
            #print ' ** assume SMILES string'
            m = obconv(s,'smi')
            m0 = obconv(s,'smi')
            m0.AddHydrogens()

        self.m0 = m0
        self.m = m

        self.objQ = mol(m0)


    def update(self, las, lbs, sg):
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
        zs1 = np.array( list(self.objQ.zs[las]) + [1,]*nh )
        chgs1 = np.array( list(self.objQ.chgs[las]) + [0,]*nh )
        tvs1 = np.array( list(self.objQ.vs[las]) + [1,]*nh )
        vars1 = vars(zs1, chgs1, tvs1, sg_U, coords_U)
        self.vars = vars1


    def get_amons(self):
        """
        tell if a given frag is a valid amon
        """

        objQ = self.objQ

        amons = []
        smiles = []

        # get amon-2 to amon-k
        g0 = ( objQ.bom > 0 ).astype(np.int)

        amons = []
        cans = []; ms = []
        a2b, b2a = objQ.get_ab()
        bs = [ set(jas) for jas in b2a ]
        for seed in generate_subgraphs(b2a, a2b, self.k):
            # lasi (lbsi) -- the i-th list of atoms (bonds)
            lasi, lbsi = list(seed.atoms), list(seed.bonds)
            #can = Chem.MolFragmentToSmiles(objQ.m, atomsToUse=lasi, kekuleSmiles=False, \
            #                               bondsToUse=lbsi, canonical=True)
            iprt = False
            bs = []
            for ibx in lbsi:
                bs.append( set(b2a[ibx]) )
                if iprt:
                    print '  -- ibx, ias2 = ', ibx, tuple(b2a[ibx])

            na = len(lasi)
            if na == 1:
                ia = lasi[0]; zi = objQ.zs[ ia ]
                iok1 = (zi in [9, 17, 35, 53])
                iok2 = ( np.any(objQ.bom[ia] == 2) ) # -S(=O)-, -P(=O)(O)- and -S(=O)(=O)-
                if np.any([iok1, iok2]):
                    continue
                can = dic_smiles[ zi ]
                if can not in cans:
                    cans.append( can )
                #    if wg:
                #        if not self.fixGeom:
                #            ms.append( ms0[can] )
                #        else:
                #            raise '#ERROR: not implemented yet'
                #else:
                #    if wg and self.fixGeom:
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
            if ir and ztot%2 == 0 and chg0 == 0:
                # ztot%2 == 1 implies a radical, not a valid amon for neutral query
                # this requirement kills a lot of fragments
                # e.g., CH3[N+](=O)[O-] --> CH3[N+](=O)H & CH3[N+](H)[O-] are not valid
                #       CH3C(=O)O (CCC#N) --> CH3C(H)O (CCC(H)) won't survive either
                #   while for C=C[N+](=O)[O-], with ztot%2 == 0, [CH2][N+](=O) may survive,
                #       by imposing chg0 = 0 solve the problem!

                tvsi0 = objQ.vs[lasi] # for N in '-[N+](=O)[O-]', tvi=4 (rdkit)
                bom0_heav = objQ.bom[lasi,:][:,lasi]
                dbnsi = (bom0_heav==2).sum(axis=0) #np.array([ (bom0_heav[i]==2).sum() for i in range(na) ], np.int)
                zsi = zs_sg
                ias = np.arange(na)
                ## 0) check if envs like '>S=O', '-S(=O)(=O)-', '-P(=O)<',
                ## '-[N+](=O)[O-]' (it's already converted to '-N(=O)(=O)', so `ndb=2)
                ## ( however, '-Cl(=O)(=O)(=O)' cannot be
                ## recognized by rdkit )
                ## are retained if they are part of the query molecule
                tvs1 = [4,6,5,5]
                dbns1 = [1,2,1,2] # number of double bonds
                zs1 = [16,16,15,7]
                istop = False
                for j,tvj in enumerate(tvs1):
                    filt = np.logical_and(tvsi0 == tvj, zsi == zs1[j])
                    jas = ias[filt].astype(np.int)
                    if len(jas) > 0:
                        dbnsj = dbnsi[jas]
                        if np.any(dbnsj != dbns1[j]):
                            istop = True; break
#                       print 'tvj, zs1[j], dbnsj, dbns1[j] = ', tvj, zs1[j], dbnsj, dbns1[j]
                if istop: continue

                self.update(lasi, lbsi, sg_heav)
                vr = self.vars
                cmg = MG( vr.zs, vr.chgs, vr.tvs, vr.g, vr.coords )
                cans_i, ms_i = cmg.update_bom(debug=True)
#               print cans_i
                for can_i in cans_i:
                    if can_i not in cans:
                        cans.append( can_i )
        return cans



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




def write_ctab(zs, chgs, bom, coords=None, sdf=None):
    """
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
    """

    na = len(zs)
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
    for i in range(na):
        px, py, pz = coords[i]
        zi = zs[i]
        ctab += fmt%(px, py, pz, ci.chemical_symbols[zi])

    for i in range(na):
        for j in range(i+1, na):
            boij = bom[i,j]
            if boij > 0:
                ctab += '%3d%3d%3d%3d\n'%(i+1,j+1,boij,0)

    ias = np.arange(na) + 1
    iasc = ias[ np.array(chgs) != 0 ]
    nac = iasc.shape[0]
    if nac > 0:
        ctab += 'M  CHG%3d'%nac
        for iac in iasc:
            ctab += ' %3d %3d'%(iac, chgs[iac-1])
        ctab += '\n'

    ctab += 'M  END'
    if sdf != None:
        with open(sdf,'w') as f: f.write(ctab)
        return
    else:
        return ctab


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


def is_edge_within_range(edge, edges, dsg, option='adjacent'):

    i1,i2 = edge
    iok = False
    if len(edges) == 0:
        iok = True
    else:
        jit = is_joint_vec(edge, edges) # this is a prerequisite
        #print ' -- jit = ', jit
        if not jit:
            for edge_i in edges:
                j1,j2 = edge_i
                ds = []
                for p,q in [ [i1,j1], [i2,j2], [i1,j2], [i2,j1] ]:
                    ds.append( dsg[p,q] )
                #print ' -- ds = ', ds
                iok1 = (option == 'adjacent') and np.any( np.array(ds) == 1 )
                iok2 = (option == 'distant') and np.all( np.array(ds) >=2 )
                if iok1 or iok2:
                    iok = True; break
    return iok

def is_adjacent_edge(edge, edges, dsg):
    return is_edge_within_range(edge, edges, dsg, option='adjacent')

def is_distant_edge(edge, edges, dsg):
    return is_edge_within_range(edge, edges, dsg, option='distant')

def get_joint(edge1, edge2):
    return np.intersect1d(edge1,edge2)

def is_joint(edge1, edge2):
    return np.intersect1d(edge1,edge2).shape[0] > 0

def is_joint_vec(edge, edges):
    return np.any([ is_joint(edge,edge_i) for edge_i in edges])

def comba(cs):
    csu = []
    for c in cs: csu += c
    return csu

def sort_edges(compl, edge_i, dsg):
    """
    sort the edges by the relative distance to nodes in `edge_i
    """
    ds = []
    for ei in compl:
        i,j = ei; k,l = edge_i
        dsi = [ dsg[r,s] for r,s in [ [i,k],[j,l],[i,l],[j,k] ] ]
        ds.append( np.min(dsi) )
    seq = np.argsort( ds )
    return [ compl[q] for q in seq ]


def edges_standalone_updated(g, irad=False):
    """
    e.g., given smiles 'cccccc', edges_1 = [(0,1),(1,2),(2,3),(3,4),(4,5)],
    possible outputs: [[(0,1),(2,3),(4,5)], i.e., 'c=cc=cc=c'

    In order to get SMILES string like 'cc=cc=cc', u have to modify this function

    Notes:
    ===================
    irad  -- If set to True, all possible combinations of double bonds
             will be output. For the above example, output will be
             'c=cc=cc=c' and 'cc=cc=cc';
             If set to False, only the formal SMILES results, i.e., the
             number of residual nodes that are not involved in double bonds
             is zero.
    """
    n = g.shape[0]
    assert n >= 3
    edges0 = [list(edge) for edge in np.array(list(np.where(np.triu(g)>0))).T];
    Gnx = nx.Graph(g)
    dsg = np.zeros((n,n), np.int)
    for i in range(n):
        for j in range(i+1,n):
            dsg[i,j] = dsg[j,i] = nx.shortest_path_length(Gnx, i,j)
    nodes0 = set( comba(edges0) )
    ess = []; nrss = []
    for edge_i in edges0:
        compl = cim.get_compl(edges0, [edge_i])
        compl_u = sort_edges(compl, edge_i, dsg)
        edges = copy.deepcopy( compl_u )
        edges_u = [edge_i, ]
        while True:
            nodes = list(set( comba(edges) ))
            nnode = len(nodes)
            if nnode == 0 or np.all(g[nodes,:][:,nodes] == 0):
                break
            else:
                edge = edges[0]
                edges_copy = copy.deepcopy( edges )
                if is_adjacent_edge(edge, edges_u, dsg):
                    edges_u.append(edge)
                    for edge_k in edges_copy[1:]:
                        if set(edge_k).intersection(set(comba(edges_u)))!=set():
                            edges.remove(edge_k)
                else:
                    edges.remove(edge)
        edges_u.sort()
        if edges_u not in ess:
            nrss_i = list(nodes0.difference(set(comba(edges_u))))
            nnr = len(nrss_i)
            if nnr > 0:
                if irad:
                    ess.append( edges_u )
                    nrss.append( nrss_i )
            else:
                ess.append( edges_u )
                nrss.append( nrss_i )
    return ess, nrss # `nrss: nodes residual, i.e., excluded in found standalone edges


def find_double_bonds(g, once=False, irad=False, debug=False):
    """
    for aromatic or conjugate system, find all the possible combs
    that saturate the valences of all atoms
    """
    ess, nrss = edges_standalone_updated(g, irad=irad)
    if once:
        edges = [ess[0], ]
    else:
        edges = ess
    return edges



## test!

if __name__ == "__main__":
    import sys, gzip

    if len(sys.argv) == 1:
        s = "NC(Cc1ccccc1)C(=O)N"
        k = 7
    elif len(sys.argv) == 2:
        s = sys.argv[1]
        k = 7
    elif len(sys.argv) == 3:
        s = sys.argv[2]
        k = int(sys.argv[1])
    else:
        raise SystemExit("""Usage: dfa_subgraph_enumeration.py <smiles> [<k>]
List all subgraphs of the given SMILES up to size k atoms (default k=5)
""")

    if not os.path.exists(s):
        obj = amon(s, k)
        cans = obj.get_amons()
        for can in cans:
            print can
    else:
        cs = []; maps = []; idxc = 0
        nmaxc = 120
        if s[-2:] in ['gz',]:
            t = gzip.open(s)
            #nmaxc = 0
            while 1:
                si = t.readline().strip(); icnt = 0
                if si == '': break
                obj = amon(si, k)
                cansi = obj.get_amons(); nci = len(cansi)
                map_i = []
                for ci in cansi:
                    if ci not in cs:
                        cs.append(ci); map_i += [idxc]; idxc += 1
                    else:
                        idxc = cs.index(ci)
                        map_i += [idxc]
                map_i += [-1,]*(nmaxc-nci)
                maps.append( map_i )
                #nmaxc = max(nmaxc, nci)

            dd.io.save(s[:-7]+'.h5', {'cans':np.array(cs), 'maps':np.array(maps)})

