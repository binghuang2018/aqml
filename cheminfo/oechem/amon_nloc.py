#!/usr/bin/env python

from itertools import chain, product
from functools import reduce
import os, sys, re, copy, ase
import ase.data as ad
from openeye.oechem import *
import numpy as np
import networkx.algorithms.isomorphism  as iso
import networkx as nx
import cheminfo.oechem.oechem as coo
from cheminfo.molecule.subgraph import *
import cheminfo.rdkit.core as cir
from cheminfo.rw.ctab import write_ctab
from rdkit import Chem
import scipy.spatial.distance as ssd
import cheminfo.openbabel.obabel as cib
import multiprocessing
import cheminfo.core as cic
import cheminfo.math as cim
import deepdish as dd
import itertools as itl
import tempfile as tpf
import cheminfo.graph as cg
from cheminfo.molecule.elements import Elements
import cml.famoneib as fa


global dsHX
#                 1.10
dsHX_normal = {5:1.20, 6:1.10, \
          7:1.00, 8:0.98, 9:0.92, 14:1.48, 15:1.42, 16:1.34, 17:1.27}
dsHX_halved = {}
for key in list(dsHX_normal.keys()): dsHX_halved[key] = dsHX_normal[key]/2.0

cnsr = {4:2, 5:3, 6:4, 7:3, 8:2, 9:1, 13:3, 14:4, 15:3, 16:2, 17:1, \
        32:4, 33:3, 34:2, 35:1,  50:4,51:3,52:2,53:1}
tvsr1 = {4:2, 5:3, 6:4, 7:3, 8:2, 9:1, 13:3, 14:4, 15:3, 16:2, 17:1, \
        32:4, 33:3, 34:2, 35:1,  50:4,51:3,52:2,53:1}

rcs0 = Elements().rcs

T,F = True,False


class ConnMat(object):

    def __init__(self, g):
        ns = g.shape
        assert len(ns) == 2 and ns[0]==ns[1]
        self.na = ns[0]
        self.g = g
        gnx = nx.from_numpy_matrix(g)
        self.gnx = gnx
        self.is_connected = nx.is_connected(gnx)

    def get_pls(self):
        """ calc shortest path lengths """
        pls = -1 * np.ones((self.na, self.na))
        np.fill_diagonal(pls,[0]*self.na)
        for i in range(self.na):
            for j in range(i+1,self.na):
                if nx.has_path(self.gnx,i,j):
                    pls[i,j]=pls[j,i]=nx.shortest_path_length(self.gnx,i,j)
        return pls

    @property
    def pls(self):
        if not hasattr(self, '_pls'):
            self._pls = self.get_pls()
        return self._pls

    @property
    def has_ring(self):
        try:
            ncyc = len( nx.algorithms.cycles.find_cycle(self.gnx) )
        except:
            ncyc = 0
        return ncyc > 0


class RawM(object):
    """
    molecule object with only `zs & `coords
    """
    def __init__(self, zs, coords):
        self.zs = zs
        self.coords = coords

    def generate_coulomb_matrix(self,inorm=False,wz=False,rpower=1.0):
        """ Coulomb matrix
        You may consider using `cml1 instead of `cm """
        na = len(self.zs)
        mat = np.zeros((na,na))
        ds = ssd.squareform( ssd.pdist(self.coords) )
        np.fill_diagonal(ds, 1.0)
        if wz:
            X, Y = np.meshgrid(self.zs, self.zs)
            diag = -1. * np.array(self.zs)**2.4
        else:
            X, Y = [1., 1.]
            diag = np.zeros(na)
        mat = X*Y/ds**rpower
        np.fill_diagonal(mat, diag)
        L1s = np.linalg.norm(mat, ord=1, axis=0)
        ias = np.argsort(L1s)
        self.cm = L1s[ias] if inorm else mat[ias,:][:,ias].ravel()


class Parameters(object):

    def __init__(self, i3d, fixGeom, k, k2, ivdw, \
                 forcefield, thresh, gopt, M, iters, reduce_namons, nproc):
        self.i3d = i3d
        self.fixGeom = fixGeom
        self.ff = forcefield
        self.k = k
        self.k2 = k2
        self.kmax = max(k,k2)
        self.ivdw = ivdw
        #self.threshDE = threshDE
        self.thresh = thresh
        self.gopt = gopt # geometry optimizer
        self.iters = iters
        self.reduce_namons = reduce_namons
        self.M = M
        self.nproc = nproc


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

def is_overcrowd(zs, bom, coords):
    ds = ssd.squareform( ssd.pdist(coords) )
    non_bonds = np.where( np.logical_and(bom==0, ds!=0.) )
    rcs = rcs0[ zs ]
    # Note that the following line of code cannot
    # correctly tell if a mol is too crowded. E.g.,
    # some high-strain mol
    #dsmin = rcs[..., np.newaxis] + [rcs] + 0.45
    dsmin = 0.75 * (rcs[..., np.newaxis] + [rcs])
    return np.any(ds[non_bonds]<dsmin[non_bonds])


class Sets(object):

    def __init__(self, param, debug=False):

        self.cans = [] #cans
        self.ms = [] #ms
        self.rmols = [] #rmols
        self.es = [] #es
        self.nsheav = [] #nsheav
        self.ms0 = [] #ms0
        self.maps = [] #maps
        self.cms = [] # coulomb matrix
        self.param = param
        self.debug = debug

    def update(self, ir, can, Mli):
        """
        update `Sets

        var's
        ==============
        Mli  -- Molecule info represented as a list
                i.e., [zs, coords, bom, charges]
        """
        zs, coords, bom, charges = Mli
        #ds = ssd.pdist(coords)
        #if np.any(ds<=0.5):
        #    print('--zs=',zs)
        #    print('--coords=',coords)
        #    raise Exception('some d_ij very samll!!')
        assert self.param.i3d
        ################# for debugging
        if self.debug:
            write_ctab(zs, charges, bom, coords, sdf='raw.sdf')
        #################
        #if not self.fixGeom:
        m0, m, ei, coords = self.optg(Mli)
        rmol = RawM(zs, coords)
        if self.param.M in ['cml1']:
            rmol.generate_coulomb_matrix(inorm=True,wz=False,rpower=1)
        nheav = (zs > 1).sum()
        self.ncan = len(self.cans)
        if can in self.cans:
            ican = self.cans.index( can )
            # for molecule with .LE. 3 heavy atoms, no conformers
            if (not self.param.fixGeom) and (not self.param.ivdw) and nheav <= 2:
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
                ms_i = self.ms[ ican ] # stores the updated geom
                rmols_i = self.rmols[ ican ] # Mols of the same graph (i.e.,conformers)
                                             # with Representation (e.g., cml1) attached
                ms0_i = self.ms0[ ican ] # stores the original geom
                nci = len(ms_i); _ics_i = np.arange(nci)
                es_i = self.es[ ican ]

                inew = True
                if self.param.M in ['cml1']: # use difference of energy as citeria
                    xs = np.array([ rmol.cm, ] )
                    ys = np.array([ ma.cm for ma in self.rmols[ican] ])
                    #print(' -- ', xs.shape, ys.shape, can)
                    _drps = ssd.cdist(xs, ys, 'cityblock')
                    #print ' can, _drps = ', can, _drps
                    drps = _drps[0]
                    filt = (drps <= self.param.thresh)
                    if np.any(filt):
                        inew = False
                        ics_i = _ics_i[filt]
                elif self.param.M in ['e','energy']: # not safe, never use this criteria
                    dEs = np.abs( np.array(es_i) - ei )
                    if np.any( dEs <= self.param.thresh ): inew = False
                else:
                    print('#ERROR: not supported `M')
                    raise

                if inew:
                    self.ms[ ican ] = ms_i + [m, ]
                    self.rmols[ ican ] = rmols_i + [ rmol, ]
                    self.ms0[ ican ] = ms0_i + [m0, ]
                    self.es[ ican ] = es_i + [ei, ]
                    self.maps.append( [ir, ican, nci] )
                else:
                    #icount = 0
                    for ic in ics_i:
                        entry = [ir,ican,ic]
                        if entry not in self.maps:
                            #print '#found entry'
                            #icount += 1
                            self.maps.append(entry)
                    #if icount > 1: print '#found multiple entries'
        else:
            #m0, m, ei, coords = self.optg(Mli)
            self.maps.append( [ir, self.ncan, 0] )
            self.cans.append( can )
            self.nsheav.append( nheav )
            self.ms.append( [m, ] )
            self.rmols.append( [rmol, ] )
            self.ms0.append( [m0, ] )
            self.es.append( [ei, ] )
            self.ncan += 1

    def update2(self, ir, can, nheav):
        """
        update mol set if we need SMILES only
        """
        self.ncan = len(self.cans)
        if can not in self.cans:
            #print '++', can #, '\n\n'
            self.maps.append( [ir, self.ncan, 0] )
            self.cans.append( can )
            self.nsheav.append( nheav )
            self.ncan += 1
        else:
            ican = self.cans.index( can )
            entry = [ir, ican, 0]
            if entry not in self.maps:
                self.maps.append( entry )
        #print(' -- maps = ', self.maps)

    def optg(self,Mli):
        """
        post process molecular fragement retrieved
        from parent molecule by RDKit
        """
        #import io2.mopac as im
        import tempfile as tpf
        zs, coords, bom, chgs = Mli
        ctab = write_ctab(zs, chgs, bom, coords)
        # get RDKit Mol first
        m0 = Chem.MolFromMolBlock( ctab, removeHs=False ) # plz keep H's
        m1 = Chem.MolFromMolBlock( ctab, removeHs=False ) # plz keep H's
        rd = cir.RDMol( m1, forcefield=self.param.ff )
        if self.param.i3d:
          if not self.param.fixGeom:
            #if not cg.is_connected_graph(bom):
            #    self.param.gopt = 'PM6-D3H4' ##
            #    print(' ** info: param.gopt switched to PM6-D3H4')
# the default case, use openbabel to do constrained optimization
            if self.param.gopt.lower() in ['obff']:
                ob1 = cib.Mol( ctab, fmt='sdf' )
                ob1.optg_c(iconstraint=3, ff="MMFF94", \
                           optimizer='cg', steps=[30,90], ic=True)
                rd = cir.RDMol( ob1.to_RDKit(), forcefield=self.param.ff )
# if u prefer to use rdkit to do FF optimization
# This seems to be a bad choice as it's likely that
# some bugs exist in RDKit code regarding FF opt
# with dihedral constrains in my system. Test it
# yourself for your own system.
            elif self.param.gopt.lower() in ['rkff']:
                if self.param.reduce_namons:
                    #print('now do constrained optg')
                    rd.optg_c(2.0,300) #300) #dev=2.0,maxIters=self.param.iters[0]) #200) #20)
                    rd.optg(maxIters=900) #900)
                    #print('now do a further optg wo constraint')
                else:
                    rd.optg_c(2.0,60) #1200)
            elif self.param.gopt.lower() in ['xtb']:
                rd.optg_c(2.0,60)
                rd.optg_xtb(acc='normal', nproc=self.param.nproc)
# if u prefer to do a partial optimization using PM7 in MOPAC
# for those H atoms and their neighboring heavy atoms
            elif self.param.gopt.lower() in ['pm7','pm6','pm6-d3h4']: #do_pm6_disp:
                # in case it's a molecular complex
                rd.optg2(meth=self.param.gopt, iffopt=T)
            else:
                raise Exception('#error: unknow geometry optimizer')
        if hasattr(rd, 'energy'):
            e = rd.energy
        else:
            e = rd.get_energy()
        m = rd.m

        if is_overcrowd(rd.zs, rd.bom, rd.coords):
            fdt = './overcrowded' # Temporary folder
            if not os.path.exists(fdt): os.mkdir(fdt)
            tsdf = tpf.NamedTemporaryFile(dir=fdt).name + '.sdf'
            print(' -- ', tsdf)
            rd.write_sdf(tsdf)
            raise Exception('#ERROR: too crowded!!')
        return m0, m, e, rd.coords

    def _sort(self):
        """ sort Mlis """
        maps = np.array(self.maps)
        ncan = len(self.cans)
        seqs = np.arange(ncan)
        nsheav = np.array(self.nsheav)
        ncs = [ len(ms_i) for ms_i in self.ms ]
        cans = np.array(self.cans)
        nsheav_u = []
        ncs_u = []
        seqs_u = []
        cans_u = []
        ms_u = []; ms0_u = []

        # now sort the amons by the number of heavy atoms
        for i in range(1, self.param.kmax+1):
            seqs_i = seqs[ i == nsheav ]
            cans_i = cans[ seqs_i ]
            seqs_j = seqs_i[ np.argsort(cans_i) ]
            seqs_u += list( seqs_j )
            for j in seqs_j:
                cans_u.append( cans[j] )
                ms_j = self.ms[j]; ms0_j = self.ms0[j]
                ncj = len(ms_j)
                ncs_u.append( ncj )
                nsheav_u.append( nsheav[j] )
                ms_u.append( ms_j ); ms0_u.append( ms0_j )

        seqs_u = np.array(seqs_u)

        # now get the starting idxs of conformers for each amon
        ias2 = np.cumsum(ncs_u)
        ias1 = np.concatenate( ([0,],ias2[:-1]) )

        # now get the maximal num of amons one molecule can possess
        #print(' size of maps: ', maps.shape)
        irs = np.unique( maps[:,0] ) # sorted now
        nt = len(irs) # 1+maps[-1,0];
        namons = []
        for i in irs: # range(nt):
            namon = (maps[:,0] == i).sum()
            namons.append(namon)
        namon_max = max(namons)

        # `maps2 stores the amon idx for each target molecule
        # (Note: any conformer is an amon)
        maps2 = -1 * np.ones((nt, namon_max),dtype=int)
        for i,ir in enumerate(irs): #range(nt):
            filt_i = (maps[:,0] == ir)
            maps_i = maps[filt_i, 1:]
            jcnt = 0
            for j in range(namons[i]):
                jcan, jc = maps_i[j,:] # `jcan: the old idx of can
                jcan_u = seqs[ seqs_u == jcan ] # new idx of can
                maps2[i, jcnt] = ias1[jcan_u] + jc
                jcnt += 1
        self.ms = ms_u
        self.ms0 = ms0_u
        self.cans = cans_u
        self.nsheav = nsheav_u
        self.ncs = ncs_u
        self.maps2 = maps2

    def _sort2(self):
        """ sort Mlis for i3d = False"""
        maps = np.array(self.maps)
        ncan = len(self.cans)
        seqs = np.arange(ncan)
        nsheav = np.array(self.nsheav)
        cans = np.array(self.cans)

        nsheav_u = []
        seqs_u = []
        cans_u = []
        # now sort the amons by the number of heavy atoms
        for i in range(1, self.param.k2+1):
            seqs_i = seqs[ i == nsheav ]
            cans_i = cans[ seqs_i ]
            seqs_j = seqs_i[ np.argsort(cans_i) ]
            seqs_u += list( seqs_j )
            for j in seqs_j:
                cans_u.append( cans[j] )
                nsheav_u.append( nsheav[j] )

        seqs_u = np.array(seqs_u)

        # now get the maximal num of amons one molecule can possess
        irs = np.unique( maps[:,0] ) # sorted now
        nt = len(irs) # 1+maps[-1,0];
        namons = []
        for i in irs: #range(nt):
            namon = (maps[:,0] == i).sum()
            namons.append(namon)
        namon_max = max(namons)

        # `maps2 stores the amon idx for each target molecule
        # (Note: any conformer is an amon)
        maps2 = -np.ones((nt, namon_max)).astype(np.int)
        for i,ir in enumerate(irs): #range(nt):
            filt_i = (maps[:,0] == ir)
            maps_i = maps[filt_i, 1:]
            jcnt = 0
            for j in range(namons[i]):
                jcan, jc = maps_i[j,:] # `jcan: the old idx of can
                jcan_u = seqs[ seqs_u == jcan ] # new idx of can
                maps2[i, jcnt] = jcan_u
                jcnt += 1
        self.cans = cans_u
        self.nsheav = nsheav_u
        self.maps2 = maps2
        self.ncs = np.ones(ncan).astype(np.int)


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


class atom_db(object):
    def __init__(self, symbol):
        wd = 'data/atoms/'
        symbs = ['B','C','N','O','Si','P','S', 'F','Cl','Br']
        assert symb in symbs, '#ERROR: no such atomic data?'
        self.oem = coo.StringM( 'data/%s.sdf'%symb ).oem


def cmp(a,b):
    if a < b:
        return -1
    elif a > b:
        return 1
    else:
        return 0

class ParentMol(coo.StringM):

    def __init__(self, string, stereo=F, isotope=F, iat=None, i3d=T, \
                 ichg=F, irad=F, k=7, k2=7, opr='.le.', nocrowd=T, \
                 iextl=F, fixGeom=F, keepHalogen=F, ivdw=F, \
                 inmr=F, debug=F, iwarn=T, warning_shown=F):
        self.warning_shown=warning_shown
        self.iextl = iextl
        coo.StringM.__init__(self, string, stereo=stereo, isotope=isotope)
        self.iwarn = iwarn
        self.k = k
        self.k2 = k2
        self.fixGeom = fixGeom
        self.nocrowd = nocrowd
        self.iat = iat
        self.keepHalogen = keepHalogen
        self.debug = debug
        self.vsa = {'.le.': [-1,0], '.eq.': [0, ]}[opr] # values accepted
        self.i3d = i3d
        self.irad = irad
        self.ivdw = ivdw
        #if not inmr:
        #    #ncbs = []
        #    #if self.i3d and self.ivdw:
        #    #    ncbs = self.ncbs #perceive_non_covalent_bonds()
        #if ivdw:
        #    self.get_gvdw()
        #    self.ncbs = self._ncbs
        a2b, b2a = self.get_ab()
        bs = [ set(jas) for jas in b2a ]
        self.a2b = a2b
        self.b2a = b2a
        self.bs = bs

    @property
    def ncbs(self):
        return super(ParentMol, self).ncbs_heav #get_ncbs()

    def get_submol(self,nodes):
        """ useful for diagnose when genearting amons """
        na1 = len(nodes)
        bonds = []
        for _i in range(na1):
            for _j in range(_i+1,na1):
                i,j = nodes[_i],nodes[_j]
                if self.bom[i,j] > 0:
                    bonds.append( self.bs.index( set([i,j]) ) )
        return nodes,bonds

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
            csi = self.bom[i]
            degree_i = np.sum( np.logical_and( csi > 0, self.zs0 > 1 ) )
            degrees.append( degree_i )

        self.degrees = degrees

        if qa is None:
            qsa = ias_za
        else:
            qsa = [ias_za[qa], ]
        self.get_rigid_and_rddtible_nodes()

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
                    nbr = ias[self.g[ja] == 1][0]
                    #if self.zs0[nbr] in [7,8,9,15,16,17]:
                        # these electronegative atoms will induce
                        # electrostatic effects (long-ranged)
                    jas_hvy.append( nbr )
                else:
                    jas_hvy.append( ja )
            #print ' -- jas_hvy = ', jas_hvy
            # get neighbors of those heavy atoms
            sg = self.g[jas_hvy,:][:,jas_hvy]
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
            for keyi in list(mapping_i.keys()):
                val_i = mapping_i[keyi]; nodes_i.append( keyi )
                mapping_i_reverse[val_i] = keyi
            if self.debug: print(' --         nodes = ', nodes_i)
            dic_i = mf_i.GetCoords()
            coords_i = []
            for j in range(mf_i.NumAtoms()): coords_i.append( dic_i[j] )
            zsi = np.array([ aj.GetAtomicNum() for aj in mf_i.GetAtoms() ])
            dsi = ssd.squareform( ssd.pdist(coords_i) )

            nodes_new = self.get_nodes_bridge(zsi, mf_i, bom_i, dsi, mapping_i_reverse, nodes_i)
            if self.debug: print(' --     new nodes = ', nodes_new)
            jas_hvy = nodes_i + nodes_new
            sg = self.g[jas_hvy,:][:,jas_hvy]
            #istop = self.extend_heavy_nodes(jas_hvy, sg)
            self.extend_heavy_nodes(jas_hvy, sg)
            jas_u = self.jas_u
            if self.debug: print(' -- jas_u = ', jas_u, ' [updated]')
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


    def get_cutout(self, lasi, cutoff=8.0):
        """
        retrieve the union of local structure within a radius
        of `cutoff of atom in `lasi
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
            csi = self.bom[i,:]
            degree_i = np.sum( np.logical_and( csi > 0, self.zs0 > 1 ) )
            degrees.append( degree_i )
        self.degrees = degrees

        self.get_rigid_and_rddtible_nodes()

        msf = []
        self.sets = []
        boms = []
        mappings = []

        jas_u = set()
        icnt = 0
        for ia in lasi:
            filt = ( ds[ia] <= cutoff )
            jas = list( ias[filt] )

            # get heavy atoms
            jas_hvy = []
            for j,ja in enumerate(jas):
                zja = self.zs0[ja]
                if zja == 1:
                    nbr = ias[self.g[ja,:] == 1][0]
                    #if self.zs0[nbr] in [7,8,9,15,16,17]:
                        # these electronegative atoms will induce electrostatic effects (long-ranged)
                    jas_hvy.append( nbr )
                else:
                    jas_hvy.append( ja )
            #print ' -- jas_hvy = ', jas_hvy
            # get neighbors of those heavy atoms
            sg = self.g[jas_hvy,:][:,jas_hvy]
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
        for keyi in list(mapping_i.keys()):
            val_i = mapping_i[keyi]; nodes_i.append( keyi )
            mapping_i_reverse[val_i] = keyi
        if self.debug: print(' --         nodes = ', nodes_i)
        dic_i = mf_i.GetCoords()
        coords_i = []
        for j in range(mf_i.NumAtoms()): coords_i.append( dic_i[j] )
        zsi = np.array([ aj.GetAtomicNum() for aj in mf_i.GetAtoms() ])
        dsi = ssd.squareform( ssd.pdist(coords_i) )

        nodes_new = self.get_nodes_bridge(zsi, mf_i, bom_i, dsi, mapping_i_reverse, nodes_i)

        if self.debug: print(' --     new nodes = ', nodes_new)
        jas_hvy = list( set(nodes_i + nodes_new) )
        sg = self.g[jas_hvy,:][:,jas_hvy]
        #istop = self.extend_heavy_nodes(jas_hvy, sg)
        #if 1339 in jas_hvy:
        #    idx = jas_hvy.index(1339)
        #    iasU = np.arange(sg.shape[0])
        #    print iasU[ sg[idx,:] > 0 ]

        self.extend_heavy_nodes(jas_hvy, sg)
        jas_u = self.jas_u

        if self.debug: print(' -- jas_u = ', jas_u, ' [updated]')
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
                if ja in self.rddtible_nodes: # saturated node
                    set_i.update( [ja,] )
                else:
                    #if ja == 36: print ' Gotha 3 !'
                    for nodes_i in self.rigid_nodes:
                        if ja in nodes_i:
                            set_i.update( nodes_i )
                            #print ' -- ja, nodes_i = ', ja, nodes_i
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
                    mf.SetCoords(ak2, self.coords[ka])
                    #print ' - ka, ', self.coords[ka]
                    bonds.append( [icnt,j,1] ); icnt += 1
                else:
                    # __don't__ add atom `ak to `mf as `ak may be added to `mf later
                    # in the for loop ``for ja in nodes_to_add`` later!!
                    if ka not in nodes_to_add:
                        # add H
                        v1 = self.coords[ka] - self.coords[ja]
                        dHX = dsHX_normal[zj];
                        coords_k = self.coords[ja] + dHX*v1/np.linalg.norm(v1)
                        ak2 = mf.NewAtom( 1 )
                        mf.SetCoords(ak2, coords_k)
                        b2 = mf.NewBond( aj2, ak2, 1)
                        bonds.append( [icnt,j,1] ); icnt += 1

        nadd = len(nodes_to_add)
        #print ' __ nodes_to_add = ', nodes_to_add
        for j in range(nadd):
            for k in range(j+1,nadd):
                #print ' j,k = ', j,k
                ja = nodes_to_add[j]; ka = nodes_to_add[k]
                ja2 = mapping[ja]; ka2 = mapping[ka]
                bo = self.bom[ja,ka]
                if bo > 0:
                    aj2 = atoms_sg[ja2]; ak2 = atoms_sg[ka2]
                    bonds.append( [j,k,bo] )
                    b2 = mf.NewBond( aj2, ak2, bo )
                    #print ' (ja,ka,bo) = (%d,%d,%d), '%(ja, ka, bo), \
                    #         '(ja2,ka2,bo) = (%d,%d,%d)'%(ja2,ka2,bo)
        assert mf.NumAtoms() == icnt
        bom_u = np.zeros((icnt,icnt), np.int)
        for bond_i in bonds:
            bgn,end,bo_i = bond_i
            bom_u[bgn,end] = bom_u[end,bgn] = bo_i

        return bom_u, mapping, mf


    def get_rigid_and_rddtible_nodes(self):
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
        (2) rddtible nodes
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
        ijs = itl.combinations( list(range(n)), 2 )
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
        sg = self.g[iasc,:][:,iasc]
        na_sg = len(iasc)
        dic_sg = dict(list(zip(list(range(na_sg)), iasc)))
        for sgi in cg.find_cliques( sg ):
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

        # now find rddtible nodes, i.e., saturated nodes with breakable connected bonds
        rddtible_nodes = list( set(nodes_hvy)^set(rigid_nodes_ravel) )

        obsolete = """
        rddtible_nodes = set()
        for pat_i in ['[CX4,SiX4,PX3,F,Cl,Br,I]', ]:
            ss = OESubSearch(pat_i)
            iok = OEPrepareSearch(m, ss)
            for match in ss.Match(m):
                for ma in match.GetAtoms():
                    rddtible_nodes.update( [ma.target.GetIdx()] )

        for pat_i in [ '[NX3;!a]', '[OX2;!a]', '[SX2;!a]', ]:
            ss = OESubSearch(pat_i)
            iok = OEPrepareSearch(m, ss)
            for match in ss.Match(m):
                for ma in match.GetAtoms():
                    ia = ma.target.GetIdx()
                    if ia not in rigid_nodes_ravel:
                        rddtible_nodes.update( [ia] )"""

        self.rddtible_nodes = list(rddtible_nodes)

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
        lasi = np.arange( len(self.ars0) )
        ias1 = lasi[filt]
        _g = self.bom[filt, :][:, filt]
        iok = False
        envs = set([])
        for csi in cg.find_cliques(_g):
            if len(csi) > 2:
                gci = _g[csi, :][:, csi]

                # set `irad to False to ensure that no atom is unsaturated in valence
                ess, nrss = edges_standalone_updated(gci, irad=F)
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
                            print('#ERROR: more than 2 different sets in nodes_residual??')
                            raise
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
                #    envs.update( set( comba( cim.get_compl_u(ess[0],comms) ) ) )
                    envs.update( set( comba( ess[0]) ) )
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
            ots = is_subg(self.oem, q, iop = 1)
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

    def generate_amons(self,submol=None):
        """
        generate all canonial SMARTS of the fragments (up to size `k)
        of any given molecule
        """
        #if self.irad:
        #    raise Exception(' Radical encountered! So far it is not supported!')

        debug = self.debug
        a2b, b2a = self.a2b, self.b2a
        bs = self.bs
        cans_u = []
        nrads = 0

        for seed in generate_subgraphs(b2a, a2b, k=self.k, submol=submol):
            # lasi (lbsi) -- the i-th list of atoms (bonds)
            lasi, lbsi = list(seed.atoms), list(seed.bonds)
            _lasi = np.array(lasi).astype(np.int)
            #iprt = False
            bs = []
            for ibx in lbsi:
                bs.append( set(b2a[ibx]) )
            #print ' atoms, bonds = ', lasi, [list(bsi) for bsi in bs]
            nheav = len(lasi)
            iaq2iaa = dict(list(zip(lasi,list(range(nheav)))))

            # constraint on number of heavy atoms
            if cmp(nheav, self.k) not in self.vsa:
                continue

            zs = self.zs[lasi]
            assert np.all(zs>1), '#ERROR: H showed up?'

            if self.iat != None: # assume input idx starts from 1
                _ja = self.iat - 1
                _zj = self.zs[_ja]
                if _zj > 1:
                    jok = ( _ja in lasi ) # self.iat is an idx starting from 1
                # otherwise, check if this H atom is connected to any heavy atom in `lasi
                else:
                    jok = False
                    for _ia in lasi:
                        if self.bom[_ia, _ja] > 0:
                            jok = True
                            break
                if not jok:
                    continue

            chgs = self.chgs[lasi] # won't be changed throughout
            zs = self.zs[lasi]
            cns = self.cns[lasi]
            cnshv = self.cnshv[lasi]
            tvs = self.tvs[lasi]
            # get the coords_q and cns before matching SMARTS to target
            coords = self.coords[lasi]

            iconjs = self.iconjs[lasi]

            #print('##1 nheav=',nheav)

            ifd_extl= F

            if nheav == 1:
                # keep HX, X=F,Cl,Br,I??
                zi = zs[0]
                symb1 = cic.chemical_symbols[zi]
                if not self.keepHalogen:
                    if symb1 in ['F','Cl','Br','I',]:
                        continue
                cni = cns[0]
                chgi = chgs[0]
                tvi = tvs[0]
                #if (tvsr1[zi] != tvi) or (cnsr[zi] != cni) # PH5 is forbidden
                if tvi!=cni or (chgi!=0): # radical. Note: now PH5 is allowed!
                    if not self.irad: # irad: accept radical or not
                        #print ' ia, zi, cni, chgi = ', lasi[0],zi,cni,chgi
                        continue
                boms = [ np.zeros((1,1),dtype=int) ]
            else:
                sg = np.zeros((nheav,nheav), np.int)
                for bij in bs:
                    ia,ja = list(bij)
                    i,j = lasi.index(ia),lasi.index(ja)
                    sg[i,j] = sg[j,i] = 1
                ## check isomorphism
                _sg = self.g[lasi,:][:,lasi]
                if not np.all(_sg==sg):
                    #print '##iso not satisfied'
                    continue

                # initialize `_bom to `sg
                _bom = sg.copy()

                cnsi = sg.sum(axis=0)
                nsh = cns - sg.sum(axis=0) # numbers (of) hydrogens (in the fragment)
                vsi = _bom.sum(axis=0)
                dvsi = tvs - (vsi+nsh)

                # radical check
                #irad = F
                if (sum(zs)+sum(nsh))%2 != 0 or (sum(chgs) != 0): #%2 !=0):
                    #irad = T # n_elec is odd -> a radical
                    nrads += 1 #print(' #Found 1 radical!' )
                    continue

# now filter out amons as described in the description section within class ParentMols()
                i_further_assessment = F
                if (self.iextl) and (nheav > 7):
                    gt = ConnMat(_bom)
                    ## rectify (PO4)n, it turns out that a dual unit is not necessary, i.e.,
                    ## we need subm of HO-P(=O)(O)-O-P(=O)(O)-OH, consisting of 8 heavy atoms
                    ## at most. The 9-atom entity can be totally described by its consistitugin
                    ## smaller unit.
                    for i in range(nheav):
                        if zs[i] in [8]: # [7,8]
                            # 1 being added below corresp. to atom `i
                            nai = 1 + (np.logical_and(gt.pls[i]>0, gt.pls[i]<=2)).sum()
                            if nai<=8 and nai==nheav:
                                # the latter criteria is crutial to remove redundant
                                # amons. E.g., when (CH3)3C-O-C(CH3)3 is selected,
                                # the redundant (CH3)3C-O-C(C)(C)CC may also be selected.
                                ifd_extl = T
                                break

                    ## now check if highly conj envs exist
                    ## should be aromatic! E.g., c6h5-CH=CH2, c6h5-CH=O
                    ## Note that the structure below is not aromatic
                    ##
                    ##           ======
                    ##          /      \
                    ##         /        \
                    ##        \\        /====
                    ##         \\______/
                    ##                 \\
                    ##                  \\
                    ##
                    ## A double bond is represented by either "===" or "//" or "\\"
                    ##
                    ## As the complete molecular graph is not available by now
                    ## postpone this to later. Now we only determine if it's potentially
                    ## interesting to be further assessed.
                    if not ifd_extl:
                        if np.all(iconjs):
                            ## cnsi .eq. number of conjugated neighbors
                            ioks = np.logical_and( np.logical_or(zs==7,zs==6), cnsi>=2 )
                            if ioks.sum() >= 6:
                                ifd_extl = T
                                i_further_assessment = T

                    if not ifd_extl:
                        continue

# neglect any fragment containing multivalent atom whose degree of heavy atom differs
# from that in query by `nhdiff and more.
# E.g., when nhdiff=2, given a query O=P(O)(O)O, only O=[PH](O)O is
# to be kept, while O=[PH3], O=[PH2]O will be skipped
                idfmulval = F
                nhdiff = 3
                dcnshv = cnshv - cnsi
                for j in range(nheav):
                    if (zs[j] in [15,16]) and (dcnshv[j] >= nhdiff):
                        idfmulval = T
                        break
                if idfmulval:
                    continue

# first retain the BO's for bonds involving any multi-valent atom, i.e.,
# atom with dvi>1. Here are a few examples that are frequently encountered:
# 1) C2 in "C1=C2=C3" & "C1=C2=N3";
# 2) N2 and N3 in "C1=N2#N3" ( -C=[N+]=[N-], -N=[N+]=[N-] )
# 3) '-S(=O)(=O)-', -Cl(=O)(=O)=O,
# By doing This, we can save a lot of work for BO perception later!
                #print 'tvs = ', tvs, ', vsi=',vsi+nsh, ', dvsi=', dvsi
                iasr_multi = []
                for _i, _ia in enumerate(lasi):
                    #if np.any([ iia in list(tsi) for tsi in self.envsC ]):
                    #    dvsi[ii] = 0
                    if dvsi[_i] > 1:
                        iasr_multi.append(_i)
                        for _ja in self.ias[self.bom[_ia]>1]:
                            if np.any(_ja==_lasi):
                                _j = iaq2iaa[_ja]
                                _bom[_i,_j] = _bom[_j,_i] = self.bom[_ia,_ja]

                # update dvsi for the 1st time
                vsi = _bom.sum(axis=0)
                dvsi = tvs - (vsi+nsh)
                #print 'dvsi = ', dvsi
                #print 'tvs = ', tvs, ', vsi=',vsi, ', nsh=',nsh
                #print 'bom = ', _bom

                # check if valence of multi-valent atoms are alright!
                # e.g., for O=C=C=C=O
                # when `lasi =  [1, 2], tvs, tvsi, dvsi =  [4 4] [3 3] [1 1]
                # is an invalid amon
                #print ' iasr_multi = ', iasr_multi
                if len(iasr_multi) > 0:
                    if np.any(dvsi[iasr_multi]!=0):
                        #print ' ** multi'
                        continue

                if np.any(dvsi>1):
                    #print 'dvi>1, implying say, [Cl][O] in query: -ClO3'
                    continue

                # now perceive double bonds
                ###print '######### lasi = ', lasi
                iok, boms = update_bom(_bom, tvs, nsh)
                #print ' iok = ', iok
                if not iok: continue

            # get coords of H's
            #lasi2 = [] # idx of H's bonded to heavy atoms
            coords2 = []
            nh = 0
            icnt = nheav
            bsxh = [] # X-H bonds

            for _i in range(nheav):
                ia = lasi[_i]
                _nbrs = self.ias[self.bom[ia]>0]
                for ja in _nbrs:
                    if np.any(ja==_lasi): continue
                    bxh = [_i,icnt]
                    if bxh not in bsxh: bsxh.append(bxh)
                    if self.zs[ja] == 1:
                        #lasi2 += [ja]
                        coords2.append( self.coords[ja] )
                    else:
                        dsHX = dsHX_normal #if self.fixGeom else dsHX_halved
                        if self.i3d:
                            coords_i = self.coords[ia]
                            v1 = self.coords[ja] - coords_i
                            #print(' ** ia, ja, v1 = ', ia, ja, v1)
                            dHX = dsHX[self.zs[ia]]
                            coords_j = coords_i + dHX*v1/np.linalg.norm(v1)
                        else:
                            coords_j = np.array([0., 0., 0.])
                        coords2.append(coords_j)
                    icnt += 1
                    nh += 1
            #print('coords2=', coords2)
            if nh > 0:
                coords = np.concatenate((coords,coords2))
            chgs = np.concatenate((chgs,[0]*nh))
            zs = np.concatenate((zs,[1]*nh))


            if self.i3d and self.fixGeom:
                ds = ssd.squareform( ssd.pdist(coords) )

            nat = nheav + nh
            #mols = []
            ishown = T
            for _bom in boms:
                bom = np.zeros((nat,nat),dtype=int)
                bom[:nheav,:nheav] = _bom
                # append H's
                for _bxh in bsxh:
                    _ia,_ja = _bxh
                    bom[_ia,_ja] = bom[_ja,_ia] = 1
                # final check
                tvs_heav = bom.sum(axis=0)[:nheav]
                #print ' tvs_heav = ', tvs_heav
                #print ' tvs = ', tvs
                if not np.all(tvs_heav-tvs==0):
                  if self.iwarn and (not self.warning_shown):
                    self.warning_shown = T
                    print(' ** [warning] ** ')
                    print('      Not all dvs==0 for the subg found, check!')
                    print('      This may happen when input is rad, but from which')
                    print('      it is still possible to extract non-rad amons ')
                    print('      Example: c1(ccc(cc1)[C](C#N)C#N)[C](C#N)C#N')
                    print('             |   ')
                  continue
                _newm = coo.newmol(zs,chgs,bom,coords)
                can = OECreateSmiString(_newm, OESMILESFlag_Canonical)
                #print 'can = ',can
                if '.' in can:
                    continue

                # when fixGeom=T, we need to reject AMONs in which
                # there exists any pair of atoms with dij<dijmax,
                # where dijmax = 1.24 * (rvdw_i + rvdw_j)
                if self.i3d and self.nocrowd: #and self.fixGeom
                    #gnb = (bom==0) # non-bonded graph
                    #np.fill_diagonal(gnb, F)
                    if is_overcrowd(zs, bom, coords):
                        fdt = './overcrowded.0' # Temporary folder
                        if not os.path.exists(fdt): os.mkdir(fdt)
                        tsdf = tpf.NamedTemporaryFile(dir=fdt).name + '.sdf'
                        print(' -- overcrowded amon written to ', tsdf)
                        write_ctab(zs, [0]*len(zs), bom, coords=coords, sdf=tsdf)
                        continue #print('## too crowded')

                Mli = [zs, coords, bom, chgs]
                if can in cans_u:
                    if (nheav <= 2) and (not self.fixGeom) and (not self.ivdw):
                        continue
                else:
                    cans_u.append( can )

                ## resume from where we left last time
                if ifd_extl and i_further_assessment:
                    #print('can=', can)
                    newm = coo.StringM(can) #_newm)
                    if newm.is_conj_amon:
                        ifd_extl = T
                    else:
                        continue

                if ifd_extl: # and (can not in cans_u):
                    print(' ##### found larger essential amons with N_I=%d: %s'%(nheav, can))
                    #ishown = F

                if submol is None:
                    yield Mli, lasi, can #, mu, can, nheav
                else:
                    yield [zs,chgs,bom,coords]



class Logger(object):

    def __init__(self, obj=None):
        if obj in ['stdout', None]:
            fid = None
            isnull = T
        else:
            assert isinstance(obj,str)
            fid = open(obj,'w')
            isnull = F
        self.isnull = isnull
        self.fid = fid

    def write(self, string):
        if self.isnull:
            print(string)
        else:
            self.fid.write(string+'\n')

    def close(self):
        if not self.isnull:
            self.fid.close()


class ParentMols(object):

    def __init__(self, strings, reduce_namons, fixGeom=F, iat=None, wg=T, i3d=T, \
                 iwa=T, k=7, iprt=T, submol=None, label=None, stereo=F, isotope=F, \
                 iextl=T, icc=None, substring=None, rc=6.4, imap=T, k2=7, \
                 opr='.le.', irc=T, iters=[90,900], M='cml1', iclean=T, \
                 thresh=0.1, wsmi=T, keepHalogen=F, nproc=1, \
                 forcefield='mmff94', gopt='xtb', nocrowd=T, \
                 ivdw=F, ivao=F, nmaxcomb=3,\
                 irad=F, ichg=F, prefix='', iwarn=T, debug=F, log=T):
#                 do_pm7=False, relaxHHV=False, \
        """
        prefix -- a string added to the beginning of the name of a
                  folder, where all sdf files will be written to.
                  It should be ended with '_' if it's not empty
        iclean -- remove redundency in retrieved submols, defaulted to T
                  For debugging purpose, e.g., to visualize the retrieved
                  local submol, set it to T
        log    -- write log file

        irc    -- T/F: relax w/wo dihedral constraints
        nocrowd-- avoid sub-structure with too close atoms (i.e., r_ij < 1.25*(r_vdw^i +r_vwd^j)

        imap   -- write maps? T/F
        iwa    -- write graph amons? T/F

        iextl  -- include larger amons with N_I > 7
                  This will reset k from 7 (the default) to 9 and then
                  filter out those amons that meet the following two conditions:
                  i) at least one O atom is involved
                  ii) There are no more than 7 heavy atoms in the local
                      fragment enclosing the O atom and all its neighboring
                      atoms within a radius of PathLength=2

        ivao   -- return vdw amons only??

        submol -- specify a substructure specifically, should be a
                  list of format [atoms,bonds]. Useful for diagnose
                  purposes.

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
        iat    -- idx of atom. If specified, any fragment centered on this
                  atom, covering all atoms within a radius of `rc will be
                  selected as the AMONs
        """

        k0 = k
        if iextl and k0 < 9:
            iextl = T
            k = 9
        else:
            iextl = F
        self.iextl = iextl

        self.debug = debug

        # label name
        if label is None:
            s1 = 'EQ' if opr == '.eq.' else ''
            svdw = ''
            if ivdw:
                svdw += '_vdw%d'%k2
            scomb = '_comb2' if nmaxcomb == 2 else ''
            if ivdw: svdw += scomb
            sthresh = '_dM%.2f'%thresh if thresh > 0 else ''
            if ivdw: svdw += sthresh
            if prefix == '':
                fdn = 'g%s%d%s'%(s1,k,svdw)
            else:
                fdn = prefix
            if self.iextl:
                fdn += '_extl'
            self.fd = fdn
            if iat is not None:
                fdn += '_iat%d'%iat # absolute idx
            fcan = fdn + '/' + fdn + '.can'
            h5f = '%s/map.h5'%fdn
        else:
            if label in ['auto']:
                label = 'g%d'%k0
                xtra = 'rat' if reduce_namons else 'raf'
                label += xtra
                if ivao: label += '_vdw'
                if self.iextl:
                    label += '_extl'
            fcan = label + '.can'
            h5f = label + '/map.h5'
            fdn = label
            for fd in [fdn, fdn+'/i-raw/']:
                if not os.path.exists(fd):
                    os.mkdir(fd)

        fin = None
        if log:
            fin = label + '.out'
        self.io = Logger(fin)


        # parameter resettings
        #if ivdw:
        #    fixGeom = T
        #    print(' ** fixGeom reset to T for vdw amons\n')


        # use mmff94 only, as there are some unresolved issues with uff,
        # one typical example is that for some structure, uff in rdkit
        # tend to favor a structure with two H's overlapped!!
        # You may manually verify this by calling optg_c1() and optg() for
        # a substructure C=CC(C=C)=C retrieved from Oc1ccc(C=O)cc1
        assert forcefield == 'mmff94', '#ERROR: DO NOT USE UFF'

        param = Parameters(i3d, fixGeom, k, k2, ivdw, \
                           forcefield, thresh, \
                           gopt, M, iters, reduce_namons, nproc)

        ncpu = multiprocessing.cpu_count()
        if nproc > ncpu:
            nproc = ncpu

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

        cans = []; nsheav = []; es = []; maps = []
        ms = []; ms0 = []

        # initialize `Sets
        ids = []
        seta = Sets(param)
        warning_shown = F
        for ir in range(nmt):
            string = strings[ir]
            if iprt:
                self.io.write('#Mid %d %s'%(ir+1, string))
                print('#Mid %d %s'%(ir+1, string))
            obj = ParentMol(string, iat=iat, i3d=i3d, k=k, k2=k2, stereo=stereo, isotope=isotope,\
                            opr=opr, fixGeom=fixGeom, nocrowd=nocrowd, \
                            iextl=iextl, irad=irad, ichg=ichg, ivdw=ivdw, \
                            keepHalogen=keepHalogen, debug=debug, iwarn=iwarn, warning_shown=warning_shown)
            warning_shown = obj.warning_shown
            if not obj.istat:
                self.io.write(' [failure to parse SMILES/kekulization]')
                continue

            if obj.is_radical():
              if iwarn:
                self.io.write(' ** warning: input mol is a radical')
              if not irad:
                raise Exception(' Consider setting irad=T... [Todo: full support of radical]')
            if obj.has_standalone_charge():
              if iwarn:
                self.io.write(' ** warning: input mol is charged species')
              if not ichg:
                raise Exception(' Consider setting ichg=T... [Todo: full support of charged species]')

            #if debug: print('##pass 1')

            ids.append(ir)
            Mlis, iass, cansi = [], [], []
            # we needs all fragments in the first place; later we'll
            # remove redundencies when merging molecules to obtain
            # valid vdw complexes
            nas = []; nasv = []; pss = []
            iass = []; iassU = []; rsc=[] # cov radius
            #c2ias = {}

            try:
                for Mli, ias, can in obj.generate_amons(submol=submol):
                    nheav = len(ias)
                    #if can in c2ias.keys():
                    #    c2ias[can] += [ias]
                    #else:
                    #    c2ias[can] = [ias]
                    kk = 15
                    if i3d:
                        # allow `kk to be larger than input `k. This is necessary
                        # as the algorithm will automatically identify the very few
                        # amons that are 1) indispensible for accurate extropolation.
                        # and 2) with N_I > k. E.g., amons with 3 HB's in AT or CG pair
                        iasU = ias + [-1,]*(kk-nheav) #
                        nasv.append(nheav)
                        Mlis.append( Mli ); iass.append( ias ); cansi.append( can )
                        iassU.append( iasU ); pss += list(Mli[1]); rsc += list( rcs0[Mli[0]] )
                        nas.append( len(Mli[0]) )
                    else:
                        #if debug: print('##can = ', can)
                        if can not in cansi:
                            cansi.append(can)
                            nasv.append(nheav)
                    #print('ias=',ias, 'can=',can)
            except:
                raise Exception('#ERROR: `generate_amons() failed!!')

            ngi = len(set(cansi)) # number of graphs (i.e., unique smiles)
            nmi = len(cansi)
            if debug: print('ngi,nmi=',ngi,nmi, ' unique cans=', set(cansi))

            nasv = np.array(nasv, np.int)
            if i3d:
                nas = np.array(nas, np.int)
                pss = np.array(pss)
                iassU = np.array(iassU, np.int)
                rsc = np.array(rsc)
                if ivdw:
                    ncbsU = []
                    for b in obj.ncbs:
                        if np.any([ set(b) <= set(ats) for ats in iassU ]):
                            print('       vdw bond: (%d,%d) deleted due to existence in cov amons'%(b[0],b[1]))
                            continue
                        ncbsU.append(b)
                    #print('ncbsU=',ncbsU)
                    ncbsU = np.array(ncbsU, np.int)

            # now combine amons to get amons complex to account for
            # long-ranged interaction. (hydrogen bond is covered as
            # well.
            Mlis2 = []; iassU2 = []; cansi2 = []
            if i3d and ivdw:
                if substring != None:
                    cliques_c = set( is_subg(obj.oem, substring, iop=1)[1][0] )
                    #print ' -- cliques_c = ', cliques_c
                    cliques = cg.find_cliques(obj.g0)
                    Mlis_centre = []; iass_centre = []; cansi_centre = []
                    Mlis_others = []; iass_others = []; cansi_others = []
                    for i in range(nmi):
                        #print ' %d/%d done'%(i+1, nmi)
                        if set(iass[i]) <= cliques_c:
                            Mlis_centre.append( Mlis[i] )
                            iass_centre.append( iass[i] )
                            cansi_centre.append( cansi[i] )
                        else:
                            Mlis_others.append( Mlis[i] )
                            iass_others.append( iass[i] )
                            cansi_others.append( cansi[i] )
                    nmi_c = len(Mlis_centre)
                    nmi_o = nmi - nmi_c
                    self.io.write(' -- nmi_centre, nmi_others = ', nmi_c, nmi_o)
                    Mlis_U = []; cansi_U = []
                    for i0 in range(nmi_c):
                        ias1 = iass_centre[i0]
                        t1 = Mlis_centre[i0]; nheav1 = (np.array(t1[0]) > 1).sum()
                        for j0 in range(nmi_o):
                            ias2 = iass_others[j0]
                            t2 = Mlis_others[j0]; nheav2 = np.array((t2[0]) > 1).sum()
                            if nheav1 + nheav2 <= k2 and check_ncbs(ias1, ias2, obj.ncbs):
                                dmin = ssd.cdist(t1[1], t2[1]).min()
                                if dmin >= 1.6: # ??
                                    cansij = [cansi_centre[i0], cansi_others[j0]]
                                    cansij.sort()
                                    cansi_U.append( '.'.join(cansij) )
                                    Mlis_U.append( merge(t1, t2) )
                    Mlis = Mlis_U; cansi = cansi_U
                    self.io.write(' -- nmi_U = ', len(Mlis))
                else:
                    self.io.write('| Now perceive vdw connectivity between standalone amons')
                    use_f90 = T # F
                    print('ncbs=', [ list(_) for _ in ncbsU ])
                    assert np.all(obj.zs[ncbsU[:]]>1), '#ERROR: only heavy atoms are allowed in `ncbs'
                    # gv   -- vdW graph connectivity between amons (if two parts
                    #         are connected by vdW bond, then assign to 1; otherwise, 0)
                    # gc   -- covalent graph connectivity between amons. Assign the
                    #         value to 1 when one amon is part of another or these two
                    #         amons are in close proximity.
                    gv = np.zeros((nmi,nmi),dtype=int)
                    gc = np.zeros((nmi,nmi),dtype=int)
                    if not use_f90:
                        for i0 in range(nmi-1):
                            _ias1 =  iassU[i0]
                            ias1 = _ias1[_ias1>-1]
                            t1 = Mlis[i0]
                            nhv1 = (np.array(t1[0])>1).sum()
                            for j0 in range(i0+1,nmi):
                                _ias2 = iassU[j0]
                                ias2 = _ias2[_ias2>-1]
                                t2 = Mlis[j0]; nhv2 = np.array((t2[0])>1).sum()
                                if nhv1 + nhv2 <= k2:
                                    #print('i0,j0=',i0,j0)  #, 'nhv1,nhv2=', nhv1,nhv2)
                                    #print('ias1,ias2=',ias1,ias2)

                                    iascplx = set( list(ias1)+list(ias2) )
                                    #print('iascplx=',iascplx)
                                    if np.any([ set(bi) <= set(iascplx) for bi in ncbsU]):
                                        #print('      =====> ')
                                        ds12 = ssd.cdist(t1[1], t2[1])
                                        rcs1 = rcs0[t1[0]]; rcs2 = rcs0[t2[0]]
                                        tfs12 = ( ds12 <= (rcs1[...,np.newaxis]+rcs2)+0.45 )
                                        if np.any(tfs12):
                                            gc[i0,j0] = gc[j0,i0] = 1
                                            continue
                                        gv[i0,j0] = gv[j0,i0] = 1
                                        #print(' ------------------ found vdw bond')
                    else:
                        gv,gc = fa.get_amon_adjacency(k2,nas,nasv,iassU.T,rsc,pss.T,ncbsU.T)
                    self.io.write('amon connectivity done')
                    #print('gv=',gv)
                    if debug:
                        print('pairs of vdw connected subgraphs: ')
                        gs1, gs2 = np.array(np.where(np.triu(gv)>0))
                        print('   idx of sg 1: ' , gs1)
                        print('   idx of sg 2: ' , gs2)
                    ims = np.arange(nmi)
                    combs = []
                    for im in range(nmi):
                        nv1 = nasv[im]
                        jms = ims[ gv[im] > 0 ]

                        if self.debug:
                            if len(jms) > 0:
                                print('im,m= ', im,nv1, cansi[im])
                                print('    |___ im,jms = ', im,jms)
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
                                        if self.debug:
                                            print('               ++ comb %s added'%comb)
                                        combs += [comb]
                            elif nmaxcomb == 3:
                                #for jm in jms:
                                #    comb = [im,jm]; comb.sort()
                                #    if comb not in combs:
                                #        combs += [comb]

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
                    self.io.write('atom indices of all amons done')
                    for comb in combs:
                        #print comb

                        ###################################################
                        # don't allow mol with N_I=1, may form too strong
                        # H-bond after optg??
                        ###################################################
                        _nat = 0
                        _nas = []
                        for ic in comb:
                            _ni = (np.array(Mlis[ic][0])>1).sum()
                            _nas.append(_ni)
                            _nat += _ni

                        imcnigt1 = T # should be molecular complex with either constituent's N_I > 1? T/F
                        #if not np.any( np.array(_nas) == 1. ):
                        if imcnigt1 and np.any( np.array(_nas) == 1. ):
                            continue

                        #if ivao and _nat <= 7:
                        #    continue

                        _cansi = [ cansi[ic] for ic in comb ]; _cansi.sort()
                        _can = '.'.join(_cansi)
                        if self.debug: print('    ++ found ', _can)
                        cansi2.append(_can)
                        _cs = [] #  for ic in comb ]
                        _iast = []
                        for ic in comb:
                            _ias = iassU[ic]
                            _iast += list(_ias[_ias > -1])
                            _cs += [Mlis[ic]]
                        Mlis2.append( merge(_cs) )
                        assert len(_iast) <= kk
                        iassU2.append( _iast + [-1]*(kk-len(_iast)) )
                    self.io.write('amons now ready for filtering')

            # return vdw amons only?
            if ivao:
                Mlis = Mlis2
                cansi = cansi2
                iassU = iassU2
            else:
                Mlis += Mlis2
                cansi += cansi2
                #print('size=', iassU.shape, np.array(iassU2).shape )
                iassU = np.array( list(iassU)+iassU2, dtype=int)

            ncan = len(cansi)
            # now remove redundancy
            if iclean:
                if i3d:
                    for i in range(ncan):
                        seta.update(ir, cansi[i], Mlis[i])
                    #seta._sort() # plz sort at last to save time
                else:
                    if imap:
                        for i in range(ncan):
                            seta.update2(ir, cansi[i], nasv[i])
                        #seta._sort2() # plz sort at last to save time

            # now update cans
            for ci in cansi:
                if ci not in cans:
                    cans.append(ci)

        ncan = len(cans)
        print('cans=',cans)

        # sort all amons
        if not i3d:
            #assert label is not None
            if imap:
                self.io.write(' -- now sort amon SMILES by N_I')
                seta._sort2()
                cans = seta.cans
                self.cans = cans
                if label is not None:
                    h5f = label + '.h5'
                    dd.io.save(h5f, {'ids': np.array(ids,dtype=int), 'maps': seta.maps2} )


            self.cans = cans
            if iwa:
              assert label is not None, '#ERROR: please specify `label'
              fcan = label+'.can'
              with open(fcan, 'w') as fid:
                fid.write('\n'.join( [ '%s'%(cans[i]) for i in range(ncan) ] ) )
        else:
            if not iclean: # debug
                self.ms = []
                self.iass = []
                for im,Mli in enumerate(Mlis):
                    zs, coords, bom, chgs = Mli
                    ctab = write_ctab(zs, chgs, bom, coords)
                    m = Chem.MolFromMolBlock( ctab, removeHs=F) # plz keep H's
                    self.ms.append(m)
                    _ias = np.array(iassU[im],dtype=int)
                    _ias2 = list(_ias[_ias>-1]); _ias2.sort()
                    self.iass.append(tuple(_ias2))
            else:
                seta._sort()
                cans = seta.cans; ncs = seta.ncs; nsheav = seta.nsheav
                self.cans = cans
                nd = len(str(ncan))

                if wg:
                    dd.io.save(h5f, {'ids': np.array(ids,dtype=int), 'maps': seta.maps2} )

                _ms = seta.ms; _ms0 = seta.ms0
                self.maps = seta.maps2
                self.nm = sum(ncs)
                if wg:
                    self.io.write(' amons are to be written to %s'%fdn)
                    for i in range(ncan):
                        ms_i = _ms[i]; ms0_i = _ms0[i]
                        nci = ncs[i]
                        labi = '0'*(nd - len(str(i+1))) + str(i+1)
                        self.io.write(' ++ %d %06d/%06d %60s %3d'%(nsheav[i], i+1, ncan, cans[i], nci))
                        if ivao: print(' ++ %d %06d/%06d %60s %3d'%(nsheav[i], i+1, ncan, cans[i], nci))
                        for j in range(nci):
                            f_j = fdn + '/frag_%s_c%05d'%(labi, j+1) + '.sdf'
                            f0_j = fdn + '/i-raw/frag_%s_c%05d_raw'%(labi, j+1) + '.sdf'
                            m_j = ms_i[j]; m0_j = ms0_i[j]
                            Chem.MolToMolFile(m_j, f_j)
                            Chem.MolToMolFile(m0_j, f0_j)
                else:
                    ms = []; ms0 = []
                    for i in range(ncan):
                        ms += _ms[i]
                        ms0 += _ms0[i]
                self.ms = ms
                self.ncan = ncan
                self.ms0 = ms0
                self.io.write(' ## summary: found %d molecular graphs, %d configurations'%(ncan, self.nm) )

    def get_matched_subm(self, ias, itype='f', otype='mol'):
        assert hasattr(self, 'iass')
        assert isinstance(ias[0], (int, np.int32, np.int64))
        if itype.lower() == 'f':
            ias = list( np.array(ias,dtype=int)-1 )
        ias.sort()
        i = tuple(ias)
        ims = np.arange(len(self.ms))
        if i in self.iass:
            idx = self.iass.index( tuple(ias) )
        else:
            print(' ** no match found, return closest match instead')
            na_share = [ len(set(i).intersection(set(j))) for j in self.iass ]
            seq = np.argsort(na_share)
            idx = seq[-1]
            print('        ias = ', self.iass[idx] )
        if otype in ['mol']:
            ot = self.ms[idx]
        elif otype in ['sdf']:
            ot = tpf.NamedTemporaryFile(dir='/tmp/').name + '.sdf'
            Chem.MolToMolFile(self.ms[idx], ot)
        return ot


def find_conjugate_chain(g,tvs,nsh):
    """
    Recursively find bonds formed between atoms with dvi=1.

    caveat!!
    ===========
    Note that previously you used an algorithm with pitfall. The
    central idea is to update graph `g by setting all BO's to 0
    for bonds associated with atoms that satisfying 1) cni=1 and
    2) have been visited. This
    This can be falsified through inspecting a submol (containing
    atoms 4-9) below (H is ignored)
                   9
                  /\\     //\
                 /  \\8_4//  \
                ||   |   |   ||
                ||  7|___|5  ||
                 \  //   \\  /
                  \//     \\/
                           6
    once we've identified the bonds with cns==1, which are
    8-9 & 5-6, we cannot simply assign BO of bonds associated
    with atoms 5,6,8,9 (i.e., bonds 8-9,8-4,8-7 and 5-4,5-6,5-7)
    to 0! In reality, only BO's of 8-9 & 5-6 should be reset to 0)
    """
    na = g.shape[0]
    ias = np.arange(na)
    ips = []
    irad = F
    gc = g.copy()
    bsv = [] # visited bonds
    icnt = 0
    while T:
        cns = g.sum(axis=0)
        dvs = tvs - (gc.sum(axis=0)+nsh)
        ###print ' + tvs, vs, dvs = ', tvs, gc.sum(axis=0)+nsh, dvs
        assert np.all(dvs<=1), '#ERROR: some dvi>=2?'
        if np.all(dvs==0):
            break
        _filt = (dvs > 0)

        # now check if the graph made up of the rest atoms is connected
        # VERY important step to account for the issue mentioned above!!!!!
        if not cg.is_connected_graph(gc[_filt][:,_filt]):
            ###print '##2'
            irad = T
            break

        f1 = (dvs>0)
        g1 = g[f1][:,f1]
        ias1 = ias[f1]
        cns1 = g1.sum(axis=0)
        f2 = (cns1==1)
        g2 = g1[f2][:,f2]
        ias2 = ias1[f2]
        #print ' +++++ ias2 = ', ias2
        nar = len(ias2)
        if nar == 0:
            break
        ias3 = np.arange(nar)
        for ia3 in ias3:
            ia = ias2[ia3]
            jas = ias[ np.logical_and(g[ia]>0,dvs>0) ]
            if len(jas) == 0: continue
            ja = jas[0]
            ip = set([ia,ja])
            # the newly found bond should either exist in the set `bsv (e.g.,
            # for a mol C=C, when ia=0, bond 0-1 would be added to `bsv; when
            # ia=1, 1-0 is identical to 0-1
            if ip in bsv:
                continue
            else:
                bsv.append(ip)
            # It's not allowed that a newly found bond shares some common atom
            # with any atom in the bond set `bsv.
            if len(ips)==0 or np.all([ _ip.isdisjoint(ip) for _ip in ips ]):
                ips.append(ip)
                #atsv.update(ip)
            else: # e.g., [CH2]C([CH2])[CH2]
                ###print '##3, ', ip
                irad = T
                break
        if irad: break

        for ip in ips:
            ib,ie = ip
            gc[ib,ie] = gc[ie,ib] = 2

        #print( ' ** ias, ips = ', ias, ips)
        icnt += 1

        ## why did I try to exit the program here??
        ## Need sometime to think about this!!
        ## Never use exit() in a while loop, please!
        ## Otherwise, it's very difficult to diagnose
        ## where has been wrong!!
        #if icnt == 3:
        #    print('########## strange case?????????')
        #    sys.exit(2)

    #if irad
    #print 'ips = ',ips
    if len(ips) > 0:
        _ips = []
        for _ip in ips:
            ia,ja = list(_ip)
            _ips.append( [ia,ja] )
        _ips.sort()
    else:
        _ips = []
    return irad, _ips

def find_bo2(_g, _tvs, _nsh, debug=F):
    """ find pairs of atoms that are associated with bo=2 """
    g = _g.copy()
    tvs = _tvs.copy()
    nsh = _nsh.copy()
    na = g.shape[0]
    ats = np.arange(na,dtype=int)
    _ats = set(range(na))
    cns = g.sum(axis=0)
    ipss = []
    irad = F
    # now conjugated double bonds
    #
    # step 1: linear chain (no ring)
    atsv = set()
    ips1 = []
    dvs = tvs - (g.sum(axis=0)+nsh)
    if np.any(dvs==1):
        irad, ipsr = find_conjugate_chain(g,tvs,nsh)
        #print ' ++ irad, ipsr = ', irad, ipsr
        if not irad:
            for ipr in ipsr:
                ia,ja = ats[ipr]
                g[ia,ja] = g[ja,ia] = 2
                #atsv.update([ia,ja])
                ips1.append([ia,ja])
    dvs = tvs - (g.sum(axis=0)+nsh)

    if irad:
        ots = [T, []]
    else:
        #atsr = _ats.difference(atsv)
        if np.all(dvs==0): #len(atsr) == 0:
            return [F,[ips1]]
        else:
            ## step 2: ring
            atsr = ats[dvs>0]
            #print ' ___ atsr = ', atsr
            gtmp = g[atsr][:,atsr]
            assert cg.is_connected_graph(gtmp), '#ERROR: not connected graph?'
            bs = np.array( np.where( np.triu(gtmp)>0 ), dtype=int ).T
            #print '++ bs = ', bs
            iok = T
            for b in bs:
                ib,ie = atsr[b]
                g1 = g.copy()
                g1[ib,ie] = g1[ie,ib] = 2
                ips = ips1+[[ib,ie]]
                dvs1 = tvs - (g1.sum(axis=0)+nsh)
                f2 = (dvs1>0)
                ats2 = ats[f2]
                na2 = len(ats2)
                irad, ips2 = find_conjugate_chain(g1,tvs,nsh) #g2,tvs2,nsh2)
                if not irad:
                    ################################################################
                    #  WARNING: Be cautious here !!
                    ################################################################
                    if len(np.array(ips2,dtype=int).ravel()) < na2:
                        # it's possible that some ring still exists
                        # in the remaining structure (when the ring is very big)
                        # E.g., for c12ccccc1cccc2, when `b is initialized to [1,2],
                        # ipsr=[[1,2],[3,4]] results by calling
                        # find_conjugate_chain(),
                        # and there still exists a ring!!
                        # The reason we skip this this is that there is always
                        # success perceiving BO's for all bonds, esp. for
                        # small molecules (e.g., AMONs)
                        continue
                    ###############################################################
                    for ip in ips2: # idx in `ips2 is absolute idx
                        ip.sort()
                        ips.append(ip) #; _atsv.update(ip)
                    #print 'b = ', b, ' ips = ', ips
                    ips.sort()
                    if ips not in ipss:
                        iok = T
                        ipss.append( ips )
            assert iok, '#ERROR: perception failure!'
            ots = [F,ipss]
    return ots

def g_from_edges(edges):
    na = np.array(edges,dtype=int).ravel().max()+1
    g = np.zeros((na,na),dtype=int)
    for edge in edges:
        i,j = edge
        g[i,j] = g[j,i] = 1
    return g

def test_find_bo2():
    # case 1: [C]([CH2])([CH2])[CH2] invalid amon
    es = [[0,1],[0,2],[0,3]]
    nsh = [0,2,2,2]
    n=4; g = np.zeros((n,n),dtype=int)
    tvs = [4,]*4
    for i,j in es: g[i,j]=g[j,i] = 1
    print(' ipss = ', find_bo2(g))
    print(' boms = ', update_bom(g, tvs, nsh))

    # case 2: c12ccccc1cccc2
    es = [ [i,i+1] for i in range(9) ]
    es += [[0,5],[9,0]]
    nsh = [0] +[1,]*4 + [0] +[1,]*4
    n=10; g = np.zeros((n,n),dtype=int)
    tvs = [4,]*n
    for i,j in es: g[i,j]=g[j,i] = 1
    print(' ipss = ', find_bo2(g))
    print(' boms = ', update_bom(g, tvs, nsh))

    # case 3: O=c1ccc(=O)cc1
    es = [[0,1],[1,2],[2,3],[3,4],[4,5],[4,6],[6,7],[7,1]]
    n=8; g = np.zeros((n,n),dtype=int)
    tvs = [2] + [4,]*4 + [2] + [4,]*2
    nsh = [0]*2 + [1]*2 + [0]*2 + [1]*2
    for i,j in es: g[i,j]=g[j,i] = 1
    print(' ipss = ', find_bo2(g))
    print(' boms = ', update_bom(g, tvs, nsh))

def update_bom(bom, tvs, nsh, debug=F):
    """ update bond order matrix """
    na = bom.shape[0]
    ias = np.arange(na)
    g = (bom>0).astype(np.int)
    cns = g.sum(axis=0)
    vs = bom.sum(axis=0)
    dvs = tvs-(vs+nsh)
    filt = (dvs==1)
    na1 = filt.sum()
    iok = T
    if na1 == 0:
        assert np.all(dvs==0), '#ERROR: some dvi>0! (case 1)'
        ots = [T, [bom]]
    else:
        if na1%2==1:
            iok = F
        else:
            g1 = g[filt][:,filt]
            cns1 = g1.sum(axis=0)
            tvs1 = tvs[filt]
            nsh1 = nsh[filt]
            ias1 = ias[filt]
            ipss = [] # ias (of) pairs's
            cs = cg.find_cliques(g1)
            nc = len(cs)
            for _csi in cs:
                csi = np.array(_csi,np.int)
                ias2 = ias1[csi]
                nai = len(csi)
                if nai%2 == 1:
                    iok = F
                    break
                elif nai == 2:
                    ipss_i = [ [ias2,] ] # one possibility
                    ipss.append(ipss_i)
                else:
                    _gi = g1[csi][:,csi]
                    _cnsi = _gi.sum(axis=0)
                    _nshi = nsh1[csi] + (cns[ias2]-_cnsi) # must-do!!
                    is_rad, ipssr_i = find_bo2(_gi, tvs1[csi], _nshi)
                    ###print ' is_rad, ipssr_i = ', is_rad, ipssr_i
                    if is_rad:
                        iok = F
                        break
                    else:
                        ipss_i = []
                        for ipsr in ipssr_i:
                            ips = []
                            for ip in ipsr: # ip -- ias of pairs
                                #print ' ip = ', ip, ',
                                ips.append( ias1[csi[ip]] )
                            ipss_i.append( ips )
                        ipss.append( ipss_i )
        if not iok:
            ots = [F, []]
        else:
            boms = []
            if len(ipss) >= 1:
                for bs in cim.products(ipss):
                    bom_i = copy.copy(bom)
                    ###print '-- bs = ', [ list(bsi) for bsi in bs ]
                    for i,bsi in enumerate(bs):
                        for bi in bsi:
                            ia1, ia2 = bi
                            bom_i[ia1,ia2] = bom_i[ia2,ia1] = 2 ## double bond
                    cnsi = bom_i.sum(axis=0)
                    dvsi = tvs-(cnsi+nsh)
                    ###print ' === tvs, tvsi, dvsi = ', tvs,cnsi+nsh,dvsi
                    assert np.all(dvsi==0), '#ERROR: some dvi>0! (case 2)'
                    boms.append(bom_i)
            else:
                print(' ########## Rare event!')
                boms.append( bom )
            ots = [T,boms]
    return ots


## test!
test_cases = """
[bing@pc-avl46 amons]$ genamon_oechem "O=C=C=C=O"
 ** set --k 7
 #Mid 1 O=C=C=C=O
  lasi =  [1, 2]
  -- bs =  [[array([0, 1])]]
   === tvs, tvsi, dvsi =  [4 4] [3 3] [1 1]
"""

if __name__ == "__main__":

    import sys, time

    args = sys.argv[1:]
    idx = 0
    ob = F
    if '-ob' in args: ob = T; idx+=1
    rk = F
    if '-rk' in args: rk = T; idx+=1
    ivdw = F
    if '-ivdw' in args: ivdw = T; idx += 1
    wg = F
    if '-wg' in args: wg = T; idx += 1
    reduce_namons = F
    if '-reduce_namons' in args: reduce_namons = T; idx += 1

    t0 = time.time()
    #objs = args[idx:]
    _args = args[idx:]
    n = len(_args)
    if n == 0:
        objs = ["C=C=S(=C=C)=[N+]=[N-]", \
                "S1(=C)(C)=CC=CC=C1", \
                "[N+]1([O-])=CC=C[NH]1", \
                "C[N+](=O)[O-]", \
                "C=[NH+][O-]", \
                "C[N+]#[C-]", \
                "C[NH2+][C-](C)C", \
                "[CH-]=[O+]C", \
                "N=[NH+][NH-]", \
                "[NH-][NH+]=C1C=C(C=C)C=C1", \
                "OP(=S)=P(=[PH2]C)C", \
                "O[N+]([O-])=[N+]([N-]C)O", \
                "OC12C3C4C1N4C32"] # the last one is highly strained, may have problem in acquring g0
                #"[NH3+]CC(=O)[O-]", \
                #"C[O-]",\
                #"C[NH3+]",\
    elif n == 1:
        f = _args[0]
        if f[-3:] in ['smi','can']:
            objs = [ si.strip() for si in file(f).readlines() ]
        else:  # either an xyz file or a SMILES string
            objs = _args
    else:
        objs = _args

    css = []
    for obj in objs:
        a = ParentMols([obj], reduce_namons, fixGeom=F, iat=None, wg=wg, i3d=i3d,\
                    k=7, label='temp', k2=7, opr='.le.', wsmi=T, irc=T, nocrowd=T, \
                   iters=[90,900], M='cml1', thresh=0.1, \
                   keepHalogen=F, forcefield='mmff94', gopt=gopt, \
                   ivdw=ivdw)
        css.append(a.cans)
    for i,cs in enumerate(css):
        print('## ', objs[i])
        print(css[i])
    print(' -- time elaped: ', time.time()-t0, ' seconds')

