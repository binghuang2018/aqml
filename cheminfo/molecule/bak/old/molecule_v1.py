#!/usr/bin/env python

import networkx as nx
import itertools as itl
#import scipy.spatial.distance as ssd
import cheminfo.molecule.elements as cce
import numpy as np
import os, sys, re, copy
from cheminfo.molecule.elements import Elements
from cheminfo.molecule.nbody import NBody
from cheminfo.rw.ctab import write_ctab
from cheminfo.rw.pdb import write_pdb
import cheminfo.math as cm
from rdkit import Chem
#import cheminfo.rdkit.amon_f as craf
import cheminfo.graph as cg
try:
    import cheminfo.fortran.famon as cf
except:
    import cheminfo.fortran.famon_mac as cf

global T,F
T=True; F=False

class Graph(object):

    def __init__(self, g):
        self.nn = g.shape[0] # number_of_nodes
        self.n = self.nn
        g1 = (g > 0).astype(np.int)
        np.fill_diagonal(g1, 0)
        self.g1 = g1
        self.nb = g1.sum()/2 # num of bonds (i.e., edges)
        self.bonds = [ list(edge) for edge in \
             np.array( list( np.where(np.triu(g)>0) ) ).T ]

    def is_connected(self):
        return nx.is_connected(self.g1)


class RawMol(object):

    def __init__(self, zs, coords):
        """ initialize a molecule"""
        self.coords = coords
        self.zs = np.array(zs, np.int)
        self.na = len(zs)
        self.ias = np.arange(self.na)
        self.pt = Elements(zs)
        self.connect()

    def connect(self):
        """
        establish connectivity between atoms from geometry __ONLY__
        """
        ps = self.coords
        rs = self.pt.rcs
        rmax = rs.max()
        ds = np.sqrt((np.square(ps[:,np.newaxis]-ps).sum(axis=2)))
        ds2 = ds * ds
        rs1, rs2 = np.meshgrid(rs,rs)
        ds2r = (rs1 + rs2 + 0.45)**2
        # step 1) get a preliminary connectivity table g
        g0 = np.logical_and( ds2 > 0.16, ds2 <= ds2r )
        self.g0 = np.array(g0,np.int)
        g = g0.astype(np.int)
        cns = g.sum(axis=0)
        #print cns

        # step 2) refine the g
        maxnbs = self.pt.maxnbs
        for ia in range(self.na):
            zi = self.zs[ia]
            if zi == 1:
                if g[ia].sum() > 1:
                    nbrs = self.ias[g[ia]>0]
                    if 1 in zs[nbrs]:
                        ja = nbrs[ zs[nbrs]==1 ]
                        g[ia,ja] = 0
                        g[ja,ia] = 0
                        cns = g.sum(axis=0)
            else:
                if cns[ia] == 1: continue
                while 1:
                    mb = NBody(self.zs, self.coords, g, rcut=0.0, unit='degree')
                    mb.get_angles([ia])
                    angs = mb.mbs3.values()
                    angmin = 180.0 if len(angs) == 0 else np.min(angs)
                    # for "C1=C2CN=C1NC2", min(angs) can reach 46.0 degree
                    if (cns[ia] > maxnbs[ia] or angmin < 50.0):
                        #some bond exceeds max valence
                        #now remove the bond with max bond length
                        jas = self.ias[g[ia]>0]
                        dsj = ds[ia,jas]
                        ja = jas[dsj==np.max(dsj)][0]
                        g[ia,ja] = g[ja,ia] = 0
                        cns = g.sum(axis=0)
                        #print ' * ia,ja = ', ia,ja
                        #print ia, zi, cns[ia],maxnbs[ia], np.concatenate(angs).min()
                        #assert cns[ia] <= maxnbs[ia], '#ERROR: still more valence than allowed'
                    else:
                        break
        self.g = g



class Mol(RawMol):

    def __init__(self, zs, coords, ican=True):
        RawMol.__init__(self, zs, coords)
        if ican:
            self.can = self.perceive_bond_order()

    def iodd_max_path_length(self, sg, source):

        _G = nx.Graph(sg)
        na = sg.shape[0]
        ias = np.arange(na)
        obsolete = """
        cns = sg.sum(axis=0)
        ias1 = ias[cns==1]
        #print ' ** ias1 = ', ias1
        #print ' ** iasc = ', iasc
        iodd = False
        for ia in ias1:
            if ia != source:
                for path in nx.all_shortest_paths(_G,source=source,target=ia):
                    if len(path)%2 == 1:
                        iodd = True #; print 'source,ia,path = ', source,ia,path
                        break
            if iodd: break
        if iodd:
            return iodd
        else:
            rings = nx.cycle_basis(_G)
            if len(rings) > 0:
                iasc = np.unique( np.concatenate(ring) )
                for ia in iasc:
                    assert ia != source
                    _ls = [ len(path) for path in nx.all_shortest_paths(_G,source=source,target=ia) ]
                    if np.max(_ls) % 2 == 1:
                        iodd = True
            return iodd"""
        ias.remove( source )
        maxl = 1
        for ia in ias:
            maxl = max(maxl, np.max([ len(path) for path in  nx.all_simple_paths(_G, source, ia)]))
        return maxl%2 == 0

    def find_bond_to_N5(self,dvs):
        ias0 = self.ias[ np.logical_and(dvs==0, self.zs==7) ]
        pairs = []
        if len(ias0) == 0:
            return pairs

        # get all subgraphs
        ias1 = self.ias[dvs==1]
        if len(ias1) == 0:
            return pairs
        amap = dict(zip(ias1, range(len(ias1))))
        _sg = self.g[ias1][:,ias1]
        iass = cg.find_cliques(_sg) # nodes are returned (relative idx)
        ng = len(iass)

        # try to find N5 and all potential neighbors with BO=2
        iok = False
        for i in ias0:
            # now check if there is any neighbor with dv=1, e.g., [NH][NH][NH]
            jas = self.ias[ np.logical_and(self.g[i]>0, dvs==1) ]
            naj = len(jas)
            if naj > 0:
                # naj=1, e.g., N_2 in [NH][NH][NH][NH]
                # naj=2, e.g., N_2 in [NH][NH][NH]
                # naj=3, e.g., N_2 in [NH][N]([CH][CH2])[NH]
                for j in jas:
                    # next check if the neighboring atoms with dv=1 form a
                    # subgraph with the maximal path length being a even number
                    for _ias in iass:
                        if amap[j] in _ias:
                            na = len(_ias)
                            if na%2 == 1: # imply the existence of R1=[NX]=R2
                                iok = True
                            else:
                                if na != 2:
                                    sg2 = _sg[_ias][:,_ias].astype(np.int32)
                                    nb = (sg2>0).sum()/2
                                    #print ' ** nb,na = ', nb,na
                                    #print ' ** sg2 = ', sg2
                                    # are there even number of double bonds
                                    _ipss = np.ones((2,nb,na/2))
                                    ipss = np.array(_ipss,dtype='int32',order='F') # ipss(2,np,nr0)
                                    nr = cf.locate_double_bonds(sg2,ipss,False)
                                    #na=shape(g,0),np=shape(ipss,1),nr0=shape(ipss,2))
                                    if nr == 0: #self.iodd_max_path_length(sg2,amap[j]):
                                        #print ' * hit', _ias, _sg[_ias]
                                        iok = True
                        if iok:
                            pair = [i,j]; pair.sort()
                            if pair not in pairs:
                                pairs.append( pair )
                            break
                            #iok = False
                    if iok: break
                if iok: break
        #print ' -- pairs = ', pairs
        return pairs

    def find_basic_standalone_bonds(self, dvi, dvs):
        """ find standalone valency-unsaturated bonds """
        ias1 = self.ias[ dvs==dvi ]
        #cns_dvi = np.zeros(self.na)
        #filt = (np.abs(dvs) >= dvi)
        #cns_dvi[filt] = np.sum(self.g[filt][:,filt],axis=0)
        #ias1 = self.ias[ np.logical_and(cns_dvi==1, dvs==dvi) ] # standalone implies cn=1

        _pairs = [] # store all pairs

        for i in ias1:
            # In the following line, abs(dvj) is used to include
            # cases involing multivalent elements, e.g., S in
            # >S([O])[O] for which a reference valency of 2 is used
            # ======================================
            # note that there is some code in famon.f90
            # of the same functionality, but the main
            # difference is: here we used `abs(dvs[j])
            # instead of dvs[j] alone (in famon.f90)
            # due to the reason that in famon.f90, all
            # reference total valencies are exact, i.e.,
            # they are derived from query molecule, which
            # have well-defined valency.
            jas0 = self.ias[ self.g[i]>0 ]
            jas = jas0[ np.abs(dvs[jas0])>=dvi ]
            naj = len(jas)
            #print ' ** i,jas0,dvsj,jas=',i,jas0,dvs[jas0],jas
            if naj == 1:
                j = jas[0]
                _pair = [i,j]; _pair.sort()
                _pairs.append( _pair )
                #break # return one pair each time
        return _pairs

    def find_next_standalone_bonds(self, dvi, dvs):
        """ get one remaining valency-unsaturated bond with
        the constituting atom satisfying abs(dv) >= dvi
        """
        if dvi == 0:
            _pairs = self.find_bond_to_N5(dvs)
            bo = 2
        else:
            _pairs = self.find_basic_standalone_bonds(dvi, dvs)
            bo = dvi + 1
        pairs = []
        # check if some common atoms are shared in the pairs
        # Exceptions include e.g., RN(=O)=O
        #_pairs = np.array(_pairs, np.int)
        #n = len(_pairs)
        #iok = True
        #for i in range(n):
        #    for j in range(i+1,n):
        #        iac = set(_pairs[i]).intersection(set(_pairs[j]))
        #        if iac != set([]) and self.zs[list(iac)[0]] not in [7,15,16,17]:
        #            # e.g., for HN-[CH]-[N]-NH-[CH]-[CH]-[CH2] or HN-[C]1-[N]-NH-[CH]-[CH]-[CH]1
        #            # two pairs share the same atom N_3, i.e., (C_2,N_3) and (N_3,N_4)
        #            iok = False
        #            break
        #    if not iok: break

        # remove pair with opposite charges, e.g., >[C-][N+]R3
        for pair in _pairs:
            i,j = pair
            if dvs[i]+dvs[j] != 0: pairs.append( pair )
        #print ' * pairs_u = ', pairs_u
        return bo, pairs

    def recursive_update_dvi(self, tvs, bom):
        for dvi in [1,2]:
            # Below a "while" clause is used due to the fact that
            # a saturation of valence in one cycle may result
            # in a new set of atoms with the same value of `dvi
            # E.g., for >C-C-C-C<, in cycle 1, atom C1 & C4 have
            # dvi=1, after saturation, the SMILES becomes
            # >C=C-C=C<; in cycle 2, atom C2 & C3 have dvi = 1,
            # and the SMILES is now modified to >C=C=C=C<. Done!
            while 1:
                dvs = tvs - bom.sum(axis=0)
                bo,pairs = self.find_next_standalone_bonds(dvi, dvs)
                if len(pairs) == 0:
                    break
                for pair in pairs:
                    i, j = pair
                    bom[i,j] = bo # dvi+1
                    bom[j,i] = bo # dvi+1
        return bom


    def update_charges(self,tvs,bom,chgs):
        """update charge of
        1) standalone valency-unsaturated atom
        2) -C(=O)=O -> -C(=O)[O-]
        3) all substructure containing v_N = 5
           e.g., -N$C -> -[N+]#[C-]
                 -N(=O)=O, ...
           Note that only -N(=O)=O, =N#N can be
           converted automatically to charged form by
           RDKit itself; all other structures, e.g.,
           C=[NH]=O,C=[NH]=NH, ... cannot be recognized!
        """
        na = self.na
        dvs = tvs - bom.sum(axis=0)
        #bom_aux = copy.deepcopy(bom)
        tvs_aux = copy.deepcopy(tvs)
        pairs1 = [] # oppositely charged atomic pairs with bo > 1
        for i in range(na):
            dvi = dvs[i]
            zi = self.zs[i]
            jas = self.ias[self.g[i]>0]
            if dvi != 0:
                dvsj = dvs[jas]
                if np.all(dvsj==0):
                    #assert abs(dvi) == 1
                    filt6 = np.logical_or(self.zs[jas]==8, self.zs[jas]==16)
                    nn6 = filt6.sum()
                    #ncn1 = (self.dvs0[jas] == 1).sum()
                    if zi == 6:
                        if dvi==-1:
                            if nn6>=1:
                                # for R=O,S
                                #     -C(=O)=R -> -C(=O)[R-] (
                                #     [C](R3)=R -> C(R3)[R-]
                                jas1 = jas[filt6]
                                j = jas1[0]
                                chgs[j] = -1
                                bom[i,j] = bom[j,i] = 1
                                #bom_aux[i,j] = bom_aux[j,i] = 1
                                tvs_aux[j] += -1
                            else:
                                chgs[i] = -1 # >[C-]-
                                tvs_aux[i] += -1
                        elif dvi==2:
                            # -[C]-OR -> -[C-]=[O+]R
                            jas1 = jas[ self.zs[jas] == 8 ]
                            assert len(jas1) == 1
                            j = jas1[0]
                            chgs[i] = -1
                            chgs[j] = 1
                            bom[i,j] = bom[j,i] = 2
                            #bom_aux[i,j] = bom_aux[j,i] = 2
                            tvs_aux[i] += -1
                            tvs_aux[j] += 1
                        else:
                            print ' ** i, zi, dvi, dvs = ', i, zi, dvi, dvs
                            print ' ** tvs = ', tvs
                            raise '#ERROR'
                    elif zi == 7: # >[N+]<
                        if dvi == -1:
                            chgs[i] = 1
                            tvs_aux[i] += 1
                        else:
                            print ' ** i, zi, dvi, dvs = ', i, zi, dvi, dvs
                            raise '#ERROR'
                    else:
                        print msg
                        raise '#unknow case'
                else:
                    jas1 = jas[dvsj==-1*dvi]
                    if len(jas1) == 1: # >[C-][N+]R3
                        j = jas1[0]
                        pair = [i,j]; pair.sort()
                        if pair not in pairs1:
                            chgs[i] = -1 * dvi
                            chgs[j] = dvi
                            tvs_aux[i] += -1 * dvi
                            tvs_aux[j] += dvi
                            pairs1.append( pair )
        return bom, chgs, tvs_aux


    def get_pairs(self, tvs, bom):
        """ get pairs of atoms that have opposite charges """
        na = self.na
        dvs = tvs - bom.sum(axis=0)
        #bom_aux = copy.deepcopy(bom)
        chgs_aux = np.array([0]*na, np.int)
        pairs = []; bo_pairs = [] # oppositely charged atomic pairs with bo > 1
        for i in range(na):
            dvi = dvs[i]
            zi = self.zs[i]
            jas = self.ias[self.g[i]>0]
            zsj = self.zs[jas]
            if dvi == 0:
                if zi==7:
                    iok = False
                    #filt1 = np.logical_and(np.logical_or(zsj==6,zsj==8), self.cns[jas]==1)
                    if self.cns[i] == 2: # identify -N$C and -->  -[N+]#[C-]
                        jas1 = jas[ np.logical_and(zsj==6,self.cns[jas]==1) ]
                        if len(jas1)== 1:
                            iok = True
                    #elif self.cns[i] == 4: # identify >[C-][N+]R3
                    #    jas1 = jas[self.dvs0[jas]==1]
                    #    #print ' <<<<< ', i, filt1.sum()
                    #    if len(jas1) == 1:
                    #        iok = True
                    #else:
                    #    pass
                    if iok:
                        j = jas1[0]
                        boij = bom[i,j]
                        if boij > 1:
                            chgs_aux[i] = 1
                            chgs_aux[j] = -1
                            pair = [i,j]; pair.sort()
                            if pair not in pairs:
                                pairs.append(pair)
                                bo_pairs.append(boij-1)
        #print ' ***** ', chgs_aux
        return pairs, bo_pairs, chgs_aux


    def tocan(self, tvs, chgs, bom, hasN5=True):
        pairs, bo_pairs, chgs_aux = self.get_pairs(tvs, bom)
        for i,chg in enumerate(chgs):
            if chg != 0:
                assert chgs_aux[i] == 0
        if pairs is not None:
            for ipair,pair in enumerate(pairs):
                i,j = pair
                bom[i,j] = bom[j,i] = bo_pairs[ipair]
        can = 'None'
        chgs += chgs_aux

        sanitize = True
        if hasN5:
            _tvs = bom.sum(axis=0)
            for ia in range(self.na):
                if _tvs[ia] == 5 and chgs[ia] == 0:
                    sanitize = False
                    break
        if self.na < 100:
            blk = write_ctab(self.zs, chgs, bom, self.coords, isotopes=[], sdf=None)
            #print blk
            m = Chem.MolFromMolBlock(blk,sanitize=sanitize) #False)
            fmt = 'sdf'
        else:
            blk = write_pdb( (self.zs,self.coords,chgs,bom) )
            m = Chem.MolFromPDBBlock(blk,sanitize=sanitize) #False)
            fmt = 'pdb'
        try:
            can = Chem.MolToSmiles(m)
        except:
            raise '#ERROR: rdkit failed to convert?'
        return can


    def reconnect(self,tvs,bom):
        """ re-connect bond for which bo was set to 0 when perceiving g
        E.g., for a SMILES string OC12C3C4C1N4C32, angle(N_6,C_4,C_5) ~ 48 degree,
              so the longest bond (N_6,C_4) was revered when calling connect() in
              class RawMol. The consequence is that N_6 & C_4 have an valency of 1
              deficient. Now reconnect these two atoms
        """
        dvs = tvs - bom.sum(axis=0)
        for i in range(self.na):
            dvi = dvs[i]
            zi = self.zs[i]
            if (zi in [6,7]) and dvi==1:
                jas0 = self.ias[ np.logical_and(self.g0[i]>0, dvs==1) ]
                if len(jas0)==1:
                    j = jas0[0]
                    if bom[i,jas0[0]] == 0:
                        bom[i,j] = bom[j,i] = 1
                        dvs = tvs - bom.sum(axis=0)
        return bom


    def perceive_bond_order(self):
        """ bond order """
        zs = self.zs
        g = self.g
        na = self.na
        ias = self.ias
        vsr = self.pt.vsr
        cns = g.sum(axis=0)
        self.cns = cns

        # now initialize `bom and then update it
        bom1 = copy.deepcopy(g) # later change it to bond-order matrix
        tvs1 = copy.deepcopy(vsr)

        can = 'None'
        dvs = tvs1 - bom1.sum(axis=0)
        self.dvs0 = copy.deepcopy(dvs)

        chgs = np.array([0,]*na, np.int)
        if np.all(dvs==0):
            can = self.tocan(tvs1,chgs,bom1)
            return can

        # locate pair of atoms with opposite charges
        # of type [A;dvi=1]~[B;dvi=-1], e.g., >[C-][N+]R3
        pairs11 = []
        for i in range(na):
            dvi = dvs[i]
            if dvi in [1,-1]:
                jas = ias[self.g[i]>0]
                filt = (dvs[jas] == -1 * dvi)
                if np.any(filt):
                    if filt.sum() != 1:
                        print ' dvs = ', dvs
                        print '#ERROR: more than one pair of (+1,-1)??', i,jas[filt]
                        sys.exit(2)
                    j = jas[filt][0]
                    bom1[i,j] += 1 # later update charge
                    pair = [i,j]; pair.sort()
                    pairs11.append(pair)

        # fix ~N-C, ~[X]-O, ~[X]-N, ~[X]-S, ~[X]-P
        for i in range(na):
            zi = zs[i]
            if cns[i] == 1 and np.any(zi == [6,7,8,15,16]):
                j = ias[g[i]>0][0]
                boij = {6:4, 7:3, 8:2, 15:3, 16:2}[zi]
                bom1[i,j] = boij
                bom1[j,i] = boij
                #print ' * hit'

        debug = T
        if debug:
            print ' *1 zs = ', zs
            print ' *1 tvs1 = ', tvs1
            print ' *1 bom1.sum(axis=0) = ', bom1.sum(axis=0)
            print ' *1 g = ', g

        if np.all(tvs1-bom1.sum(axis=0)==0):
            can = self.tocan(tvs1,chgs,bom1)
            return can

        # fix all valencies of atoms that have dvi == 1 or 2
        _bom = copy.deepcopy(bom1)
        bom1 = self.recursive_update_dvi(tvs1, _bom)
        dvs = tvs1 - bom1.sum(axis=0)

        if debug:
            print ' *2 zs = ', zs
            print ' *2 tvs1 = ', tvs1
            print ' *2 bom1.sum(axis=0) = ', bom1.sum(axis=0)

        if np.all(tvs1-_bom.sum(axis=0)==0):
            can = self.tocan(tvs1,chgs,bom1)
            return can

        _tvs = [ [vi] for vi in tvs1 ]
        # now update the values of `_tvs1 for multi-valent atom
        for ia in range(na):
            zi = self.zs[ia]
            dvi = dvs[ia]
            msg = 'Unknow valence  ia=%d, zi=%d, dvi=%s'%(ia,zi,dvi)
            if zi in [7,]:
                # dv = 3-4 = -1
                #    -[N*]-[N] (new: -[N*]#N)
                #    -NX3
                # dv = 3-5 = -2
                #    -N([O])[O] (new: R-N(=O)=O)
                #    -N([C])  (new: R-N$C)
                #    >C-[N*]-[N] (new: >C=N#N)
                if dvi < 0:
                    if dvi == -1:
                        if cns[ia] in [3,2]: # cn=3: >N[O] (new: >N=O; eventually: =[NR]=O)
                            _tvs[ia] = [5]
                        elif cns[ia] == 4:
                            if np.any(ia==np.array(pairs11,np.int)):
                                _tvs[ia] = [5]
                            #else:
                            #    pass # >[N+]< (correct charge later) #pass #_tvs[ia] = [3]
                        else:
                            print msg
                            raise '#ERROR'
                    elif dvi == -2:
                        _tvs[ia] = [5]
                    else:
                        print msg
                        raise '#ERROR'
                else: #if dvi > 0:
                    pass
            elif zi in [15]:
                # dv = 3-5 = -2
                #     PR5
                #     >P(=O)-
                # dv = 3-4 = -1
                #     P-[P*]#P
                if dvi < 0:
                    if dvi in [-1,-2]:
                        _tvs[ia] = [5]
                    else:
                        print msg
                        raise '#ERROR'
                elif dvi > 0:
                    print msg
                    raise '#ERROR'
            elif zi in [16,]:
                # dvi = 2-3= -1
                #    >S[C]<
                #    >[C]S([C]<)[C]<, =[C](-R)S([C](R)(=R'))=C(R)R'
                # dvi = 2-4 = -2
                #    >S[O] (new: >S=O)
                #    >S([C]<)[C]<
                # dvi = 2-5 = -3
                #    -S([O])[O] (new: -S(=O)=O)
                #    >S([C]<)[O] (new: >S([C]<)=O)
                # dvi = 2-6 = -4
                #    >S([O])[O] (new: >S(=O)=O)
                #    [O]S([O])[O] (new: O=S(=O)=O)
                if dvi <= -3:
                    _tvs[ia] = [6]
                else:
                    if dvi == -2:
                        if cns[ia] == 3:
                            # >S=O, >S=C<, ..
                            _tvs[ia] = [4]
                        elif cns[ia] == 4:
                            _tvs[ia] = [6]
                        else:
                            print msg
                            raise '#ERROR'
                    elif dvi == -1:
                        # =========================
                        _tvs[ia] = [6,4]
                        # =========================
                    else:
                        print msg
                        raise '#ERROR'
            else:
                pass # normal cases

        #print zs
        #print dvs
        #print _tvs

        nrmax = na/2
        nbmax = (g>0).sum()/2
        icon = False
        # if any of tvs[i] is a list, multiple choices of
        # `tvs exist, we have to try each of them
        tvs = cm.products(_tvs)
        istat = False
        for tvsi in tvs:
            _bom1 = copy.deepcopy(bom1)
            _bom2 = self.reconnect(tvsi,_bom1)
            _bom,_chgs,tvsi_aux = self.update_charges(tvsi,_bom2,chgs)

            if debug:
                print _bom
                print ' * tvsi = ', tvsi
                print ' * tvsi_aux = ', tvsi_aux
                print ' * _bom.sum() = ', _bom.sum(axis=0)
                print ' * _chgs = ', _chgs
            #if np.all(tvsi+_chgs-_bom.sum(axis=0)==0):
            if np.all(tvsi_aux-_bom.sum(axis=0)==0):
                can = self.tocan(tvsi, _chgs, _bom)
                return can

            # update bom
            iok, bom = cf.update_bom(nrmax,nbmax,zs,tvsi_aux,_bom,icon)
            if iok: # and np.all(tvsi+chgs-bom.sum(axis=0)==0):
                istat = True
                break

        if istat:
            can = self.tocan(tvsi_aux,_chgs,bom)
        else:
            print ' [failure]'
        return can


## test!

if __name__ == "__main__":
    import sys
    import ase.io as aio
    import tempfile as tpf
    import cheminfo.rdkit.RDKit as cir

    args1 = sys.argv[1:]

    trial = True
    if '-d' in args1 or '-debug' in args1:
        trial = False
        for _idx,_arg in enumerate(args1):
            if _arg[:2] == '-d':
                break
        args = args1[_idx+1:]
    else:
        args = args1

    n = len(args)
    if n == 0:
        objs = ["C=C=S(=C=C)=[N+]=[N-]", \
                "S1(=C)(C)=CC=CC=C1", \
                "[N+]1([O-])=CC=C[NH]1", \
                "C[N+](=O)[O-]", \
                "C=[NH+][O-]", \
                "C[N+]#[C-]", \
                "C[NH2+][C-](C)C", \
                "[NH3+]CC(=O)[O-]", \
                "C[O-]",\
                "C[NH3+]",\
                "[CH-]=[O+]C", \
                "N=[NH+][NH-]", \
                "[NH-][NH+]=C1C=C(C=C)C=C1", \
                "OC12C3C4C1N4C32"] # the last one is highly strained, may have problem in acquring g0
    elif n == 1:
        f = args[0]
        if f[-3:] in ['smi','can']:
            objs = [ si.strip() for si in file(f).readlines() ]
        else:  # either an xyz file or a SMILES string
            objs = args
    else:
        objs = args

    isf = False
    nobj = len(objs)
    for i,obj in enumerate(objs):
        if not os.path.isfile(obj):
            f = tpf.NamedTemporaryFile(dir='/tmp').name + '.xyz'
            m0 = cir.RDMol(obj, doff=True)
            m0.write_xyz(f)
        else:
            isf = True
            f = obj
        atoms = aio.read(f)
        can = 'None'
        if nobj > 1:
            if trial:
                try:
                    m = Mol(atoms.numbers, atoms.positions, ican=True)
                    can = m.can
                except:
                    pass
            else:
                m = Mol(atoms.numbers, atoms.positions, ican=True)
                can = m.can
        else:
            print f
            m = Mol(atoms.numbers, atoms.positions, ican=True)
            can = m.can

        if isf:
            print i+1, f, can
        else:
            print i+1, f, obj, can

