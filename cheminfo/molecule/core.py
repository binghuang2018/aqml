#!/usr/bin/env python

import os, sys, re, copy
import numpy as np
try:
  import indigo
except:
  pass
import aqml.cheminfo as co
import aqml.cheminfo.graph as cg
import networkx as nx
import aqml.cheminfo.molecule.geometry as cmg
from aqml.cheminfo.molecule.elements import Elements

#__all__ = [ 'rawmol_indigo' ]

T,F = True,False


class Graph(object):

    def __init__(self, g):

        g1 = (g > 0).astype(np.int)
        np.fill_diagonal(g1, 0)
        self.g1 = g1
        self.nb = g1.sum()/2 # num of bonds (i.e., edges)
        self.bonds = [ list(edge) for edge in \
             np.array( list( np.where(np.triu(g)>0) ) ).T ]
        self.gnx = nx.from_numpy_matrix(g1)

    @property
    def is_connected(self):
        if not hasattr(self, '_ic'):
            self._ic = nx.is_connected(self.gnx)
        return self._ic

    def get_pls(self):
        """ calc shortest path lengths """
        _na = g.shape[0] # number_of_nodes
        pls = -1 * np.ones((_na, _na))
        np.fill_diagonal(pls,[0]*_na)
        for i in range(_na):
            for j in range(i+1,_na):
                if nx.has_path(self.gnx,i,j):
                    pls[i,j]=pls[j,i]=nx.shortest_path_length(self.gnx,i,j)
        return pls

    @property
    def pls(self):
        if not hasattr(self, '_pls'):
            self._pls = self.get_pls()
        return self._pls

    def get_shortest_path(self, i, j):
        return list( nx.shortest_path(self.gnx, i, j) )

    def get_paths(self, i, j):
        """ return shortest paths connecting tow nodes i & j """
        paths = []
        for p in nx.all_shortest_paths(self.gnx, source=i, target=j):
            paths += list(p)
        return paths






class RawMol(Graph):

    def __init__(self, obj, ivdw=False, scale=1.0, iprt=F):
        """ initialize a molecule"""
        if isinstance(obj,(tuple,list)):
            zs, coords = obj
        else:
            sa = obj.__str__()
            if ('atoms' in sa) or ('molecule' in sa):
                zs, coords = obj.zs, obj.coords
            else:
                raise Exception('#ERROR: input obj not supported')
        self.obj = obj
        self.iprt = iprt
        self.scale = scale
        self._coords = np.array(coords)
        self._zs = np.array(zs, np.int)
        self.symbols = [ co.chemical_symbols[zi] for zi in zs ]
        self._na = len(zs)
        self._ias = np.arange(self._na)
        self.pt = Elements( list(zs) )

        self.cns0 = np.array([ co.cnsr[zi] for zi in self._zs ], dtype=int)

        self.connect()
        #if (not self.is_connected) or ivdw:
        #    self.connect_vdw_inter(scale=scale)

    def connect(self):
        """
        establish connectivity between atoms from geometry __ONLY__
        """
        ps = self._coords
        rs = self.pt.rcs
        rmax = rs.max()
        ds = np.sqrt((np.square(ps[:,np.newaxis]-ps).sum(axis=2)))
        self.ds = ds
        ds2 = ds * ds
        rs1, rs2 = np.meshgrid(rs,rs)
        ds2r = (rs1 + rs2 + 0.45)**2
        # step 1) get a preliminary connectivity table g
        g0 = np.logical_and( ds2 > 0.16, ds2 <= ds2r )
        self.g0 = np.array(g0,np.int)
        g = g0.astype(np.int)
        cns = g.sum(axis=0)
        #print('cns=',cns)

        # step 2) refine the g
        maxnbs = self.pt.maxnbs
        for ia in range(self._na):
            zi = self._zs[ia]
            if zi == 1:
                if g[ia].sum() > 1:
                    jas = self._ias[g[ia]>0]
                    if 1 in self._zs[jas]:
                        ja = jas[ self._zs[jas]==1 ]
                    else:
                        # remove the longer bond
                        ds2i = ds2[ia,jas]
                        maxd = ds2i.max()
                        ja = jas[ds2i==maxd]
                    g[ia,ja] = 0
                    g[ja,ia] = 0
                    cns = g.sum(axis=0)
            else:
                if cns[ia] == 1: continue
                while 1:
                    gg = cmg.GraphGeometry(self.obj, g)
                    mbs3 = gg.get_angles([ia], 'degree')
                    angs = mbs3.values()
                    #print('angs=', angs)
                    angmin = 180.0 if len(angs) == 0 else np.min([ min(angs_i) for angs_i in angs ])
                    # for "C1=C2CN=C1NC2", min(angs) can reach 46.0 degree
                    #print( 'angmin=',angmin, angs)
                    if (cns[ia] > maxnbs[ia] or angmin < 45): # 50.0):
                        #some bond exceeds max valence
                        #now remove the bond with max bond length
                        jas = self._ias[g[ia]>0]
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
        #gnx = nx.from_numpy_matrix(g)
        #self.gnx = gnx
        #self.is_connected = nx.is_connected(gnx)
        Graph.__init__(self, g)

    @property
    def nscu(self):
        if not hasattr(self, '_nscu'):
            self._nscu = self.get_nscu()
        return self._nscu

    def get_nscu(self):
        cns = self.g.sum(axis=0)
        zs = self._zs

        scus = {}

        ias2 = self._ias[ np.logical_and(cns==2, zs==6) ] # now ...=C=C=...
        g1 = self.g[ias2][:,ias2]
        clqs = []
        if len(g1)>0:
            raise Exception('Todo: cannot distinguish between -C#C-C#C- and >C=C=C=C< ...')
            cns1 = g1.sum(axis=0)
            for cq in cg.Graph(g1).find_cliques():
                ias_end = ias2[cns1==1]
                ias_others = ias2[cns1>1]
                for ie in ias_end:
                    jas = []

        vis1 = set() # visited nodes
        ias1 = self._ias[ np.logical_and(cns==1, zs>1) ]
        for ia1 in ias1:
            if ia1 in vis1: continue
            z1 = zs[ia1]
            jas = self._ias[self.g[ia1]>0]
            if z1 in [8,16,34,52]:
                assert len(jas)==1
                ja = jas[0]
                nnbr1 = self.g[ja,ias1].sum()
                #nbrsj = self._ias[self.g[ja]>0]
                if nnbr1 == 1:
                    t = [ co.chemical_symbols[zi] for zi in [zs[ia1],zs[ja]] ]
                    seti = [ia1,ja]
                    scu = '{s[0]}={s[1]}'.format(s=t)
                elif nnbr1 == 2:
                    nbrsj = ias1[ self.g[ja,ias1]>0 ]
                    seti = [ja, nbrsj[0], nbrsj[1]]
                    zsj = zs[nbrsj]; zsj.sort()
                    zsi = [zsj[0], zs[ja], zsj[1]]
                    t = [ co.chemical_symbols[zi] for zi in zsi ]
                    scu = '{s[0]}={s[1]}={s[2]}'.format(s=t)
                else:
                    raise Exception('Todo... Possibly ClO4 group?')
            elif z1 in [6,]: # -[N+]#[C-]
                assert len(jas)==1
                seti = [ia1, jas[0]]
                scu = '[{}+]#[C-]'.format( co.chemical_symbols[zs[jas[0]]] )
            elif z1 in [1,9,17,35,53]:
                seti = [ia1]
                scu = co.chemical_symbols[z1]
            else:
                raise Exception('#ERROR: what the hell happend?')
            vis1.update(seti)
            ## now update counts
            if scu not in scus:
                scus[scu] = [ set(seti) ]
            else:
                scus[scu].append( set(seti) )

        iasr = np.setdiff1d(self._ias, list(vis1))
        for ia in iasr:
            scu = '%s%d'%( co.chemical_symbols[zs[ia]], cns[ia] )
            seti = [ia]
            if scu not in scus:
                scus[scu] = [set(seti)]
            else:
                scus[scu].append( set(seti) )
        return scus

    def get_fragments(self):
        mols = []
        if self.is_connected:
            mols = [ co.atoms(self._zs, self._coords) ]
        else:
            for sg in nx.connected_component_subgraphs(self.gnx):
                idx = list(sg.nodes())
                mols.append( co.atoms(self._zs[idx], self._coords[idx]) )
        return mols

    @property
    def fragments(self):
        if not hasattr(self, '_fragments'):
            self._fragments = self.get_fragments()
        return self._fragments


    def connect_vdw_inter(self): #,scale=1.0):
        """
        add vdw bond between standalone submols in the system
        """
        g2 = np.zeros((self._na,self._na))
        if not self.is_connected:
            rvdws = np.array([ self.pt.rvdws ])
            rs2max = (rvdws+rvdws.T) * self.scale
            nncb = 0
            for i in range(self._na):
                for j in range(i+1,self._na):
                    if self.pls[i,j] < 0:
                        if rs2max[i,j] >= self.ds[i,j]:
                            g2[i,j]=g2[j,i]=1; nncb += 1
            # now check CN of hydrogen atoms
            cns2 = g2.sum(axis=0)
            iash = self._ias[self._zs==1]
            nncb_r = 0 # number of bond to be removed
            for ia in iash:
                msg = '#ERROR: multiple bonds detected for H-%d'%(ia+1)
                if cns2[ia]>1: #msg
                    # now keep only the vdw bond with shortest length
                    jas = self._ias[g2[ia]>0]
                    dsj = self.ds[ia,jas]
                    seq = np.argsort(dsj)
                    _jas = jas[seq[1:]]
                    nncb_r += len(_jas)
                    g2[ia,_jas] = 0
                    g2[_jas,ia] = 0
                    cns2 = g2.sum(axis=0)
            # now break vdw bond between heavy atoms ??
            ias_heav = self._ias[self._zs>1]
            na_heav = len(ias_heav)
            for i in range(na_heav):
                for j in range(i+1,na_heav):
                    ia,ja = ias_heav[i],ias_heav[j]
                    g2[ia,ja]=g2[ja,ia]=0
            nncb = (g2>0).sum()/2
            #print( np.where( g2 > 0 ) )
            if self.iprt: print(' ## found %d non-covalent bonds'%nncb )
        self.g2 = g2


    def connect_vdw_intra(self):
        """ connect intramolecular vdw bonds
        E.g, hydrogen bond
        """
        raise Exception('Todo:')
        return


    def get_conj_envs(self):
        dvs = self.g.sum(axis=0) - self.cns0
        cond = (dvs < 0)
        iasc = self._ias[cond] # map relative idx in `sg to absolute idx in parent mol
        sg = self.g[cond][:,cond]
        sgnx = nx.from_numpy_matrix(sg)
        sub_graphs = nx.connected_component_subgraphs(sgnx)
        cliques = []
        for i, _sg in enumerate(sub_graphs):
            cliques.append( list(iasc[list(_sg.nodes())]) )
        #print('cliques=',cliques)
        cenvs = []
        for ic, csi in enumerate(cliques):
            nai = len(csi)
            if nai == 1:
                # e.g., O in O=PX3, O=[SX2]=O
                a1 = csi[0]
                for a2 in np.setdiff1d(self._ias[self.g[a1]>0], csi):
                    if (dvs[a2] != 0) and (a2 not in csi):
                        csi.append( a2 )
            else:
                # add neighboring atoms that are i) saturated and ii) one of N, O, P, S, As, Se
                # e.g., N in -C(=O)N<, P in c1c[PH]cc1
                for ia in csi:
                    for ja in self._ias[self.g[ia]>0]:
                        if (ja != ia) and (dvs[ja] == 0) and (self._zs[ja] in [7,8,15,16,33,34]) and (ja not in csi):
                            csi += [ja]
            if len(csi) > 1:
                #print('csi=', csi)
                csi.sort()
                cenvs.append(csi)
        nc = len(cenvs)
        for i in range(nc):
            for j in range(i+1,nc):
                if len( set(cenvs[i]).intersection( set(cenvs[j]) ) ) > 0:
                    raise Exception('#ERROR: some conj envs share the same atom??')
        return cenvs

    def get_acmap(self):
        """ map atom to idx of conj env
        must be called after calling get_conj_envs()
        """
        acmap = {} #cas = []
        for i,envs in enumerate(self.cenvs):
            acmap.update( dict(zip(envs, [i]*len(envs))) )
        return acmap

    @property
    def acmap(self):
        if hasattr(self, '_acmap'):
            cmap = self._acmap
        else:
            cmap = self.get_acmap()
        return cmap


    @property
    def cenvs(self):
        """
          conjugated environments
        """
        if hasattr(self, '_cenvs'):
            cenvs = self._cenvs
        else:
            cenvs = self.get_conj_envs()
        return cenvs


    def connect_conj(self):
        """
          connectivity matrix between conjugated atoms
        """
        gc = np.logical_or( self.g, self.g2 )
        for csi in self.cenvs:
            nai = len(csi) #cg[np.ix_(csi,csi)] = 1
            for i in range(nai):
                for j in range(i+1,nai):
                    gc[i,j] = gc[j,i] = 1
        self.gc = gc


    def get_slatm_nbrs(self, ias=None, itype='c'):
        """
        variable-cutoff SLATM

        get neighbors that are connected to `ia in SLATM representation
        based on graph.

        Principle:
        a) for C-, Si-sp3, include PL=1 neighbors only (i.e., covalent bond)
        b) for atom in conj env, include PL=2 neighbors

        Todo: voronoi diagram

        vars
        ===================
        itype : 'c' or 'f', for which idx starts with 0 and 1, respectively
        """
        ds2 = self.ds**2
        rs = self.pt.rvdws
        rs1, rs2 = np.meshgrid(rs,rs)
        ds2r = (rs1 + rs2)**2
        g3 = (ds2 <= ds2r) #np.logical_and( ds2 > 0.16, ds2 <= ds2r )
        g3 = np.array(g3, np.int)

        dvs = self.g.sum(axis=0) - self.cns0
        cas = [] # conjugated atoms set
        for envs in self.cenvs:
            cas += envs

        cnshv = np.array([ (self._zs[self.g[i]>0]>1).sum() for i in self._ias ], dtype=int)

        nbrs = []
        if ias is None:
            ias = self._ias
        else:
            ias = np.array(ias, dtype=int) #- 1 #
            if itype in ['f','fortran']:
                ias -= 1

        ias_only_nbrs = [] # atoms to which only bonded neighbors are considered to be connected
        g3s = []
        for i in ias:
            g3i = g3[i]
            cnbrs = [i]+list(self._ias[self.g[i]>0]) # connected neighbors
            nnbrs = list(np.setdiff1d(self._ias, cnbrs)) # non-connected neighbors
            if dvs[i] == 0:
                if i not in cas:
                    if self._zs[i] in [6,14]: # for C or Si-sp3, no matter what
                        ias_only_nbrs.append( i )
                        for j in nnbrs:
                            g3i[j] = 0
                            #print(' ** i,j=',i,j, ' set gij=0')
                    elif self._zs[i] in [8,]: # e.g., O in >[P(=O)]O[P(=O)]<
                        if cnshv[i] == 2:
                            # now find the neighbors which is in contact with `i and in a conjugated env
                            nbrs2 = []
                            for j in cnbrs:
                                if j in cas:
                                    nbrs2 += self.cenvs[ self.acmap[j] ]
                            for j in nbrs2:
                                g3i[j] = 1
                            for j in np.setdiff1d(self._ias, cnbrs+nbrs2):
                                g3i[j] = 0 # previously, it's set to 1 as r_ij < r_i^vdw + r_j^vdw
                                #print(' ***** i,j=',i,j, ' set gij=0')
            else:
                # e.g., >[PX]=O, >S(=O)(=O)
                if self._zs[i] in [15,16]:
                    #print('i=', i, 'cnbrs=',cnbrs)
                    ias_only_nbrs.append( i )
                    for j in nnbrs:
                        g3i[j] = 0
            g3s.append(g3i)

        nbrs = []
        print('ias_only_nbrs=', ias_only_nbrs)
        for i,g3i in enumerate(g3s):
            for j in ias_only_nbrs:
                if self.g[i,j] == 0 and (j!=i):
                    #print(' ++ i,j = ', i,j)
                    g3i[j] = 0
            nbrs_i = self._ias[g3i.astype(np.bool)]
            if itype in ['f']:
                nbrs_i += 1
            nbrs.append( list(nbrs_i) ) # + [-1]*(self._na-len(nbrs_i)) )
        return nbrs #np.array(nbrs, dtype=int) #dict(zip(ias+1, nbrs))

    def nbrs_slatm(self, i):
        assert i>0, '#ERROR: i starts from 1'
        return list(1+self._ias[ self.g_slatm[i-1] > 0 ])

    def connect_vdw(self):
        self.connect_vdw_inter()
        self.connect_vdw_intra()

    def connect_all(self):
        self.connect_vdw()
        self.connect_conj()


class rawmol_indigo(object):

    """
    build a Indigo mol object from scratch
    """

    def __init__(self, obj):
        assert isinstance(obj,(list,tuple)), "#ERROR: obj not a list/tuple?"
        assert len(obj)==4, "#ERROR: `obj should .eq. (zs,chgs,bom,coords)"
        zs, coords, chgs, bom = obj
        # unfortunately, Indigo cannot load file formats other than sdf/mol
        # so we manually construct molecule here
        na = len(zs)
        ias = np.arange(na).astype(np.int)
        newobj = indigo.Indigo()
        m = newobj.createMolecule()
        ats = []
        for ia in range(na):
            ai = m.addAtom(co.chemical_symbols[zs[ia]])
            ai.setXYZ( coords[ia,0],coords[ia,1],coords[ia,2] )
            ai.setCharge(chgs[ia])
            ats.append(ai)
        for ia in range(na):
            ai = ats[ia]
            jas = ias[ np.logical_and(bom[ia]>0, ias>ia) ]
            for ja in jas:
                bi = ai.addBond(ats[ja], bom[ia,ja])
        self.m = m

    def tocan(self, nostereo=True, aromatize=True):
        """
        one may often encounter cases that different SMILES
        correspond to the same graph. By setting aromatize=True
        aliviates such problem
        E.g., C1(C)=C(C)C=CC=C1 .eqv. C1=C(C)C(C)=CC=C1
        """
        m2 = self.m.clone()
        if nostereo: m2.clearStereocenters()
        if aromatize: m2.aromatize()
        can = m2.canonicalSmiles()
        return can

