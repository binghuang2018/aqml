

import aqml.cheminfo as co
import networkx as nx
import itertools as itl
import scipy.spatial.distance as ssd
import multiprocessing
import numpy as np
import ase.io as aio
import ase.data as ad
import ase, os, sys, re, copy
import aqml.cheminfo as co
import aqml.cheminfo.core as cc
import aqml.cheminfo.math as cim
import aqml.cheminfo.graph as cg
from aqml.cheminfo.molecule.elements import Elements
import aqml.cheminfo.molecule.core as cmc
import aqml.cheminfo.molecule.geometry as GM
import aqml.cheminfo.molecule.nbody as MB
import aqml.cheminfo.rw.ctab as crc
import functools as tools
import tempfile as tpf
from rdkit import Chem
import aqml.cheminfo.rw as crw
import aqml.cheminfo.rw.ctab as crwc

# allowed reference states for amons generation so far
# (you may considering modifying it to allow for more
#  diverse chemistries in the future. E.g., R[Cl](=O)(=O)(=O)
#  is not supported yet)
tvsr = { 1:[1],  4:[2],   5:[3], \
         6:[4],  7:[3,5], 8:[2],      9:[1], \
        14:[4], 15:[3,5],16:[2,4,6], 17:[1,3,5,7], \
        32:[4], 33:[3,5],34:[2,4,6], 35:[1,3,5,7], \
                51:[3,5],52:[2,4,6], 53:[1,3,5,7]}

## Taken from https://github.com/rdkit/rdkit/blob/master/rdkit/Chem/Lipinski.py
# SMARTS of HBond donoar #### aaaa
sma_hbd = '[$([N;!H0;v3]),$([N;!H0;+1;v4]),$([O,S;H1;+0]),$([n;H1;+0])]'
# changes log for HAcceptorSmarts:
#  v2, 1-Nov-2008, GL : fix amide-N exclusion; remove Fs from definition
sma_hba = '[$([O,S;H1;v2]-[!$(*=[O,N,P,S])]),' +\
           '$([O,S;H0;v2]),$([O,S;-]),$([N;v3;!$(N-*=!@[O,N,P,S])]),' +\
           '$([nH0,o,s;+0])]'

# reference coordination number
#cnsr = {1:1, 3:1, 4:2, 5:3, 6:4, 7:3, 8:2, 9:1, 13:3, 14:4, 15:3, 16:2, 17:1, 35:1, 53:1}

# reference X-H distance (X=heavy atom)
dsxh = {5:1.20, 6:1.10, 7:1.00, 8:0.98, 9:0.92, 14:1.48, \
        15:1.42, 16:1.34, 17:1.27, 35:1.41, 53:1.61}

T,F = True,False

THRESH_TOR_PLANAR = 6.0 # degree
THRESH_DHA_ANGLE = 90. # degree



class EGraph(object):

    """ Enhanced Graph object """

    def __init__(self, bom, tvs, nsh, debug=F):
        self.tvs = tvs
        self._bom = bom
        na = bom.shape[0]
        self.na = na
        self.ias = np.arange(na)
        g = (bom>0).astype(np.int)
        self.g = g
        self.cns = g.sum(axis=0)
        self.nsh = np.array(nsh, dtype = int)
        self.debug =debug

    def find_conjugate_chain(self, bom, tvs, nsh):
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
        na = bom.shape[0]
        ias = np.arange(na)
        ips = []
        irad = F
        g = (bom > 0).astype(int)
        gc = bom.copy()
        cns = g.sum(axis=0)
        bsv = [] # visited bonds
        icnt = 0
        while T:
            if self.debug: print('## cycle ',icnt+1, ' length of tvs, gc, nsh =', len(tvs),len(gc),len(nsh))
            dvs = tvs - (gc.sum(axis=0)+nsh)
            if self.debug:
                print('    vs= ', gc.sum(axis=0)+nsh, ' dvs= ', dvs)
            assert np.all(dvs<=1), '#ERROR: some dvi>=2?'
            if np.all(dvs==0):
                break
            _filt = (dvs > 0)

            # now check if the graph made up of the rest atoms is connected
            # VERY important step to account for the issue mentioned above!!!!!
            if cg.Graph(gc[_filt][:,_filt]).has_standalone_atom:
                if self.debug:
                    print('    ** graph now contains standalone atom, exit!')
                irad = T
                break

            f1 = (dvs>0)
            g1 = g[f1][:,f1]
            ias1 = ias[f1]
            cns1 = g1.sum(axis=0)
            f2 = (cns1==1)
            g2 = g1[f2][:,f2]
            ias2 = ias1[f2]
            if self.debug:
                print('    ias2 = ', ias2)
            if len(ias2) == 0:
                break
            for i,ia in enumerate(ias2):
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
                    if self.debug: print('    ##3, ', ip)
                    irad = T
                    break
            if self.debug: print('    ips= ', ips)
            if irad: break
            for ip in ips:
                ib,ie = ip
                gc[ib,ie] = gc[ie,ib] = 2
            icnt += 1

            ## why did I try to exit the program here??
            ## Need sometime to think about this!!
            ## Never use exit() in a while loop, please!
            ## Otherwise, it's very difficult to diagnose
            ## the program!!
            #if icnt == 3:
            #    print('########## strange case?????????')
            #    sys.exit(2)

        #if irad
        if self.debug:
            print('ips = ',ips)
        if len(ips) > 0:
            _ips = []
            for _ip in ips:
                ia,ja = list(_ip)
                _ips.append( [ia,ja] )
            _ips.sort()
        else:
            _ips = []
        return irad, _ips

    def find_bo2(self, _g, _tvs, _nsh):
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
            irad, ipsr = self.find_conjugate_chain(g,tvs,nsh)
            if self.debug:
                print(' irad, ipsr = ', irad, ipsr)
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
                if self.debug:
                    print(' ___ atsr = ', atsr)
                gtmp = g[atsr][:,atsr]
                go = cg.Graph(gtmp)
                if go.has_standalone_atom:
                    if self.debug:
                        print('    ** graph now contains standalone atom, exit!')
                    return [F,[]]
                else:
                    assert len(go.find_cliques()) == 1
                    bs = np.array( np.where( np.triu(gtmp)>0 ), dtype=int ).T
                    if self.debug:
                        print('++ bs = ', bs)
                    iok = T
                    for b in bs:
                        if self.debug: print('## b= ', b)
                        ib,ie = atsr[b]
                        g1 = g.copy()
                        g1[ib,ie] = g1[ie,ib] = 2
                        ips = ips1+[[ib,ie]]
                        dvs1 = tvs - (g1.sum(axis=0)+nsh)
                        f2 = (dvs1>0)
                        ats2 = ats[f2]
                        na2 = len(ats2)
                        irad, ips2 = self.find_conjugate_chain(g1,tvs,nsh) #g2,tvs2,nsh2)
                        if self.debug: print('    irad=', irad, ' ips2= ', ips2)
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

    @property
    def mesomers(self):
        if not hasattr(self, '_mesomers'):
            self._mesomers = self.get_mesomers()
        return self._mesomers


    @property
    def bs2_simple(self):
        """ explicit double bonds (simple), e.g., C=C in C=CC=C;
        while bonds in benzene are called non-simple  """
        if not hasattr(self, '_bs2'):
            ips2 = []
            # first, locate all conjugate atoms that are part of a chain
            irad, ips2 = self.find_conjugate_chain(self._bom, self.tvs, self.nsh)
            if self.debug:
                print(' irad, ips2 = ', irad, ips2)
            self._bs2 = ips2
        return self._bs2


    def get_mesomers(self):
        """
        get mesomeric structures as bond sets
        """
        mesomers = []
        bom = self._bom
        # first, locate all conjugate atoms that are part of a chain
        for ipr in self.bs2_simple:
            ia,ja = self.ias[ipr]
            bom[ia,ja] = bom[ja,ia] = 2

        # update `vs
        vs = bom.sum(axis=0)
        dvs = self.tvs - (vs + self.nsh)
        filt = (dvs == 1)
        if filt.sum() <= 1:
            return []
        g1 = self.g[filt][:,filt]
        cns1 = g1.sum(axis=0)
        tvs1 = self.tvs[filt]
        nsh1 = self.nsh[filt]
        ias1 = self.ias[filt]
        cs = cg.Graph(g1).find_cliques()
        if self.debug:
            print('cs= ', cs)
        nc = len(cs)

        iok = T # an indicator if the submol is fine in tvs, nsh and bom

        ipss = [] # ias (of) pairs's
        for _csi in cs:
            csi = np.array(_csi, dtype=int)
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
                #print('gi=', _gi)
                _cnsi = _gi.sum(axis=0)
                _nshi = nsh1[csi] + (self.cns[ias2]-_cnsi) # must-do!!
                is_rad, ipssr_i = self.find_bo2(_gi, tvs1[csi], _nshi)
                if self.debug:
                    print(' is_rad, ipssr_i = ', is_rad, ipssr_i)
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

        N = len(ipss)
        if N > 0 and iok:
            mesomers = cim.products(ipss)
        return mesomers


    def get_boms(self):
        """
        now update the input bom (i.e., self.bom)
        """
        bom = self._bom.copy()
        for ipr in self.bs2_simple:
            ia,ja = self.ias[ipr]
            bom[ia,ja] = bom[ja,ia] = 2
        # update bom
        self._bom = bom

        vs = self._bom.sum(axis=0)
        dvs = self.tvs - (vs + self.nsh)
        filt = (dvs == 1)
        na1 = filt.sum()

        boms = []
        if na1 % 2 == 1:
            return boms
        else:
            if na1 == 0:
                assert np.all(dvs == 0), '#ERROR: some dvi /= 1! (case 1)'
                boms.append( bom )
            else:
                for bs in self.mesomers:
                    bom_i = copy.copy( bom )
                    #print('   === bs = ', [ list(bsi) for bsi in bs ])
                    for i,bsi in enumerate(bs):
                        for bi in bsi:
                            ia1, ia2 = bi
                            bom_i[ia1,ia2] = bom_i[ia2,ia1] = 2 ## double bond
                    cnsi = bom_i.sum(axis=0)
                    dvsi = self.tvs - (cnsi + self.nsh)
                    #print('   === tvs, nsh, dvsi = ', self.tvs, self.nsh, dvsi)
                    #print('   === bonds = ', np.array( np.array(np.where(np.triu(bom_i>0)) ).T, dtype=int) )
                    assert np.all(dvsi==0), '#ERROR: some dvi > 0! (case 2)'
                    boms.append(bom_i)
        return boms

    @property
    def ats_meso(self):
        """
        get atoms that are part of a mesomeric ring (genericly aromatic system)
        Note that atoms are pruned, i.e., only those atoms that belong to any
        resonated ring is kept!

        Note that the structure below is not aromatic in nature
        but understood by OEChem to be aromatic!
        (A double bond is represented by either "===" or "//" or "\\")

                  ======
                 /      \
                /        \
               \\        /====
                \\______/
                        \\
                         \\

        """
        if not hasattr(self, '_ats_meso'):
            ats = []
            if len(self.mesomers) > 1:
                combs = []
                for bs in self.mesomers:
                    comb = [] #set()
                    for bsi in bs:
                        for _bi in bsi:
                            bi = list(_bi); bi.sort()
                            comb.append( tuple(bi) )
                    _comb = set(comb)
                    assert _comb not in combs
                    combs.append(_comb)
                # now find the common bonds
                comm = set.intersection( *combs )
                bsr = combs[0] - comm
                for b in bsr:
                    ats += list(b)
            self._ats_isomeric = ats
        return self._ats_isomeric

    @property
    def meso(self):
        if not hasattr(self, '_meso'):
            meso = np.array([F]*self.na, dtype=bool)
            for ia in self.ats_meso:
                meso[ia] = T
            self._meso = meso
        return self._meso

    @property
    def boms(self):
        if not hasattr(self, '_boms'):
            boms = self.get_boms()
            self._boms = boms
        return self._boms

    @property
    def istat(self):
        if not hasattr(self, '_istat'):
            self._istat = ( len(self.boms) > 0 )
        return self._istat



def create_rdmol(zs, chgs, bom, coords, sort_atom=F, sanitize=True, removeHs=False):
    """ create a RDKit molecule object from scratch info """
    om = crw.rawmol(zs, chgs, bom, coords)
    return om.build(sort_atom=sort_atom, sanitize=sanitize, removeHs=removeHs)


class newmol(object):

    """ construct a new mol """

    def __init__(self, zs, chgs, bom, coords=None, ds=None, pls=None, \
                 dsref=None, nprocs=1, scale_cov=1.0, scale_vdw=1.2, \
                 sanitize=True, removeHs=False, debug=F):
        """
        build a RDKit mol object from (zs, chgs, bom, coords)

        :param zs: nuclear charges (integers)
        :type zs: list
        :param chgs: formal atomic charges (integers)
        :type chgs: list
        :param bom: bond order matrix
        :type bom: array
        :param ds: interatomic distance matrix. Default: None.
        :type ds: array
        :param dsref:  (as reference) maximally allowed interatomic distance
                        of intramolecular vdw bond. E.g., {'hh':2.1, 'ho':2.28}. To be used when
                        determining if the mol is too crowded. Default: None
        :type dsref: dict
        """

        self.nprocs = nprocs
        #if isinstance(obj, list):
        #zs, chgs, bom, coords = obj
        na = len(zs)
        self.na = na
        self._ds = ds
        self._pls = pls

        # info -> sdf -> rdmol


        newm = create_rdmol(zs, chgs, bom, coords, sanitize=sanitize, \
                            removeHs=removeHs)
        self.zs = zs
        self.chgs = chgs
        self.bom = bom
        self.coords = coords

        ats = []
        for ai in newm.GetAtoms():
            ats.append(ai)

        self.mol = newm
        self.dsref = dsref
        self.scale_vdw = scale_vdw
        self.scale_cov = scale_cov
        self.debug = debug
        #self.tvs = bom.sum(axis=0)
        #self.ias = np.arange(na)
        #self.nav = (self.zs>1).sum()
        #self.iasv = self.ias[self.zs>1]
        #self.zsv = self.zs[self.iasv]
        #self.zsu = np.unique(zs)
        #self.g = (self.bom>0).astype(np.int)
        #self.cns = self.g.sum(axis=0)
        #self.cnsv = np.array([(self.zs[self.g[ia]>0]>1).sum() for ia in self.ias])
        #self.cnsvv = self.cnsv[self.iasv]


    def get_hs(self, iasv0, i3d=T):
        """ get all hydrogen atoms attached to heavy atoms,
        including the existing ones (in q) and newly appended
        ones (to saturate valency of heavy atoms)
        """
        _iasv0 = np.array(iasv0)
        ias0 = _iasv0[self.zs[_iasv0]>1]
        nav = len(ias0)

        # get coords of H's
        coords = []
        nh = 0
        icnt = nav
        bsxh = [] # X-H bonds
        bsxh_visited = []

        # existing H atoms (in query)
        for i,ia in enumerate(ias0):
            nbrs = self.ias[ np.logical_and(self.zs==1, self.bom[ia]>0) ]
            for ja in nbrs:
                bxh = set([ia,ja])
                if (bxh not in bsxh_visited):
                    bsxh_visited.append(bxh)
                    bsxh.append([i,icnt])
                    coords.append( self.coords[ja] )
                    icnt += 1
                    nh += 1

        # new H atoms added to saturate valence of heavy atoms
        for i,ia in enumerate(ias0):
            nbrs = np.setdiff1d(self.ias[ np.logical_and(self.zs>1, self.bom[ia]>0) ], ias0)
            for ja in nbrs:
                b = set([ia,ja])
                if (b not in bsxh_visited):
                    bsxh_visited.append(b)
                    bxh = [i,icnt]
                    bsxh.append(bxh)
                    if i3d:
                        coords_i = self.coords[ia]
                        v1 = self.coords[ja] - coords_i
                        dxh = dsxh[self.zs[ia]]
                        coords_j = coords_i + dxh*v1/np.linalg.norm(v1)
                    else:
                        coords_j = np.array([0., 0., 0.])
                    coords.append(coords_j)
                    icnt += 1
                    nh += 1

        return bsxh, coords


    def get_subm(self, iasv0, bom0=None, i3d=T):
        """
        get a submol given a list of atomic indices
        """
        _iasv0 = np.array(iasv0)
        iasv0 = _iasv0[self.zs[_iasv0]>1]
        zs = self.zs[iasv0]
        chgs =self.chgs[iasv0]
        coords = self.coords[iasv0]
        if bom0 is None:
            bom0 = self.bom[iasv0][:,iasv0]
        bsxh, coords_h = self.get_hs(iasv0, i3d=T)
        nh = len(coords_h)
        nav = len(iasv0)
        na = nh + nav
        bom = np.zeros((na,na))
        bom[:nav,:nav] = bom0
        if nh > 0:
            coords = np.concatenate((self.coords[iasv0], coords_h))
            for bxh in bsxh:
                i, j = bxh
                bom[i,j] = bom[j,i] = 1
            chgs = np.concatenate((chgs,[0]*nh))
            zs = np.concatenate((zs,[1]*nh))
        return newmol(zs, chgs, bom, coords)


    @property
    def gmeso(self):
        if not hasattr(self, '_gmeso'):
            nsh = self.cns - self.cnsv
            iasv = self.iasv
            dvsi = self.tvs[iasv] - self.cns[iasv]
            _bom = self.gv.copy()
            # first retain the BO's for bonds involving any multi-valent atom, i.e.,
            # atom with dvi>1. Here are a few examples that are frequently encountered:
            # 1) C2 in "C1=C2=C3" & "C1=C2=N3";
            # 2) N2 and N3 in "C1=N2#N3" ( -C=[N+]=[N-], -N=[N+]=[N-] )
            # 3) '-S(=O)(=O)-', -Cl(=O)(=O)=O,
            # By doing This, we can save a lot of work for BO perception later!
            #print 'tvs = ', tvs, ', vsi=',vsi+nsh, ', dvsi=', dvsi
            for i,ia in enumerate(iasv):
                if dvsi[i] > 1:
                    for ja in self.ias[self.bom[ia]>1]:
                        _bom[ia,ja] = _bom[ja,ia] = self.bom[ia,ja]
                        #print('ia,ja,bo=',ia,ja,self.bom[ia,ja])
            self._gmeso = EGraph(_bom, self.tvs[iasv], nsh[iasv], debug=self.debug)
        return self._gmeso

    @property
    def ias(self):
        if not hasattr(self, '_ias'):
            self._ias = np.arange(self.na)
        return self._ias


    @property
    def nav(self):
        if not hasattr(self, '_nav'):
            self._nav = (self.zs>1).sum()
        return self._nav


    @property
    def tvs(self):
        if not hasattr(self, '_tvs'):
            self._tvs = self.bom.sum(axis=0)
        return self._tvs


    @property
    def iasv(self):
        if not hasattr(self, '_iasv'):
            self._iasv = self.ias[self.zs>1]
        return self._iasv


    @property
    def zsv(self):
        if not hasattr(self, '_zsv'):
            self._zsv = self.zs[self.iasv]
        return self._zsv


    @property
    def zsu(self):
        if not hasattr(self, '_zsu'):
            self._zsu = np.unique(self.zs)
        return self._zsu

    @property
    def gv(self):
        if not hasattr(self, '_gv'):
            self._gv = self.g[self.iasv][:,self.iasv]
        return self._gv

    @property
    def g(self):
        if not hasattr(self, '_g'):
            self._g = (self.bom>0).astype(np.int)
        return self._g


    @property
    def cns(self):
        if not hasattr(self, '_cns'):
            self._cns = self.g.sum(axis=0)
        return self._cns


    @property
    def cnsv(self):
        if not hasattr(self, '_cnsv'):
            self._cnsv = np.array([(self.zs[self.g[ia]>0]>1).sum() for ia in self.ias])
        return self._cnsv

    @property
    def cnsvv(self):
        if not hasattr(self, '_cnsvv'):
            self._cnsvv = self.cnsv[self.iasv]
        return self._cnsvv

    @property
    def can(self):
        if not hasattr(self, '_can'):
            _ = Chem.MolFromMolBlock( Chem.MolToMolBlock(self.mol) )
            self._can = Chem.MolToSmiles(_, isomericSmiles=F)
        return self._can

    def write_ctab(self, f, dir='./'):
        """ write mol to file ends with, e.g., xyz, pdb, sdf ... """
        if not os.path.exists(dir):
            os.system('mkdir -p %s'%dir)
        if f is None:
            opf = tpf.NamedTemporaryFile(dir=dir).name + '.sdf'
        else:
            opf = dir+'/'+f
        self.opf = opf
        if os.path.exists(opf):
            os.system('cp %s %s-1'%(opf,opf))
        fmt = opf[-3:]
        crwc.write_ctab(self.zs, self.chgs, self.bom, coords=self.coords, sdf=opf)

    @property
    def symbols(self):
        if not hasattr(self, '_symb'):
            self._symb = [ co.chemical_symbols[zi] for zi in self.zs ]
        return self._symb

    @property
    def valences(self):
        """ total valences """
        if not hasattr(self,'_val'):
            self._val = self.bom.sum(axis=0) - self.chgs
        return self._val

    @property
    def ds(self):
        iok = hasattr(self,'_ds')
        if (not iok) or (iok and self._ds is None):
            self._ds = ssd.squareform( ssd.pdist(self.coords) )
        return self._ds

    def _union(self, sets):
        so = np.array([], dtype=int)
        for i,si in enumerate(sets):
            if len(si) > 0:
                so = np.union1d(so, list(si))
        return so #tools.reduce(np.union1d, sets)


    @property
    def iasr3(self):
        if not hasattr(self, '_iasr3'):
            _sets = [ x for x in self.rings if len(x) == 3 ]
            self._iasr3 = self._union( _sets )
        return self._iasr3

    @property
    def iasr4(self):
        if not hasattr(self, '_iasr4'):
            _sets = [ x for x in self.rings if len(x) == 4 ]
            self._iasr4 = self._union( _sets )
        return self._iasr4

    @property
    def iasr5(self):
        if not hasattr(self, '_iasr5'):
            _sets = [ x for x in self.rings if len(x) == 5 ]
            self._iasr5 = self._union( _sets )
        return self._iasr5

    @property
    def iasr6(self):
        if not hasattr(self, '_iasr6'):
            _sets = [ x for x in self.rings if len(x) == 6 ]
            self._iasr6 = self._union( _sets )
        return self._iasr6

    @property
    def iasr56(self):
        if not hasattr(self, '_iasr56'):
            iasmult = self.iasP5orS6
            iasr5 = self.iasr5
            iasr5nmv = np.setdiff1d(iasr5, iasmult)
            iasr6 = self.iasr6
            iasr6nmv = np.setdiff1d(iasr6, iasmult)
            self._iasr56 = np.union1d(iasr5nmv,iasr6nmv)
        return self._iasr56

    @property
    def i5r(self): # contains 5-membered ring?
        if not hasattr(self, '_i5r'):
            self._i5r = ( len(self.iasr5)>0 )
        return self._i5r

    @property
    def i6r(self): # contains 6-membered ring?
        if not hasattr(self, '_i6r'):
            self._i6r = ( len(self.iasr6)>0 )
        return self._i6r

    @property
    def nbo1(self):
        """ get number of bonds satisfying
        a) BO=1
        b) hyb=1 for both atoms in the bond """
        if not hasattr(self, '_nbo1'):
            nb = 0
            ias2 = set(list(self.iasv[self.hybsv==2]))
            bos1 = np.array(np.where(np.triu(self.bom[self.iasv][:,self.iasv])==1), dtype=int).T
            for bo1 in bos1:
                if not (set(bo1) <= ias2):
                    nb += 1
            self._nbo1 = nb
        return self._nbo1

    @property
    def resonated(self):
        """ normal resonace due to Pi-electron delocalization, e.g.,
        benzene ring """
        if not hasattr(self, '_resonated'):
            self._resonated = self.get_is_resonated()
        return self._resonated

    def get_is_resonated(self):
        """ a mol is resonated if the number of alternave structures .lt. 1 """
        import aqml.cheminfo.rdkit.resonance as crr
        obj = crr.ResonanceEnumerator(self.rdmol, kekule_all=T)
        tf = ( obj.nmesomers > 1 )
        return tf

    @property
    def resonated_cs(self):
        """ resonance due to Charge Seperation (cs)
        This exists mainly in aromatic mols containing -C(=O)N-
        """
        if not hasattr(self, '_resonated_cs'):
            self._resonated_cs = self.get_is_resonated_cs()
        return self._resonated_cs

    def get_is_resonated_cs(self):
        patt = '[#6;a](=[#8,#16])[#7;a]'
        tf, _ = is_subg(self.mol, patt)
        return tf

    # 5-membered aromatic ring with substitutes is also considered as resonated!

    def get_istrains(self):
        istrains = np.array([F]*self.nav, dtype=bool)
        _ias = set()
        for ats in self.atsr_strain:
            _ias.update(ats)
        istrains[list(_ias)] = T
        return istrains

    @property
    def istrains(self):
        if not hasattr(self, '_istrains'):
            self._istrains = self.get_istrains()
        return self._istrains

    @property
    def is_conj_amon(self):
        """
        is it a benzene ring (Ph) with attached functional groups (R) mesomeric?
        """
        if not hasattr(self, '_meso'):
            self._meso = self.get_is_conj_amon()
        return self._meso


    def get_is_conj_amon(self):
        """
        Tell if the molecule is a conjugated amon (to be selected for training) through
        mainly three criterias:
          a) are all heavy atoms conjugated or hyper-conjugated (via sigma or pi)?
             E.g., c6h5-CH3: sigma hyper-conj; c6h5-OH: pi hyper-conj
          b) resonated? I.e., number of resonance structures > 1
          c) contains 6-membered aromatic ring?

        :return: A list of all possible resonance forms of the molecule.
        :rtype: list of rdkit.Chem.rdchem.Mol
        """
        zsh = self.zs[self.zs>1]
        iconjs = self.iconjs
        nav = self.nav
        NI = 8 # 9
        if not (nav<=NI and all(iconjs) and np.any(self.aromatic) and self.i6r): # (self.i5r or self.i6r)):
            return F
        #if (self.i5r and self.i6r):
        #    return F # E.g., nucleobase A (Adenine) without -NH2 group
        if nav <= 7:
            return T
        patt = '[a]-[A;#6,#7,#8,#9,#16,#17]'
        imth, _idx = is_subg(self.mol, patt)
        idx = [ i[-1] for i in _idx ]
        n = len(idx)

        tf = F
        #if self.i5r:
        #    if n <= 2 and nav <= 8:
        #        if (iconjs==3).sum() <= 1:
        #            # allow at most 1 attached group interacting via sigma-hyperconj
        #            tf = T  # E.g., C-c1ccnN1(C=O)
        if self.i6r:
            if (self.resonated or self.resonated_cs):
                if n == 0:
                    tf = T
                elif n == 1:
                    #if self.resonated:
                    #    if all(iconjs==1):
                    #        tf = T # Ph-R (R is -CH=O, -CH=CH2, -C#N, -C=N#N, -N=N#N, ...)
                    #    else:
                    #        zs_hc = zsh[iconjs==2] #  hyperconjugated due to Lone-p-electron and Pi e
                    #        nhc = len(zs_hc)
                    #        # for R = {-C(=O)F, -C(=O)Cl, }
                    #        # Don't include Ph-C(=O)N<, as C(=O)N is not coplanar with Ph after optg
                    #        if nhc==1 and (zs_hc[0] in [9,16,17]):
                    #            tf = T
                    if self.resonated_cs:
                        tf = T
        return tf

    @property
    def strained(self):
        if not hasattr(self, '_strained'):
            self._strained = self.get_strained()
        return self._strained

    def get_strained(self):
        """ get the status of strain for each atom """
        _sets = []
        for atsi in self.rings: #self.iasr3, self.iasr4 ]
            if len(atsi) in [3,4,5,7]:
                _sets.append(atsi)
        iats = self._union(_sets)
        tfs = np.array([F]*self.na, np.bool)
        if len(iats) > 0:
            print('iats=',iats)
            tfs[iats] = T
        return tfs

    @property
    def aseobj(self):
        if not hasattr(self, '_ase'):
            self._ase = ase.Atoms(self.zs, self.coords)
        return self._ase

    def has_standalone_charge(self):
        """ is there any atom with standalone charge? """
        hsc = False
        iasc = self.ias[self.chgs!=0]
        nac = len(iasc)
        if nac > 0:
            chgsc = self.chgs[iasc]
            gc = self.g[iasc][:,iasc]
            cliques = cg.Graph(gc).find_cliques()
            for csi in cliques:
                if np.sum(chgsc[csi])!=0:
                    hsc = True
                    break
        return hsc

    def is_radical(self):
        """ is the mol a radical? """
        irad = False
        if sum(self.zs)%2 == 1:
            irad = True
        else:
            vs = self.bom.sum(axis=0) - self.chgs
            for i in range(self.nav):
                if vs[i] not in tvsr[ self.zs[i] ]:
                    print( self.zs[i], vs[i], tvsr[self.zs[i]])
                    irad = True
                    break
        return irad

    @property
    def saturated(self):
        if not hasattr(self, '_sat'):
            self._sat = self.get_is_saturated()
        return self._sat

    def get_is_saturated(self):
        sat = np.array([T]*self.na, dtype=np.bool)
        for i in self.iasv:
            if self.valences[i] not in tvsr[ self.zsv[i] ]:
                sat[i] = F
        return sat

    @property
    def aromatic(self):
        """
        Tell if a mol is aromatic by checking the existence of any aromatic atom
        """
        if not hasattr(self, '_aromatic'):
            #for ai in self.mol.GetAtoms():
            _aromatic = np.array([ ai.GetIsAromatic() for ai in self.mol.GetAtoms() ], np.bool)
            self._aromatic = _aromatic
        return self._aromatic




    dct_hyb = { Chem.rdchem.HybridizationType.SP3: 3, \
                Chem.rdchem.HybridizationType.SP2: 2, \
                Chem.rdchem.HybridizationType.SP: 1, \
                Chem.rdchem.HybridizationType.S: 0, \
                Chem.rdchem.HybridizationType.UNSPECIFIED: 0}

    @property
    def hybs(self):
        """ hybridization states """
        if not hasattr(self, '_hybs'):
            hybs = []
            ia = 0
            for ai in self.mol.GetAtoms():
                hi = ai.GetHybridization()
                #print('hyb=', hi)
                if self.zs[ia]==7 and self.cns[ia]==3:
                    hyb = 3
                    if ia in self.iasNsp2: hyb = 2
                else:
                    hyb = self.dct_hyb[hi]
                hybs.append(hyb)
                ia += 1
            hybs = np.array(hybs, dtype=int)
            self._hybs = hybs #[self.iasv]
        return self._hybs

    @property
    def hybsv(self):
        """ hybridization states """
        if not hasattr(self, '_hybsv'):
            self._hybsv = self.hybs[self.iasv]
        return self._hybsv

    @property
    def icisclose(self):
        """
        detect if the following cis-structure exists. Due to the fact that the two
        H's are too close, barely representing any local structure in query (typically
        a 5- or 6-membered ring, thus it's not gonna be considered as an amon!

                R2         R3
                 \        /
                  \______/
                  //    \\
          R1 ____//      \\___ R4
                 \        /
                  \      /
                   H     H               (Note: C atom in the chain may also be N atom)

            Just realized that this structure can be avoided by constraining not
            to break a 6-membered ring by removing two atoms!!
            Or alternatively, ensure that d(H,H) > 1.8 Ang is satisfied for any extracted
            local structure in the first place, without geom opt in any form (partial or full).
        """
        tf = F
        #ts = is_subg(self.mol, '[H][#6,#7;]~?~?[H]' )[1] # including >N-C=R (R=C<,N-,O)
        return tf


    @property
    def irddt(self):
        """
        Is this mol too rddtible to be chosen as an amon? T/F
        Typically for a rddtible amon, it is easy to predict its properties accurately.

         """
        if not hasattr(self, '_irddt'):

            irddt = F
            iash = self.iasv

            cobj = self.cobj # self.cobj_ext # _ext: HB included
            pls = cobj.pls

            zs = self.zs
            icjs = self.iconjs

            pls_heav = pls[iash][:,iash]
            L0 = np.max(pls_heav)
            print('  max path length = ', L0)

            iasmult = self.iasP5orS6
            iasr5 = self.iasr5
            iasr5nmv = np.setdiff1d(iasr5, iasmult)
            iasr6 = self.iasr6
            iasr6nmv = np.setdiff1d(iasr6, iasmult)
            iasr56 = np.union1d(iasr5nmv,iasr6nmv)

            cnsv5 = self.cnsv[iasr5nmv]
            cnsv6 = self.cnsv[iasr6nmv]
            cnsv56 = self.cnsv[iasr56]

            Lm = 3
            infd = T

            if infd and L0>=Lm:
                if len(iasr5nmv)>0:
                    if (np.any(cnsv5>3) or (cnsv5>=3).sum()>=2) or \
                            ((cnsv5>=3).sum()==1 and (self.cnsv>=3).sum()==2):
                        # if it's a 5-membered conj or non-conj ring with either
                        # a) more than 1 attached -R, or
                        # b) heavy degree of any atom >3, or
                        # c) two CN=3 atoms, one in 5-membered ring, the other not in any ring
                        #  return as rddtible directly
                        print(' r5 rddt')
                        irddt = T; infd = F

                    if infd and (np.any(cnsv6>3) or (cnsv6>=3).sum()>=2) or \
                            ((cnsv6>=3).sum()==1 and (self.cnsv>=3).sum()==2):
                        # if it's a 5-membered conj or non-conj ring with either
                        # a) more than 1 attached -R, or
                        # b) heavy degree of any atom >3, or
                        # c) two CN=3 atoms, one in 5-membered ring, the other not in any ring
                        #  return as rddtible directly
                        print(' r6 rddt')
                        irddt = T; infd = F

                if infd:
                    ic4s = np.logical_and(self.zsv==6, self.cnsvv==4)
                    icn3s = (self.cnsvv==3)
                    nc4 = ic4s.sum()
                    ncn3 = icn3s.sum()
                    #print(' ____ ', L0>=Lm,nc4>=1,len(iasr56)==0)
                    if (nc4>=1 or ncn3>=2) and len(iasr56)==0:
                        # I.e., CC(C)(C)-C(C)C is considered to be rddtible
                        print(' non-ring rddt')
                        irddt = T; infd = F

            # remove any 6- or 7-membered ring comprising of atoms that are all sp3-hyb
            if infd and np.all(icjs==0) and self.nav>=6 and (self.cnsvv>=2).sum()>=6:
                irddt = T; infd = F

            if not infd:
                self._irddt = irddt
                return self._irddt
            else:
                if L0 > Lm:
                    #raise Exception('Todo: current implementation not perfect!')
                    tfs = (np.triu(pls_heav) > Lm)
                    pairs = np.array( np.where(tfs), dtype=int ).T
                    #print('pairs=', pairs)
                    paths = []
                    for i,atsi in enumerate(pairs):
                        ia1, ia2 = [ int(vi) for vi in iash[atsi] ]
                        path = cobj.get_shortest_path(ia1,ia2)
                        paths.append(path)

                    # a) L0=Lm+1 and
                    # b) only one such path and
                    # c) such path comprises of atoms that are all conj (iconj could be 1,2)
                    impl4conj = F # is maximal pl .eq. 4 and associated atoms (along the path) are conj
                    atsr56 = list( set( list(iasr5) + list(iasr6) ) )
                    #print('path=',path)#'iconjs=',self.iconjs[path], 'atsr56=',atsr56)
                    if len(paths)==1 and L0==Lm+1:
                        path = paths[0]
                        atsrc = np.intersect1d(path,atsr56) # ats in ring, that are common to path[0] and atsr56
                        if np.any(icjs[path]==1) and len(atsrc)>0 and (cnsv56>=3).sum()<2:
                            impl4conj = T
                    print( '  impl4conj=',impl4conj)

                    # maxpl = 4, one end bond has to satisfy BO>1
                    # E.g., CCOC=O
                    impl4ebc = F
                    if L0==Lm+1 and self.nav<=5:
                        iebcs = []
                        #print('paths = ', paths)
                        for path in paths:
                            #print('      path=', path)
                            if len(path)==5: # PL=4
                                b1, b2 = path[:2], path[-2:]
                                iebc = F
                                #if (np.all(icjs[b1]==1) and np.any(zs[b1]==6)) or \
                                #        (np.all(icjs[b2]==1) and np.any(zs[b2]!=6)):
                                    # np.any(zs[b1/b2]==6) would exclude C=CC=CC
                                if np.all(icjs[b1]==1) or np.all(icjs[b2]==1):
                                    iebc = T
                                iebcs.append(iebc)
                        #print('  iebcs=',iebcs, 'L0=',L0, 'nav=',self.nav, self.zs)
                        if np.all(iebcs):
                            impl4ebc = T
                    print('  impl4ebc = ', impl4ebc)

                    impl4 = (not impl4conj) and (not impl4ebc) #and (not impl4ni6ccn4)

                    # contains two POX3/SO2X2 ?
                    imlt2 = np.any([ len( set(iasmult).intersection(set(path)) ) > 1 for path in paths ])
                    #print('imlt2=', imlt2)

                    # contains -C(=O)-NX-C(=O)- ?
                    inco2 = (self.inco2 and L0==4)
                    #print('inco2=', inco2)

                    # conj ring with -R containing no more than 1 sp3-hybridized heavy atom ?
                    iconjr = (self.iconjr and (self.iconjs!=1).sum()<=1)
                    #print('iconjr=',iconjr)

                    ############ Note the two lines of code below can be replaced by the lines
                    ############ containing the keywor `newm.ats_ict0ex2`
                    # non-planar conj chain with N_I<=6 ?
                    # covering X=C/C=C\C=Y, X,Y could be C,N,O,S
                    #ictnp = self.is_conj_torsions_planar
                    #inpc = (np.all(self.iconjs==1) and ictnp and self.nav<=6)

                    if impl4 and (not imlt2) and (not iconjr) and (not inco2): # and (not inpc):
                        irddt = T

            self._irddt = irddt
        return self._irddt

    @property
    def iconjr(self):
        """ if there exists 5 or 6-membered conjugated ring in the mol """
        if not hasattr(self, '_iconjr'):
            nca = (self.iconjs==1).sum()
            self._iconjr = ( (self.i5r and nca>=5) or (self.i6r and nca>=6) )
        return self._iconjr











    @property
    def ats_ncopl(self):
        """
        return a list of atoms that are ``assumed'' to be coplanar,
        but actually not. E.g.,a conformer of C=COC=O, in which the
        torsional angle of the first 4 atoms may be ~60 degree. Note
        that for its global minima, da=180 for all torsions, that's
        what we mean by ``assumed''
        """
        if not hasattr(self, '_ats_ncopl'):
            ias = []
            for t in self.torsions.keys():
                da = self.torsions[t]
                if np.all(self.iconjs[list(t)]>0) and da>THRESH_TOR_PLANAR \
                        and da<180.-THRESH_TOR_PLANAR:
                    ias.append(t)
            self._ats_ncopl = ias
        return self._ats_ncopl

    @property
    def irigid(self):
        if not hasattr(self, '_irig'):
            self._irig = self.get_is_rigid()
        return self._irig

    def get_is_rigid(self):
        irig = F
        if np.all(self.cnsv[ np.setdiff1d(self.iasv, self.iasr3456) ] <= 1):
            irig = T
        return irig

    @property
    def i_ncopl_rddt(self):
        """
        check if the subm (formed by `ats_ncopl) is rddtible or not?
        """
        irddt = F
        pl0 = self.plvmax
        nhv = self.nav
        iasv = self.iasv
        icjs = self.iconjs
        if self.ats_ncopl: #  nhv, pl0, iconjs, cnsv =  6 4.0 [2 1 1 1 1 3] [2 1 2 1 3 1]
            #print(' nhv, pl0, iconjs, cnsv = ', nhv, pl0, self.iconjs[iasv], self.cnsv[iasv])
            if pl0==3:
                if nhv==4:
                    # check if it's c=cc=c and its derivatives
                    # These guys turn planar after optg, so it's safe to remvoe them
                    print('    found bent \\\\__// and alike (e.g., cis C=NC=C), skip!')
                    irddt = T
            elif pl0<=4:
                # check if it's C=CC=C-R (R=C,N,O)
                # The reason to remove these guys is that they can be described accurately
                # by its smaller constitutes
                if nhv==5:
                    print('   found bent C=CC=C-R (R=C,N,O), skip!')
                    irddt = T
            elif pl0 <= 6:
                #icjs12 = np.logical_and( icjs[iasv]>0, icjs[iasv]<=2 )
                if nhv<=7 and np.all(icjs) and ((icjs[iasv]>1).sum()<2) and \
                        (self.cnsv[iasv]>2).sum()<=1:
                    print('    found bent C=C\C=C/C=C or alike, keep it!')
                else:
                    irddt = T
                    print('    `ats_ncopl, pl0<=5 but rddt (nhv>7 or other cases); skip!')
            else:
                irddt = T
                print('    pl0>6, skip!')
        return irddt

    @property
    def ats_meso6ic1(self):
        """ atoms that are in a 6-membered mesomeric ring """
        if not hasattr(self, '_ats_ar6ic1'):
            ats = []
            for ia in self.iasr6:
                if self.iconjs[ia]==1: # and self.aromatic[ia]:
                    ats.append(ia)
            self._ats_ar6ic1 = ats
        return self._ats_ar6ic1

    @property
    def ats_cr5(self):
        """ atoms that are in a 5-membered ring and conjugated (iconj=1/2) """
        if not hasattr(self, '_ats_cr5'):
            ats = []
            for ia in self.iasr5:
                if self.iconjs[ia] in [1,2]: #self.aromatic[ia]:
                    ats.append(ia)
            self._ats_cr5 = ats
        return self._ats_cr5

    @property
    def ats_cr6(self):
        """ atoms that are in a 6-membered ring and conjugated (iconj=1/2) """
        if not hasattr(self, '_ats_cr6'):
            ats = []
            for ia in self.iasr6:
                if self.iconjs[ia] in [1,2]: #self.aromatic[ia]:
                    ats.append(ia)
            self._ats_cr6 = ats
        return self._ats_cr6

    @property
    def ats_cr56(self):
        """ atoms that are in a 5-membered ring and conjugated (iconj=1/2) """
        if not hasattr(self, '_ats_cr56'):
            self._ats_cr56 = list( set(self.ats_cr5 + self.ats_cr6) )
        return self._ats_cr56


    @property
    def ats_ic4ex2(self):
        """ envs (as a list of atoms) that are conjugated with
        a) at least two atoms are attached to each
           of the two end sp2-hyb atoms and

        Note: N in envs -NO2, -C(=O)N- are interpretated as sp2-hyb
              by OEChem
        """
        if not hasattr(self, '_ats_ic4ex2'):
            patt = '[^2;!X1]~[^2]~[^2]~[^2;!X1]'
            ifd, idxs = is_subg(self.mol, patt )
            ats = []
            for idx in idxs:
                a,b,c,d = idx
                nbrs_a = np.setdiff1d(self.ias[self.g[a]>0], idx)
                nbrs_d = np.setdiff1d(self.ias[self.g[d]>0], idx)
                if len(np.intersect1d(nbrs_a, nbrs_d))>0:
                    continue
                iok = T
                tf1 = np.all(self.g[[a,a,b],[c,d,d]]==0)
                tf2 = (len(np.intersect1d(idx,self.iasN3))==0)
                if not (tf1 and tf2):
                    continue
                for (i,j) in itl.product(nbrs_a, nbrs_d):
                    if self.g[i,j] > 0:
                        iok = F
                        break
                if not iok:
                    continue
                else:
                    ats.append(idx)
            self._ats_ic4ex2 = ats
        return self._ats_ic4ex2


    @property
    def ats_ic4ex2c(self):
        """ envs (as a list of atoms) that are conjugated with
        a) at least two atoms are attached to each
           of the two end sp2-hyb atoms and
        b) the distance between the two end atoms is the shortest
           among all distances of pairs of atoms that are made up
           of (e1/e2, a2), where e1/e2 is one of the end atoms and
           `a2 is the other atom that satisfies PL(a2,e1/e2) >= 3
        """
        if not hasattr(self, '_ats_ic4ex2c'):
            ats = []
            for idx in self.ats_ic4ex2:
                a,b,c,d = idx
                ats1 = np.setdiff1d(self.ias[self.g[a]>0], idx)
                nbrs_a = ats1[ self.cns[ats1] == 1 ]
                ats2 = np.setdiff1d(self.ias[self.g[d]>0], idx)
                nbrs_d = ats2[ self.cns[ats2] == 1 ]
                ispair = F
                for (i,j) in itl.product(nbrs_a, nbrs_d):
                    dij = self.ds[i,j]
                    #nbrs_i = np.setdiff1d(self.ias[self.pls[i]>=3], [j])
                    #nbrs_j = np.setdiff1d(self.ias[self.pls[j]>=3], [i])
                    #if len(nbrs_i)==0 or len(nbrs_j)==0:
                    #    continue
                    #ds1 = self.ds[i, nbrs_i]
                    #d1 = np.min(ds1)
                    #ds2 = self.ds[j, nbrs_j]
                    #d2 = np.min(ds2)
                    #ias1 = nbrs_i[ds1<dij]; na1 = len(ias1)
                    #ias2 = nbrs_j[ds2<dij]; na2 = len(ias2)
                    ##print('i,j,dij=',i,j,dij, 'd1,d2=',d1,d2)
                    #ias1_rm = []
                    #for ia1 in ias1:
                    #    if self.pls[i,ia1] <= 4:
                    #        ias1_rm.append(ia1)
                    #ias2_rm = []
                    #for ia2 in ias2:
                    #    if self.pls[j,ia2] <= 4:
                    #        ias2_rm.append(ia2)
                    #ias1f = np.setdiff1d(ias1, ias1_rm)
                    #ias2f = np.setdiff1d(ias2, ias2_rm)
                    #print('  ###', ias1f,ias2f)
                    #if len(ias1f)==0 and len(ias2f)==0:
                    if dij <= self.rsvdw_ref[i,j] * 1.15: # scaled by 1.15
                        ispair = T
                        break
                if not ispair:
                    continue
                else:
                    ats.append(idx)
            self._ats_ic4ex2c = ats
        return self._ats_ic4ex2c

    @property
    def ats_ict0ex2(self):
        if not hasattr(self, '_ats_ict0ex2'):
            self._ats_ict0ex2 = self.get_ats_ict0ex2() #THRESH_TOR_PLANAR)
        return self._ats_ict0ex2

    def get_ats_ict0ex2(self, thresh_tor_planar=30.0):
        """ envs (as a list of atoms) that contain
        a) `ats_ic4ex2
        b) conjugated torsion with dihedral angle close to 0 (<= 6. degree)
        """
        ats = []
        if self.ats_ic4ex2:
            for idx in self.ats_ic4ex2:
                a,b,c,d = idx
                ang = self.get_dihedral_angle(idx)
                #print('  ang=', ang)
                if ang < thresh_tor_planar: #THRESH_TOR_PLANAR:
                    ats.append(idx)
        return ats

    @property
    def ats_ict0ex2gc(self):
        """ envs (as a list of atoms) that contain
        a) `ats_ict0ex2
        b) geometry clash (gc) is assumed to happen if the
           two middle atoms in the conj torsion are not part
            of a strained ring (e.g., 3, 4, or 5-membered ring)
        """
        if not hasattr(self, '_ats_ict0ex2gc'):
            ats = []
            if self.ats_ict0ex2:
                for idx in self.ats_ict0ex2:
                    a,b,c,d = idx
                    if set(idx) <= self.iasra:
                        continue
                    ats1 = np.intersect1d([b,c], self.iasr345)
                    #ats2 = np.intersect1d([b,c], self.iasr6)
                    if len(ats1)==0:
                        ats.append(idx)
            self._ats_ict0ex2gc = ats
        return self._ats_ict0ex2gc


    @property
    def ats_ict0ex2vb(self):
        """ envs that are `ats_ict0ex2 and
        c) there exist one pair of end X atoms that satisfies
           d(x1,x2) < rvdw(x1) + rvdw(x2)
        """
        if not hasattr(self, '_ats_ict0ex2vb'):
            ats = []
            for idx in self.ats_ict0ex2:
                iasnb1 = np.setdiff1d(self.ias[self.bom[idx[0]]>0], idx)
                iasnb2 = np.setdiff1d(self.ias[self.bom[idx[-1]]>0], idx)
                if len( np.intersect1d(iasnb1,iasnb2) ) > 0:
                    # skip five membered ring
                    continue
                dspi = self.ds[iasnb1][:,iasnb2]
                dspiref = self.rsvdw_ref[iasnb1][:,iasnb2] #* 1.12 # scaled by 1.120
                iaspi = np.array( np.where(dspi < dspiref), dtype=int ).T
                #print('iaspi= ', iaspi)
                # now check if the end two atoms are bonded
                if len(iaspi) > 0:
                    for iasp in iaspi:
                        i1, i2 = iasp
                        ib, ie = iasnb1[i1], iasnb2[i2]
                        if (self.bom[ib,ie]==0):
                            #print('ib,ie=', ib,ie)
                            ats.append(idx)
            self._ats_ict0ex2vb = ats
        return self._ats_ict0ex2vb

    @property
    def ict0ex2vb(self):
        """ is there any env as in `ats_ict0ex2vb """
        if not hasattr(self, '_ict0ex2'):
            self._ict0ex2vb = (len(self.ats_ict0ex2vb) > 0)
        return self._ict0ex2vb

    @property
    def ats_cc4bent(self):
        """ is it a cis- conjugated chain with 4 heavy atoms? """
        if not hasattr(self, '_ats_cc4bent'):
            tf = F
            ats = []
            if self.ats_ic4ex2:
                for atsi in self.ats_ic4ex2:
                    ang = self.get_dihedral_angle(atsi)
                    if ang > THRESH_TOR_PLANAR and ang < 90.:
                        ats.append( atsi )
            self._ats_cc4bent = ats
        return self._ats_cc4bent

    @property
    def i_ats3_conn_cc4bent(self):
        """ is there any atom that is i) sp3-hybridised and
        ii) connected to any of `ats_cc4bent """
        if not hasattr(self, '_i_ats3_cc4bent'):
            tf = F
            if self.ats_cc4bent:
                for atsi in self.ats_cc4bent:
                    istop = F
                    if self.nav == 4:
                        tf = T ## found \\___//, or structures alike
                        break
                    for aj in atsi:
                        nbrsj = self.iasv[ self.gv[aj] > 0 ]
                        print('nbrsj=', nbrsj, self.zs[nbrsj])
                        cnsj = self.cns[nbrsj]
                        cnsrj = np.array([ co.cnsr[zi] for zi in self.zs[nbrsj] ], dtype=int)
                        if np.any( cnsj==cnsrj ):
                            tf = T
                            istop = T
                            break
                    if istop:
                        break
            self._i_ats3_icc4bent = tf
        return self._i_ats3_icc4bent


    @property
    def iasP5orS6(self):
        """ get idx of atoms that are i) P with valency of 5 or ii) S with valency of 6 """
        if not hasattr(self, '_iasP5orS6'):
            self._iasP5orS6 = self.get_iasP5orS6()
        return self._iasP5orS6

    def get_iasP5orS6(self):
        iasPS = []
        ts = is_subg(self.mol, '[#15,#16;X4]' )[1]
        for tsi in ts:
            ia = tsi[0]
            if self.hybs[ia] == 3:
                iasPS.append( tsi[0] )
        return iasPS

    @property
    def inco2(self):
        """ if there exists env N-C(=O)-, where N could be sp2 or sp3 hybridized"""
        if not hasattr(self, '_inco2'):
            patt = '[#6](=[#8])~[#7]~[#6](=[#8,#16])'
            tf, _ = is_subg(self.mol, patt)
            self._inco2 = tf
        return self._inco2


    @property
    def iasNsp2(self):
        """ N atoms that are sp2-hyb but CN .eq. 3, i.e.,
        atoms that satisfy:
        i) bonded to aromatic atom or
        ii) in -N-C=O
        """
        if not hasattr(self, '_iasNsp2'):
            ias = set()
            #ts = is_subg(self.mol, '[#7;X3;A]~[a]' )[1]
            #for tsi in ts:
            #    ias.update( tsi[:1] )
            patt = '[#7;X3]~[#6]=[#8]'
            ts = is_subg(self.mol, patt)[1]
            for tsi in ts:
                ias.update( tsi[:1] )
            # N in -NO2 is sp2-hyb
            for ia in self.ias[ np.logical_and(self.zs==7, self.cns==3) ]:
                if self.bom[ia].sum() > 3:
                    ias.update([ia])
            # N-aromatic
            ias.update( self.ias[ tools.reduce(np.logical_and, (self.aromatic, self.zs==7, self.cns==3)) ] )
            self._iasNsp2 = ias
        return self._iasNsp2


    @property
    def iasN3(self):
        if not hasattr(self, '_iasN3'):
            self._iasN3 = self.get_iasN3()
        return self._iasN3

    def get_iasN3(self):
        """
        get atomic idx of N atoms satisfying
            a) CN = 3
            b) not in an aromatic env
            c) not =[NX]=, e.g., -NO2
        Purpose: to identify hyper-conjugated atoms (with
        normally conjugated atoms as neighbors) before calling
        function `get_iconjs()
        """
        iasN3 = []
        ts = is_subg(self.mol, '[#7;X3;A]' )[1] # including >N-C=R (R=C<,N-,O)
        for tsi in ts:
            ia = tsi[0]
            if self.tvs[ia]==3: #hybs[ia] == 2:
                # OEChem assigns a hyb of 2 to N in envs such
                # as >N-C=R, -N(=O)=O
                iasN3.append( tsi[0] )
        return iasN3

    @property
    def iconjs_native(self):
        """
        Determine the conjugation state of atoms by coordination number (CN)
        This automatically exclude -NO2, >P(=O)-, >[S(=O)=O] as conj env"""
        if not hasattr(self, '_iconjs2'):
            cnsr = np.array([ co.cnsr[zi] for zi in self.zs ], dtype=int)
            self._iconjs2 = (self.cns < cnsr).astype(np.int)
        return self._iconjs2

    @property
    def iconjs(self):
        """ is conjugated atomic env? """
        if not hasattr(self, '_iconjs'):
            self._iconjs = self.get_iconjs()
        return self._iconjs

    @property
    def iconjsv(self):
        """ is conjugated atomic env? """
        if not hasattr(self, '_iconjsv'):
            self._iconjsv = self.iconjs[self.iasv]
        return self._iconjsv

    def get_iconjs(self):
        """
        iconj - integer indicating the types of conjugation
                could be one of 0, 1, 2 or 3 (all atoms involved
                should be ideally co-planar
             0: not conjugated at all
             1: conjugated (normal case)
             2: p-pi conjugation, e.g., O-sp3 in Ph-OH, N in >N-C(=O)-
             3: sigma-pi conjugation, e.g., X in >C=C(X)-
        """
        _iconjs = np.logical_and(self.hybs<3, self.hybs>0)
        cas = self.ias[_iconjs] # conjugated atoms
        ncas = np.setdiff1d(self.iasv, cas) #non-conj atoms
        #print('cas=',cas, 'ncas=',ncas)
        iconjs = _iconjs.astype(int)
        for i in ncas:
            zi = self.zs[i]
            nbrsi = cas[ self.g[i,cas]>0 ]
            if len(nbrsi)==0: continue
            #print('i, zi, nnb, cnr=', i,zi,nnb,cc.cnsr[zi])
            #if (zi in [7,16,]): # for Ph-SH, Ph-NH2, conj between Ph and R is very effective
            #    if np.any(self.aromatic[self.ias[self.g[i]>0]]):
            #        iconjs[i] = 2
            if zi in [6,]: #14]:
                iconjs[i] = 3
            elif zi in [7,8,16,]: # 9,15,17,35,53]:
                if self.cns[i] == co.cnsr[zi]:
                    # hyper-conj via lone electron pair (p electron) and Pi electron
                    iconjs[i] = 2

        # reset `iconj to 0 for N in env -C(=O)N
        #for i in self.iasN3:
        #    iconjs[i] = 2
        return iconjs

    @property
    def iasra(self):
        """ all ring atoms (up to 7 membered ring) """
        if not  hasattr(self, '_iasra'):
             ats = set()
             for si in self.rings:
                 ats.update(si)
             self._iasra = ats
        return self._iasra


    @property
    def crings(self):
        """ rings with atoms conjugated """
        if not  hasattr(self, '_crings'):
            crs = []
            icjs = self.iconjs
            for r0 in self.rings:
                r = list(r0)
                if np.all( np.logical_or(icjs[r]==1, icjs[r]==2) ):
                    crs.append(r)
            self._crings = crs
        return self._crings

    @property
    def rings_v(self):
        """ rings of amon, with all possible vdw bonds included """
        if not  hasattr(self, '_ringsv'):
            self._ringsv = self.mext.rings
        return self._ringsv

    @property
    def crings_v(self):
        """ rings with atoms conjugated in an amon, with all possible vdw bonds included.
        Note that crings_v cannot be obtained by calling self.mext.crings, as the `iconjs
        value has changed when assigning a BO=1 to vdw bond """
        if not  hasattr(self, '_cringsv'):
            crs = []
            icjs = self.iconjs
            for r0 in self.rings_v:
                r = list(r0)
                if np.all( np.logical_or(icjs[r]==1, icjs[r]==2) ):
                    crs.append(r)
            self._cringsv = crs
        return self._cringsv


    @property
    def rings(self):
        """ rings with atoms conjugated """
        if not  hasattr(self, '_rings'):
            self._rings = self.get_rings()
        return self._rings

    def get_rings(self, namin=3, namax=7, remove_redudant=T):
        """ get ring nodes for ring
        made up of `namin- to `namax atoms

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
            ifd, idxs = is_subg(self.mol, pat_i )
            for idx in idxs:
                if idx:
                    #print('idx=',idx)
                    _set = set(idx)
                    if _set not in sets:
                        sets.append(_set)
        if remove_redudant:
            # now remove those rings that are union of smaller rings
            n = len(sets)
            sets_remove = []
            ijs = itl.combinations( list(range(n)), 2 )
            sets2 = []
            for i,j in ijs:
                set_ij = sets[i].union( sets[j] )
                if (set_ij in sets) and (set_ij not in sets_remove):
                    sets_remove.append( set_ij )
            sets2 = cim.get_compl(sets, sets_remove)
        else:
            sets2 = sets
        return sets2

    @staticmethod
    def is_part_of_smaller_ring(seti, sets):
        """ check if the ring atoms `seti are all part of
        any of the smaller rings """
        na = len(seti)
        _sets = []
        for setj in sets:
            if len(setj) < na:
                _sets.append(setj)
        _seti = list(seti)
        iss = []
        for a in _seti:
            ispart = F
            for setj in _sets:
                if a in setj:
                    ispart = T
                    break
            iss.append( ispart )
        return np.all(iss)

    @property
    def atsr_strain(self):
        if not hasattr(self, '_atsr_strain'):
            self._atsr_strain = self.get_atsr_strain()
        return self._atsr_strain

    def get_atsr_strain(self):
        atsr = self.rings #get_(namin=3, namax=7, remove_redudant=T)
        atso = []
        for i, _atsi in enumerate(atsr):
            na = len(_atsi)
            atsi = list(_atsi)
            # no matter what, if all atoms in this ring are constitutes
            # of even smaller ring, then exclude this ring
            if i>0 and newmol.is_part_of_smaller_ring(_atsi, atsr[:i]):
                continue
            else:
                tf = np.all(self.iconjs[atsi])
                if na == 5:
                    if not tf:
                        continue
                elif na == 6:
                    continue
                elif na > 6:
                    # include 7- or higher-membered ring iff the ring comprises
                    # atoms that are all conj
                    if not tf:
                        continue
            atso.append( _atsi )
        return atso

    @property
    def gobj(self):
        """ graph object """
        if not hasattr(self, '_gobj'):
            self._gobj = cg.Graph(self.g, nprocs=self.nprocs)
        return self._gobj

    @property
    def pls(self):
        iok = hasattr(self, '_pls')
        if (not iok) or (iok and self._pls is None):
            self._pls = self.gobj.pls
        return self._pls

    @property
    def plvmax(self):
        if not hasattr(self, '_plvmax'):
            self._plvmax =  np.max(self.pls[self.iasv][:, self.iasv])
        return self._plvmax

    @property
    def cobj(self):
        """ connectivity object, no hydrogen bond """
        if not hasattr(self, '_cobj'):
            self._cobj = cmc.Graph(self.g)
        return self._cobj

    @property
    def mext(self):
        """ extended mol object, with connectivity between
        atoms bonded by vdw interaction appended """
        if not hasattr(self, '_mext'):
            bomv = self.bom.copy()
            #for i,h,j in self.iasdha:
            for i,j in self.hbsv_ext:
                bomv[i,j] = bomv[j,i] = 1
            _mext = newmol(self.zs, self.chgs, bomv, self.coords, sanitize=False)
            self._mext = _mext
        return self._mext

    @property
    def cobj_ext(self):
        """ connectivity object, hydrogen bond included """
        if not hasattr(self, '_cobj_ext'):
            g2 = self.g
            if np.any(self.ghb > 0):
                g2 = np.logical_or(self.g>0, self.ghb>0).astype(int)
            self._cobj_ext = cmc.Graph( g2 )
        return self._cobj_ext

    @property
    def plshb(self):
        if not hasattr(self, '_plshb'):
            self._plshb = self.cobj_ext.pls
        return self._plshb


    @property
    def torsions(self):
        if not hasattr(self, '_tor'):
            obj = MB.NBody( (self.zs, self.coords), self.g, icn=F, unit='degree', iheav=T, idic4=T)
            tor = obj.get_dihedral_angles()
            self._tor = tor
        return self._tor




    @property
    def is_conj_torsions_planar(self):
        if not hasattr(self, '_ictp'):
            self._ictp = self.get_is_conj_torsions_planar()
        return self._ictp

    def get_is_conj_torsions_planar(self):
        iconjs = (self.hybsv < 3) # don't exclude N3 in >N-C(=O)-, as we did in get_is_conj_amon()
        if iconjs.sum() < 4: return T
        iasv = self.iasv[iconjs]
        gv = self.g[iasv][:,iasv]
        zsv = self.zs[iasv]
        cnsv = self.cns[iasv]
        #print('cnsv=',cnsv)
        coordsv = self.coords[iasv]
        cs = cg.Graph(gv).find_cliques()
        istat = T
        for csi in cs:
            #print('csi=',csi)
            gi = gv[csi][:,csi]
            zsi = zsv[csi]
            coordsi = coordsv[csi]
            cnsi = cnsv[csi]
            #print('cnsi=',cnsi)
            obj = MB.NBody( (zsi,coordsi), gi, icn=T, unit='degree', cns=cnsi) #
            mbs4 = obj.get_dihedral_angles()
            #print('mbs4=', obj.mbs4)
            _vs = []
            for k in mbs4:
                _vs += list(mbs4[k])
            vals = np.abs( np.array(_vs) )
            dvals = np.min( [ vals, np.abs(vals-180.) ], axis=0 )
            #print('vals=',vals, 'dvals=',dvals)
            self._dangs = vals
            if np.any(dvals > self.thresh_TOR_PLANAR):
                #print('  angs = ', vals)
                istat = F
                break
        return istat





    @property
    def L(self): # maximal length of a molecule
        return np.max(self.pls)

    @property
    def L0(self): # maximal length of a molecule (heavy atoms only)
        return np.max(self.pls[self.iasv][:,self.iasv])

    @property
    def H(self): # height of a molecule
        return self.nav/(1.0 + self.L)

    def sort(self):
        """
        In a mol, H atom may appear in between heavy atoms, causing trouble
        later. Now we sort atoms to fix the potential problems.
        """
        _ias = np.concatenate((self.ias[self.zs>1], self.ias[self.zs==1]))
        zs = self.zs[_ias]
        chgs = self.chgs[_ias]
        coords = self.coords[_ias]
        _bom = self.bom.copy()
        bom = _bom[_ias][:,_ias]
        self.mol = create_rdmol(zs, chgs, bom, coords)
        self.zs = zs
        self.chgs = chgs
        self.coords = coords
        self.bom = bom
        self.tvs = bom.sum(axis=0)
        self.isorted = T

    @property
    def b2a(self):
        """
        The correspendence between bond idx and atom pairs
        Note: returned idxs of atoms are relative, i.e.,
              for mol CH3OH with zs = [6,1,1,1,8,1],
              iasv = [0,4] (absolute idx),
              while bondsv=[0,1] (relative idx)
        """
        if not hasattr(self, '_b2a'):
            bonds = []
            for i in range(self.nav):
                for j in range(i+1, self.nav):
                    i0 = self.iasv[i]; j0 = self.iasv[j]
                    if self.bom[i0,j0] > 0:
                        bonds.append( (i,j) )
            self._b2a = bonds #, dtype=int)
        return self._b2a

    @property
    def a2b(self):
        """
        The correspondence between atom and its associated bonds
        Note: atomic idx is relative! (not absolute)
        """
        if not hasattr(self, '_a2b'):
            _a2b = -1 * np.ones((self.nav, 6), np.int) # assume maximally 6 bonds
            for i,ia in enumerate(self.iasv):
                icnt = 0
                for ja in self.iasv[self.gv[i]>0]:
                    j = np.where(ja==self.iasv)[0][0]
                    b = [i,j]; b.sort(); b = tuple(b)
                    ib = self.b2a.index(b)
                    _a2b[i,icnt] = ib
                    icnt += 1
            self._a2b = _a2b
        return self._a2b

    def get_ab(self):
        """
        -- For heav atoms only
        get atoms and bonds info
        a2b: bond idxs associated to each atom
        b2a: atom idxs associated to each bond
        """
        b2a = [] #np.zeros((self.nb,2), np.int)
        b2idx = {}
        ib = 0 # idx of bonds involving heavy atoms only
        for i in range(self.nav):
            for j in range(i+1,self.nav):
                if self.bom[i,j] > 0:
                    if self.zs[i] > 1 and self.zs[j] > 1:
                        b2idx['%d-%d'%(i,j)] = ib
                        b2a.append( [i,j] )
                        ib += 1
        b2a = np.array(b2a, np.int)
        a2b = -1 * np.ones((self.nav, 6), np.int) # assume maximally 6 bonds
        for ia in self.iasv:
            icnt = 0
            for ja in self.ias[ np.logical_and(self.bom[ia]>0,self.zs>1)]:
                pair = [ia,ja]; pair.sort()
                ib2 = b2idx['%d-%d'%(pair[0],pair[1])]
                assert ib2 <= ib  #np.all( self.zs[b2a[ib]] > 1 ):
                a2b[ia, icnt] = ib2
                icnt += 1
        return a2b, b2a

    def copy(self):
        return create_rdmol(self.zs, self.chgs, self.bom, self.coords, sort_atom=F)

    def clone(self):
        return self.copy()

    def check_valence_states(self):
        cnsDic = {5:[3], 6:[4], 7:[3], 8:[2], 9:[1], 14:[4], 16:[2,6], 17:[1]}
        bom = self.bom
        zs = []; chgs = []
        iok = True
        for ai in self.mol.GetAtoms():
            zi = ai.GetAtomicNum()
            if zi==1: continue
            chgi = ai.GetFormalCharge()
            nhi = ai.GetExplicitHCount() + ai.GetImplicitHCount()
            tvi = ai.GetValence()
            ia = ai.GetIdx()
            _bo = bom[ia]
            bosi = list( _bo[_bo>0] ); bosi.sort(reverse=True); sbo=['%s'%si for si in bosi]
            msg = ' ia=%d, zi=%d, tvi=%d, chgi=%d, bosi=%s'%(ia,zi,tvi,chgi,''.join(sbo))
            if tvi-chgi not in cnsDic[zi]:
                print(msg) # check if valency is ok
                iok = False
                break
        return iok

    def neutralise(self):
        """
        neutralise a molecule, typically a protein
        """
        raise Exception('To be implemented for rdkit. For OEChem, check ../oechem/core.py')
        m = self.copy()
        zs = self.zs
        numHs = []; tvs = []; atoms = []
        for ai in m.GetAtoms():
            atoms.append( ai )
            numHs.append( ai.GetExplicitHCount() + ai.GetImplicitHCount() )
            tvs.append( ai.GetValence() )

        self.get_charged_pairs()
        for i in range(self.na):
            ai = atoms[i]
            ci = self.charges[i]
            nhi = numHs[i]
            if ci != 0:
                if i not in self.cpairs.ravel():
                    msg = ' zi = %d, tvi = %d, ci = %d, neib = %d'%(self.zs[i], tvs[i], ci, cnsDic[zs[i]])
                    assert tvs[i] - ci == cnsDic[zs[i]], msg
                    if nhi == 0 and ci > 0:
                        # in the case of >[N+]<, i.e., N with CoordNum = 4
                        # we don't have to do anything
                        continue
                    ai.SetFormalCharge( 0 )
                    ai.SetImplicitHCount( nhi - ci )
                    print('i, zi, ci, nH = ', i, self.zs[i], ci, numHs[i])
                else:
                    print('atom %d in a bond like ~[A+]~[B-]'%i)
        # note that H attached to sp2-hybridized and charged N, e.g., N in =[NH2+] won't
        # be removed by OERemoveFormalCharge(), now we do this manually
        self.mol = m

    def annihilate_chgs(self):
        """
        chgs in some SMILES can be totally skipped.
        _____________________________________________
         group         SMILES          purged SMILES
        -------       --------        ---------------
        -NO2        -[N+](=O)[O-]        -N(=O)=O
        >CN2        >C=[N+]=[N-]         >C=N#N
        -NC         -[N+]#[C-]           -N$C (i.e., BO(N-C) = 4)
        _____________________________________________

        In cases like [NH3+]CCC(=O)[O-], the chgs retain.
        """
        na = self.na
        ias = np.arange(na)
        chgs = self.chgs
        zs = self.zs
        bom = self.bom
        if np.any(chgs!= 0):
            ias1 = ias[chgs==1]
            ias2 = ias[chgs==-1]
            bs = {} # store the idx of bonds for update of BO
            cs = {} # store the idx of atom for update of charge
            irev = False # revise `bom and `chgs?
            visited = dict( zip(range(na), [False,]*na) )
            for ia1 in ias1:
                for ia2 in ias2:
                    bo12 = bom[ia1,ia2]
                    if bo12 > 0:
                        assert not (visited[ia1] or visited[ia2]), \
                                  '#ERROR: visted before?'
                        visited[ia1] = visited[ia2] = True
                        bo12 += 1
                        irev = True
                        pair = [ia1,ia2]; pair.sort()
                        bs[ tuple(pair) ] = bo12
                        bom[ia1,ia2] = bom[ia2,ia1] = bo12
                        cs[ia1] = chgs[ia1] - 1
                        cs[ia2] = chgs[ia2] + 1
                        chgs[ia1] = cs[ia1]
                        chgs[ia2] = cs[ia2]
            if irev:
                csk = cs.keys()
                for ai in self.mol.GetAtoms():
                    idx = ai.GetIdx()
                    if idx in csk:
                        ai.SetFormalCharge( cs[idx])
                bsk = bs.keys()
                for bi in self.mol.GetBonds():
                    ias12 = [bi.GetBgnIdx(), bi.GetEndIdx()]
                    ias12.sort()
                    ias12u = tuple(ias12)
                    if ias12u in bsk:
                        bi.SetOrder( bs[ias12u] )
        self.bom = bom
        self.chgs = np.array(chgs)

    @property
    def rsvdw(self):
        if not hasattr(self, '_rsvdw'):
            self._rsvdw = Elements().rvdws[self.zs]
        return self._rsvdw

    @property
    def rscov(self):
        if not hasattr(self, '_rscov'):
            self._rscov = Elements().rcs[self.zs]
        return self._rscov

    @property
    def rsvdw_ref(self):
        if not hasattr(self, '_rsvdw_ref'):
            self._rsvdw_ref = [self.rsvdw] + self.rsvdw[..., np.newaxis]
        return self._rsvdw_ref

    @property
    def geom(self):
        if not hasattr(self, '_geom'):
            self._geom = GM.Geometry(self.coords, ds=self.ds)
        return self._geom

    def get_dihedral_angle(self, qats):
        return self.geom.get_dihedral_angle(qats, unit='degree')

    @property
    def iasdha(self):
        """ a list of idx of atoms that form hydrogen bond
        A typical hydrogen bond comprises of 3 parts: D...H-A"""
        if hasattr(self,'_iasdha'):
            return self._iasdha
        ifdhbd, _iashbd = is_subg(self.mol, sma_hbd) #, iop=1)
        ifdhba, _iashba = is_subg(self.mol, sma_hba) #, iop=1)
        iashbd = [ i[0] for i in _iashbd ]
        iashba = [ i[0] for i in _iashba ]
        if not all([ifdhbd,ifdhba]):
            self._iasdha = []
            return self._iasdha
        nad = len(iashbd); naa = len(iashba)
        zs = self.zs
        ds = self.ds
        dhas = []
        for i0 in range(nad):
            for j0 in range(naa):
                i,j = iashbd[i0], iashba[j0]
                if i==j: continue
                # now find H's attached to HB donor
                iash = self.ias[ np.logical_and(self.zs==1, self.g[i]>0) ]
                for h in iash:
                    rvdw0 = self.rsvdw[[h,j]].sum()
                    rc0 = self.rscov[[h,j]].sum()
                    ang = self.geom.get_angle([i,h,j], unit='degree')
                    #print(' ------ zi,zj = ', zs[i],zs[j])
                    iDHA = self.get_is_dha_valid([zs[i],zs[j]],ang)
                    if ds[h,j] > rc0 and ds[h,j] <= rvdw0 and iDHA:
                        dhas.append( [i,h,j] )
        self._iasdha = dhas
        return self._iasdha

    @property
    def ghb(self):
        if not hasattr(self, '_ghb'):
            ghb = np.zeros((self.na,self.na))
            for i,h,j in self.iasdha:
                ghb[h,j] = ghb[j,h] = 0.5
            self._ghb = ghb
        return self._ghb

    @property
    def ghbv(self):
        if not hasattr(self, '_ghbv'):
            ghbv = np.zeros((self.nav,self.nav))
            for i,h,j in self.iasdha:
                ghbv[i,j] = ghbv[j,i] = 0.5
            self._ghbv = ghbv
        return self._ghbv

    @property
    def hbv2hb(self):
        if not hasattr(self, '_hbv2hb'):
            dct = {}
            for i,h,j in self.iasdha:
                bv = [i,j]; b = [h,j]
                bv.sort(); b.sort()
                assert bv not in dct, '#ERROR: for HB D-H...A, (D,A) <-> (H,A) is not 1-to-1'
                dct[bv] = b
            self._hbv2hb = dct
        return self._hbv2hb

    @property
    def hb2hbv(self):
        if not hasattr(self, '_hb2hbv'):
            dct = {}
            for i,h,j in self.iasdha:
                bv = [i,j]; b = [h,j]
                bv.sort(); b.sort()
                assert b not in dct, '#ERROR: for HB D-H...A, (D,A) <-> (H,A) is not 1-to-1'
                dct[b] = bv
            self._hb2hbv = dct
        return self._hb2hbv

    @property
    def hbs(self):
        if not hasattr(self, '_hbs'):
            t = []
            for i,h,j in self.iasdha:
                b = [h,j]; b.sort()
                t.append(b)
            self._hbs = t
        return self._hbs

    @property
    def chbs(self):
        """
        HB's in a quasi-conjugated env, e.g.,
        hydrogen bonds in GC and AT base pairs
        """
        if not hasattr(self, '_chbs'):
            _chbs = []
            for i,h,j in self.iasdha:
                b = [i,j]; hb = [h,j]; hb.sort()
                for cr in self.crings_v:
                    if set(b) < set(cr):
                        _chbs.append(hb)
                        break
                #ias_mext = self.mext.iasr56
                #icjs_mext = self.iconjs[ias_mext] ## note that the self.mext.iconjs != self.iconjs
                #ats_cr56 = ias_mext[np.logical_or(icjs_mext==1, icjs_mext==2) ]
                #if set(b) < set(ats_cr56):
                #    _chbs.append(hb)
            self._chbs = _chbs
        return self._chbs



    @property
    def hbs_ext(self):
        """
        extended HB bonds (including all possible non-covalent
        H...X bonds that are in co-planar and conjugated envs,
        e.g., hydrogen bonds in GC and AT base pairs.
        Apparently, chbs \in chbs_ext
        """
        if not hasattr(self, '_hbs2'):
            _hbs = []
            for b in self.ncbs:
                if (self.zs[b]==1).sum()==1:
                    _hbs.append(b)
            self._hbs2 = _hbs
        return self._hbs2


    @property
    def hbsv_ext(self):
        """
        extended HB bonds (associated heavy atoms only)
        """
        if not hasattr(self, '_hbsv2'):
            _hbs = []
            idx = np.array([0,1], dtype=int)
            for b in self.hbs_ext:
                zi, zj = self.zs[b]
                if zi==1:
                    ia0 = b[0]; ib = b[1]
                else:
                    ia0 = b[1]; ib = b[0]
                ia = self.ias[self.g[ia0]>0][0]
                hb = [ia,ib]; hb.sort()
                _hbs.append(hb)
            self._hbsv2 = _hbs
        return self._hbsv2


    @property
    def hmap(self):
        if not hasattr(self, '_hmap'):
            _hmap =  {}
            for ia in self.ias:
                if self.zs[ia] > 1:
                    _hmap[ia] = ia
                else:
                    nbrs = self.ias[self.g[ia]==1]
                    assert len(nbrs)==1
                    _hmap[ia] = nbrs[0]
            self._hmap = _hmap
        return self._hmap


    @property
    def iasvv(self):
        if not hasattr(self, '_iasvv'):
            _ias = set()
            for b in self.ncbs:
                _ias.update( [ self.hmap[ia] for ia in b ] )
            self._iasvv = _ias
        return self._iasvv

    @property
    def iasvv_inter(self):
        if not hasattr(self, '_iasvv2'):
            _ias = set()
            for b in self.ncbs_inter:
                _ias.update( [ self.hmap[ia] for ia in b ] )
            self._iasvv2 = list(_ias)
        return self._iasvv2



    @property
    def chbs_ext(self):
        """
        extended HB bonds (including all possible non-covalent
        H...X bonds that are in co-planar and conjugated envs,
        e.g., hydrogen bonds in GC and AT base pairs.
        Apparently, chbs \in chbs_ext
        """
        if not hasattr(self, '_chbs2'):
            _chbs = []
            for b in self.hbs_ext:
                #ias_mext = self.mext.iasr56
                #icjs_mext = self.iconjs[ias_mext] ## note that the self.mext.iconjs != self.iconjs
                #ats_cr56 = ias_mext[ np.logical_or(icjs_mext==1, icjs_mext==2) ]
                #print('b=',b, 'ats_cr56=',ats_cr56)
                bv = [ self.hmap[ja] for ja in b ]
                for cr in self.crings_v:
                    if set(bv) < set(cr):
                        _chbs.append(b)
                        break
                #if set(bv) < set(ats_cr56):
                #    _chbs.append(b)
            self._chbs2 = _chbs
        return self._chbs2


    def get_is_dha_valid(self, pair, ang):
        """ is it a valid hydrogen bond, of form `D --- H-A`,
        where D: Donor, A: Acceptor """
        iok = F
        pair.sort()
        #if pair in [ [7,7], [7,8], [8,8] ]:
        angref = 180.
        #else:
        #    raise Exception('#Todo')
        if abs(ang-angref) <= THRESH_DHA_ANGLE:
            iok = T
        return iok

    @property
    def ats_pipistack(self):
        if not hasattr(self, '_pipi'):
            self._pipi = self.get_ats_pipistack()
        return self._pipi

    def get_ats_pipistack(self):
        """ Todo: """
        raise Exception('Todo')
        ats = []
        return set(ats)

    @property
    def ocs(self):
        if not hasattr(self, '_ocs'):
            self._ocs = self.get_ocs()
        return self._ocs

    def get_ocs(self):
        patt = '[#6;X3]=O'
        _, idx = is_subg(self.mol, patt)
        return idx

    @property
    def ocns(self):
        """
        return idx of atoms matching pattern '[#7][#6;X3]=O'
        """
        if not hasattr(self, '_ocns'):
            self._ocns = self.get_ocns()
        return self._ocns

    def get_ocns(self):
        patt = '[#7][#6;X3]=O'
        _, idx = is_subg(self.mol, patt)
        return idx

    @property
    def ncbs(self):
        """ non-covalent bonds """
        if not hasattr(self, '_ncbs'):
            self._ncbs = self.get_ncbs()
        return self._ncbs

    @property
    def ncbsv(self):
        if not hasattr(self, '_ncbsv'):
            self._ncbsv = self.get_ncbsv()
        return self._ncbsv

    @property
    def gvdw(self):
        if not hasattr(self, '_gvdw'):
            na = self.na
            gv = np.zeros((na,na))
            for seti in self.ncbs:
                i,j = list(seti)
                gv[i,j] = gv[j,i] = 0.2
            self._gvdw = gv
        return self._gvdw

    @property
    def gvdwv(self):
        """ vdw bond graph, connecting associated heavy atoms for
        a vdw bond """
        if not hasattr(self, '_gvdwv'):
            na = self.na
            gvv = np.zeros((na,na))
            for seti in self.ncbsv:
                i,j = list(seti)
                gvv[i,j] = gv[j,i] = 0.2
            self._gvdwv = gvv
        return self._gvdwv


    zsvdw = set([1,7,8,9,15,16,17,33,34,35,53])
    def get_ncbs(self):
        """ get all non-covalent bonds (including both
        intra- and inter-molecular)

        Attention should be paid to the var `plmin, which is
        defaulted to 4.

        pl = 4 corresponds to the structure below

                         H .
                       /    .
                      /       .
                     O         O--H
                     \        /
                      \      /
                       CH---CH
                       /     \
                      /       \

        while pl=5 may be associated with the following intra-molecular
        vdw bond
                        H ... H
                       /       \
                      /         \
               H3C---CH          C---CH3
                     \          /
                      \        /
                       CH----CH
                       /      \
                      /        \
        """
        rsvdw = self.rsvdw

        # Param `scale_vdw is vital for determining vdw amons
        # It's found that when d(H-H) reaches 2.424 (r_vdw of H is 1.10 Angstrom) in
        # octadecane dimer (L7 dataset), the E_disp can still reach up to > 10 kcal/mol
        # Thus a scale factor > 2.424/2.20 (~1.1) should be applied!
        scale_vdw = 1.20 #self.scale_vdw

        m = self.mol
        ds = self.ds
        g = self.g
        na = self.na
        ias = self.ias
        pls = self.pls
        ncbs = []
        gv = np.zeros((self.na,self.na)) ###### gv
        for i in range(na):
            for j in range(i+1,na):
                pair = [i,j]
                pair.sort()
                zsij = self.zs[pair]
                cnsij = self.cns[pair]
                zi,zj = zsij
                #nH = (np.array([zi,zj]) == 1).sum()
                dijmin = self.rscov[[i,j]].sum() + 0.45
                dijmax = ( rsvdw[i]+rsvdw[j] ) * scale_vdw
                dij = ds[i,j]
                if g[i,j]==0 and dij<dijmax and dij>dijmin:
                    pl = pls[i,j]
                    plmin = 4
                    if pl==0 or pl>=plmin:
                        #print('i,j,zi,zj=', i,j,zi,zj)
                        if np.any( np.logical_and(zsij==6, cnsij==4) ): #set([zi,zj]) < self.zsvdw:
                            # no vdw bond is gonna be formed between any atom and C-sp3
                            #raise Exception('#ERROR: found vdw bond between an atom and sp3-C??')
                            continue
                        #else:
                        #    raise Exception('Todo')
                        #    if (set([i,j]) <= self.ats_pipistack):
                        #        # pi-pi stacking
                        #        raise Exception('Todo')
                        #    else:
                        #        continue
                        if pair not in ncbs:
                            gv[i,j] = gv[j,i] = 0.2
                            ncbs.append(pair)

        #print(' ** temporary ncbs = ', ncbs)

        # For submol matching the pattern A1-B1-X1 ... X2-B2-A2
        # remove vdw bond B1-X2 if X2 is hydrogen atom and/or
        # X1-B2 bond if X1 is hydrogen atom
        # This happens when X1 and X2 are very close to each other
        #_seta = set()

        ncbs2 = []
        # criteria used to determine if an atom pair (i,j) is a vdw bond:
        # d(i,j) <= d(i,l) & d(j,i)<=d(j,k) for any l \in cov neighbors of j,
        # k \in neighbors of i
        for b in ncbs:
            i, j = b
            fli = np.logical_and(pls[i]>0, pls[i]<=2)
            flj = np.logical_and(pls[j]>0, pls[j]<=2) # pl must >0!!
            nbrsi = ias[fli]
            nbrsj = ias[flj]
            if np.all(ds[i,j]<=ds[i,nbrsj]) and np.all(ds[j,i]<=ds[j,nbrsi]):
                ncbs2.append(b)

        Todo = """_ncbs2 = []
        b2a_h = {}
        _ats_h = set()
        # criteria used to determine if an atom pair (i,j) is a vdw bond:
        # d(i,j) <= d(i,l) & d(j,i)<=d(j,k) for any l \in cov neighbors of j,
        # k \in neighbors of i
        for b in ncbs:
            i, j = b
            zsb = self.zs[b]
            fli = np.logical_and(pls[i]>0, pls[i]<=2)
            flj = np.logical_and(pls[j]>0, pls[j]<=2) # pl must >0!!
            nbrsi = ias[fli]
            nbrsj = ias[flj]
            if np.all(ds[i,j]<=ds[i,nbrsj]) and np.all(ds[j,i]<=ds[j,nbrsi]):
                _ats_h.update( np.array(b,dtype=int)[zsb==1] )
                _ncbs2.append(b)
                for iz in range(2):
                    ia = b[iz]; ib = b[1-iz]
                    za = self.zs[ia]
                    if za == 1:
                        if ia in b2a_h:
                            b2a_h[ia].update( [ib] )
                        else:
                            b2a_h[ia] = set([ib])
        # further ensure that each H can form at most 1 HB
        ats_h = np.array(list(_ats_h), dtype=int)
        bs_remove = []
        ncbs2 = []
        for ia in ats_h:
            if ia in b2a_h:
                nbrs_v = list(b2a_h[ia])
                #print('ia=',ia, 'nbrs_vdw=',nbrs_v)
                if len(nbrs_v)==1:
                    seti = [ia,nbrs_v[0]]; seti.sort()
                    #print('seti=', seti)
                    if seti not in ncbs2:
                        ncbs2.append( seti )
                else:
                    dsi = self.ds[ia, nbrs_v]
                    seq = np.argsort(dsi)
                    inbr = nbrs_v[seq[0]]
                    for ir in seq[1:]:
                        bvi = [ia,nbrs_v[ir]]; bvi.sort()
                        if bvi not in bs_remove:
                            bs_remove.append(bvi)
                    seti = [ia,inbr]; seti.sort()
                    if seti not in ncbs2:
                        ncbs2.append(seti)
        _ncbs2 = []
        for bvi in ncbs2:
            if bvi not in bs_remove:
                _ncbs2.append(bvi)
        ncbs2 = _ncbs2.copy() """

        return ncbs2


    def get_ncbsv(self):
        """ get the pair of heavy atoms associated with
        each of the non-covalent bonds """
        g = self.g
        gvv = np.zeros((self.na,self.na))
        ncbsv = []
        for b in self.ncbs:
            i,j = list(b)
            zi = self.zs[i]
            zj = self.zs[j]
            iu = i
            if zi == 1:
                iu = self.ias[g[i]==1][0]
            ju = j
            if zj == 1:
                ju = self.ias[g[j]==1][0]
            if iu != ju and g[iu,ju]==0:
                p = [iu,ju]; p.sort()
                if p not in ncbsv:
                    ncbsv.append(p)
        return ncbsv


    @property
    def ncbs_inter(self):
        if not hasattr(self, '_ncbs2'):
            ncbs2 = []
            for b in self.ncbs:
                i,j = list(b)
                if self.pls[i,j] == 0:
                    ncbs2.append(b)
            self._ncbs2 = ncbs2
        return self._ncbs2

    @property
    def ncbs_intra(self):
        if not hasattr(self, '_ncbs1'):
            ncbs1 = []
            for b in self.ncbs:
                i,j = list(b)
                if self.pls[i,j] > 0:
                    ncbs1.append(b)
            self._ncbs1 = ncbs1
        return self._ncbs1

    @property
    def rsmax(self):
        """ reference interatomic distance (rs) to be used
        for determining if other mol is crowd """
        if not hasattr(self, '_rsmax'):
            dct = {}
            for b in self.ncbs_intra:
                i,j = b
                dij = self.ds[i,j]
                sij = [ self.symbols[_] for _ in b ]
                sij.sort()
                k = ''.join(sij)
                if k in dct:
                    dct[k] += [dij]
                else:
                    dct[k] = [dij]
            rsmax = {}
            for k in dct:
                rmin = np.min(dct[k])
                rsmax[k] = rmin
            self._rsmax = rsmax
        return self._rsmax

    def get_sbond(self, zsij):
        """ 0, 1 --> HO """
        zsij.sort()
        sij = [ co.chemical_symbols[_] for _ in zsij ]
        return ''.join(sij)


    @property
    def icrowd_cov(self):
        """ is there any bond whose rij is shorter than the minimal
        value of sum of covalent raidus? """
        if not hasattr(self, '_icrowdc'):
            crowd = F
            non_bonds = np.where( np.logical_and(self.bom==0, self.ds!=0.) )
            dsmin = ([self.rscov] + self.rscov[..., np.newaxis]) * 1.25 # self.scale_cov
            if np.any(self.ds[non_bonds] < dsmin[non_bonds]):
                crowd = T
            self._icrowdc = crowd
        return self._icrowdc

    @property
    def icrowd_vdw(self):
        """ is there any pair of atoms too close to each other? """
        if not hasattr(self, '_icrowd'):
            crowd = F
            if self.dsref: # highly recommended!!!!!
                dsref = self.dsref
            else:
                print("   Use with caution! `dsref to be obtained as summation of rvdw")
                dsref = {}
                nzu = len(self.zsu)
                for i in range(nzu):
                    for j in range(i,nzu):
                        zsij = [ self.zsu[_] for _ in [i,j] ]
                        k = self.get_sbond(zsij)
                        dsref[k] = Elements().rvdws[zsij].sum()
                # reset dmin of HH to 1.90 Angstrom
                #dsref['HH'] = 1.90

            #######################################################
            scale = self.scale_vdw # You may need to adjust this value
            #######################################################

            for b in self.ncbs:
                i,j = b
                dij = self.ds[i,j]
                zsij = [ self.zs[_] for _ in [i,j] ]
                k = self.get_sbond(zsij)
                if k not in self.dsref:
                    print('   new vdw bond introduced in amons --> too crowded, skip!')
                    crowd = T
                else:
                    if self.ds[i,j] < self.dsref[k] * scale:
                        crowd = T
                        break
            self._icrowd = crowd
        return self._icrowd


def is_subg(t, q, qidxs=[], woH=False):
    """
    check if `q is a subgraph of `t

    if iop .lt. 0, then the matched atom indices in `t will also
    be part of the output
    """
    if type(t) is str:
        m = Chem.MolFromSmarts(t)
    else:
        m = t

    if woH:
        m = Chem.RemoveHs(m)

    q = Chem.MolFromSmarts(q)
    idxs = m.GetSubstructMatches(q)

    ifd = F
    if idxs:
        ifd = T

    return ifd, idxs


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

