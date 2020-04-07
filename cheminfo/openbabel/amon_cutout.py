
from itertools import chain, product
import os, sys, re, copy, ase
import ase.data as ad
from openeye.oechem import *
import numpy as np
import networkx as nx
import networkx.algorithms.isomorphism  as iso
import aqml.cheminfo.oechem.OEChem as oe
from rdkit import Chem
import scipy.spatial.distance as ssd
import aqml.cheminfo.openbabel.obabel as cib
import multiprocessing
import aqml.cheminfo.math as cim
import cml.sd as dd
import itertools as itl
import tempfile as tpf
#tsdf = tpf.NamedTemporaryFile(dir=tdir)
import aqml.cheminfo.fortran.famoneib as fa

global dsHX
dsHX = {5:1.20, 6:1.10, 7:1.00, 8:0.98, 9:0.92, 14:1.48, 15:1.42, 16:1.34, 17:1.27}


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


    def get_atoms_within_cutoff(self, qa=None, za=None, cutoff=3.6):
        """
        For now, for prediction of NMR only

        retrieve atoms around atom `ia-th H atom within a radius of
        `cutoff.

        This function will be used when dealing with large molecules
        like proteins where long-range interactions are significant.
        The related properties include NMR shifts.
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
                    print 'atom indices of all amons done'
                    for comb in combs:
                        #print comb
                        cans_i = [ cans[ic] for ic in comb ]; cans_i.sort()
                        cans.append('.'.join(cans_i))
                        ts_i = [ Mlis[ic] for ic in comb ]
                        Mlis.append( merge(ts_i) )
                    print 'amons now ready for filtering'
            #else:
            #    #


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
                    #print ' ++ i, cans[i] = ', i,cans[i]
                    seta.update2(ir, cans[i], Mlis[i])
                seta._sort2()
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
        dd.io.save('%s/maps.pkl'%fdn, {'maps': maps} )

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

