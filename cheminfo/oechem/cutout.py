#!/usr/bin/env python

import tempfile as tpf
import os, sys, re, copy, ase
import numpy as np
import itertools as itl
import cheminfo.oechem.oechem as coo
import cheminfo as co
import cheminfo.graph as cg
import cheminfo.oechem.core as coc
import scipy.spatial.distance as ssd

T, F = True, False

class setset(object):

    def __init__(self, sets):
        self.sets = sets

    def get_idx(self, seti):                                     
        idx = None
        for j, setj in enumerate(self.sets):
            if seti <= setj: 
                idx = j
                break
        return idx
    
    def update(self, seti, nested=T):
        """ 
        append a new set `seti` if it is not a subset of any
        set in `sets`
        """
        sets = self.sets.copy()
        if len(sets) == 0:
            sets.append( seti )
        else:
            if not np.any( [ seti <= setj for setj in sets ] ):
                intersected = [ seti.intersection(set_j) for set_j in sets ]
                istats = np.array([ si != set() for si in intersected ])
                nset = len(sets)
                idxs = np.arange( nset )
                if np.any(istats):
                    #assert istats.astype(np.int).sum() == 1
                    for iset in idxs:
                        if istats[iset]:
                            sets[iset] = seti.union( sets[iset] )
                else:
                    sets.append( seti )
        self.sets = sets



class ParentMol(coo.StringM):

    def __init__(self, string, debug=False, nprocs=24):

        self.debug = debug

        coo.StringM.__init__(self, string, debug=debug, nprocs=nprocs)

        if os.path.exists(string) and self.na > 100: # save `ds and `pls
            ftmp = string[:-4]+'.npz'
            if os.path.exists(ftmp):
                print(' ++ read `ds and `pls from file: ', ftmp)
                dt = np.load(ftmp)
                self._ds = dt['ds']
                self._pls = dt['pls']
            else:
                ds = self.ds #ssd.squareform( ssd.pdist(self.coords) )
                pls = self.pls #cg.Graph(self.g, nprocs=self.nproc
                print(' ++ save `ds and `pls to file ', ftmp)
                np.savez(ftmp, ds=ds, pls=pls)

    @property
    def rgs(self):
        if not hasattr(self, '_rgs'):
            self._rgs = self.get_rigid_groups()
        return self._rgs


    def get_rigid_groups(self):
        """
        get rigid groups of atoms, including
        i) ring structures, 3- to 6-membered ring
        ii) conjugated envs, e.g., C=C=C, C=N#N

        Note that function §update_sets won't merge two rings
        in, say Naphthalene. This is exactly what we demand!
        """

        sets = setset([])

        # first search for rings
        for set_r in self.rings: #(namin=3, namax=7):
            sets.update(set_r)

        for patt in ['[CX3;!a](=O)[#7;!a]', '[^2;!a]~[a]', '[^2;X1]=[*]' ]:
            _, ssi, _ = coc.is_subg(self.mol, patt, iop=1)
            for si in ssi:
                sets.update(set(si))

# first retain the BO's for bonds involving any multi-valent atom, i.e.,
# atom with dvi>1. Here are a few examples that are frequently encountered:
# 1) C2 in "C1=C2=C3" & "C1=C2=N3";
# 2) N2 and N3 in "C1=N2#N3" ( -C=[N+]=[N-], -N=[N+]=[N-] )
# 3) '-S(=O)(=O)-', -Cl(=O)(=O)=O,
# By doing This, we can save a lot of work for BO perception later!
        dvs = self.tvs - self.cns
        for ia in self.ias:
            if dvs[ia] > 1:  # patterns like '[*]=[*]#[*]' and '[*]=[*]=[*]' are thus covered!
                atsi = set([ia])
                for ja in self.ias[self.bom[ia]>1]:
                    if dvs[ja] > 0:
                        atsi.update([ja])
                if atsi not in sets:
                    sets.update(atsi)

        for b in self.b2a:
            i, j = [ self.iasv[_] for _ in b ]
            if self.bom[i,j] > 1:
                si = set([i,j])
                sets.update(si)

        return sets.sets



    @property
    def a2rg(self):
        """
        a map from atom idx (heavy atom) to groups of atoms that are rigid
        """
        if not hasattr(self, '_a2gr'):
            dct = {}
            for rg in self.rgs:
                for ia in rg:
                    if ia not in dct:
                        dct[ia] = rg
                    else:
                        dct[ia].update(rg)
            self._a2gr = dct
        return self._a2gr


    def get_atoms_within_cutoff(self, rcut, k=999, iasq=None, zsq=None):
        """
        For now, for prediction of NMR only

        retrieve atoms around atom `ia-th atom within a radius of
        `rcut.

        This function will be used when dealing with large molecules
        like proteins where long-range interactions are significant.
        The related properties include NMR shifts.
        """

        if np.all([obj is None for obj in [iasq,zsq]]):
            iasq = list(self.ias)
            print('    each of the atoms in q is to be processed')
        else:
            if zsq is not None:
                iasq = []
                for zq in zsq:
                    iasq += list(self.ias[self.zs == zq])
            print('  iasq=',iasq, 'zsq=',zsq)

        iasq.sort()
        iasq = np.array(iasq, dtype=int)
        naq = len(iasq)

        print('  There are %d matched atoms'%naq)

        iasvq = iasq[self.zs[iasq]>1]

        sets = [] # mols as lists of atoms
        ms = []; maps = []; imol = 0
        for i0,iaq in enumerate(iasq):
            print('  now {}/{}: iaq = {}, zi = {}'.format(i0+1, naq, iaq, self.zs[iaq]))
            seti = set() # the ultimate list of heavy atoms, to be used for extract subm

            _jas = self.ias[self.ds[iaq]<=rcut]
            _gi = self.g[_jas][:,_jas]
            _zsj = self.zs[_jas]
            _cnsrj = np.array([ co.cnsr[_zj] for _zj in _zsj ], dtype=int)
            _cnsj = _gi.sum(axis=0)

            # exclude standalone H/F/Cl/Br/I
            jas = _jas[ np.logical_and(_cnsrj!=1, _cnsj>0) ]
            jasv = jas[self.zs[jas]>1]
            seti.update(jasv)


            visited = []
            loop = 0
            print('    now iteratively update neighboring atoms ')
            while True: #for _ in range(6): ## do this twice!! This is essential!!
                print('      ++ %d-th loop'%(loop+1))
                jasv = list(seti)

                for ja in jasv:
                    if ja in self.a2rg:
                        #print('ja, rg=', ja, self.a2rg[ja])
                        seti.update( self.a2rg[ja] )
                    #else:
                    #    print('     ja=%d not found in a2rg'%ja)

                jasv = np.array(list(seti), dtype=int)
                gv = self.g[jasv][:,jasv]

                # the extracted molecular fragments (i.e., `mf) may contain
                # several disconnected components, now add some atoms
                # to re-connecting these standalone entities
                go = cg.Graph(gv)
                cs = go.find_cliques()
                nc = len(cs)
                gc = np.zeros((nc,nc))
                for ic in range(nc):
                    for jc in range(ic+1,nc):
                        atsi = jasv[cs[ic]]; atsj = jasv[cs[jc]]
                        for ia in atsi:
                            for ja in atsj:
                                sij = set([ia,ja])
                                if sij not in visited:
                                    visited.append(sij)
                                    if 1 <= self.pls[ia,ja] <= 3:
                                        for path in self.gobj.get_shortest_paths(ia,ja):
                                            #print('ia,ja=',ia,ja, 'path=',path)
                                            seti.update(path)
                if len(seti) == len(jasv):
                    break
                loop += 1

            if len(seti) > k:
                print('    skip this submol as nav>k')
                continue
            
            atsv = list(seti)
            jasv = np.array(atsv, dtype=int)
            if seti not in sets:
                sets.append(seti)
                mi = self.get_subm(atsv)
                dsi = ssd.cdist([self.coords[iaq]], mi.coords)[0]
                if self.debug:
                    for csi in cg.Graph(self.g[atsv][:,atsv]).cliques:
                        _m = self.get_subm(jasv[csi])
                        if _m.zs.sum()%2 != 0:
                            print('subm is a radical for csi=',csi)
                            fdt = '%s/Dropbox/Trash/'%(os.environ['HOME'])
                            fn = tpf.NamedTemporaryFile(dir=fdt).name + '.sdf'
                            fsdf = os.path.basename(fn)
                            _m.write_ctab(fsdf, dir=fdt)
                            print(' please check file ', fn)
                            raise Exception('??')
                assert mi.zs.sum()%2 == 0
                ms.append(mi)
                ia2 = mi.ias[dsi<=0.001][0]
                maps.append( [iaq, imol, ia2] )
                imol += 1
            else:
                im = sets.index(seti)
                mi = ms[im]
                dsi = ssd.cdist([self.coords[iaq]], mi.coords)[0]
                ia2 = mi.ias[dsi<=0.001][0]
                maps.append( [iaq, im, ia2] )
            print('    up to now, %s submols found'%(len(sets)))
        print( 'summary: naq={}, found {} submols'.format(naq, len(sets)))
        return ms, np.array(maps, dtype=int)





if __name__ == "__main__":

    import sys
    import argparse  as ap



    ps = ap.ArgumentParser()
    ps.add_argument('-a','-all', dest='all', action='store_true', help='all atoms in target mol are to be inquired to retrieve local env')
    #ps.add_argument('-ga', action='store_true', help='generate amons')
    ps.add_argument('-debug', action='store_true')
    ps.add_argument('-f', '-file', '-lb','-label', dest='lb', default='amons', type=str, help='generate amons')
    ps.add_argument('-k', nargs='?', type=int, help='maximal num of heavy atoms allowed (N_I) in an amon')
    ps.add_argument('-rcut', nargs='?', default=6.0, type=float)
    ps.add_argument('-q', '-query', dest='q', nargs='?', type=str, help='query mol(s) as sdf/pdb file(s)')

    ag = ps.parse_args() # sys.argv[1:]

    print(' -- now excuting %s'%( ' '.join(sys.argv[:]) ) )

    print('q={}, rcut={}'.format(ag.q, ag.rcut))
    obj = ParentMol(ag.q, debug=ag.debug)
    ms = []
    iasq = None; zsq = [1, 6, 7]
    if ag.all:
        zsq = None
    else:
        print(' ++ zsq to be inquired: ', zsq)

    k = 999
    if ag.k:
        k = ag.k

    ms, maps = obj.get_atoms_within_cutoff(ag.rcut, k=k, iasq=iasq, zsq=zsq)

    fd = '%s-rc%.1f/'%(ag.lb, ag.rcut)
    if not os.path.exists(fd):
        os.mkdir(fd)
    fmap = '%s/maps.txt'%fd
    np.savetxt(fmap, maps, fmt='%d')
    print('map file saved to ', fmap)

    nm = len(ms)
    for i in range(nm):
        mi = ms[i]
        fi = 'piece_%04d.sdf'%(i+1)
        mi.write_ctab(fi, fd)



