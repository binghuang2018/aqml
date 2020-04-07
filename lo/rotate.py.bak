#!/usr/bin/env python

from time import time
from functools import reduce
import numpy as np
from pyscf import gto, scf
from aqml.cheminfo.base import *
import aqml.cheminfo.graph as cg
from aqml.cheminfo.molecule.elements import *
from aqml.cheminfo.molecule.molecule import *
from pyscf.lo.orth import pre_orth_ao_atm_scf
from pyscf.tools import molden

class atoms(object):
    """
    get HF orbitals for free atoms
    """
    def __init__(self, zs, basis):
        return


lmap = {'s':0, 'p':1, 'd':2, 'f':3, 'g':4}
mmap = {'':0, \
        'x':0, 'y':1, 'z':2, \
        'xy':0, 'yz':1, 'z^2':2, 'xz':3, 'x2-y2':4 }
nmap = {1:1, 6:2, 7:2, 8:2, 9:2, 15:3,16:3,17:3} # for MINBASIS

class pyscf_object(object):

    def __init__(self, symbs, coords, basis, spin=0):
        self.spin = spin
        str_m = '' #'O 0 0 0; H 0 0 1; H 0 1 0'
        for i, si in enumerate(symbs):
            x, y, z = coords[i]
            str_m += '%s %.8f %.8f %.8f; '%(si, x, y, z)
        str_m = str_m[:-2]
        mol = gto.M(atom=str_m, basis=basis, verbose=0, spin=spin)
        self.mol = mol
        self.na = len(symbs)
        self.nbf = mol.nao
        self.zs = [ chemical_symbols.index(symbs[0]) ] + [1,]*(self.na-1)

    def get_ao_map(self):
        ids = self.mol.offset_ao_by_atom()[:, 2:4]
        ibs, ies = ids[:,0], ids[:,1]
        self.ibs = ibs
        self.ies = ies
        idx = [ np.arange(ibs[i],ies[i]) for i in range(self.na) ]
        self.idx = idx
        imap = np.zeros(self.nbf).astype(np.int)
        for i in range(self.na):
            imap[ibs[i]:ies[i]] = i
        self.imap = imap
        labels = self.mol.ao_labels()
        nlms = []
        for _lb in labels:
            lb = _lb.strip().split()[-1]
            n = int(lb[0])
            l = lmap[lb[1]]
            m = mmap[lb[2:]]
            nlms.append( [n,l,m] )
        self.nlms = np.array(nlms, np.int)

    def get_hao(self):
        """
        C: central atom
        L: ligands (i.e., hydrogen atoms)
        """
        mol = self.mol
        c = pre_orth_ao_atm_scf(mol) #
        _s = mol.intor_symmetric('int1e_ovlp')
        s = reduce( np.dot, (c.conjugate().T, _s, c) )
        #print ' ** s = ', s

        self.get_ao_map()
        _idxc = self.idx[0] # AO idx of the central atom
        _iaosr = np.arange(len(_idxc))
        _nlms = self.nlms[_idxc]

         # to be changed if a bsis other than minimal bst is used
        #flt = np.logical_and(_nlms[:,0]==nmap[self.zs[0]], _nlms[:,1]>0)
        flt = ( _nlms[:,0]==nmap[self.zs[0]] )
        iaosr = _iaosr[flt]

        idxc = _idxc[ flt ] # AO's to be rotated
        #idxc_f = _idxc[ _nlms[:,1]==0 ] # AO not to be rotated

        idxs = np.arange(self.nbf)
        idxl = np.setdiff1d(idxs, _idxc) # AO idx of ligands
        s1 = s[idxl][:,idxc] # s[ibl:iel][:, ibc:iec]
        #s1 = s[idxc][:,idxl] # s[ibc:iec][:,ibl:iel]

        #
        # get hybridized atomic orbitals that maximizes \sum_i < \phi_i | \psi_i >
        # where
        s2 = np.dot(s1,s1.T)
        eigs, _U = np.linalg.eigh( s2 )
        U = _U.T; 
        print ' ** U = ', U
        #print ' ** U^T * eigs * U = ', reduce( np.dot, (U.T, np.diag(eigs), U) ) - s2
        #print ' ** eigs = ', eigs
        assert np.all(eigs>0), '#ERROR: not all eigvalues are >0?'
        smax = np.sqrt(eigs).sum()
        #print ' ** maximal ovlp = %.4f'%smax
        diag = np.diag(1./np.sqrt(eigs))
        #R = reduce( np.dot, (U.T, diag, U, s1, c[idxc][:,idxc]) ) # final rotation matrix
        R = reduce( np.dot, (U.T, diag, U, s1) )#, c[idxc][:,idxc]) ) # final rotation matrix
        #print ' ** RR^T = ', np.dot(R,R.T)
        self.R = R
        #print ' ** R = ', R
        return iaosr, R


def update_vs(v,vs):
    if len(vs) == 0:
        return v
    else:
        vu = v
        if np.all(np.dot([v],np.array(vs).T)<0):
            vu = -v
        return vu


def get_v12_sp3(cs): # sp3 O
    assert len(cs)==2
    c0 = cs[0]
    vs = [(ci-c0)/np.linalg.norm(ci-c0) for ci in cs[1:] ]
    _v1 = -(vs[0]+vs[1])
    _v2 = np.cross(vs[0],vs[1])
    v1,v2 = [ v/np.linalg.norm(v) for v in [_v1,_v2] ]
    c12 = np.dot(vs[0],vs[1])/np.product(np.linalg.norm(vs,axis=1)) # cos(vs[0],vs[1])
    theta = np.arcos(c12)
    cos1, sin1 = np.cos(theta/2.), np.sin(theta/2.)
    v1u = cos1*v1 + sin1*v2
    v2u = cos1*v1 - sin1*v2
    return v1u,v2u

def get_v_sp3(cs): # sp3 N
    assert len(cs)==4
    c0 = cs[0]
    vs = [(ci-c0)/np.linalg.norm(ci-c0) for ci in cs[1:] ]
    c1u,c2u,c3u = [ c0+v/np.linalg.norm(v) for v in vs ]
    v1,v2 = c2u-c1u, c3u-c1u
    v = np.cross(v1,v2)
    vu = v
    if np.dot(v,vs[0])>0: vu = -v
    return vu/np.linalg.norm(vu)

def get_v3(cs): # sp2 C, e.g., =C<
    assert len(cs)==4
    c0 = cs[0]
    vs = [(ci-c0)/np.linalg.norm(ci-c0) for ci in cs[1:] ]
    c1u,c2u,c3u = [ c0+v/np.linalg.norm(v) for v in vs ]
    v1,v2 = c2u-c1u, c3u-c1u
    v = np.cross(v1,v2)
    return v/np.linalg.norm(v)

def get_v2(cs): # sp2 N in =N-
    assert len(cs)==3
    c0 = cs[0]
    vs = [(ci-c0)/np.linalg.norm(ci-c0) for ci in cs[1:] ]
    v = np.cross(vs[0],vs[1])
    v1 = -(vs[0]+vs[1])
    return v/np.linalg.norm(v), v1/np.linalg.norm(v1)

def get_v12(_vx,_vz): # sp2 O, e.g., =O
    # _vz specifies the z direction
    _vy = np.cross(_vz,_vx)
    vy,vx = [ _v/np.linalg.norm(_v) for _v in [_vy,_vx] ]
    cos1 = np.cos(np.pi/3.)
    sin1 = np.sin(np.pi/3.)
    v1 = cos1*vx-sin1*vy
    v2 = -cos1*vx-sin1*vy
    return v1,v2


class config_adapted_hao(RawMol):

    """
    configuration adapted hybridized ANO
    """

    def __init__(self, zs, coords, basis='cc-pvdz'): # fn='test'):
        self.rcs = Elements().rcs
        self.basis = basis
        #self.fn = fn
        #assert np.sum(self.zs)%2 == 0, '#ERROR: spin polarised?'
        RawMol.__init__(self, list(zs), coords)

        spin = sum(self.zs)%2
        symbs = [ chemical_symbols[zi] for zi in self.zs ]
        OBJ = pyscf_object(symbs, coords, basis, spin=spin)
        self.mol = OBJ.mol
        self.nbf = OBJ.mol.nao
        ids = OBJ.mol.offset_ao_by_atom()[:, 2:4]
        ibs, ies = ids[:,0], ids[:,1]
        self.aoidxs = [ np.arange(ibs[i],ies[i]) for i in range(self.na) ]
        self.T0 = pre_orth_ao_atm_scf(OBJ.mol) #
        self.T = np.eye(OBJ.mol.nao)

        _cnsr = {1:1, 6:4, 7:3, 8:2}
        cnsr = np.array([_cnsr[zi] for zi in self.zs],np.int)
        cns = self.g.sum(axis=0)
        dvs = cnsr - cns

        bidxs = []
        #print 'dvs = ', dvs
        assert np.all(dvs>=0)

        # first add H's to sp3 N and O
        for ia in self.ias:
            zi = self.zs[ia]
            jas = self.ias[self.g[ia]>0]
            d = np.sum( self.rcs[ [1,zi] ] )
            if zi==7 and cns[ia]==3:
                v = get_v_sp3( self.coords[ [ia]+list(jas) ] )
                bidxs.append( [ia,self.coords[ia]+v*d] )
            elif zi==8 and cns[ia]==2:
                v1,v2 = get_v12_sp3( self.coords[ [ia]+list(jas) ] )
                for v in [v1,v2]:
                    bidxs.append( [ia,self.coords[ia]+v*d] )

        # add H's to sp2 C, N and O
        _jas = self.ias[dvs==1]; #print _jas
        if len(_jas) > 0:
            _jasr = cg.find_cliques(self.g[_jas][:,_jas])
            for kdxr in _jasr:
                naj = len(kdxr)
                assert naj%2==0
                jas = _jas[kdxr]
                #print ' * jas = ', jas
                cnsj = cns[jas]
                seq = np.argsort(cnsj)
                vs = []
                for _j in range(naj):
                    j = seq[_j-1]
                    ja = jas[j]
                    #print '  |__ ja = ', ja
                    zj = self.zs[ja]
                    jas2 = self.ias[self.g[ja]>0]
                    nbr = len(jas2)
                    d = np.sum( self.rcs[ [1,zj] ] )
                    if nbr==3 and zj==6:
                        v = get_v3(self.coords[ [ja]+list(jas2) ])
                        vu = update_vs(v,vs); vs.append(vu)
                        bidxs.append( [ja,self.coords[ja]+vu*d] )
                        #print '  |__ dot(v,vs) = ', np.dot([vu],np.array(vs).T)
                    elif nbr==2 and zj==7:
                        v,v1 = get_v2(self.coords[ [ja]+list(jas2) ])
                        for _v in [v,v1]:
                            vu= update_vs(_v,vs); vs.append(vu)
                            bidxs.append( [ja,self.coords[ja]+vu*d] )
                    elif nbr==1 and zj==8:
                        ja2 = jas2[0]
                        vz = vs[list(jas).index(ja2)]
                        vx = self.coords[ja2]-self.coords[ja]
                        v1,v2 = get_v12(vx,vz)
                        for _v in [v,v1,v2]:
                            vu = update_vs(_v,vs); vs.append(vu)
                            bidxs.append( [ja,self.coords[ja]+vu*d] )
                    else:
                        raise '#unknown case'

        nadd = len(bidxs)
        na = self.na
        if nadd > 0:
            na2 = na + nadd
            g2 = np.zeros((na2,na2)).astype(np.int)
            g2[:na, :na] = self.g
            ih = na
            cs2 = [] # coords of H's
            for bidx in bidxs:
                ia, ci = bidx
                g2[ih,ia] = g2[ia,ih] = 1
                cs2.append(ci)
                ih += 1

            zs = np.concatenate((self.zs,[1,]*nadd))
            coords = np.concatenate((self.coords,cs2))
            self.zs = zs
            self.coords = coords
            self.g = g2
            self.ias = np.arange(na2)
            self.na = na2


    def run(self):
        #
        for iac in range(self.na):
            if self.zs[iac] > 1:
                jas = self.ias[ self.g[iac]>0 ]
                zi = self.zs[iac]
                si = chemical_symbols[zi]
                zs1 = [ zi ] + [1,]*len(jas)
                symbs1 = [ si, ] + [ 'H@2' ]*len(jas)
                basis1 = { si: self.basis, 'H@2':'sto-3g' }
                origin = self.coords[iac]
                coords1 = [ origin ]
                for j in jas:
                    dest = self.coords[j]
                    v = dest - origin
                    d = np.sum( self.rcs[ [1,self.zs[j]] ] )
                    coords1.append( origin + d * v/np.linalg.norm(v) )
                spin = sum(zs1)%2
                subm = pyscf_object(symbs1, coords1, basis1, spin=spin)
                iaosr, R = subm.get_hao()

                t1 = self.aoidxs[iac]
                t = t1[iaosr]
                ib,ie = t[0],t[-1]+1
                self.T[ib:ie,ib:ie] = R.T
                #molden.from_mo(subm.mol, '%s_%03d.molden'%(self.fn, iac+1), subm.c2)
        self.t = np.dot(self.T0, self.T)


class pairwise_rotate(RawMol):
    """
    general purpose orbital rotation algo
    """

    def __init__(self, zs, coords, basis, spin=0):

       RawMol.__init__(zs, coords)
       symbs = [ chemical_symbols[zi] for zi in zs ]
       obj = pyscf_object(symbs, coords, basis, spin=spin)
       self.mol = obj.mol


    def get_max_ovlp_bst(self, iac):

       mol = self.mol
       g = self.g

       # get maximally overlapped atomic orbitals
       # i on atom A: n_A(i) = \sum_{\mu \in A} (C^\mu_i)^2
       p = True #False
       StartTime = time()

       na = mol.natm
       ias = np.arange(na)
       nbf = mol.nao
       bfs = np.eye(nbf)

       bfs = bfs.copy() # <- leave original orbitals alone.

       # pyscf mol
       t = mol.offset_ao_by_atom()[:, 2:]
       ibs, ies = t[:,0], t[:,1]
       idx = [ range(ibs[i],ies[i]) for i in range(na) ]
       imap = np.zeros(nbf).astype(np.int)
       for i in range(na):
           imap[ibs[i]:ies[i]] = i

       labels = mol.ao_labels()
       orbs = []
       for label in labels:
           orbs.append( label.strip().split()[-1] )
       s = mol.intor_symmetric('int1e_ovlp')
       #print ' * s = ', s

       ###
       #iac = 0 # central atom

       idxc = idx[iac]
       nbfc = len(idxc)

       idx2 = np.concatenate([ range(ibs[i],ies[i]) for i in ias[g[iac]>0] ])

       if p: print("   {0:^5s} {1:^14s} {2:^11s} {3:^8s}".format("ITER.","LOC(Orbital)","GRADIENT", "TIME"))
       Converged = False
       for it in range(2048):
          # calculate value of functional (just for checking)
          if it == 0:
              s1 = s
          else:
              s1 = reduce(np.dot, (bfs.conj().T,s,bfs))

          #s1 = np.abs(s1)

          #L = np.abs(s1[idxc][:,idx2]).sum()
          _L = s1[idxc][:,idx2]
          #print '* L = ', _L
          L = _L.sum()

          # calculate orbital rotation angles. It's a bit tricky
          # because the functional depends in 4th order on the
          # rotation angles. References:
          # [1] Knizia, J. Chem. Theory Comput., http://dx.doi.org/10.1021/ct400687b
          # [2] Pipek, Mezey, J. Chem. Phys. 90, 4916 (1989); http://dx.doi.org/10.1063/1.456588
          fGrad = 0.
          for i in range(nbf):
             for j in range(i):
                if imap[i] != imap[j]: continue
                if orbs[i][1]=='s' or orbs[j][1]=='s': continue # don't rotate any pair of orbitals involving "s" channel
                ia1 = imap[i]
                if ia1 == iac:
                    ias2 = ias[g[ia1]>0]
                else:
                    ias2 = [iac]

                _idx2 = np.concatenate([ range(ibs[ia],ies[ia]) for ia in ias2 ])

                s1 = reduce(np.dot, (bfs.conj().T,s,bfs))

                a = s1[i,_idx2].sum()
                b = s1[j,_idx2].sum()
                Aij = a - b
                Bij = a + b
                #print 'i,j,Aij,Bij = ',i,j,Aij,Bij
                #print 'Aij**2+Bij**2=', Aij*Aij+Bij*Bij

                if (Aij**2 + Bij**2 < 1e-10) and False:
                    # no gradient for this rotation.
                    continue
                else:
                    # Calculate 2x2 rotation angle phi.
                    # This correspond to [2] (12)-(15), re-arranged and simplified.
                    phi = 1.*np.arctan2(Bij,-Aij)
                    fGrad += Bij**2
                    # ^- Bij is the actual gradient. Aij is effectively
                    #    the second derivative at phi=0.

                    # 2x2 rotation form; that's what PM suggest. it works
                    # fine, but I don't like the asymmetry.
                    cs = np.cos(phi)
                    ss = np.sin(phi)
                    Ci = 1. * bfs[:,i] # 1. * bfs[:,i]
                    Cj = 1. * bfs[:,j] # 1. * bfs[:,j]
                    bfs[:,i] =  cs * Ci + ss * Cj
                    bfs[:,j] = -ss * Ci + cs * Cj

                    # Further notes on this vs [2]:
                    # With the use of atan2, the entire discussion in [2] about the
                    # correct phases and further angle transformations can be ignored
                    # (atan2 chooses the right phase, and has the right angular
                    # transformation form).
                    #
                    # The trick here to obtaining the current formulas is to never
                    # write the 2x2 rotation equation in angle form in the first
                    # place, but insert and TrigReduce the ansatz
                    #
                    #     phi := (1/4) atan(x)
                    #
                    # right at the start (and solve for x instead of phi). This leads
                    # to algebraic equations (rather than trigonometric ones) which
                    # Mathematica happily chews through without further intervention.

                # update ovlp matrix
                #s = s1

          fGrad = fGrad**.5
          if p: print(" {0:5d} {1:14.8f} {2:11.2e} {3:8.2f}".format(it+1, L, fGrad, time()-StartTime))
          if fGrad < 1e-8:
             Converged = True
             break

       Note = "MOA/2x2, %i iter; Final gradient %.2e" % (it+1, fGrad)
       if not Converged:
          print("\nWARNING: Iterative localization failed to converge!"\
                "\n         %s" % Note)
       else:
          if p: print()
          print(" Iterative localization: %s" % Note)
       #print("Localized orbitals deviation from orthogonality: %8.2e" % la.norm(np.dot(bfs.T, bfs) - np.eye(nOcc)))
       return bfs[idxc][:,idxc]



if __name__ == '__main__':

  from ase import Atoms
  import ase.io as aio
  import os, sys
  import scipy.spatial.distance as ssd

  home = os.environ['HOME']
  np.set_printoptions(precision=3,suppress=True)

  s0 = """
  x = 1.09/np.sqrt(3.)
  c1,s1 = np.cos(np.pi/3), np.sin(np.pi/3)
  zs = [6, 1, 1, 1, 1]
  #coords = np.array( [[ 0,  0,  0],
  #                    [ -x*c1,  x*s1,  0],
  #                    [ -x*c1, -x*s1,  0],
  #                    [ x,  0, 0],
  #                    [ 0, 0, x*2] ])

  coords = np.array( [[ 0,  0,  0],
                      [ -x,  -x,  -x],
                      [ x, x, -x],
                      [ x,  -x, x],
                      [ -x, x, x] ])
  m = Atoms(zs,coords)
  """

  #s0 = """
  fns = ['ch4',] #'c2h6','c3h8','c4h10']
  #_fns = ['c2h4','c4h8','c6h12','c8h16', 'c12h14']
  #fns = _fns[2:]
  for fn in fns:
    #fn = fns[3]
    print ' molecule=', fn
    m = aio.read('%s.xyz'%fn)

    a = 0.0
    v = [1.,1.,1.]
    if a != 0.0:
        m.rotate(v, a*np.pi/180.) # rotate `a degree around vector `v
    #"""
    zs, coords = m.numbers, m.positions
    #print ' * ds = ', ssd.squareform( ssd.pdist(coords) )
    #sys.exit(2)

    na = len(zs)
    obj = config_adapted_hao(zs, coords, basis='sto-3g')
    obj.run()
    B = obj.t
    print ' * B = ', B

    # pyscf run
    m = obj.mol
    s1 = m.intor_symmetric('int1e_ovlp')
    s2 = reduce(np.dot, (B.T,s1,B))
    #print ' * s2 = ', s2

    mf = scf.RHF(m)
    mf.run()
    occ = mf.mo_occ
    nmo = (occ>0).sum()

    e1 = mf.e_tot
    es = mf.mo_energy[nmo-1:nmo+1]
    print ' e_tot, e_homo, e_lumo = %.4f %.4f %.4f'%(e1, es[0], es[1])

    # in pyscf, orbital \chi_k = \sum_i \phi_i * A_{ik}
    # thus, \rho = \sum_k \chi_k^{*} \chi_k
    #            = \sum_{k,i,j} A_{ik}^* S_{ij} A_{jk}
    #            = \sum_{i,j} (A.T * S * A)_kk
    #
    #            = \sum_{k,i,j} A_{jk} * (A'^T)_{ki} S_{ij}
    #            = \sum_{i,j} (\sum_k A_{jk} * (A'^T)_{ki}) S_{ij}
    #            = \sum_{i,j} D_{ji}*S_{ij}
    # where D = A * A.T (summed over all occupied orbitals, indexed by `k)
    #
    A = mf.mo_coeff
    print ' A = '
    print A
    #occ[nmo] += 1
    #print ' occ = ', occ

    #dm1 = reduce( np.dot, (A, np.diag(occ), A.T) )
    dm1 = mf.make_rdm1(A, mo_occ=occ)
    ne1 = np.trace( np.dot(dm1,s1) )
    e1_u = mf.energy_tot(dm=dm1)
    print ' dE = ', e1_u - e1
    #print ' a) n = ', ne1, ', csc=', reduce(np.dot, (A.T,s1,A))

    C = np.linalg.solve(B, A)
    print ' C = ',
    print C
    #C = np.dot(A, np.linalg.inv(B))

    dm2 = reduce( np.dot, (C, np.diag(occ), C.T) )
    print ' dm2 = '
    print dm2
    ne2 = np.trace( np.dot(dm2,s2))
    msg = ' ** ne1=%.2f, ne2=%.2f'%(ne1,ne2)
    assert np.abs(ne2-ne1)<=1e-8, msg
    #print ' N_e = ', ne2, ', csc=', reduce(np.dot, (C.T,s2,C))

    #idx = [0,1,2,3,4, 11,12,13]
    n = int(fn[1])
    #idx = [0,1,2,3,4] + [ 5*n+j for j in range(2) ]
    #idx = range(10) #+ range(30,35) #idx = [7*5+i for i in range(5)] + [48,49]
    #print ' dm2 = ', dm2[idx][:,idx]
    #print ' C = ', C[idx][:,idx]

