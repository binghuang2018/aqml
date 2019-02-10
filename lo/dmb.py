

import ase.visualize as av
import ase, copy
import numpy as np
import ase.io as aio

import itertools as itl
import scipy.spatial.distance as ssd

from cheminfo.molecule.core import *
from cheminfo.molecule.elements import *

import cheminfo.lo.rotate as clr
import cheminfo.lo.dm as cld
from cheminfo.lo.dmx import *
from cheminfo.lo.dmml import *

#reload(cld)
import os,sys
import numpy.linalg as LA

global rcs
rcs = Elements().rcs

np.set_printoptions(precision=4,suppress=True)

def normalize(_vs, _signs=None):
    _vsu = []
    for i,vi in enumerate(_vs):
        vo = np.array(vi)/LA.norm(vi)
        if _signs is not None:
            vo *= _signs[i]
        _vsu.append( vo )
    return np.array(_vsu)

def get_hyb_map(vsp1, vsb1):
    idx = []; vals = []
    for i1,vp1 in enumerate(vsp1):
        _vals = []
        for i2,vb1 in enumerate(vsb1):
            #_vals.append( np.abs(np.dot(vp1,vb1)) )
            _vals.append( np.dot(vp1,vb1) )
        seq = np.argsort(_vals)
        _id = seq[-1]
        #if not (_vals[seq[-1]] > _vals[seq[-2]]):
        #print ' _vals = ', _vals
        idx.append( _id ); vals.append( _vals[_id] )
    return idx #, vals

def get_dm_obj(obj, basis='sto-3g', meth='b3lyp', idx=None, idx2=None, iprt=False, \
               scaling_factor=1.0, rotate_angle=0.0, v=[1.,1.,1.]):
    spin=0; a=0.; verbose=3
    if isinstance(obj,str):
        assert os.path.exists(obj)
        m = aio.read(obj)
    else:
        m = obj
    m.rotate(rotate_angle*np.pi/180,v)
    zs, coords = m.numbers, m.positions
    coords *= scaling_factor
    #fno = fn[:-4] + '.out'
    obj2 = cld.density_matrix(zs, coords, output=None, basis=basis, meth=meth, \
                         spin=spin, verbose=verbose, iprt=iprt)
    obj2.calc_ca_dm(idx=idx, idx2=idx2)
    return obj2

def get_dm_aa(obj, ia, ja, bst='sto-3g', ihao=True):
    assert bst=='sto-3g'
    zs = obj.zs
    #nheav = (np.array(zs)>1).sum()
    aoidxs = []
    _aoidxs = obj.aoidxs
    for i in [ia,ja]:
        aoidxs.append( list(_aoidxs[i]) )
    ias1, ias2 = aoidxs
    #print 'ias1=', ias1, ', ias2=',ias2
    if ihao:
        _dm = obj.dm1
    else:
        _dm = obj.dm0
    return _dm[ias1][:,ias2]


def retrieve_bond_env(b, m, iPL='1,2'):
    assert type(m) is RawMol
    #assert b[1]>b[0], '#ERROR: no need to consider the case I>J for bond pair [I,J] since \gamma_{IJ} = \gamma_{JI}'
    ias0 = np.arange(m.na)
    #print('total atoms:', m.na)
    g1,g2 = m.g,m.g2
    na = m.na
    g = np.logical_or(g1,g2).astype(np.int)
    cns1 = g1.sum(axis=0)
    cns = g.sum(axis=0)
    ias = set([])
    ia1,ia2= b
    iasb = [ia1 ]
    if ia2 != ia1: iasb += [ia2]
    for i in iasb:
        cni = cns1[i]
        if iPL in ['1,2']:
            if cni == 1: # get neighbors {j} for which PL(i,j) <= 2
                fil = (g1[i]>0) # np.logical_or(g[i]>0,g2[i]>0)
                ias_pl1 = ias0[fil]
                ias.update( ias_pl1 )
                # now get neighbors with PL = 2
                for ja in ias_pl1:
                    fil = (g1[ja]>0) # np.logical_or(g[ja]>0,g2[ja]>0)
                    ias.update( ias0[fil] )
                # now get vdw-bonded neighbors
                fil_vdw = (g2[i]>0)
                ias_pl1_vdw = ias0[fil_vdw]
                ias.update( ias_pl1_vdw )
            else: # get neighbors {j} for which PL(i,j) = 1
                fil = (g[i]>0)# np.logical_or(g[i]>0,g2[i]>0)
                ias_pl1 = ias0[fil]
                ias.update( ias_pl1 )
        elif iPL in ['2']:
            fil = (g1[i]>0) #np.logical_or(g[i]>0,g2[i]>0)
            ias_pl1 = ias0[fil]
            ias.update( ias_pl1 )
            # now get neighbors with PL = 2
            for ja in ias_pl1:
                fil = (g1[ja]>0) # np.logical_or(g[ja]>0,g2[ja]>0)
                ias.update( ias0[fil] )
            # now get vdw-bonded neighbors
            if cni == 1:
                fil_vdw = (g2[i]>0)
                ias_pl1 = ias0[fil_vdw]
                ias.update(ias_pl1)
                for ja in ias_pl1:
                    fil = (g1[ja]>0)
                    ias.update( ias0[fil] )
            else:
                pass # no action
        else:
            raise '#ERROR: unknown `iPL'

    ias_new = []
    bonds1 = []
    ia_new = m.na
    coords = m.coords
    zs = m.zs
    for i in iasb:
        zi = m.zs[i]
        cni = cns[i]
        # append H atoms to CN-deficient atoms, e.g., >N-, >O, -F, ...
        if zi==8:
          if cni==2:
            triple = [i]+list(ias0[g1[i]>0])
            cs = m.coords[triple]
            v1,v2 = clr.get_v12_sp3(cs)
            d = np.sum( rcs[ [1,zi] ] )
            ias_new += [ia_new,ia_new+1]
            bonds1+= [[i,ia_new],[i,ia_new+1]]
            ia_new += 2
            zs = np.concatenate( (zs,[1,1]) )
            coords = np.concatenate( (coords, [m.coords[i]+v1*d,m.coords[i]+v2*d]) )
          elif cni==3:
            coords3 = m.coords[ ias0[g[i]>0] ]
            vs3 = [ coord-m.coords[i] for coord in coords ]
            nrms3 = [ np.linalg.norm(vi) for vi in vs3 ]
            centre = np.mean([ vs3[iv]/nrms3[iv] for iv in range(3) ], axis=0)
            vh = m.coords[i]-centre
            nrmh = vh/np.linalg.norm(vh)
            d = np.sum( rcs[ [1,zi] ] )
            ias_new += [ia_new,]
            bonds1+= [[i,ia_new],]
            ia_new += 1
            zs = np.concatenate( (zs,[1,]) ) 
            coords = np.concatenate( (coords, [m.coords[i]+vh*d]) )
          elif cni==4:
            pass #sys.exit('#ERROR:')
          else:
            sys.exit('#ERROR: `cni not supported')
    gu = np.zeros((ia_new,ia_new))
    #print('total atoms:', ia_new)
    #print('neighbors of atom 6: ', ias0[g1[6]>0])
    gu[:na,:na] = g[:na,:na]
    #print('new bonds=',bonds1)
    for b1 in bonds1:
        ib1,ie1 = b1
        gu[ib1,ie1] = gu[ie1,ib1] = 1
    ias0 = np.concatenate( (ias0,ias_new) )
    #print('neighbors of atom 6: ', ias0[gu[6]>0])

    # check CN's
    cnsu = gu.sum(axis=0).astype(np.int)
    #print('cnsu=',cnsu)
    for i in iasb:
        if np.any(zs[i]==np.array([6,7,8,9])):
            if cnsu[i]-4 != 0: 
                sys.exit('#ERROR: cni!=4??')

    #print 'ias=',ias
    iasr = list(ias.difference(set(iasb)))
    iasr.sort()
    iasU = iasb + iasr + ias_new
    #print iasU
    tags = iasU
    o = m.coords[ia1]
    mi = ase.Atoms(numbers=zs[iasU], positions=coords[iasU]-o, tags=tags)

    # now rotate mol s.t. bond `b is aligned along  z-axis
    if ia2 != ia1:
        v0 = mi.positions[1]
        vz = [0,0,1]
        v = v0/np.linalg.norm(v0)
        dp = np.dot(vz,v); #print 'dp=',dp
        ang = np.arccos(dp) * 180./np.pi
        if dp != 1:
            vnrm = np.cross(v,vz)
            mi.rotate(ang,vnrm) # always counter-clockwise rotation
    return mi,gu[iasU][:,iasU]


class bob(object):

    def __init__(self, r, q):
      self.r = r
      self.q = q

    def get_bob_query(m,rcut=3.6):
      """ get bob of query molecule
      as well as numbers of each type of bonds. The latter will be used
      to fix the length of bob vector for the reference molecule (AMONS)"""
      na = len(m)
      zs = m.numbers
      esa = zs #-1. * zs**2.4
      zs1 = np.unique(zs)
      nz1 = len(zs1)
      izs = np.arange(nz1)
      boa = np.zeros((na,nz1))
      for i in range(na):
          boa[i,izs[zs[i]==zs1][0]] = esa[i]

      zpairs = [ (zi,zi) for zi in zs1 ] + list( itl.combinations(zs1,2) )
      dic0 = {}
      for zpair in zpairs: dic0[zpair] = []
      ds = ssd.squareform(ssd.pdist(m.positions))

      dics = []; ns = []
      for i in range(na):
          dic = dic0.copy()
          for j in range(na):
              if i==j or ds[i,j]>rcut: continue
              zi,zj = zs[i],zs[j]
              pair = (min(zi,zj), max(zj,zi))
              if pair in dic.keys():
                  dic[pair] +=[ ds[i,j] ]# [zi*zj/ds[i,j]]
          _ns = []
          for zpair in zpairs:
              _bob = dic[zpair]
              _l = len(_bob)
              if _l > 0: _bob.sort(reverse=True)
              _ns.append( _l )
          ns.append(_ns); dics.append(dic)
      nbs = np.max(ns, axis=0) #; print 'nbs=',nbs
      l = sum(nbs)
      idx2 = np.cumsum(nbs)
      idx1 = np.array([0]+list(idx2[:-1]),dtype=int)
      bob = np.zeros((na,l))
      for i in range(na):
          dic = dics[i]
          for j,zpair in enumerate(zpairs):
              _bob = dic[zpair]
              lenj = len(_bob)
              ib = idx1[j]; ie = ib+lenj
              bob[i, ib:ie] = _bob
      bob = np.concatenate((boa,bob), axis=1)
      return zs1, zpairs,idx1,idx2,bob

    def get_bob_ref(m,zs1,zpairs,idx1,idx2,rcut=3.6):
      na = len(m)
      zs = m.numbers
      zs1 = np.unique(zs)
      nz1 = len(zs1)

      esa = zs # -1. * zs**2.4
      izs = np.arange(nz1)
      boa = np.zeros((na,nz1))
      for i in range(na):
          boa[i,izs[zs[i]==zs1][0]] = esa[i]

      nb = len(zpairs)
      counts = [0,]*nb
      ds = ssd.squareform(ssd.pdist(m.positions))
      l = idx2[-1]
      _bob = np.zeros((na,l))
      for i in range(na):
          for j in range(na):
              if i==j or ds[i,j]>rcut: continue
              zi,zj = zs[i],zs[j]
              pair = (min(zi,zj), max(zj,zi))
              ipair = zpairs.index( pair )
              ib = idx1[ipair] + counts[ipair]
              _bob[i,ib] = ds[i,j] # zi*zj/ds[i,j]
              counts[ipair] += 1
      bob = np.zeros((na,l))
      for i in range(nb):
          ib, ie = idx1[i],idx2[i]
          t = _bob[:,ib:ie]
          t.sort(axis=1)
          bob[:,ib:ie] = t
      bob = np.concatenate((boa,bob), axis=1)
      return bob

    def get_mapped_idxs_bob(ref, q, xref, xq, icase=2):
      # permutate idxs of atoms in `q so that d(BOB_i, BOB_j) is minimized
      n1,n2 = len(ref),len(q)
      #assert n1 == n2
      #n = max(n1,n2)
      #cmr = get_cm(ref,n,iz=T)
      #cmq = get_cm(q,n,iz=T)
      tags_r = ref.get_tags()
      tags_q = q.get_tags()
      dsx = []
      if icase==1:
          raise '#not implemented yet'
      else:
          # bonds
          perms = list( itl.permutations( list(np.arange(2,n1)) ) )
          for _perm in perms:
              perm = [0,1]+list(_perm) #+ list(np.arange(n1,n))
              vr = xref[ tags_r[perm] ]
              vq = xq[ tags_q[:n1] ]
              dsx.append( np.sum(np.abs(vr-vq)) )
      return np.array(dsx), perms


def get_cm(m,n,iz=False):
    na = len(m)
    cm = np.zeros((n,n))
    ds = ssd.squareform(ssd.pdist(m.positions))
    np.fill_diagonal(ds, 1.0)
    if iz:
        zs = m.numbers
        X, Y = np.meshgrid(zs, zs)
        diag = list( -np.array(zs)**2.4) + [0.,]*(n-na)
    else:
        X, Y = 1., 1.
        diag = [0.,]*n
    cm[:na,:na] = X*Y/ds
    np.fill_diagonal(cm, diag)
    return cm

def get_angle_xy_plane(v0,v):
    v0[2]=0.; v[2]=0.
    v0 = v0/np.linalg.norm(v0)
    v = v/np.linalg.norm(v)
    dp = np.dot(v0,v); #print 'dp=',dp
    ang = np.arccos(dp) * 180./np.pi
    # be cautious here!
    nrm = np.dot(np.cross(v0,v), [0,0,1])
    c = 360.
    #print('ang=',ang, ', nrm=',nrm)
    if np.sign(nrm) == -1: ango = -1.*ang + c
    return ang # np.sign(nrm)*ang

def get_rmsd(ref,q):
    """ calculate rmsd of two molecules
    Assume that they have been centered and aligned
    Note: the first atom is always located at the center
    and the 2nd atom being aligned to +z axis"""
    na1, na2 = len(ref), len(q)
    ias1, ias2 = np.arange(na1), np.arange(na2)
    zs1, zs2 = ref.numbers, q.numbers
    fl1, fl2 = (zs1>1), (zs2>1)
    nheav1, nheav2 = fl1.sum(), fl2.sum()
    zs1_heav, zs2_heav= zs1[fl1], zs2[fl2]
    ias1_heav, ias2_heav = ias1[fl1], ias2[fl2]
    n1 = max(nheav1,nheav2)
    n2 = min(nheav1,nheav2)
    ps1, ps2 = ref.positions, q.positions

    ds = []
    #for _perm in itl.permutations(np.arange(1,n1)):
    #    perm = list(_perm)
    #    i1, i2 = ias1_heav[:n2], ias2_heav[perm[:n2]]
    #    ds.append( np.sqrt( np.sum( np.sum( (ps1[i1]-ps2[i2])**2, axis=1 ) * ws ) ))

    perms = list( itl.permutations(np.arange(n1)) )
    if nheav1 > nheav2:
        for _perm in perms:
            perm = list(_perm)
            i1, i2 = ias1_heav[perm], ias2_heav
            ws = []
            for i in range(n2):
                dz = zs1[i1[i]]-zs2[i2[i]]
                if dz==0:
                    ws.append( 1.0 )
                else:
                    ws.append(abs(dz))
            ws += list(zs1[i1[n2:]])
            rs = list(np.sum( (ps1[i1[:n2]]-ps2[i2])**2, axis=1 )) + list(np.sum(ps1[i1[n2:]]**2, axis=1))
            ds.append( np.sqrt( np.sum( np.array(rs) * np.array(ws) ) ))
    else:
        for _perm in perms:
            perm = list(_perm)
            i1, i2 = ias1_heav, ias2_heav[perm]
            ws = []
            for i in range(n2):
                dz = zs1[i1[i]]-zs2[i2[i]]
                if dz==0:
                    ws.append( 1.0 )
                else:
                    ws.append(abs(dz))
            ws += list(zs2[i2[n2:]])
            rs = list(np.sum( (ps1[i1]-ps2[i2[:n2]])**2, axis=1 )) + list(np.sum(ps2[i2[n2:]]**2, axis=1))
            ds.append( np.sqrt( np.sum( np.array(rs) * np.array(ws) ) ))
    dmin = min(ds)
    #print( 'permutation=', perms[ ds.index(dmin) ] )
    return min(ds)

def calc_ang1(v1,v2):
    """
    if ang(v1,v2) > 90, return 180-ang
    else return ang
    """
    ang = np.arccos( np.abs(np.dot(v1,v2))/(np.linalg.norm(v1)*np.linalg.norm(v2)) )
    return ang * 180./np.pi

def get_mapped_idxs_cm(ref,gref,q,gq, ref2, q2, icase=2, debug=False):
    """
    ref2, q2: reference and query local geometry with PL=2 (w.r.t. the central atom)
    """
    # permutate idxs of atoms in `q so that d(BOB_i, BOB_j) is minimized
    n1,n2 = len(ref),len(q)
    assert n1 == n2
    n = max(n1,n2)
    xref = get_cm(ref,n,iz=F)
    xq = get_cm(q,n,iz=F)
    dsx = []
    istart = 2; bonds_r = [0,1]
    if icase==1:
        istart = 1; bonds_r = [0]
    perms = list( itl.permutations( list(np.arange(istart,n1)) ) )
    for _perm in perms:
        perm = bonds_r+list(_perm)
        vr = xref[perm][:,perm]
        vq = xq
        dsx.append( np.sum(np.abs(vr-vq)) )

    if debug: print('n1=',n1)
    if debug: print('dsx=',dsx)

    # prune `perms
    seq = np.argsort(dsx)
    dmin = np.min(dsx)
    perms_c = [] # chosen `perms
    #print('ref.ps = ',ref.positions, 'q.ps=',q.positions)
    angs_c = []
    for i in seq:
        if abs(dsx[i]-dmin) < 0.3: ##########################################################
            perm = perms[i]
            # now compute the rotation angles
            angs = []
            for j in range(istart,n):
                posr,posq = ref.positions[perm[j-istart]],q.positions[j]
                if np.all([ np.linalg.norm(vj[:2])<= 0.36 for vj in [posr,posq]]): continue
                #print('a1=',perm[j-istart],'a2=',j,'pos1=',posr,'posq=',posq)
                _ang = get_angle_xy_plane(posr,posq)
                assert not np.isnan(_ang), '#ERROR: ang=NaN??'
                angs.append(_ang)
            std = np.std(angs)
            ang = np.mean(angs)
            if debug: print( 'perm=',perm, ', std=', std, ', angs=',np.array(angs))
            if std < 30.: #15.:
                perms_c.append(perm)
                angs_c.append(ang)
    nperm = len(perms_c)
    #print( 'perms_c=', perms_c, ', angs_c=',angs_c)
    if nperm == 0:
        print( ' * Warning: you may need mirror symmetry to rectify this!' )
        print( '            For now, we simply neglect such case, i.e., skip' )
        print( '            this very bond as a training point')
        return []
    elif nperm == 1:
        perm_out = list(perms_c[0])
    else:
        diffs = []
        #av.view(q2)
        for j,perm in enumerate(perms_c):
            copy_ref2 = ref2.copy()
            copy_ref2.rotate(angs_c[j], 'z')
            #av.view(copy_ref2)
            rmsd = get_rmsd(copy_ref2,q2)
            #print( 'perm=',perm, ', rsmd=',rmsd)
            diffs.append( rmsd )
        seq = np.argsort(diffs)
        dmin = diffs[seq[0]]
        if debug: print('diffs=',diffs)
        #assert diffs[seq[1]]-dmin > 0.2 #??
        perm_out = list(perms_c[seq[0]]  )
    # the relative idxs of the first two atoms are retained

    # at last, get tag idx
    tags_r = ref.get_tags()
    iasr = np.arange(len(ref))
    iasq = np.arange(len(q))
    tags_q = q.get_tags()

    iasr_f = tags_r[np.array(bonds_r+perm_out, dtype=int)]
    iasq_f = tags_q
    dic = dict(zip(iasq_f,iasr_f))

    #print( 'iasr_f = ', iasr_f)
    #print( 'iasq_f = ', iasq_f)

    idxs = []
    if debug: print('bonds_r=',bonds_r)
    for ia in bonds_r:
        idx = [0]  # maybe valid for sto-3g only
        if ref.numbers[ia] > 1:
            nbrs_ref_i = tags_r[ iasr[ gref[ia]>0 ] ]
            nbrs_q_i = tags_q[ iasq[ gq[ia]>0 ] ]
            if debug: print('nbrs_ref_i=',nbrs_ref_i, 'nbrs_q_i=',nbrs_q_i)
            c = copy.copy(nbrs_q_i)
            c.sort()
            jdx = [ dic[j] for  j in c ]
            jdx2 = copy.copy(jdx)
            jdx2.sort()
            #print('jdx=',jdx, ', jdx2=',jdx2)
            kdx = []
            for k in jdx:
                kdx.append( jdx2.index(k)+1 )
            idx += kdx
        if debug: print('idx=',idx)
        idxs.append( idx )
    return idxs


def get_mapping(mr,br,mq,bq,debug=False):
    m1,m4 = mr,mq
    zs = m1.numbers
    coords = m1.positions
    rawm_ref = RawMol(list(zs), coords)
    zs = m4.numbers
    coords = m4.positions
    rawm_q = RawMol(list(zs), coords)
    b = br #[2,3] #[1,8] # [0,9]
    sm_ref,gref = retrieve_bond_env(b, rawm_ref, iPL='1,2') #'1,2')
    sm_ref2,gref2 = retrieve_bond_env(b, rawm_ref, iPL='2')
    #av.view(sm_ref2)
    #av.view(sm_ref)

    b2 = bq #[4,5] # [2,13] # [5,15]
    sm_q,gq = retrieve_bond_env(b2, rawm_q, iPL='1,2') #'1,2')
    sm_q2,gq2 = retrieve_bond_env(b2, rawm_q, iPL='2')
    #av.view(sm_q2)
    #av.view(sm_q)

    ots = get_mapped_idxs_cm(sm_ref,gref,sm_q,gq,sm_ref2, sm_q2, icase=len(set(b)), debug=debug)
    if len(ots) == 1:
        ots = ots*2
    return ots

def get_shuffle(a,b):
    n1,n2 = a.shape
    s1,s2 = range(1,n1), range(1,n2)
    seq1 = [ [0]+list(si) for si in list(itl.permutations(s1)) ]
    seq2 = [ [0]+list(si) for si in list(itl.permutations(s2)) ]
    n1 = len(seq1)
    n2 = len(seq2)
    d = 999.
    for i in range(n1):
        for j in range(n2):
            i1 = seq1[i]; i2 = seq2[j]
            di = np.mean(np.abs(b[i1][:,i2]-a))
            if di < d:
                d = di
                i1o = i1; i2o = i2
    return i1o,i2o,d

def ready_pyscf(fs,scaling_factors=None,rotate_angles=None,v=[1.,1.,1]):
    objs1 = []
    for i,f1 in enumerate(fs):
        print( ' now ', f1)
        scaling_factor = 1.0
        rotate_angle = 0.0
        if scaling_factors is not None:
            scaling_factor = scaling_factors[i]
            print('     + scaling_factor = ', scaling_factor)
        if rotate_angles is not None:
            rotate_angle = rotate_angles[i]
            print('     + rotate_angle = ', rotate_angle)
        objs1 += [ get_dm_obj(f1,scaling_factor=scaling_factor,rotate_angle=rotate_angle,v=v)  ]
    return objs1

def get_dmxs(fs, objs, brs, bq, debug=False):
    """
    return: updated `brsc (i.e., BondS of Ref Chosen for training)
    """
    o1,o2 = objs
    #fs = [ 'test/'+fi+'.xyz' for fi in ['c06h14', 'c12h26'] ] #'c07h16', 'c08h18',
    ms = [ aio.read(f) for f in fs ]
    m1, m2 = ms

    mr,mq = m1,m2
    brsc = [] # BondS of Ref Chosen for training
    y1 = []
    dm2 = get_dm_aa(o2,bq[0],bq[1])
    y2 = np.array([ dm2.ravel() ])
    for br in brs:
        dm1 = get_dm_aa(o1,br[0],br[1])
        ots = get_mapping(mr,br,mq,bq,debug=debug)
        if len(ots) == 0: continue
        i1, i2 = ots
        dm1u = dm1[i1][:,i2]
        y1.append(dm1u.ravel()); brsc.append(br)
        print( 'bond=(%3d,%3d)'%(br[0],br[1]), ' max deviation: %.5f'% np.max(np.abs(dm1u-dm2)) )
    y1 = np.array(y1)
    return brsc, y1, y2

def get_newidxs(fs, objs, br, bq):
    o1,o2 = objs

    #fs = [ 'test/'+fi+'.xyz' for fi in ['c06h14', 'c12h26'] ] #'c07h16', 'c08h18',
    ms = [ aio.read(f) for f in fs ]
    m1, m2 = ms

    mr,mq = m1,m2
    #br,bq = [1,3],[3,5]
    dm1 = clb.get_dm_aa(o1,br[0],br[1])
    dm2 = clb.get_dm_aa(o2,bq[0],bq[1])

    ots = clb.get_mapping(mr,br,mq,bq)
    i1, i2 = ots
    print( ' max deviation: ', np.max(np.abs(dm1[i1][:,i2]-dm2)) )
    return ots


def get_lc(fs, objs, ims1, ims2, brs, bq, racut=3.6,rbcut=4.8):

    zs = []; coords = []; nas = []
    for fi in fs:
        mi = aio.read(fi)
        nas.append(len(mi)); zs += list(mi.numbers); coords += list(mi.positions)
    nas = np.array(nas,np.int)
    zs = np.array(zs,np.int)
    coords = np.array(coords)
    #racut,rbcut = 3.6,4.8 # 4.8, 4.8
    xd = XData(nas, zs, coords)
    xd.get_x(param={'racut':racut,'rbcut':rbcut})
    xs = np.array( xd.xsb )
    assert not np.any(np.isnan(xs))

    obj = dmml(xd)

    opt = 'ii'
    brsc, y1, y2 = get_dmxs(fs, objs, brs, bq)
    idxs_x1 = xd.get_idx(brsc, ims=ims1, opt=opt)
    idxs_x2 = xd.get_idx([bq], ims=ims2, opt=opt)
    x1, x2 = xs[idxs_x1], xs[idxs_x2]

    n2 = 1
    n1s = np.arange(1,len(brsc)+1)
    errs = []
    for n1 in n1s:
        ds2, y2_est = obj.krr(x1[:n1],y1[:n1],x2,kernel='g',c=1.0,l=1.e-8)
        dy2 = np.abs(y2_est-y2)
        denom = n2 * dy2.shape[1]
        mae, rmse, errmax = np.sum(dy2)/denom, np.sqrt(np.sum(dy2**2)/denom), np.max(dy2)
        errs.append([mae, rmse, errmax])
        print('  n1,  mae, rmse, delta_max = %d, %.5f %.5f %.5f'%(n1, mae, rmse, errmax) )
    return n1s, errs

def test(fs,objs,br,bq,debug=False):
    np.set_printoptions(precision=4,suppress=True)

    o1,o2 = objs

    #fs = [ 'test/'+fi+'.xyz' for fi in ['c06h14', 'c12h26'] ] #'c07h16', 'c08h18',
    ms = [ aio.read(f) for f in fs ]
    m1, m2 = ms

    mr,mq = m1,m2
    #br,bq = [1,3],[3,5]
    dm1 = get_dm_aa(o1,br[0],br[1])
    dm2 = get_dm_aa(o2,bq[0],bq[1])

    ots = get_mapping(mr,br,mq,bq,debug=debug)
    print('i1,i2=',ots)
    if len(ots) == 0:
        print('## we have to skip this bond')
    else:
        i1,i2 = ots
        ddm = dm1[i1][:,i2]-dm2
        print(ddm.T)
        print('mae=%.6f'% np.mean(np.abs(ddm)), 'max error=%.6f'% np.max(np.abs(ddm)))

    # manually shuffle the idxs to minimize ddm so as to check if the i1 & i2
    # obtained above is correct.
    i1,i2,d = get_shuffle(dm2,dm1)
    ddm = dm2[i1][:,i2] - dm1
    print(ddm)
    print('mae=%.6f'% np.mean(np.abs(ddm)), 'max error=%.6f'% np.max(np.abs(ddm)))
    print('i1,i2=',i1,i2)

