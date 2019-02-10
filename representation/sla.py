#!/usr/bin/env python

"""
SLATM repr generation: for test purpose only
"""

import numpy as np
import ase
import ase.data as ad
import scipy.spatial.distance as ssd
import itertools as itl

#import pyximport
import numpy as np
#pyximport.install(setup_args={'include_dirs':[np.get_include()]})
#import _bop_and_bot as mbc


from time import gmtime, strftime

global zgns
zgns = {1:1, 6:4, 7:5, 8:6, 9:7, 14:4, 15:5, 16:6, 17:7}

def get_date(*strings):
    s = ''
    for si in strings:
        s += si; break

    print('%60s'%s, ' ', strftime("%Y-%m-%d %H:%M:%S", gmtime()))

def get_mbtypes(zs, nzmax, isymb=False):
    ntypes = []
    zs = np.array(zs); nzmax = np.array(nzmax)
    ntypes.append(len(zs))

    boas = [ [zi,] for zi in zs ]
    bops = [ [zi,zi] for zi in zs ] + list( itl.combinations(zs,2) )
    ntypes.append(len(bops))

    bots = []
    for i in zs:
        for bop in bops:
            j,k = bop
            tas = [ [i,j,k], [i,k,j], [j,i,k] ]
            for tasi in tas:
                if (tasi not in bots) and (tasi[::-1] not in bots):
                    nzsi = [ (zj == tasi).sum() for zj in zs ]
                    if np.all(nzsi <= nzmax):
                        bots.append( tasi )
    ntypes.append(len(bots))

    mbtypes = boas + bops + bots
    if isymb:
        mbtypes_u = []
        for mbtype in mbtypes:
            mbtypes_u.append( [ ad.chemical_symbols[zi] for zi in mbtype ])
        mbtypes = mbtypes_u

    return mbtypes,ntypes

def update_m(m, ia, rcut=9.0, pbc=None):
    """
    retrieve local structure around atom `ia
    for periodic systems (or very large system)
    """

    c = m.cell
    v1, v2, v3 = c
    ls = ssd.norm(c, axis=0)

    nns = []; ns = []
    for i,li in enumerate(ls):
        n1_doulbe = rcut/li
        n1 = int(n1_doulbe)
        if n1 - n1_doulbe == 0:
            n1s = list(range(-n1, n1+1)) if pbc[i] else [0,]
        elif n1 == 0:
            n1s = [-1,0,1] if pbc[i] else [0,]
        else:
            n1s = list(range(-n1-1, n1+2)) if pbc[i] else [0,]

        nns.append(n1s)

    #get_date(' # 001,  ')
    #print ' -- nns = ', nns

    n1s,n2s,n3s = nns

    n123s_ = np.array( list( itl.product(n1s,n2s,n3s) ) )
    n123s = []
    for n123 in n123s_:
        n123u = list(n123)
        if n123u != [0,0,0]: n123s.append(n123u)

    nau = len(n123s)
    n123s = np.array(n123s, np.float)
    #print ' -- n123s = ', n123s

    coords = m.positions; zs = m.numbers; ai = m[ia]; cia = coords[ia]
    na = len(m)
    if na == 1:
        ds = np.array([[0.]])
    else:
        ds = ssd.squareform( ssd.pdist(coords) )

# also return `idxs0, which stores the environments of the atoms in the
# "NEW" molecule corresponding to the atom index in the OLD small unit cell
    idxs0 = []

    #print ' -- na, ds = ', na,ds
    mu = ase.Atoms([], cell=c); mu.append( ai ); idxs0.append( ia )
    for i in range(na) :
        di = ds[i,ia]
        if di <= rcut:
            if di > 0:
                mu.append( m[i] ); idxs0.append( i )

# add new coords by translation
            #print ' !! '
            #ts = np.dot(n123s, c); print 'size(ts) = ', ts.shape
            ts = np.zeros((nau,3))
            for iau in range(nau):
                ts[iau] = np.dot(n123s[iau],c)

            coords_iu = coords[i] + ts #np.dot(n123s, c)
            #print ' !!2'
            dsi = ssd.norm( coords_iu - cia, axis=1);
            #print ' -- dsi = ', dsi
            filt = np.logical_and(dsi > 0, dsi <= rcut); nx = filt.sum()
            mii = ase.Atoms([zs[i],]*nx, coords_iu[filt,:])
            for aii in mii: mu.append( aii ); idxs0.append( i )

    return mu, idxs0


def get_boa(z1, zs_):
    return z1*np.array( [(zs_ == z1).sum(), ])
    #return -0.5*z1**2.4*np.array( [(zs_ == z1).sum(), ])

def get_sbop(mbtype, m, zsm=None, local=False, ia=None, normalize=True, sigma=0.05, \
             rcut=4.8, dgrid=0.03, ipot=True, cspeed=[True,False], iprt=False, \
             zg=False, pbc='000', rpower=6):
    """
    zg -- group number of Z
    """

    if cspeed[1]: # the first entry won't be used below (it's for getting idxs of 3-body terms)
        import mbc

    z1, z2 = mbtype

    if local:
        assert ia != None, '#ERROR: plz specify `za and `ia '

    if zsm is None: zsm = m.numbers
    if pbc != '000':
        #get_date(' #1, ia = %s '%ia)
        assert local, '#ERROR: for periodic system, plz use atomic rpst'
        m, idxs0 = update_m(m, ia, rcut=rcut, pbc=pbc)
        zsmu = [ zsm[i] for i in idxs0 ]; zsm = zsmu

        # after update of `m, the query atom `ia will become the first atom
        ia = 0

    na = len(m)
    coords = m.positions
    ds = ssd.squareform( ssd.pdist(coords) )

    ias = np.arange(na)
    ias1 = ias[zsm == z1]
    ias2 = ias[zsm == z2]

    if z1 == z2:
        #if local:
        #    if za != z1:
        #        dsu = []
        #    else:
        #        ias2u = np.setdiff1d(ias1,[ia,])
        #        dsu = np.array([ ds[ia,i] for i in ias2u ])
        #else:
        ias12 = list( itl.combinations(ias1,2) )
    else:
        #if local:
        #    if za not in [z1,z2]:
        #        dsu = []
        #    elif za == z1:
        #        dsu = np.array([ ds[ia,i] for i in ias2 ])
        #    elif za == z2:
        #        dsu = np.array([ ds[i,ia] for i in ias1 ])
        #    else:
        #        raise '#ERROR'
        #else:
        ias12 = itl.product(ias1,ias2)

    # !!!!!!!!!!!!
    # the following 3 lines of code are not compatible with
    # the `Z now, cause they are all added by 1000
    # !!!!!!!!!!!
    if zg: # use group number (or num_valence_electron) instead of Z
        zsm = np.array([ zgns[zi] for zi in m.numbers ])
        z1, z2 = [ zgns[zi] for zi in mbtype ]


    if local:
        dsu = []; icnt = 0
        for j1,j2 in ias12:
            if ia == j1 or ia == j2:
                dsu.append( ds[j1,j2] )
            icnt += 1
    else:
        dsu = [ ds[i,j] for (i,j) in ias12 ]

    dsu = np.array(dsu)

    #print ' -- (d_min, d_max) = (%.3f, %.3f)'%(np.min(ds), np.max(ds))

    # bop potential distribution
    r0 = 0.1
    nx = (rcut - r0)/dgrid + 1
    xs = np.linspace(r0, rcut, nx)
    ys0 = np.zeros(xs.shape)

    # update dsu by exluding d > 6.0
    nr = dsu.shape[0]
    if nr > 0:
        dsu = dsu[ dsu <= rcut ]
        nr = len(dsu)

    #print ' -- dsu = ', dsu

    coeff = 1/np.sqrt(2*sigma**2*np.pi) if normalize else 1.0
    #print ' -- now calculating 2-body terms...'
    if ipot:
        # get distribution of 2-body potentials
        # unit of x: Angstrom
        c0 = (z1%1000)*(z2%1000)*coeff
        #print ' -- c0 = ', c0
        if cspeed[1]:
            ys = mbc.calc_es_bop(c0, sigma, xs, dsu, ys0)
        else:
            ys = ys0
            for i in range(nr):
                ys += ( c0/(xs**rpower) )*np.exp( -0.5*((xs-dsu[i])/sigma)**2 )
        ys *= dgrid
    else:
        # print distribution of distances
        c0 = coeff
        if cspeed[0]:
            ys = mbc.calc_rs_bop(c0, sigma, xs, dsu, ys0)
        else:
            ys = ys0
            for i in range(nr):
                ys += c0*np.exp( -0.5*((xs-dsu[i])/sigma)**2 )

    return xs, ys

def vang(u,v):
    cost = np.dot(u,v)/(np.linalg.norm(u) * np.linalg.norm(v))
# sometimes, cost might be 1.00000000002, then np.arccos(cost)
# does not exist!
    u = cost if abs(cost) <= 1 else 1.0
    return np.arccos( u )

def cvang(u,v):
    return np.dot(u,v)/np.sqrt(np.dot(u,u)*np.dot(v,v))

def get_sbot(mbtype, m, zsm=None, local=False, ia=None, normalize=True, sigma=0.05, label=None, \
             rcut=4.8, dgrid=0.0262, ipot=True, cspeed=[True,False], iprt=False, \
             zg=False, pbc='000'):

    """
    sigma -- standard deviation of gaussian distribution centered on a specific angle
            defaults to 0.05 (rad), approximately 3 degree
    dgrid    -- step of angle grid
            defaults to 0.0262 (rad), approximately 1.5 degree
    """

    if np.any(cspeed):
        import mbc

    #get_date(' Program starts ')

    z1, z2, z3 = mbtype

    if local:
        assert ia != None, '#ERROR: plz specify `za and `ia '

    if zsm is None: zsm = m.numbers
    if pbc != '000':
        assert local, '#ERROR: for periodic system, plz use atomic rpst'
        m, idxs0 = update_m(m, ia, rcut=rcut, pbc=pbc)
        zsm = [ zsm[i] for i in idxs0 ]

        # after update of `m, the query atom `ia will become the first atom
        ia = 0

    na = len(m)
    coords = m.positions
    dsr = ssd.pdist(coords)
    #print ' -- minimal distance is %.2f'%( dsr.min() )
    ds = ssd.squareform( dsr )
    dminThresh = 0.5
    print(' -- dminThresh = %.2f'%dminThresh)
    for i in range(na):
        for j in range(i+1,na):
            if ds[i,j] <= dminThresh:
                print(' I, J, R_IJ = %6d, %6d, %12.6f'%(i,j,ds[i,j]))
    #get_date(' ds matrix calc done ')

    ias = np.arange(na)
    ias1 = ias[zsm == z1]; n1 = len(ias1)
    ias2 = ias[zsm == z2]; n2 = len(ias2)
    ias3 = ias[zsm == z3]; n3 = len(ias3)
    tas = []

    #print ' -- len(zsm) = ', len(zsm)

    if local:
        ia2 = ia
        if zsm[ia2] == z2:
            ias1u = ias1[ np.logical_and( ds[ias1,ia2] > 0, ds[ias1,ia2] <= rcut ) ]
            ias3u = ias3[ np.logical_and( ds[ias3,ia2] > 0, ds[ias3,ia2] <= rcut ) ]
            for ia1 in ias1u:
                for ia3 in ias3u:
                    d13 = ds[ia1,ia3]
                    if d13 > 0 and d13 <= rcut:
                        tasi = [ia1,ia2,ia3]
                        iok1 = (tasi not in tas)
                        iok2 = (tasi[::-1] not in tas)
                        if iok1 and iok2:
                            tas.append( tasi )
        tas = np.array(tas)
        #print ' -- tas = ', tas
    else:
        if cspeed[0]:
            # get the size of `tas first before calling
            # cython function
            if z1 == z2 and z3 == z2:
                ntas0 = n2*(n1-1)*(n3-2)
            elif z1 == z2 and z3 != z2:
                ntas0 = n2*(n1-1)*n3
            elif z1 != z2 and z3 == z2:
                ntas0 = n2*n1*(n3-1)
            elif z1 != z2 and z3 != z2:
                ntas0 = n2*n1*n3
            else:
                raise ' #unknow case??'

            tas0 = np.zeros((ntas0,3),np.int)

            ias1 = np.array(ias1) #, np.int32)
            ias2 = np.array(ias2) #, np.int32)
            ias3 = np.array(ias3) #, np.int32)

            ias2u = np.zeros(n2, np.int)
            ias3u = np.zeros(n3, np.int)
            tas = mbc.get_tidxs(ias1, ias2, ias3, ds, tas0, rcut,  ias2u,ias3u)
            #print ' -- tas = ', tas
        else:
            for ia1 in ias1:
                ias2u = ias2[ np.logical_and( ds[ia1,ias2] > 0, ds[ia1,ias2] <= rcut ) ]
                for ia2 in ias2u:
                    filt1 = np.logical_and( ds[ia1,ias3] > 0, ds[ia1,ias3] <= rcut )
                    filt2 = np.logical_and( ds[ia2,ias3] > 0, ds[ia2,ias3] <= rcut )
                    ias3u = ias3[ np.logical_and(filt1, filt2) ]
                    for ia3 in ias3u:
                        tasi = [ia1,ia2,ia3]
                        iok1 = (tasi not in tas)
                        iok2 = (tasi[::-1] not in tas)
                        if iok1 and iok2:
                            tas.append( tasi )

    # problematic with new Z, e.g., 1089 is actually Au (89)
    if zg: # use group number instead of Z
        zsm = np.array([ zgns[zi] for zi in m.numbers ])
        z1, z2, z3 = [ zgns[zi] for zi in mbtype ]

   #if local:
   #    tas_u = []
   #    for tas_i in tas:
   #        if ia == tas_i[1]:
   #            tas_u.append( tas_i )
   #    tas = tas_u
   ##print ' -- tas = ', np.array(tas)

    #get_date(' enumerating triples of atoms done ')

    d2r = np.pi/180 # degree to rad
    a0 = -20.0*d2r; a1 = np.pi + 20.0*d2r
    nx = int((a1-a0)/dgrid) + 1
    xs = np.linspace(a0, a1, nx)
    ys0 = np.zeros(nx, np.float)
    nt = len(tas)

    # u actually have considered the same 3-body term for
    # three times, so rescale it
    prefactor = 1.0/3

    # for a normalized gaussian distribution, u should multiply this coeff
    coeff = 1/np.sqrt(2*sigma**2*np.pi) if normalize else 1.0

    if iprt: get_date(' -- now calculating 3-body terms...')

    tidxs = np.array(tas, np.int)
    if ipot:
        # get distribution of 3-body potentials
        # unit of x: Angstrom
        c0 = prefactor*(z1%1000)*(z2%1000)*(z3%1000)*coeff
        if cspeed[1]:
            ys = mbc.calc_es_bot(c0, sigma, coords, xs, tidxs, ds, ys0)
        else:
            ys = ys0
            for it in range(nt):
                i,j,k = tas[it]
                # angle spanned by i <-- j --> k, i.e., vector ji and jk
                u = coords[i]-coords[j]; v = coords[k] - coords[j]
                ang = vang( u, v ) # ang_j
                #print ' -- (i,j,k) = (%d,%d,%d),  ang = %.2f'%(i,j,k, ang)
                cak = cvang( coords[j]-coords[k], coords[i]-coords[k] ) # cos(ang_k)
                cai = cvang( coords[k]-coords[i], coords[j]-coords[i] ) # cos(ang_i)
                ys += c0*( (1.0 + 1.0*np.cos(xs)*cak*cai)/(ds[i,j]*ds[i,k]*ds[j,k])**3 )*\
                                    ( np.exp(-(xs-ang)**2/(2*sigma**2)) )
        ys *= dgrid
    else:
        # print distribution of angles (unit: degree)
        sigma = sigma/d2r
        xs = xs/d2r
        c0 = 1

        if cspeed[1]:
            ys = mbc.calc_angs_bot(c0, sigma, coords, xs, tidxs, ds, ys0)
        else:
            ys = ys0
            for it in range(nt):
                i,j,k = tas[it]
                # angle spanned by i <-- j --> k, i.e., vector ji and jk
                ang = vang( coords[i]-coords[j], coords[k]-coords[j] )/d2r
                ys += c0*np.exp( -(xs-ang)**2/(2*sigma**2) )

    if iprt: get_date(' -- 3-body terms done')

    return xs, ys

def get_sla(m, zsm, mbtypes, ias=None, local=False, normalize=True, sigmas=[0.05,0.05], \
           dgrids=[0.03,0o3], rcut=4.8, iprt=False, noprt=False, alchemy=False, \
           cspeed=[True,False], zg=False, pbc='000', rpower=6):
    """
    smooth LATM rpst
    """

    if local:
        mbs = []
        na = len(m)
        if ias == None:
            ias = list(range(na))

        X2Ns = []
        for ia in ias:
            if not noprt: print('               -- ia = ', ia + 1)
            n1 = 0; n2 = 0; n3 = 0
            mbs_ia = np.zeros(0)
            icount = 0
            for mbtype in mbtypes:
                if iprt: print('  ++ mbtype, len(mbtype) = ', mbtype,len(mbtype))
                if len(mbtype) == 1:
                    mbsi = get_boa(mbtype[0], np.array([zsm[ia],])) #print ' -- mbsi = ', mbsi
                    if alchemy:
                        n1 = 1
                        n1_0 = mbs_ia.shape[0]
                        if n1_0 == 0:
                            mbs_ia = np.concatenate( (mbs_ia, mbsi), axis=0 )
                        elif n1_0 == 1:
                            mbs_ia += mbsi
                        else:
                            raise '#ERROR'
                    else:
                        n1 += len(mbsi)
                        mbs_ia = np.concatenate( (mbs_ia, mbsi), axis=0 )
                elif len(mbtype) == 2:
                    #print ' 001, pbc = ', pbc
                    mbsi = get_sbop(mbtype, m, zsm=zsm, local=local, ia=ia, normalize=normalize, \
                                    sigma=sigmas[0], dgrid=dgrids[0], rcut=rcut, \
                                    iprt=iprt, cspeed=cspeed, zg=zg, pbc=pbc, \
                                    rpower=rpower)[1]
                    mbsi *= 0.5 # only for the two-body parts, local rpst
                    #print ' 002'
                    if alchemy:
                        n2 = len(mbsi)
                        n2_0 = mbs_ia.shape[0]
                        if n2_0 == n1:
                            mbs_ia = np.concatenate( (mbs_ia, mbsi), axis=0 )
                        elif n2_0 == n1 + n2:
                            t = mbs_ia[n1:n1+n2] + mbsi
                            mbs_ia[n1:n1+n2] = t
                        else:
                            raise '#ERROR'
                    else:
                        n2 += len(mbsi)
                        mbs_ia = np.concatenate( (mbs_ia, mbsi), axis=0 )
                else: # len(mbtype) == 3:
                    mbsi = get_sbot(mbtype, m, zsm=zsm, local=local, ia=ia, normalize=normalize, \
                                    sigma=sigmas[1], dgrid=dgrids[1], rcut=rcut, \
                                    iprt=iprt, cspeed=cspeed, zg=zg, pbc=pbc)[1]
                    if alchemy:
                        n3 = len(mbsi)
                        n3_0 = mbs_ia.shape[0]
                        if n3_0 == n1 + n2:
                            mbs_ia = np.concatenate( (mbs_ia, mbsi), axis=0 )
                        elif n3_0 == n1 + n2 + n3:
                            t = mbs_ia[n1+n2:n1+n2+n3] + mbsi
                            mbs_ia[n1+n2:n1+n2+n3] = t
                        else:
                            raise '#ERROR'
                    else:
                        n3 += len(mbsi)
                        mbs_ia = np.concatenate( (mbs_ia, mbsi), axis=0 )

            mbs.append( mbs_ia )
            X2N = [n1,n2,n3];
            if X2N not in X2Ns:
                X2Ns.append(X2N)
        assert len(X2Ns) == 1, '#ERROR: multiple `X2N ???'
    else:
        n1 = 0; n2 = 0; n3 = 0
        mbs = np.zeros(0)
        for mbtype in mbtypes:
            if iprt: print('    ---- mbtype = ', mbtype)
            if len(mbtype) == 1:
                mbsi = get_boa(mbtype[0], zsm)
                if alchemy:
                    n1 = 1
                    n1_0 = mbs.shape[0]
                    if n1_0 == 0:
                        mbs = np.concatenate( (mbs, [sum(mbsi)] ), axis=0 )
                    elif n1_0 == 1:
                        mbs += sum(mbsi )
                    else:
                        raise '#ERROR'
                else:
                    n1 += len(mbsi)
                    mbs = np.concatenate( (mbs, mbsi), axis=0 )
            elif len(mbtype) == 2:
                mbsi = get_sbop(mbtype, m, zsm=zsm, normalize=normalize, sigma=sigmas[0], \
                                dgrid=dgrids[0], rcut=rcut, zg=zg, rpower=rpower,
                                cspeed=cspeed)[1]

                if alchemy:
                    n2 = len(mbsi)
                    n2_0 = mbs.shape[0]
                    if n2_0 == n1:
                        mbs = np.concatenate( (mbs, mbsi), axis=0 )
                    elif n2_0 == n1 + n2:
                        t = mbs[n1:n1+n2] + mbsi
                        mbs[n1:n1+n2] = t
                    else:
                        raise '#ERROR'
                else:
                    n2 += len(mbsi)
                    mbs = np.concatenate( (mbs, mbsi), axis=0 )
            else: # len(mbtype) == 3:
                mbsi = get_sbot(mbtype, m, zsm=zsm, normalize=normalize, sigma=sigmas[1], \
                                 cspeed=cspeed, dgrid=dgrids[1], rcut=rcut, zg=zg)[1]
                if alchemy:
                    n3 = len(mbsi)
                    n3_0 = mbs.shape[0]
                    if n3_0 == n1 + n2:
                        mbs = np.concatenate( (mbs, mbsi), axis=0 )
                    elif n3_0 == n1 + n2 + n3:
                        t = mbs[n1+n2:n1+n2+n3] + mbsi
                        mbs[n1+n2:n1+n2+n3] = t
                    else:
                        raise '#ERROR'
                else:
                    n3 += len(mbsi)
                    mbs = np.concatenate( (mbs, mbsi), axis=0 )

        X2N = [n1,n2,n3]

    return mbs,X2N

def get_pbc(m, d0 = 3.6):

    pbc = []

    c = m.cell
    ps = m.positions
    na = len(m); idxs = np.arange(na)

    for ii in range(3):
        psx = ps[:,ii]; xmin = min(psx)
        idxs_i = idxs[ psx == xmin ]
        ps1 = ps[idxs_i[0]] + c[ii]
        if np.min( ssd.cdist([ps1,], ps)[0] ) < d0:
            pbc.append( '1' )
        else:
            pbc.append( '0' )

    return ''.join(pbc)


def get_cns(m):
    """
    get the CNs of each atom in `m
    """

    thresh_ds = {78:3.2,}

    pbc = get_pbc(m)

    na0 = len(m); zs0 = m.numbers; cnsi = []
    if pbc != '000':
      for i in range(na0):
        thresh_d = thresh_ds[zs0[i]]
        mi = update_m(m, i, rcut=9.0, pbc=pbc)[0]
        coords = mi.positions
        na = len(mi); idxs = np.arange(na)
        ds = ssd.squareform( ssd.pdist(coords) )
        cni = np.logical_and(ds[0] > 0, ds[0] < thresh_d).sum()
        cnsi.append( cni )
    else:
        coords = m.positions
        ds = ssd.squareform( ssd.pdist(coords) )
        for i in range(na0):
            thresh_d = thresh_ds[zs0[i]]
            cni = np.logical_and(ds[i] > 0, ds[i] < thresh_d).sum()
            cnsi.append( cni )

    return cnsi

def count_unique_numbers(ns):

    nsu = np.unique(ns)
    nsu.sort()
    cnts = []
    for ni in nsu:
        cnts.append( (ni == ns).sum() )

    return nsu, cnts

def get_slas(ms, h5f=None, mIds=[0,-1], local=False, normalize=True, \
             sigmas=[0.05,0.05], dgrids=[0.03,0o3], rcut=4.8, Y=None, \
             iwrite=True, iprt=False, noprt=False, alchemy=False, \
             cspeed=[True,False], zg=False, rpower=6, icn=False, \
             imol=True):

    zsmax = set()
    nm = len(ms)
    if iprt: print(' -- nm = ', nm)

    if icn:
        thresh_ds = {78:3.0,}

        cnsmax = set()
        nas = []; cns = []; zs = []
        for m in ms:
            na = len(m); nas.append(na)
            coords = m.positions; idxs = np.arange(na);
            zsi = m.numbers; zs.append( zsi )
            cnsi = get_cns(m)
            cns.append( cnsi ); cnsmax.update( cnsi )

        zs_u = []; zs_ravel = []; zsmax_u = set()
        cnsmax = list(cnsmax)
        print(' -- cnsmax = ', cnsmax)
        for i in range(nm):
            na = nas[i]

            zsi = zs[i]
            cnsi = cns[i]; zsi_u = []
            for j in range(na):
                cnj = cnsi[j]
                idxj = cnsmax.index(cnj)
                zju = (idxj+1)*1000 + zsi[j]
                zsi_u.append( zju )
            #print '        ++ i, nai, nai2 = ', i,na,len(zsi_u)
            zs_ravel += zsi_u; zs_u.append( zsi_u )
            zsmax_u.update( zsi_u )

        zsmax = zsmax_u
        zs = zs_u
    else:
        zs_ravel = []; zs = []; nas = []
        for m in ms:
            #print ' -- m=', m
            zsi = m.numbers; zsil = list(zsi)
            zs.append( zsi ); na = len(m); nas.append(na)
            zsmax.update( zsil ); zs_ravel += zsil #[ 1000 + zi for zi in zsi ]

    zsmax = np.array( list(zsmax) )
    nass = []
    for i in range(nm):
        zsi = zs[i]
        nass.append( [ (zi == zsi).sum() for zi in zsmax ] )

    nzmax = np.max(np.array(nass), axis=0)
    nzmax_u = []
    if not imol:
        for nzi in nzmax:
            if nzi <= 2:
                nzi = 3
            nzmax_u.append(nzi)
        nzmax = nzmax_u

    #print ' -- zsmax, nzmax = ', zsmax, nzmax

    mbtypes,ntypes= get_mbtypes(zsmax, nzmax)
    if iprt:
        ntt = len(mbtypes)
        for iit in range(ntt):
            print(' -- mbtypes, ntypes = ', mbtypes, ntypes)

    im1 = mIds[0]
    im2 = nm if mIds[-1] == -1 else mIds[-1]

    #ia1 = sum(nas[:im1]); ia2 = sum(nas[:im2])
    #print ' -- ia1,ia2 = ', ia1,ia2
    #print ' -- nasi =', nas[im1:im2]
    #print ' -- zsi = ', len(zs_ravel), len(zs), zs_ravel[ia1:ia2]


    X = []
    for j in range(im1,im2): #enumerate(ms[im1:im2]):
        if not noprt: print('  -- im = %d '%(j+1))
        if icn: print('   -- cns_unique = ', set(cns[j]))
        mj = ms[j]
        pbc = '000' if imol else get_pbc(mj)
        zsm = np.array( zs[j] ) ### must use an array as input
        #print zsm, zsm.shape

        Xi,X2N = get_sla(mj, zsm, mbtypes, local = local, normalize=normalize, \
                         sigmas=sigmas, dgrids=dgrids, rcut=rcut, iprt=iprt,\
                         cspeed=cspeed, noprt=noprt, alchemy=alchemy, zg=zg,\
                         pbc=pbc, rpower=rpower)
        if local:
            for Xij in Xi: X.append( Xij )
        else:
            X.append( Xi )

    X = np.array(X); X2N = np.array(X2N); nas = np.array(nas); zsu = np.array(zs_ravel)
    #print ' -- shape(X) = ', X.shape
    #print ' -- X2N = ', X2N

    if (Y is None) or (Y.shape[0] == 0):
        Yu = np.array([0,])
    else:
        Yu = Y[im1:im2]

    if local:
        ia1 = sum(nas[:im1]); ia2 = sum(nas[:im2])
        dic = {'X':X.T, 'Y':Yu.T, 'X2N':X2N, 'nas':nas[im1:im2], 'zs':zsu[ia1:ia2] }
    else:
        dic = {'X':X, 'Y':Yu, 'X2N':X2N, 'nas':nas, 'zs':zsu}

    import deepdish as dd
    if iwrite:
        print(' -- now writing h5 file: %s'%h5f)
        assert h5f != None, '#ERROR: Plz specify `h5f'
        dd.io.save(h5f, dic, compression=True)

    return X

