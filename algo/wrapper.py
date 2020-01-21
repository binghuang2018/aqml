#!/usr/bin/env python

import io2, os, sys
import algo.krr as krr 
import io2.gaussian as iog
import representation.slatm_x as sl
import numpy as np
import cheminfo.molecule.nbody as by
import cheminfo.molecule.geometry as cmg
import cheminfo.core as cc
import algo.qml as aqml
import argparse as ap

T,F = True,False

__all__ = ['glc']


def varparser(ipt):
    ps = ap.ArgumentParser()

    if isinstance(ipt,str):
        ipt = ipt.split()

    ps.add_argument('-p', '--property', dest='p', nargs='?', type=str, help='property to be trained/test')
    ps.add_argument('-wd', dest='w0', nargs='?', default='./', type=str, help='current working directory, default is "./"')
    
    ps.add_argument('-rcut','--rcut', nargs='?', default=4.8, type=float, help='SLATM cutoff radius, default is 4.8 Ang')

    ps.add_argument('-train','--train', nargs='*', type=str, help='Name of the folder(s) containing all training mols')
    ps.add_argument('-exclude','--exclude', dest='_exclude', nargs='*', type=str, help='molecular idxs to be excluded for training')
    ps.add_argument('-test', '--test', nargs='*', type=str, help='Name of the folder(s) containing all test molecules')
    ps.add_argument('-n2', '--n2', nargs='?', type=int, help='Number of test molecules; must be specified when no test folder ia avail')
    #ipt = '--wd %s --train %s --test %s -p %s'
    ag = ps.parse_args(ipt) # sys.argv[1:] )
    
    info = ''
    for key in ['train',]:
        if not hasattr(ag, key):
            info += ' --%s is missing\n'%key
    if info != '':
        print(info)
        raise Exception('Please comply!')

    exclude = []   
    if ag._exclude: #hasattr(ag, '_exclude'):
        for idx in ag._exclude:
            if '-' in idx:
                ib, ie = idx.split('-') # input mol idx starts from 1
                exclude += [ j for j in range(int(ib)-1,int(ie)) ]
            else:
                exclude += [ int(idx) ]
    ag.exclude = exclude

    if not hasattr(ag, 'test'):
        if not hasattr(ag, 'n2'):
            raise Exception('#ERROR: `n2 not seen as input. For the case of no -test option, `n2 must be specified')
        ag.test = None
    return ag 


def glc(ipt):
    
    ag = varparser(ipt)
    
    rcut = ag.rcut
    fs = []; n2 = 0
    w0 = ag.w0
    for _wd in ag.train:
        wd = _wd if _wd[-1] != '/' else _wd
        print('_wd=',_wd)
        if not os.path.exists(w0+wd):
            try:
                wd += '_extl/'
                assert os.path.exists(w0+wd)
            except:
                raise Exception('#ERROR: either %s or %s does not exist'%(w0+_wd, w0+wd))
        fsi = io2.cmdout('ls %s/%s/*.xyz'%(w0,wd))
        assert len(fsi) > 0
        fs += fsi

    use_fmap = F
    print('test=',ag.test)
    if ag.test is not None:
        n2 = 0
        for wd in ag.test:
            fsi = io2.cmdout('ls %s/%s/*.xyz'%(w0,wd))
            assert len(fsi) > 0
            fs += fsi
            n2 += len(fsi)
        if n2>1: use_fmap = T
    else:
        n2 = ag.n2

    xp = {'coeffs':[1.0]}
    #xp = {'coeffs':[1.0], 'saves':[T,T,T], 'reuses':[T,T,T]}
    namax = 9 # 8 #7
    #n2 = 1

    if ag.p is None:
        # not explicitely given by user, then detect it from input file
        l2 = open(fs[0]).readlines()[1]
        ps = [ si.split('=')[0] for si in l2.split() ]
        if len(ps) > 1:
            print(' Property list: ', ps )
            raise Exception('#ERROR: pls specify one property!')
        print('  set property to "%s"'% ps[0])
        ag.p = ps[0]

    if ag.p in ['energy']:
        unit = 'h'
    else:
        # 'lmp2vtz' #'tpssdef2tzvp'
        unit = 'kcal'

    obj = aqml.qml(fs, fitmorse=F, property_names=[ag.p], iae=F, \
                   xparam=xp, rcut=rcut, unit=unit, no_strain=F, prog='orca',\
                   check_boundary=F)
    #obj.test_target(fmap=wds[0]+'map.h5', llambdas=[1e-2, 1e-4, 1e-8])
    fmap = w0+ag.train[0]+'/map.h5' if use_fmap else None
    obj.test_target(n2=n2, fmap=fmap, izeff=F, icg=F, cab=F, namax=namax, \
                    exclude=ag.exclude, llambdas=[1e-2, 1e-4, 1e-8])
    return obj


if __name__ == "__main__":

    import sys, time

    args = sys.argv[:]
    for i in range(3):
        print('')
    print(' now running:')
    print(' '.join(args))
    obj = glc(args[1:])

    #maes = np.array(obj.maes['1,0'])
    #err_a = maes[:,6]
    #fst = io2.cmdout2('ls %s/f*z'%wds[-1] )
    #for i,fi in enumerate(fst):
    #    if abs(err_a[i]) > 1.0:
    #        print(i+1, fi, err_a[i])


