
from rdkit import Chem
import os, sys
import numpy as np
import ase.units as au
import ase, copy
import multiprocessing
import copyreg
import types as _TYPES
from .emorse import *


global dic_bonds, dic_atypes, h2kc, c2j
dic_bonds = { Chem.BondType.SINGLE:1.0,
              Chem.BondType.DOUBLE:2.0,
              Chem.BondType.TRIPLE:3.0,
              Chem.BondType.AROMATIC:1.5,
              Chem.BondType.UNSPECIFIED:0.0}
dic_atypes =  { 'H': ['H',], \
                'F': ['F',], \
                'Cl': ['Cl',],\
                'O': ['O_3',  'O_2',  'O_R',], \
                'S': ['S_3', 'S_2',  'So3',  'S_R', ], \
                'N': ['N_3', 'N_2', 'N_1', 'N_R', ], \
                'C': ['C_3', 'C_2', 'C_1',  'C_R', ]}
h2kc = au.Hartree * au.eV/(au.kcal/au.mol)
c2j = au.kcal/au.kJ

## register instance method
## otherwise, the script will stop when calling the function
## multiprocessing() with error:
## ``TypeError: can't pickle instancemethod objects
def _reduce_method(m):
    if m.__self__ is None:
        return getattr, (m.__self__.__class__, m.__func__.__name__)
    else:
        return getattr, (m.__self__, m.__func__.__name__)
copyreg.pickle(_TYPES.MethodType, _reduce_method)




class BAML(object):
    def __init__(self, strings, forcefield='UFF', groupBys='mmmmm', nproc=1, dic=None):

        self.nproc = nproc
        sids = []
        for groupBy in groupBys:
            sid = groupBy.lower(); sids.append( sid )
            assert sid in ['n','m'], '#ERROR: "z" & "m" are the only two options!'
        groupBys = ''.join(sids)
        self.groupBys = groupBys
        if dic == None:
            dic = esb # from emorse.py
        self.objs = []
        if type(strings) is str:
            strings = [strings, ]
        self.nm = len(strings)
        if nproc == 1:
            for i, string in enumerate(strings):
                print(i+1, string)
                ipt = [string, dic, forcefield, groupBys]
                self.objs.append( self.processInput(ipt) )
        else:
            pool = multiprocessing.Pool(processes=nproc)
            ipts = [ [string, dic, forcefield, groupBys] for string in strings ]
            self.objs = pool.map(self.processInput, ipts)

    def processInput(self, ipt):
        string, dic, forcefield, groupBys = ipt
        obj = RDMol(string, forcefield=forcefield)
        obj.get_atypes()
        obj.get_atom_contrib(groupBy=groupBys[0])
        obj.get_bond_contrib(groupBy=groupBys[1], dic=dic)
        obj.get_vdw_contrib(groupBy=groupBys[2])
        obj.get_angle_contrib(groupBy=groupBys[3])
        obj.get_torsion_contrib(groupBy=groupBys[4])
        return obj

    def generate_baml(self):
        # boa
        types1 = []; es1 = []
        types2 = []; es2 = []
        types2v = []; es2v = []
        types3 = []; es3 = []
        types4 = []; es4 = []
        for obj in self.objs:
            types1.append( obj.types1 )
            es1.append( obj.es1 )
            types2.append( obj.types2 )
            es2.append( obj.es2 )
            types2v.append( obj.types2v )
            es2v.append( obj.es2v )
            types3.append( obj.types3 )
            es3.append( obj.es3 )
            types4.append( obj.types4 )
            es4.append( obj.es4 )

        if self.nproc == 1:
            boas = self.bin1(types1, es1)
            bops = self.bin1(types2, es2)
            bovs = self.bin1(types2v, es2v)
            bots = self.bin1(types3, es3)
            boqs = self.bin1(types4, es4)
        else:
            info1 = self.bin_info(types1)
            pool = multiprocessing.Pool(processes=nproc)
            ipts = [ info1 + [types1[j], es1[j]] for j in range(self.nm) ]
            boas = pool.map(self.bin, ipts)

            info2 = self.bin_info(types2)
            pool = multiprocessing.Pool(processes=nproc)
            ipts = [ info2 + [types2[j], es2[j]] for j in range(self.nm) ]
            bops = pool.map(self.bin, ipts)

            info2v = self.bin_info(types2v)
            pool = multiprocessing.Pool(processes=nproc)
            ipts = [ info2v + [types2v[j], es2v[j]] for j in range(self.nm) ]
            bovs = pool.map(self.bin, ipts)

            info3 = self.bin_info(types3)
            pool = multiprocessing.Pool(processes=nproc)
            ipts = [ info3 + [types3[j], es3[j]] for j in range(self.nm) ]
            bots = pool.map(self.bin, ipts)

            info4 = self.bin_info(types4)
            pool = multiprocessing.Pool(processes=nproc)
            ipts = [ info4 + [types4[j], es4[j]] for j in range(self.nm) ]
            boqs = pool.map(self.bin, ipts)
        X = np.concatenate( (boas, bops, bovs, bots, boqs), axis=1 )
        self.X = X

    def bin_info(self, types):
        """
        step 1 for multiprocessing
        """
        nm = self.nm
        # find all the possible unique types
        types_u = set([])
        for i in range(nm):
            types_u.update( types[i] )
        types_u = list( types_u )
        # find the number of each torsion type
        ntu = len(types_u)
        ns = np.zeros((nm, ntu), np.int32)
        for i in range(nm):
            types_i = types[i]
            if len(types_i) > 0:
                for j in range(ntu):
                    ns[i,j] = (np.array(types_i) == types_u[j]).sum()
        # get the size of descriptor array
        # i.e.,  maximal value of each column of `ns
        nsmax = np.max(ns, axis = 0)
        # size of the n-th body term
        N = np.sum(nsmax)
        # indexing array
        ies = np.cumsum(nsmax) # beginning indices
        ibs = np.concatenate(([0,], ies[:-1]))
        return ntu, types_u, ies, ibs

    def bin(self, ipt):
        """
        step 2 for multiprocessing
        """
        ntu, types_u, ies, ibs, types_i, es_i = ipt
        nti = len(types_i)
        its0 = np.arange(nti)
        boN_i = np.zeros( ies[-1] )
        for j in range(ntu):
            its = its0[ types_u[j] == types_i ]
            ib = ibs[j]; ie = ies[j]
            esj = np.array(es_i)[its]; esj.sort()
            nej = len(esj)
            boN_i[ib:ib+nej] = esj #[::-1]
        return boN_i

    def bin1(self, types, es):
        """
        one-step procedure using 1 core
        purpose: To bin the many-body terms
                 based on atomic types
        """
        nm = self.nm
        # find all the possible unique types
        types_u = set([])
        for i in range(nm):
            types_u.update( types[i] )
        types_u = list( types_u )
        # find the number of each torsion type
        ntu = len(types_u)
        ns = np.zeros((nm, ntu), np.int32)
        for i in range(nm):
            types_i = types[i]
            if len(types_i) > 0:
                for j in range(ntu):
                    ns[i,j] = (np.array(types_i) == types_u[j]).sum()
        # get the size of descriptor array
        # i.e.,  maximal value of each column of `ns
        nsmax = np.max(ns, axis = 0)
        # size of the n-th body term
        N = np.sum(nsmax)
        # indexing array
        ies = np.cumsum(nsmax) # beginning indices
        ibs = np.concatenate(([0,], ies[:-1]))
        # now the representation: bag of N-th body terms
        boN = np.zeros((nm, N))
        for i in range(nm):
            types_i = np.array( types[i] )
            #print ' -- types_i = ', types_i
            #print ' -- tyeps_u = ', types_u
            nti = len(types_i)
            if nti > 0:
                its0 = np.arange(nti)
                for j in range(ntu):
                    #its = its0[ types_u[j] == types_i ]
                    filt = ( types_u[j] == types_i )
                    #print ' -- types_u[j], nti, len(es[i]) = ', types_u[j], nti, len(es[i])
                    esj = np.array(es[i])[filt]
                    #print ' --esj = ', esj
                    esj.sort()
                    nej = len(esj)
                    ib = ibs[j]; ie = ies[j]
                    boN[i, ib:ib+nej] = esj #[::-1]
        return boN

