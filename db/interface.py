#!/usr/bin/env python

from ase import Atoms
import numpy as np

class UniversalCalculator(object):

    def __init__(self, dct):
        self.dct = dct
        self.name = dct['name']

    def todict(self):
        param = {}
        for k in ['name', 'version', ]:
            param[k] = self.dct[k]
        return param

    def check_state(self, atoms, tol=1e-15):
        """Check for system changes since last calculation."""
        return [] #True #compare_atoms(self.atoms, atoms)

    def get_property(self, prop, atoms, allow_calculation=False):
        res = None
        if prop in self.dct.keys():
            res = self.dct[prop]
        return res



import io2.molpro_reader as ir
from ase.db import connect
import aqml.cheminfo.oechem.oechem as coo
import aqml.cheminfo.molecule.molecule as cmm


all_properties = ['energy', 'forces', 'stress', 'dipole',
                  'charges', 'magmom', 'magmoms', 'free_energy']

def perceive_smiles(zs, coords):
    m = cmm.Mol(zs, coords, ican=True)
    return m.can


class molpro(object):

    def __init__(self, dbf, fs):

        db = connect(dbf) #'mono.db')

        for f in fs:
            print(' processing ', f)
            obj = ir.Molpro(f, keys=['dipole','homo'], units=['h','a'])
            ps0 = obj.props
            m = Atoms(obj.zs, obj.coords)

            eps = {}
            for ep in obj.energetic_props:
                eps[ep] = ps0[ep]
            ps = {'forces': obj.props['forces'], \
                  'dipole': obj.props['dipole'],
                  'energy': np.array(eps)}

            c = UniversalCalculator( obj.props )
            m.set_calculator( c )
            id = db.write(atoms=m)

            dt = {}
            can = perceive_smiles(obj.zs, obj.coords)
            dt['can'] = can
            #for p in ['nheav', 'homo', 'lumo', 'gap',] + obj.energetic_props:
            #    dt[p] = ps0[p]
            nheav, homo, lumo, gap= [ ps0[p] for p in ['nheav', 'homo', 'lumo', 'gap',] ]
            db.update(id, nheav=nheav)
            db.update(id, can=can)
            db.update(id, homo=homo)
            db.update(id, lumo=lumo)
            db.update(id, gap=gap)



if __name__ == '__main__':
    import os, sys, io2

    args = sys.argv[1:]
    dbf = args[0]
    objs = args[1:]
    fs = []
    for obj in objs:
        if os.path.isdir(obj):
            fs += io2.cmdout('ls %s/*.out'%fd)
        elif os.path.isfile(obj):
            fs += [obj]
    od = molpro(dbf, fs)


