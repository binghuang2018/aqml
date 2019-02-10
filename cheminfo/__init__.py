

import numpy as np
import os
import cheminfo.rw.xyz as rx
try:
    from importlib import reload
except:
    pass

__all__ = ['chemical_symbols', 'chemical_symbols_lowercase', 'valence_electrons', 'atomic_numbers',
           'cnsr', 'T', 'F']


__zs =  [1,3,4,5,6,7,8,9,11,12,13,14,15,16,17, 31,32,33,34,35, 49,50,51,52,53, 81,82,83,84,85]
__ves = [1,1,2,3,4,3,2,1, 1, 2, 3, 4, 3, 2, 1,  3, 4, 3, 2, 1,  3, 4, 3, 2, 1,  3, 4, 3, 2, 1]
valence_electrons = dict(zip(__zs,__ves))

T,F = True,False

chemical_symbols = ['X',  'H',  'He', 'Li', 'Be',
                    'B',  'C',  'N',  'O',  'F',
                    'Ne', 'Na', 'Mg', 'Al', 'Si',
                    'P',  'S',  'Cl', 'Ar', 'K',
                    'Ca', 'Sc', 'Ti', 'V',  'Cr',
                    'Mn', 'Fe', 'Co', 'Ni', 'Cu',
                    'Zn', 'Ga', 'Ge', 'As', 'Se',
                    'Br', 'Kr', 'Rb', 'Sr', 'Y',
                    'Zr', 'Nb', 'Mo', 'Tc', 'Ru',
                    'Rh', 'Pd', 'Ag', 'Cd', 'In',
                    'Sn', 'Sb', 'Te', 'I',  'Xe',
                    'Cs', 'Ba', 'La', 'Ce', 'Pr',
                    'Nd', 'Pm', 'Sm', 'Eu', 'Gd',
                    'Tb', 'Dy', 'Ho', 'Er', 'Tm',
                    'Yb', 'Lu', 'Hf', 'Ta', 'W',
                    'Re', 'Os', 'Ir', 'Pt', 'Au',
                    'Hg', 'Tl', 'Pb', 'Bi', 'Po',
                    'At', 'Rn', 'Fr', 'Ra', 'Ac',
                    'Th', 'Pa', 'U',  'Np', 'Pu',
                    'Am', 'Cm', 'Bk', 'Cf', 'Es',
                    'Fm', 'Md', 'No', 'Lr']

chemical_symbols_lowercase = [ si.lower() for si in chemical_symbols ]

# reference coordination numbers
cnsr = { 1:1,  3:1,  4:2,  5:3,  6:4,  7:3,  8:2, 9:1, \
        11:1, 12:2, 13:3, 14:4, 15:3, 16:2, 17:1, \
         35:1, 53:1}

atomic_numbers = {}
for Z, symbol in enumerate(chemical_symbols):
    atomic_numbers[symbol] = Z


class atoms(object):
    """
    a single molecule
    """
    def __init__(self,zs,coords,props=None):
        if isinstance(zs[0],str):
            zs = [ chemical_symbols.index(_) for _ in zs ]
        self._zs = list(zs)
        self.zs = np.array(zs, np.int)
        self._symbols = [ chemical_symbols[zi] for zi in zs ]
        self.symbols = np.array(self._symbols)
        self._coords = [ list(coords_i) for coords_i in coords ]
        self.coords = np.array(coords)
        self.na = len(zs)
        self.nheav = (self.zs>1).sum()
        self.props = {}
        if props is not None:
            self.props = {'E':props} if isinstance(props,(float,np.float64)) else props

    def write(self, f):
        props = self.props
        symbols = self._symbols
        coords = self._coords
        if os.path.exists(f):
            #print('exist!')
            nas, _symbols, _coords, nsheav, _props = rx.read_xyz_simple(f, property_names=['a'])
            if (nas[0] == self.na):
                dxyz = np.abs(np.array(coords)-_coords)
                if np.all(dxyz < 0.0001):
                    #print('################')
                    props.update( _props )
        #print('props=',props)
        so = '%d\n'%( self.na )

        icol = 0 # i-th atomic property
        if 'chgs' in props:
            so += 'chgs=%d '%icol
            icol += 1
        if 'nmr' in props:
            so += 'nmr=%d '%icol
            icol += 1
        if 'cls' in props:
            so += 'cls=%d '%icol
            icol += 1
        if 'grads' in props:
            so += 'grads=%d '%icol

        if len(props) > 0:
            for key in props.keys():
                if key not in ['chgs','nmr','cls','grads']:
                    so += '%s=%s '%(key,str(props[key]))
        so += '\n'
        icnt = 0
        for si,(x,y,z) in zip(symbols, coords):
            chgi = ''; nmri = ''; gradi = ''; clsi = ''
            if 'chgs' in props:
                chgi = ' %9.4f'%props['chgs'][icnt]
            if 'nmr' in props:
                nmri = ' %9.4f'%props['nmr'][icnt]
            if 'cls' in props:
                clsi = ' %9.4f'%props['cls'][icnt]
            if 'grads' in props:
                fxi,fyi,fzi = props['grads'][icnt]
                gradi = ' {:8.4f} {:8.4f} {:8.4f}'.format(fxi,fyi,fzi)
            so += '{:>6} {:15.8f} {:15.8f} {:15.8f}{chg}{nmr}{cls}{grad}\n'.format(si,x,y,z,chg=chgi,nmr=nmri,cls=clsi,grad=gradi)
            icnt += 1
        with open(f,'w') as fid: fid.write(so)

class molecule(atoms):
    """
    a single molecule read from a file or ...
    """
    def __init__(self, obj, props=None, isimple=F):
        nas, zs, coords, nsheav, props = obj2m(obj, props, isimple=isimple)
        atoms.__init__(self, zs, coords, props=props)


