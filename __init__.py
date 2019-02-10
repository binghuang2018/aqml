


try:
    from importlib import reload
except:
    pass

__all__ = ['chemical_symbols', 'valence_electrons', 'atomic_numbers',
           'cnsr', 'T', 'F', 'gen_cls_property']


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

# reference coordination numbers
cnsr = { 1:1,  3:1,  4:2,  5:3,  6:4,  7:3,  8:2, 9:1, \
        11:1, 12:2, 13:3, 14:4, 15:3, 16:2, 17:1, \
         35:1, 53:1}

atomic_numbers = {}
for Z, symbol in enumerate(chemical_symbols):
    atomic_numbers[symbol] = Z



# the function below is useful
# for generating python codes for
# property function of Class object
def gen_cls_property(s1=None):

    if s1 is None:
        s1 = """
#self.ias = np.arange(na)
#self.nheav = (self.zs>1).sum()
#self.tvs = self.bom.sum(axis=0)
#self.iasv = self.ias[self.zs>1]
#self.zsv = self.zs[self.iasv]
#self.zsu = np.unique(zs)
#self.g = (self.bom>0).astype(np.int)
#self.cns = self.g.sum(axis=0)
#self.cnsv = np.array([(self.zs[self.g[ia]>0]>1).sum() for ia in self.ias])
#self.cnsvv = self.cnsv[self.iasv]"""

    for si in s1.strip().split('\n'):
        k0, v = si.split(' = ')
        k = k0[6:]
        print("""    
        @property
        def %s(self):
            if not hasattr(self, '_%s'):
                self._%s = %s
            return self._%s\n"""%(k, k, k, v, k))


