"""
Golden data set 2 from experiments
many De's are not availabe

The following contains a database of small molecules
based on experiments

geometry info extracted from:
http://cccbdb.nist.gov/expdata.asp


D0, De extracted from:
[book] "Molecular Electronic Theory, 2015, chap 15, page 855
unit: kJ/mol

D0: Experimental atomization energies
De: estimated nonrelativistic equilibrium atomization energies
"""

from ase import Atoms
from ase.symbols import string2symbols

molecule_names = ['C2',]

data = {

'C2':{
    'name':'c2',
    'smiles': 'n/a',
    'symbols': 'CC',
    're': 1.2425,
    'magmoms': None,
    'positions': """0 0 0.62125
                    0 0 -0.62125"""},

'CH2NH':{
    'charges': None,
    'magmoms': None,
    'symbols': 'CNHHH',
    'positions': """0.1	0.5875	0
                    0.1	-0.6855	0
                    -0.9	1.1946	0
                    1.0	1.1231	0
                    -0.9	-1.0438	0"""},

}



