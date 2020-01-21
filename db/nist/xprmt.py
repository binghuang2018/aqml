"""
Golden data set from experiments
All data (esp. energies) are precise!


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

molecule_names = ['CO', 'CH2O', 'C2H4', 'O3', 'C2H2', 'CO2', 'H2', 'H2O', 'CH2', 'CH4']

data = {

'H2':{
    'name':'h2',
    'smiles': '[H][H]',
    'symbols': 'HH',
    'magmoms': None,
    're': 0.7414,
    'D0': 432.07,
    'De': 458.04,
    'positions': """0 0 0.3707
                    0 0 -0.3707"""},


'CH2':{
    'name': 'CH_2 (\tilde{X}^3B_1)',
    'smiles':'[CH2]',
    'symbols': 'CHH',
    'magmoms': [ 2.,  0.,  0.],
    'charges': None,
    'D0': 713.11,
    'De': 757.06,
    'positions': """0       0   0.1027
                    1.0042  0   -0.3081
                    -1.0042 0   -0.3081"""},


'O3':{
    'name':'O_3 (^1A_1)',
    'smiles': 'O=[O+][O-]',
    'symbols': 'OOO',
    'magmoms': None,
    'D0': 595.02,
    'De': 616.24,
    'positions': """ 0.0000  0.0  0.0000
                     1.0885  0.0 -0.6697
                     -1.0885 0.0 -0.6697"""},

'H2O':{
    'name': 'H_2O',
    'smiles': 'O',
    'symbols': 'OHH',
    'magmoms': None,
    'D0': 917.78,
    'De': 975.28,
    'positions': """0   0   0.1173
                    0.7572  0   -0.4692
                    -0.7572 0   -0.4692"""},

'CO':{
    'D0': 1071.79,
    'De': 1086.70,
    'symbols': 'CO',
    'magmoms': None,
    're': 1.1282,
    'positions': """0 0 0.0641
                    0 0 -0.0641"""},

'CH2O':{
    'D0': 1494.73,
    'De': 1566.58,
    'symbols': 'OCHH',
    'magmoms': None,
    'smiles': 'C=O',
    'positions': """0   0   1.205
                    0   0   0
                    0.9429  0   -0.5876
                    -0.9429 0   -0.5876"""},

'CO2':{
    'D0': 1597.92,
    'De': 1632.46,
    'magmoms': None,
    'symbols': 'COO',
    'positions': """0   0   0
                    0   0   1.1621
                    0   0   -1.1621"""},

'C2H2':{
    'D0':1627.16,
    'De':1697.84,
    'magmoms': None,
    'smiles': 'C#C',
    'symbols': 'CCHH',
    'positions': """0   0   0.6013
                    0   0   -0.6013
                    0   0   1.6644
                    0   0   -1.6644"""},

'CH4':{
    'D0':1642.24,
    'De':1759.33,
    'magmoms': None,
    'smiles': 'C',
    'symbols': 'CHHHH',
    'positions': """0   0   0
                    0.6276  0.6276  0.6276
                    0.6276  -0.6276 -0.6276
                    -0.6276 0.6276  -0.6276
                    -0.6276 -0.6276 0.6276"""},

'C2H4':{
    'D0':2225.53,
    'De':2359.82,
    'magmoms': None,
    'smiles': 'C=C',
    'symbols': 'CCHHHH',
    'positions': """0   0   0.6695
                    0   0   -0.6695
                    0.9289  0   1.2321
                    -0.9289 0   1.2321
                    0.9289  0   -1.2321
                    -0.9289 0   -1.2321"""},

}



