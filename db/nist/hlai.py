"""
The following contains a database of small molecules
based on high-level ab initio (abbr. hlai) method

Data are extracted from nist database
http://cccbdb.nist.gov/geom1.asp

"""

from ase import Atoms
from ase.symbols import string2symbols

molecule_names = ['C2', 'H2CCCCH2', 'C4H4', 'C2H3CCH', 'C6H6', 'C5H4CH2', 'CH3CCCCCH3', 'C6F6','C2F2','N2F2',]


data = {
'H2CCCCH2': {
    'name': 'Butatriene',
    'symmetry': 'd2h',           # new property
    'state': '1Ag',              # new property
    'gs': True,                  # new property
    'method': 'CCSD(T)/6-31+G**',# new property
    'database': 'nist',
    'symbols': 'CCCCHHHH',
    'magmoms': None,
    'charges': None,
    'enthalpy': 'n/a',
    'thermal correction': 'n/a',
    'ionization energy': 'n/a',
    'positions': """0	0	0.6409
                    0	0	-0.6409
                    0	0	1.9746
                    0	0	-1.9746
                    0	0.9276	2.5356
                    0	-0.9276	2.5356
                    0	0.9276	-2.5356
                    0	-0.9276	-2.5356"""},

'C4H4': {
    'name': 'cyclobutadiene',
    'symmetry': 'd2h',           # new property
    'state': '1Ag',              # new property
    'gs': True,                  # new property
    'method': 'CCSD(T)=FULL/aug-cc-pVTZ', # new property
    'database': 'nist',
    'symbols': 'CCCCHHHH',
    'magmoms': None,
    'charges': None,
    'enthalpy': 'n/a',
    'thermal correction': 'n/a',
    'ionization energy': 'n/a',
    'positions': """0	0.6713	0.7831
                    0	0.6713	-0.7831
                    0	-0.6713	0.7831
                    0	-0.6713	-0.7831
                    0	1.4332	1.5427
                    0	1.4332	-1.5427
                    0	-1.4332	1.5427
                    0	-1.4332	-1.5427"""},


'C2H3CCH': {
    'name': '1-Buten-3-yne',
    'symmetry': 'c1',           # new property
    'state': '1A',              # new property
    'gs': True,                  # new property
    'method': 'CCSD=FULL/aug-cc-pVTZ',
    'database': 'nist',
    'symbols': 'CCCCHHHH',
    'magmoms': None,
    'charges': None,
    'enthalpy': 'n/a',
    'thermal correction': 'n/a',
    'ionization energy': 'n/a',
    'positions': """0.5847	-0.5602	0
                    0	0.7419	0
                    -0.1213	-1.6874	0
                    -0.4593	1.8534	0
                    1.6629	-0.5966	0
                    0.3698	-2.6465	0
                    -1.1992	-1.6758	0
                    -0.8581	2.8334	0"""},


'C6H6': {
    'name': 'Benzene',
    'smiles': 'c1ccccc1',
    'symmetry': 'd6h',           # new property
    'state': '1A1g',              # new property
    'gs': True,                  # new property
    'method': 'CCSD(T)=FULL/aug-cc-pVTZ',
    'database': 'nist',
    'symbols': 'CCCCCCHHHHHH',
    'magmoms': None,
    'charges': None,
    'enthalpy': 'n/a',
    'thermal correction': 'n/a',
    'ionization energy': 'n/a',
    'positions': """0	1.392	0
                    1.2055	0.696	0
                    1.2055	-0.696	0
                    0	-1.392	0
                    -1.2055	-0.696	0
                    -1.2055	0.696	0
                    0	2.4722	0
                    2.141	1.2361	0
                    2.141	-1.2361	0
                    0	-2.4722	0
                    -2.141	-1.2361	0
                    -2.141	1.2361	0"""},


'C5H4CH2': {
    'name': 'Fulvene',
    'smiles': 'C1=CC=CC1=C',
    'symmetry': 'c2v',           # new property
    'state': '1A1',              # new property
    'gs': True,                  # new property
    'method': 'CCSD(T)/6-31G*',
    'database': 'nist',
    'symbols': 'CCCCCCHHHHHH',
    'magmoms': None,
    'charges': None,
    'enthalpy': 'n/a',
    'thermal correction': 'n/a',
    'ionization energy': 'n/a',
    'positions': """0	0	0.7624
                    0	0	2.1146
                    0	1.1827	-0.1259
                    0	-1.1827	-0.1259
                    0	0.741	-1.4125
                    0	-0.741	-1.4125
                    0	0.9285	2.686
                    0	-0.9285	2.686
                    0	2.2128	0.2234
                    0	-2.2128	0.2234
                    0	1.3569	-2.3099
                    0	-1.3569	-2.3099"""},


'CH3CCCCCH3': {
    'name': '2,4-Hexadiyne',
    'smiles': 'CC#CC#CC',
    'symmetry': 'd3h',           # new property
    'state': '1A1',              # new property
    'gs': True,                  # new property
    'method': 'MP2/cc-pVTZ',
    'database': 'nist',
    'symbols': 'CCCCCCHHHHHH',
    'magmoms': None,
    'charges': None,
    'enthalpy': 'n/a',
    'thermal correction': 'n/a',
    'ionization energy': 'n/a',
    'positions': """0	0	0.6836
                    0	0	-0.6836
                    0	0	1.9063
                    0	0	-1.9063
                    0	0	3.3622
                    0	0	-3.3622
                    0	1.0186	3.7476
                    -0.8821	-0.5093	3.7476
                    0.8821	-0.5093	3.7476
                    0	1.0186	-3.7476
                    0.8821	-0.5093	-3.7476
                    -0.8821	-0.5093	-3.7476"""},

'C2':{
    'name':'c2',
    'smiles': 'n/a',
    'symbols': 'CC',
    'charges': None,
    'magmoms': None,
    'positions': """0 0 0.62125
                    0 0 -0.62125"""},

'C6F6':{
    'smiles': 'FC1=C(F)C(F)=C(F)C(F)=C1F',
    'symmetry': 'D6h',
    'charges': None,
    'symbols': 'CCCCCCFFFFFF',
    'magmoms': None,
    'method': 'MP2=FULL/6-31G*',
    'positions': """0	1.3913	0
                    1.2049	0.6956	0
                    1.2049	-0.6956	0
                    0	-1.3913	0
                    -1.2049	-0.6956	0
                    -1.2049	0.6956	0
                    0	2.731	0
                    2.3651	1.3655	0
                    2.3651	-1.3655	0
                    0	-2.731	0
                    -2.3651	-1.3655	0
                    -2.3651	1.3655	0""" },

'C2F2':{
    'smiles': 'FC#CF',
    'symbols': 'CCFF',
    'charges': None,
    'magmoms': None,
    'method': 'MP2=FULL/6-31G*',
    'positions': """0	0	0.6002
                    0	0	-0.6002
                    0	0	1.9001
                    0	0	-1.9001"""},

'N2F2':{
    'name': '(Z)-Difluorodiazene',
    'smiles': 'FN=NF',
    'charges': None,
    'symbols': 'FNNF',
    'magmoms': None,
    'method': 'MP2=FULL/6-31G*',
    'positions': """0	1.2151	-0.5425
                    0	0.6163	0.6975
                    0	-0.6163	0.6975
                    0	-1.2151	-0.5425"""},

'C5H5N':{
    'charges': None,
    'magmoms': None,
    'method': 'MP2=FULL/6-31G*',
    'symbols': 'NCCCCCHHHHH',
    'positions': """0	0.0	1.4247
0	0.0	-1.3861
0	1.1	0.7202
0	-1.1	0.7202
0	1.2	-0.6729
0	-1.2	-0.6729
0	0.0	-2.4729
0	2.1	1.3073
0	-2.1	1.3073
0	2.2	-1.183
0	-2.2	-1.183"""},

'C6H5NH2':{
    'charges': None,
    'magmoms': None,
    'method': 'MP2=FULL/6-31G*',
    'symbols': 'CCCCCCNHHHHHHH',
    'positions': """-0.0068	0.9	0
0.0065	0.2	1.2058
0.0065	-1.2	1.2025
0.0098	-1.9	0
0.0065	-1.2	-1.2025
0.0065	0.2	-1.2058
0.0651	2.3	0
0.0089	0.8	2.1492
0.0099	-1.7	2.1494
0.0095	-3.0	0
0.0099	-1.7	-2.1494
0.0089	0.8	-2.1492
-0.339	2.8	-0.8305
-0.339	2.8	0.8305"""},

'ONNO':{
    'charges': None,
    'magmoms': None,
    'smiles': 'O=NN=O',
    'method': 'MM2', #'CCSD(T)=FULL/cc-pVTZ',
    'symbols': 'NONO',
    'positions': """-2.1965	0.7563	0.3440
-1.0707	0.5130	0.0192
-2.9751	1.6273	-0.4275
-4.1009	1.8706	-0.1027"""},

'ONNO-E':{
    'charges': None,
    'magmoms': None,
    'smiles': 'O=NN=O',
    'method': 'MM2', #'CCSD(T)=FULL/cc-pVTZ',
    'symbols': 'NONO',
    'positions': """-2.1965	0.7563	0.3440
-2.2000	-0.4171	0.1086
-2.9751	1.6273	-0.4275
-3.6440	1.1985	-1.3225"""},

'CH2NCH3':{
    'charges': None,
    'magmoms': None,
    'method': 'MP2=FULL/6-31G*',
    'symbols': 'CNCHHHHH',
    'positions': """-1.073	-0.4037	0
0	0.5588	0
1.1865	0.1015	0
-0.7429	-1.4555	0
-1.6985	-0.2348	0.8789
-1.6985	-0.2348	-0.8789
1.435	-0.9697	0
2.0244	0.7964	0"""},

'NH2NO':{
    'charges': None,
    'magmoms': None,
    'smiles': 'NN=O',
    'method': 'MP2=FULL/6-31G*',
    'symbols': 'ONNHH',
    'positions': """-1.1272	0.2245	0.0052
-0.1393	-0.5188	0.0036
1.0209	0.1521	-0.0434
0.9932	1.1615	0.0919
1.8525	-0.3897	0.1455"""},

'CH2NNO':{
    'charges': None,
    'magmoms': None,
    'smiles': 'C=NN=O',
    'method': 'MM2',
    'symbols': 'NONCHH',
    'positions': """-2.1965	0.7563	0.3440
-1.0707	0.5130	0.0192
-2.9751	1.6273	-0.4275
-4.1920	1.8903	-0.0764
-4.6036	1.4421	0.8037
-4.7870	2.5559	-0.6661"""},

'CH2CHNH2':{
    'smiles': 'C=CN',
    'symbols': 'CCNHHHHH',
    'charges': None,
    'magmoms': None,
    'method': 'MP2=FULL/6-31G*',
    'positions': """1.2485	-0.1991	0.0182
0.0687	0.4359	-0.0005
-1.1904	-0.1679	-0.0971
1.3184	-1.2824	0.0279
2.1735	0.3613	0.0003
0.0303	1.5226	-0.0201
-1.1798	-1.1559	0.1362
-1.9134	0.3092	0.4294"""},

'HCCNO':{
    'smiles': 'C#CN=O',
    'method': 'MM2',
    'charges': None,
    'magmoms': None,
    'symbols': 'NOCCH',
    'positions': """-3.0118	1.4074	-0.6510
-3.9737	2.1095	-0.7706
-2.7188	0.7767	0.6440
-2.4799	0.2616	1.7026
-2.2671	-0.1972	2.6454"""},

'HCONCH2':{
    'smiles': 'O=CN=C',
    'method': 'MM2',
    'charges': None,
    'magmoms': None,
    'symbols': 'OCHNCHH',
    'positions': """-3.7786	2.1286	-1.1027
-3.0167	1.3786	-0.4389
-2.1368	0.9730	-0.8929
-3.3356	1.0597	0.9602
-2.5525	0.2888	1.6426
-2.7846	0.0567	2.6610
-1.6725	-0.1168	1.1886"""},

}



