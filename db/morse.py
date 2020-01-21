
"""
Morse potential paramters built manually from CRC

suported atom types
==============================================================
atom [#1]     H_          Generic hydrogen
atom [#6]     C_3         Generic sp3 C
atom [C^2]    C_2         sp2 non-aromatic C
atom [C^1]    C_1         sp hybridized C
atom [c]      C_R         aromatic C
atom [#6H1D3r5] C_R       aromatic C (cyclopentadienyl ring)
atom [#8]     O_3         generic, sp3 hybridized O
atom [O^2]    O_2         sp2 hybridized O
atom [O^1]    O_1         sp hybridized O
atom [o]      O_R         aromatic O

@ extra types for GDB9
atom [#7]     N_3         Generic sp3 N
atom [N^2]    N_2         sp2 non-aromatic N
atom [N^1]    N_1         sp hybridized N
atom [n]      N_R         aromatic 

atom [#9]     F_          generic F

atom [#15]    P_3+3       generic phosphorus
atom [#15D5]  P_3+5       formal charge +5
atom [#16]    S_3+2       generic S
atom [#16+4]  S_3+4       S+4  ( SO2 )
atom [#16+6]  S_3+6       S+6  ( SO3 / H2SO4 )
atom [S^2]    S_2         non-aromatic sp2 S
atom [s]      S_R         aromatic S
atom [#17]    Cl
==============================================================
"""

from math import *

global dic
dic = {'H_':'H', 'C_R':'c', 'C_2':'C^2', 'C_3':'C^3', 'C_1':'C^1', \
       'O_3':'O^3', 'O_2':'O^2', 'O_R':'o', }

def unique_patterns0():
    """
    canonicalize the dictionary
    """

    db_morse = {'[C^3][C^3]': 324, \
    '[C^3][O^3]': 321.9, \
    '[C^3][H]': 410.5, \
    '[O^3][O^3]': 137.7, \
    '[O^2]=[O^2]': 488.4, \
    '[O^3][H]': 458.9, \
    '[H][H]': 432.1, \
    '[C^1]=[O^2]': 799, \

    '[C^1][C^1]': 445.2, \
    '[C^1]#[C^1]': 748.2, \
    '[C^1]#[N^1]': 826.5, \
    '[C^1][H]': 439.2, \

    '[C^3][C^1]': 411.8, \
    '[C^2]=[C^1]': 598.47, \

    '[C^2]=[O^2]': 700.6, \
    '[C^1][C^2]': 403.4, \
    '[C^3][C^2]': 356.7, \

    '[C^2][C^2]': 366.7, \
    '[C^2]=[C^2]': 623.1, \
    '[C^2][H]': 400.6, \
    '[c][H]': 400.6, \
    '[c][c]': 509.9, \

    '[C^3][O^2]': 366.6, \
    '[C^2][O^2]': 425.9, \
    '[O^2][H]': 483.6, \

    '[C^3][c]': 345, \
    '[C^2][c]': 367.8, \
    '[C^1][c]': 409.7, \
    '[c][O^2]': 340.3,\
    '[c][o]': 422.7, }

    keys = db_morse.keys()
    db_upd = {}
    for key in keys:
        key1 = key[1:-1]
        i1 = key1.index(']')
        i2 = key1.index('[')
        s1 = key1[:i1]; s2 = key1[i2+1:]
        s12 = [s1,s2]
        s12.sort()
        s1, s2 = s12
        key2 = ''.join([ s1, key1[i1+1:i2], s2 ])
        db_upd[key2] = db_morse[key]

    return db_upd

def is_aromatic(s):
    return (s in ['c','n','o','s','[c]','[n]','[o]','[s]',])

def get_unique_pattern0(btype):
    """
    """
    s1, s2, bo = btype

    t12 = [dic[s1], dic[s2]]
    t12.sort()
    s1u, s2u = t12

    isar1 = is_aromatic(s1u)
    isar2 = is_aromatic(s2u)

    pat0 = ''.join(t12)

    if bo == '1.50':
        if not (isar1 and isar2):
            print btype
            raise '##error'
        pat = pat0
    elif bo == '1.00':
        pat = pat0 
    elif bo == '2.00':
        if isar1 or isar2:
            pat = pat0
        else:
            pat = '='.join(t12)
    elif bo == '3.00':
        if isar1 or isar2:
            pat = pat0
        else:
            pat = '#'.join(t12)
    else:
        print 'BO = ', bo 
        raise '##Unknown bond order'

    return pat 


def get_emr0(btype, r, re, kb):
    """
    calculate Morse energy
    """
    db_morse = unique_patterns0()

    pat = get_unique_pattern0(btype)
    keys = db_morse.keys()
    if pat not in keys:
        print ' pattern %s not in keys of `db_morse'%pat 
        print '\n'
        raise '##ERROR'

    De = db_morse[pat]
    
    a = (kb/(2*De))**0.5

#    print ' btype, r, re, kb, a = ', btype, r, re, kb, a

    Dr = De*( (exp(-a*(r-re)) - 1.0)**2 - 1.0 )

    return Dr, pat 


def unique_patterns1():
    """
    canonicalize the dictionary
    """

    db_morse = {\
'C_3-C_3-1.00': 324, \
'C_3-O_3-1.00': 321.9, \
'C_3-H-1.00': 410.5, \
'O_3-O_3-1.00': 137.7, \
'O_2-O_2-2.00': 488.4, \
'O_3-H-1.00': 458.9, \
'H-H-1.00': 432.1, \
'C_1-O_2-2.00': 799, \

'C_1-C_1-1.00': 445.2, \
'C_1-C_1-3.00': 748.2, \
'C_1-N_1-3.00': 826.5, \
'C_1-H-1.00': 439.2, \

'C_3-C_1-1.00': 411.8, \
'C_2-C_1-2.00': 598.47, \

# O=C=O
'C_2-O_2-2.00': 700.6, \

# C=C=C
'C_1-C_2-1.00': 403.4, \
'C_3-C_2-1.00': 356.7, \

'C_2-C_2-1.00': 366.7, \
'C_2-C_2-2.00': 623.1, \
'C_2-H-1.00': 400.6, \
'C_R-H-1.00': 400.6, \
'C_R-C_R-1.50': 509.9, \

'C_3-O_2-1.00': 366.6, \
'C_2-O_2-1.00': 425.9, \
'O_2-H-1.00': 483.6, \

'C_3-C_R-1.00': 345, \

# the same as C_2-C-2-2.00 for C_2-C_R-2.00
'C_2-C_R-2.00': 367.8, \
'C_2-C_R-1.00': 367.8, \

'C_1-C_R-1.00': 409.7, \
'C_R-O_2-1.00': 340.3,\
'C_R-O_R-1.50': 422.7, }
    keys = db_morse.keys()
    dic_upd = {}
    for key in keys:
      a1,a2, bo = key.split('-')
      a12 = [a1,a2]; a12.sort()
      dic_upd[ '-'.join( a12 + ['%.2f'%eval(bo), ] ) ] = db_morse[key]

    return dic_upd 


def get_unique_pattern1(btype):
    """
    """
    dic_news = {'H_':'H', 'F_':'F', 'S_3+2':'So3', }

    s1, s2, bo = btype

    t12 = []
    for si in [s1,s2]:
        if si in ['H_','F_','S_3+2', ]:
            t12.append( dic_news[si] )
        else:
            t12.append( si )

    t12.sort()
    s1u, s2u = t12

    pat = '-'.join( t12 + [ '%.2f'%eval(bo), ] )

    return pat 


def get_emr1(btype, r, re, kb):
    """
    calculate Morse energy
    """
    db_morse = unique_patterns1()

    pat = get_unique_pattern1(btype)
    keys = db_morse.keys()
    if pat not in keys:
        print ' pattern %s not in keys of `db_morse'%pat 
        print '\n'
        raise '##ERROR'

    De = db_morse[pat]
    
    a = (kb/(2*De))**0.5

#    print ' btype, r, re, kb, a = ', btype, r, re, kb, a

    Dr = De*( (exp(-a*(r-re)) - 1.0)**2 - 1.0 )

    return Dr, pat 


def get_BEs(datf):
    conts = file(datf).readlines()
    dic_BEs = {}
    for cont in conts:
        #print cont.split()
        a1,a2,bo, eb_str = cont.split()
        a12 = [a1,a2]; a12.sort()
        key_u = '-'.join( a12 + [ '%.2f'%eval(bo), ] )
        dic_BEs[key_u] = eval(eb_str)

    return dic_BEs 


def get_emr(btype, r, re, kb, bdE_version='_ob_uff'):
    """
    calculate Morse energy
    """

    if type(btype) is list:
        btype = get_unique_pattern1(btype)

    prefix = '/home/bing/workspace/ML/'
    fn = prefix + 'BEs%s.dat'%bdE_version
    dic_BEs = get_BEs(fn)
    btypes = dic_BEs.keys()

    if btype == 'N_R-O_3-1.00':
        btype = 'N_R-O_2-1.00'

    if btype not in btypes:
        print ' bond type %s not in keys of `db_morse'%btype
        print '\n'
        raise '##ERROR'

    De = dic_BEs[btype]
    
    a = (kb/(2*De))**0.5

#    print ' btype, r, re, kb, a = ', btype, r, re, kb, a

    Dr = De*( (exp(-a*(r-re)) - 1.0)**2 - 1.0 )

    return Dr, btype 


