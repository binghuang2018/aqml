
import numpy as np
import ase.data as ad
import os, re, unicodedata
from math import *

global dot, blank, linefeed, bo3
dot = '\xb7'
blank = '\xa0'
linefeed = '\n'
bo3 = '\u2261' # triple bond representation

def read_data(obj):
    return np.loadtxt(obj)

def get_symbols(strs, case='normal'):
    # 'OCH2CClBrF' --> ['O','C','H','H','C','Cl','Br','F']

    # redirect to function `str2symbs
    # cause the original codes is incapable of
    # handling molecules with certain atoms whose
    # number exceeds 10.
    syms = str2symbs(strs, case=case)

    # these are the original codes
    obsolete_codes = """
    syms=[]
    ns = len(strs)
    numl = [str(n) for n in range(1,101)]
    i = 0
    strs += '   '
    for ii in xrange(ns):
        if i > ns - 1:
            continue

        if i < ns:
            s1,s2,s3 = strs[i:i+3]
            #print s1,s2,s3,s3!=' '
            if s2 not in numl:
                if s2!=' ':
                    if s2==s2.lower():
                        s=s1+s2
                        if s3!=' ':
                            if s3 in numl:syms+=[s,]*eval(s3);i+=3#;print 'case1',syms,'s3=',s3,[s,]*eval(s3)
                            else:syms+=[s,];i+=2
                        else:syms+=[s,];i+=2
                    else:
                        syms+=[s1,];i+=1#;print syms
                else:
                    if s1!=' ':syms+=[s1,];i+=1#;print syms
            else:
                if s2!=' ':
                    syms+=[s1,]*eval(s2)#;print syms,'s2=',s2,[s1,]*eval(s2)
                    i+=2
        #i+=1
        #print 'i=',i
    for sym in syms:
        if sym not in ad.chemical_symbols:
            print 'sym = ',sym
            raise 'non-formal formula'
    """

    return syms


def str2symbs(strs, case='organic', count_num=1):

    # 'OCH2CClBrF' --> ['O','C','H','H','C','Cl','Br','F']

    import re

    s0 = ad.chemical_symbols
    if case in ['organic',]:
        symbs1 = ['H','C','N','O','F','P','S','Cl','Br','I'] + ['c','n','o','p','s',]

    if count_num == 0:
        #print ' ** Neglecting the numbers in the Mol'
        #print '    used for refining smiles string'
        strs = re.sub('\d','',strs)

    # step1, --> ['OCH', 'CClBrF']
    raw_symbs = re.split('\d+', strs)
    nr = len(raw_symbs)

    n = len(strs)

    ns0 = []; symbs = []
    ints = [ str(s) for s in range(1,201) ]

    # step2, get `symbs and `ns
    cnt = 0
    while True:
        if cnt > n - 1:
            break

        s3 = strs[cnt:cnt+3]
        s2 = strs[cnt:cnt+2]
        s1 = strs[cnt:cnt+1]

        # maximal number of one type of atom is 200
        if s3 in ints:
            ns0.append( int(s3) )
            cnt += 3
        elif s2 in symbs1:
            symbs.append( s2 )
            cnt += 2
        elif s2 in ints:
            ns0.append( int(s2) )
            cnt += 2
        elif s1 in symbs1:
            symbs.append( s1 )
            cnt += 1
        elif s1 in ints:
            ns0.append( int(s1) )
            cnt += 1
        else:
            print('#ERROR: how could it happen?')
            raise

    nsymb = len(symbs)
    ns = [1,]*nsymb
    cnt = 0; cnt1 = -1
    while True:
        if cnt > n - 1:
            break

        s3 = strs[cnt:cnt+3]
        s2 = strs[cnt:cnt+2]
        s1 = strs[cnt:cnt+1]

        # maximal number of one type of atom is 200
        if s3 in ints:
            ns[cnt1] = int(s3)
            cnt += 3
        elif s2 in symbs1:
            cnt += 2; cnt1 += 1
        elif s2 in ints:
            ns[cnt1] = int(s2)
            cnt += 2
        elif s1 in symbs1:
            cnt += 1; cnt1 += 1
        elif s1 in ints:
            ns[cnt1] = int(s1)
            cnt += 1
        else:
            print('#ERROR: how could it happen?')
            raise

    symbs_upd = []
    for i in range(nsymb):
        symbs_upd += [symbs[i],]*ns[i]

    symbs = symbs_upd
    for symb in symbs:
        if symb not in symbs1:
            print('ERROR: symb `%s does not exist!'%symb)
            sys.exit(1)
            #raise 'non-formal formula'

    return symbs_upd

def remove_element(ls, option):
    """
    For `ls being list:
    a. remove the blank element in a list `ls
       whose unicode is u'\\xa0'
    b. remove the line feed u'\\n'
    c. remove odd/even elements

    For `ls being string:
    a. remove certain string `option from the given string
        when `option is a str
    b. replace the first str in `option by the second one in `option
        if `option is a list of length 2.
    """

    def canonicalize(s):
        if s in ['blank', 'ublank', blank]:
            s2 = blank
        elif s in ['linefeed', 'ulinefeed', linefeed]:
            s2 = linefeed
        elif s in ['dot', 'udot', dot]:
            s2 = dot
        elif s in ['bo3', 'ubo3', bo3]:
            s2 = bo3
        else:
            s2= s
        return s2

    if type(ls) is list:
        nl = len(ls)
        cnt = 0
        ls2 = []
        if type(option) not in [str, str]:
            print('not allowed')
            raise

        if option in ['blank', blank]:
            for l in ls:
                iok1 = (l == blank)
                if (not iok1):
                    ls2.append(l)
        elif option == ['linefeed', linefeed]:
            for l in ls:
                iok2 = (l == ''.join([linefeed,]*len(l)))
                if (not iok2):
                    ls2.append(l)
        elif option == 'odd':
            while 1:
                if cnt >= nl:
                    break
                else:
                    if cnt % 2 != 0:
                        ls2.append(ls[cnt])
                    cnt += 1
        elif option == 'even':
            while 1:
                if cnt >= nl:
                    break
                else:
                    if cnt % 2 == 0:
                        ls2.append(ls[cnt])
                    cnt += 1
        else:
            print('##no such option')
            raise
    elif type(ls) in [str, str]:
        if type(option) in [str, str]:
            o2 = canonicalize(option)

            if o2 in ls:
                ls2 = ''.join(ls.split(o2))
            else:
                ls2 = ls
                #raise 'the specified str does not exist in the given str'
        elif (type(option) is list) and (len(option) == 2):
            o1, o2 = option
            oo1 = canonicalize(o1)
            oo2 = canonicalize(o2)
            ls2 = oo2.join(ls.split(oo1))
        else:
            print('invalid option')
            raise
    else:
        print('not supported type of `ls')
        raise
    return ls2

def remove_element_exotic(s1):
    """
    remove some exotic str in a chemical formula
    """
    srms = []
    if '(%s)'%dot in s1:
        srm = '(%s)'%dot; srms.append(srm)
    else:
        if dot in s1:
            srm = dot; srms.append(srm)

    if '(&middot)' in s1:
        srms.append('(&middot)')
    else:
        if '&middot' in s1:
            srms.append('&middot')

    if '(.)' in s1:
        srms.append('(.)')
    else:
        if '.' in s1:
            srms.append('.')

    if ':' in s1:
        srms.append(':')
    if '*' in s1:
        srms.append(':')

    #print ' -- srms = ', srms

    for srm in srms:
        s1 = remove_element(s1, srm)

    # replace triple bond symbol with #
    s2 = remove_element(s1, ['ubo3', '#'])

    return s2

def is_decodable(s1):
    """
    try hard to decode the input chemical formula
    useful for recognizing those strings from nist database
    """
    for s in ['-','=','#',]:
        s1 = remove_element(s1, s)

    if ('(' in s1) and (')' in s1):
        while True:
            if not ('(' in s1):
                break
            else:
                # get functional group and its number
                try:
                    fg, N = re.search('\(([A-Za-z0-9]+)\)(\d?)', s1).groups()
                    if N == '':
                        s2 = ''.join(s1.split('(%s)'%fg))
                    else:
                        smd = '(%s)%s'%(fg, N)
                        smd2 = fg*int(N)
                        s2 = smd2.join(s1.split(smd))
                    s1 = s2
                except: # for cases such as O(^3P)
                    break

    iok = True
    try:
        symbs = get_symbols(s1)
    except:
        iok = False

    return iok

def get_indices(one_character, long_str):
    ns = len(long_str)
    indices = []
    for n in range(ns):
        if one_character == long_str[n]:
            indices.append(n)

    return indices

def is_gs_rxn(rxn):
    """
    check if the reaction happens under ground-state
    for strings downloaded from nist database
    """
    rs, ps = rxn
    species = rs + ps
    is_gs = True
    for s1 in species:
        try:
            s1u = str(s1)
        except:
            is_gs = False

        if ('(' in s1) and (')' in s1):
            indicesL = get_indices('(', s1)
            indicesR = get_indices(')', s1)
            nLs = len(indicesL)
            if nLs != len(indicesR):
                print('#ERROR: left and right paratheses not consistent')
                raise

            for jL in range(nLs):
                contjL = s1[indicesL[jL] + 1:indicesR[jL]]
                #try:
                #    symbs = get_symbols(contjL)
                #except:
                #    is_gs = False

                # cases to be excluded:
                # HO2 + H -> O2(1DELTA) + H2
                # H + O -> OH(A2Sigma+)
                # Cl(2P3/2) + BrONO2 -> Products
                # CH2(X3B_1) + OH -> CH + H2O
                if ('Sigma' in contjL) or ('X3B' in contjL) or \
                       (re.search('[Dd][Ee][Ll][Tt][Aa]',contjL) is not None) or \
                       (re.match('[1-9][SPDFGH]',contjL[:2]) is not None):
                   is_gs = False

    return is_gs

def is_none(s1):
    if type(s1) is list:
        iok = (s1 == [None, ]*len(s1))
    elif s1 is None:
        iok = True
    else:
        iok = False
    return iok

def unicode2(s1):
    return unicodedata.normalize('NFKD', str(s1)).encode('ascii','ignore')

def get_nsmax(formulas):
    """
    get the maximum number of each element in the order of
    H, C, N, O, F, P, S, Cl, Br, I
    """

    ss0 = ['H','C','N','O','F','P','S','Cl', 'Br', 'I']
    ns0 = [0,]*len(ss0)

    nas = []
    nmol = len(formulas)
    for j, sj in enumerate(formulas):
        sjs = str2symbs(sj)
        nasj = []
        for sk in ss0:
            nak = 0
            for sl in sjs:
                if sl == sk:
                    nak += 1
            nasj.append(nak)

        nas.append(nasj)

    nas_ = np.array(nas)
    nsmax = np.max(nas_,0)

    return nsmax

def concatenate(symbs):
    """
    ['C','H','H','H','H'] --> 'CH4'

    """

    ns = len(symbs)

    formula = ''

    groups = [[],]

    cntg = 0
    cnt = 0
    while True:
        if cnt > ns - 1:
            break

        gs1 = groups[cntg]
        gs2 = []
        sj = symbs[cnt]
        if sj not in gs1:
            gs2.append(sj)
            groups.append(gs2)
            cntg += 1
        else:
            gs1.append(sj)

        cnt += 1

    print(groups[1:])
    for gi in groups[1:]:
        si = gi[0]; ni = len(gi)
        if ni == 1:
            formula += '%s'%si
        else:
            formula += '%s%d'%(si,ni)

    return formula


def haskey(kvs, keys, idx):
  haskey = False
  for k in keys:
    if k in kvs:
      haskey = True
      idx += 1
  return haskey,idx

def parser(kvs, keys, v0, idx, iprt=True):
  """
  kvs: key_value list
  """

  v = v0

  haskey = False
  for k in keys:
    if k in kvs:
      haskey = True
      v = kvs[ kvs.index(k) + 1 ]; idx += 2

  if not haskey and iprt:
    print(' ** set -%s %s'%(keys[-1],v0))
  return haskey, v, idx

