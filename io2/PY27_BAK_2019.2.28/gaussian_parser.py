
import numpy as np
import util as ut

def get_line_number(f,s):
    return int( ut.cmdout2("grep -n '%s' %s | tail -1 | sed 's/:/ /g' | awk '{print $1}'"%(s,f) )

def get_dm(f,ia1,ia2):
    # total number of basis set
    nt = int( ut.cmdout("grep 'cartesian basis functions' %s | head -1 | awk '{print $3}'"%f) )

    n1 = get_line_number(f, '     Density Matrix:') + 1
    n2 = get_line_number(f, '    Full Mulliken population analysis')

    cs = file(f).readlines()[n1-1:n2]

    dm = np.zeros((nt,nt))

    n0 = 5 # number of columns for DM in Gaussian output file
    nl = n2-n1+1
    nlr = nl # number of lines remaining
    i5 = 0
    ns = [0,]
    nbk = nl/5 if nl%5 == 0 else nl/5 + 1
    for i in range(nbk):
        nli = nt - i5*5 + 1; ns.append(nli)
    assert nl == sum(ns), '#ERROR: Shit happens?'
    ins = np.cumsum(ns)
    for i in range(nbk):
        i1 = ins[i]
        i2 = ins[i+1]
        ics = np.array(cs[i1].split()).astype(np.int) # indices of columns
        csi = cs[i1+1:i2]; ni = len(csi)
        if i == 0:
            ibs = []; bst = []; symbs = []
            # get basis --> atom Idx
            for j in range(ni):
                sj = csi[j]
                sj1 = sj[:22].strip().split(); sj2 = sj[22:].strip().split()
                nj1 = len(sj1); nj2 = len(sj2)
                if len(sj1) == 4:
                    ibs.append(j); symbs.append(sj1[2]); bst.append(sj1[3])
                ir = int(sj1[0])
                dm[ir-1,ics[:nj2]] = np.array(sj2).astype(np.float)
        else:
            for j in range(ni):
                sj = csi[j]
                sj1 = sj[:22].strip().split(); sj2 = sj[22:].strip().split()
                ir = int(sj1[0])
                dm[ir-1,ics[:nj2]] = np.array(sj2).astype(np.float)

    ibs_u = np.array( ibs + [nt,] ).astype(np.int)
    na = len(ibs)
    ibs1 = ibs_u[:na]; ibs2 = ibs_u[1:na+1]
    p,q,r,s = ibs1[ia1],ibs2[ia1], ibs1[ia2],ibs2[ia2]
    nb1 = q-p; nb2 = s-r
    dmij = dm[p:q,r:s]
    so = ''
    for p in range(nb1):
        for q in range(nb2):
            so += '%9.5f'%dmij[p,q]
        so += '\n'
    print so

def __main__():
    import os,sys

    args = sys.argv[1:]
    f = args[0]
    ia1 = int( args[1] )
    ia2 = int( args[2] )
    get_dm(f, ia1, ia2)

