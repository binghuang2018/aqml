
"""
Valence, Coordination_number, Pi electron_num and c/r^2
representation (VCP) of atom in molecule, where c, r are
the column and row number of the corresponding element
in periodic table

VCP is designed for prediction of melting & boiling
points of molecule

"""

from openeye.oechem import *
import aqml.cheminfo.OEChem as cio
import numpy as np
import itertools as itl

global const
const = np.sqrt(0.5)/4.0

def get_overlap2(v1,v2,sigma):
    # overlap of two Gaussian distr
    return np.exp(-(v1-v2)**2/(4.0*sigma**2)) #/(2.*sigma*np.sqrt(np.pi))


def get_overlap4(v1,v2,v3,v4,sigma):
    # overlap of 4 Gaussian distr
    vs = np.array([v1,v2,v3,v4])
    return np.exp( -(4.*np.sum(vs**2)-np.sum(vs)**2)/(8.0*sigma**2) ) #*const*(sigma**2*np.sqrt(np.pi))**(-1.5)


class dist(object):

    def __init__(self, strings, nproc=1):

        objs = cio.StringsM( strings, igroup=True, iPL=True, nproc=nproc ).objs
        self.objs = objs
        self.nm = len(objs)

    def get_dist(self, s1, s2):
        """
        get distance between two molecules: a summation
        of all atomic pair similarity
        """
        n1 = s1.nha
        n2 = s2.nha
        self.dsi = np.zeros((n1,n2))
        for ia in range(n1):
            for ja in range(ia,n1):
                self.dsi[ia,ja] = self.get_atom_dist(s1,ia,s2,ja)

    def get_atom_dist(self,s1,ia,s2,ja, sigmas, power=2, debug=False):
        n = power
        gs1 = s1.groups
        vcps1 = [ gs1[i][0] for i in np.arange(s1.nha) ]
        gs2 = s2.groups
        vcps2 = [ gs2[i][0] for i in np.arange(s2.nha) ]
        vcp1 = vcps1[ia]
        vcp2 = vcps2[ja]
        sigmas_a = sigmas[:-1]
        sigma_r = sigmas[-1]
        # one-body distance
        vf2 = np.vectorize(get_overlap2)
        if vcp1 == vcp2:
            d2_A1 = 0.0
        else:
            di = np.product( vf2( vcp1, vcp1, sigmas_a ) )
            dj = np.product( vf2( vcp2, vcp2, sigmas_a ) )
            dk = np.product( vf2( vcp1, vcp2, sigmas_a ) )
            d2_A1 = di + dj - 2*dk
            #print ' di, dj, dk = ', di,dj,dk
        #print ' ----- d2_A1 = ', d2_A1
        # two-body parts
        ias1_raw = np.arange(s1.nha)
        ias2_raw = np.arange(s2.nha)
        pls1 = s1.PLs
        pls2 = s2.PLs
        ias1 = ias1_raw[ np.logical_and(pls1[ia] <= 3, pls1[ia] > 0) ]
        ias2 = ias2_raw[ np.logical_and(pls2[ja] <= 3, pls2[ja] > 0) ]
        vf4 = np.vectorize(get_overlap4)
        d2_A2 = 0.
        #print ' ia, ias1, PLs1 = ', ia, ias1, pls1[ia,ias1]
        #print ' ja, ias2, PLs2 = ', ja, ias2, pls2[ja,ias2]
        if not (s1 == s2 and ia == ja):
            for b1 in ias1:
                d1 = pls1[ia,b1]+1.
                di = np.product( vf4(vcps1[ia],vcps1[b1],vcps1[ia], \
                               vcps1[b1],sigmas_a) )*get_overlap2(d1,d1,sigma_r)/(d1*d1)**n
                if debug: print ' + 1-1 (ia,b1,ia,b1) = ', (ia,b1,ia,b1), ', di = ', di
                d2_A2 += di
            for b1,b2 in itl.combinations(ias1, 2):
                d1 = pls1[ia,b1]+1.; d2 = pls1[ia,b2]+1.
                dii = 2*np.product( vf4(vcps1[ia],vcps1[b1],vcps1[ia], \
                           vcps1[b2],sigmas_a) )*get_overlap2(d1,d2,sigma_r)/(d1*d2)**n
                d2_A2 += dii
                if debug: print ' + 1-1 (ia,b1,ia,b2) = ', (ia,b1,ia,b2), ', dii = ', dii
            for b1 in ias2:
                d1 = pls2[ja,b1]+1.
                dj = np.product( vf4(vcps2[ja],vcps2[b1],vcps2[ja], \
                               vcps2[b1],sigmas_a) )*get_overlap2(d1,d1,sigma_r)/(d1*d1)**n
                d2_A2 += dj
                if debug: print ' + 2-2 (ja,b1,ja,b1) = ', (ja,b1,ja,b1), ', dj = ', dj
            for b1,b2 in itl.combinations(ias2, 2):
                d1 = pls2[ja,b1]+1.; d2 = pls2[ja,b2]+1.
                djj = 2*np.product( vf4(vcps2[ja],vcps2[b1],vcps2[ja], \
                           vcps2[b2],sigmas_a) )*get_overlap2(d1,d2,sigma_r)/(d1*d2)**n
                d2_A2 += djj
                if debug: print ' + 2-2 (ia,b1,ia,b1) = ', (ia,b1,ia,b1), ', djj = ', djj
            for b1 in ias1:
                for b2 in ias2:
                    d1 = pls1[ia,b1]+1.; d2 = pls2[ja,b2]+1.
                    dij = -2.*np.product(vf4(vcps1[ia],vcps1[b1],vcps2[ja],vcps2[b2],\
                         sigmas_a))*get_overlap2(d1,d2,sigma_r)/(d1*d2)**n
                    d2_A2 += dij
                    #print ' - ', vcps1[ia],vcps1[b1],vcps2[ja],vcps2[b2]
                    if debug: print ' + 1-2 (ia,b1,ja,b2) = ', (ia,b1,ja,b2), ',dij = ', dij

        if d2_A2 < 0 and d2_A2 > -1.e-6: d2_A2 = 0.
        if d2_A1 < 0 and d2_A1 > -1.e-6: d2_A1 = 0.
        if debug: print ' -- d2_A1, d2_A2 = ', d2_A1, d2_A2
        return np.sqrt( d2_A1 + d2_A2 )

    def get_molecule_dist(self, s1, s2, sigmas, PLc=3, power=2):

        n = power
        gs1 = s1.groups
        na1 = s1.nha
        gs2 = s2.groups
        vcps1 = [ gs1[i][0] for i in np.arange(s1.nha) ]
        vcps2 = [ gs2[i][0] for i in np.arange(s2.nha) ]
        na2 = s2.nha
        sigmas_a = sigmas[:-1]
        sigma_r = sigmas[-1]

        #if s1 == s2:
        #   d12 = 0.
        if 1: #else:
            pls1 = s1.PLs
            pls2 = s2.PLs

            # one-body distance
            vf2 = np.vectorize(get_overlap2)
            d2_A1 = 0.
            bs1 = []
            for i in range(na1):
                d2_A1 += np.product( vf2( vcps1[i], vcps1[i], sigmas_a ) )
                for j in range(i+1,na1):
                    d2_A1 += 2.*np.product( vf2( vcps1[i], vcps1[j], sigmas_a ) )
                    cij = pls1[i,j]
                    if cij > 0 and cij <= PLc: bs1.append( [i,j] )
            bs2 = []
            for i in range(na2):
                d2_A1 += np.product( vf2( vcps2[i], vcps2[i], sigmas_a ) )
                for j in range(i+1,na2):
                    d2_A1 += 2.*np.product( vf2( vcps2[i], vcps2[j], sigmas_a ) )
                    cij = pls2[i,j]
                    if cij > 0 and cij <= PLc: bs2.append( [i,j] )
            for i in range(na1):
                for j in range(na2):
                    d2_A1 -= 2.*np.product( vf2( vcps1[i], vcps2[j], sigmas_a ) )

            vf4 = np.vectorize(get_overlap4)
            d2_A2 = 0.
            for i,j in bs1:
                for k,l in bs1:
                    d1 = pls1[i,j] + 1
                    d2 = pls1[k,l] + 1
                    di = np.product( vf4(vcps1[i],vcps1[j],vcps1[k], \
                               vcps1[l],sigmas_a) )*get_overlap2(d1,d2,sigma_r)/(d1*d2)**n
                    d2_A2 += di
            for i,j in bs2:
                for k,l in bs2:
                    d1 = pls2[i,j] + 1
                    d2 = pls2[k,l] + 1
                    di = np.product( vf4(vcps2[i],vcps2[j],vcps2[k], \
                               vcps2[l],sigmas_a) )*get_overlap2(d1,d2,sigma_r)/(d1*d2)**n
                    #print ' + (ia,b1,ia,b1) = ', (ia,b1,ia,b1), ', di = ', di
                    d2_A2 += di

            for i,j in bs1:
                for k,l in bs2:
                    d1 = pls1[i,j] + 1
                    d2 = pls2[k,l] + 1
                    di = np.product( vf4(vcps1[i],vcps1[j],vcps2[k], \
                               vcps2[l],sigmas_a) )*get_overlap2(d1,d2,sigma_r)/(d1*d2)**n
                    d2_A2 += -2.*di
        #print 'd2_A1, d2_A2 = ', d2_A1, d2_A2
        if d2_A2 < 0 and d2_A2 > -1.e-6: d2_A2 = 0.
        if d2_A1 < 0 and d2_A1 > -1.e-6: d2_A1 = 0.

        return np.sqrt(d2_A1 + d2_A2)
