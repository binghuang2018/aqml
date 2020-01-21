

# unit: Hartree

"""
hfcbsv23z  : HF/vdz, HF/vtz --> HF/cbs
mp2cbsv23z : MP2/vdz, MP2/vtz --> MP2/cbs
cc2cbsmp2v23z : E[cc2/cbs] = E[HF/cbs] + E_corr[cc2/cbs]
                           = E[HF/cbs] +  E_corr[cc2/vdz] + ( E_corr[cc2/cbs] - E_corr[cc2/vdz] )
                           ~ E[HF/cbs] +  E_corr[cc2/vdz] + ( E_corr[mp2/cbs] - E_corr[mp2/vdz] )
                           = E[mp2/cbs] + E[cc2/vdz] - E[mp2/vdz] )
cc2cbsv23z : CCSD(T)/vdz, ccsd(t)/vtz --> ccsd(t)/cbs

"""

T, F = True, False

#def update_dct(adic, dct):
import numpy as np
import io2

## reference atomic energies
## H2, CH4, N2, O2, F2, Cl2, Br2, I2, P4, S8
e_atoms0 = {}


class extrapolate(object):

    _alpha = {'v23z': 4.42, 'v34z': 5.46, }
    _beta = {'v23z': 2.46, 'v34z': 3.05}
    _card = {'v23z':[2,3], 'v34z':[3,4]}

    def __init__(self, es, basis, unit='h'): #, etype='corr'):
        """
        Note: input energy (`es) has to be either HF energies or other type of energies
        """
        self.es = es
        self.basis = basis
        uc = io2.Units()
        self.const = {'kcal':uc.h2kc, 'h':1.0}[unit]

    @property
    def hfcbs(self):
        if not hasattr(self, '_hfcbs'):
            self._hfcbs = self.get_hfcbs()
        return self._hfcbs

    def get_hfcbs(self):
        e1, e2 = self.es
        a = self._alpha[self.basis]
        cn1, cn2 = self._card[self.basis]
        A = (e1-e2)/(np.exp(-a*np.sqrt(cn1))-np.exp(-a*np.sqrt(cn2)))
        e_hf = e1 - A*np.exp(-a*np.sqrt(cn1))
        return e_hf * self.const

    @property
    def corrcbs(self):
        if not hasattr(self, '_corrcbs'):
            self._corrcbs = self.get_corrcbs()
        return self._corrcbs

    def get_corrcbs(self):
        e1, e2 = self.es
        b = self._beta[self.basis]
        cn1, cn2 = self._card[self.basis]
        e_corr = (e1 * cn1**b - e2 * cn2**b)/(cn1**b - cn2**b)
        return e_corr * self.const



class adb(object):

    def __init__(self, dct):
        self.dct = dct

    def update(self, newdct):
        for k in newdct:
            if k not in self.dct.keys():
                self.dct.update( {k:newdct[k]} )
            else:
                v2 = self.dct[k] # e.g., v2 = {'H': {'hfvdz': ..}, 'C': {'hfvdz': ...}}
                newv2 = newdct[k]
                for k2 in newv2.keys():
                    if k2 not in v2.keys():
                        self.dct[k].update( {k2:newv2[k2]} )
                    else:
                        v3 = v2[k2] # e.g., v3 = {'hfvdz':...}
                        newv3 = newv2[k2]
                        for k3 in newv3.keys():
                            if k3 not in v3.keys():
                                self.dct[k][k2].update( {k3:newv3[k3]} )
                            else:
                                if abs(v3[k3]-newv3[k3]) > 0.0004: # in Hartree,  ~ 0.25 kcal/mol
                                    print(" dct['%s']['%s']['%s']: default=%.8f, new=%.8f"%(k,k2,k3, v3[k3], newv3[k3]))
                                    raise Exception('#ERROR: inconsistent')

    def update_ecbs(self):
        """ update `dct with extrapolated energies: HF/cbs, CC2/cbs ... """
        _bsts = ['vdz','vtz','vqz']
        for prog in ['orca','g09']:
          d = self.dct[prog]
          for el in d.keys():
            dl = d[el]
            ms0 = list(dl.keys())
            _mshf = ['hf'+si for si in _bsts]
            cbsts = ['v23z','v34z']
            for m in ['hf','mp2','cc2']:
                _ms = [m+si for si in _bsts ]
                for i in range(2):
                    cbst = cbsts[i]; ms = _ms[i:i+2] # 'v23z'; ms = _ms[:2]
                    mshf = _mshf[i:i+2]
                    if set(ms) <= set(ms0):
                        esi = [dl[ms[0]],dl[ms[1]]] if m=='hf' else [dl[ms[0]]-dl[mshf[0]],dl[ms[1]]-dl[mshf[1]]]
                        o = extrapolate(esi, cbst)
                        e = o.hfcbs if m=='hf' else o.corrcbs + self.dct[prog][el]['hfcbs'+cbst]
                        se = m+'cbs'+cbst
                        #if m=='mp2':
                        #    print( {se:e} )
                        if se not in dl.keys():
                            self.dct[prog][el].update({se:e})
                        else:
                            if abs(e-dl[se])>0.0004:# Hartree
                                print(" dct['%s']['%s']['%s']: default=%.8f, new=%.8f"%(prog,el,se, self.dct[prog][el][se], e))
                                raise Exception('inconsistent?')
            # cc2cbsmp2vNz
            for i in range(2):
                cbst = cbsts[i]
                bst = _bsts[i]
                #mshf = _mshf[i:i+2]
                #eshf = [ dl[mi] for mi in mshf ]
                #hfcbs = extrapolate(esi, cbst)
                meths = ['mp2cbs'+cbst] + [ mi+bst for mi in ['mp2','cc2'] ]
                mp2cbs, cc2, mp2 = [ self.dct[prog][el][_mt] for _mt in ['mp2cbs'+cbst, 'cc2'+bst, 'mp2'+bst ] ]
                se = 'cc2cbsmp2'+cbst
                e = mp2cbs + cc2 - mp2
                if se not in self.dct[prog][el]:
                    self.dct[prog][el].update({se:e})
                else:
                    if abs(e-self.dct[prog][el][se]) > 0.0005:
                        print(" dct['%s']['%s']['%s']: default=%.8f, new=%.8f"%(prog,el,se, self.dct[prog][el][se], e))
                        raise Exception('inconsistent?')


_adic = \
{
  'orca':{
        'Br': {'hfvtz': -2572.445145464, 'hfvqz': -2572.448122325, 'hfcbsv34z': -2572.449019265,
               'cc2vtz': -2572.6612460819997, 'cc2vqz': -2572.733440121, 'cc2cbsv34z': -2572.783612043},
        'Cl': {'hfvtz': -459.48543389, 'hfvqz': -459.489094587, 'hfcbsv34z': -459.49019757,
               'cc2vtz': -459.671808342, 'cc2vqz': -459.693309006, 'cc2cbsv34z': -459.707112074},
        'C': {'hfvtz': -37.691569173, 'hfvqz': -37.69330782, 'hfcbsv34z': -37.693831681,
              'cc2vtz': -37.780761758, 'cc2vqz': -37.786540054, 'cc2cbsv34z': -37.789939699},
        'F': {'hfvtz': -99.40552486, 'hfvqz': -99.413770134, 'hfcbsv34z': -99.416254467,
              'cc2vtz': -99.620360706, 'cc2vqz': -99.650258019, 'cc2cbsv34z': -99.668156212,
             },
        'H': {'hfvdz': -0.499279204052, 'hfvtz':-0.499808808990, 'hfvqz':-0.499944283175, 'hfcbsv23z':-0.49998105, 'hfcbsv34z':-0.49998510,
              'mp2vdz':-0.499279204052, 'mp2vtz':-0.499808808990, 'mp2vqz':-0.499944283175, 'mp2cbsv23z':-0.49998105, 'hfcbsv34z':-0.49998510,
              'cc2vdz':-0.499279204052, 'cc2vtz':-0.499808808990, 'cc2vqz':-0.499944283175, 'cc2cbsv23z':-0.49998105, 'hfcbsv34z':-0.49998510,
             },
        'N': {'hfvtz': -54.40068621, 'hfvqz': -54.40371796, 'hfcbsv34z': -54.40463143,
              'cc2vtz': -54.51470784, 'cc2vqz': -54.52482389, 'cc2cbsv34z': -54.53078061},
        'O': {'hfvtz': -74.811756619, 'hfvqz': -74.817294694, 'hfcbsv34z': -74.818963337,
              'cc2vtz': -74.973961494, 'cc2vqz': -74.993565588, 'cc2cbsv34z': -75.005247683},
        'P': {'hfvtz': -340.71630581, 'hfvqz': -340.718721232, 'hfcbsv34z': -340.719449009,
              'cc2vtz': -340.821020567, 'cc2vqz': -340.827920685, 'cc2cbsv34z': -340.83184107},
        'S': {'hfvtz': -397.509268872, 'hfvqz': -397.512581195, 'hfcbsv34z': -397.513579211,
              'cc2vtz': -397.653348383, 'cc2vqz': -397.667115266, 'cc2cbsv34z': -397.675555773},
        },

}

adic = adb(_adic)


dct1 = {'orca':{
    'Br':{'hfvdz':-2572.36837409, 'hfvtz':-2572.44514625, 'mp2vdz':-2572.47307716, 'mp2vtz':-2572.63951333, 'cc2vdz':-2572.48718132, 'hfcbsv23z':-2572.47011380, 'cc2cbsmp2v23z':-2572.73097898, } ,
    'Cl':{'hfvdz':-459.47114583, 'hfvtz':-459.48543443, 'mp2vdz':-459.58094420, 'mp2vtz':-459.64336303, 'cc2vdz':-459.59889601, 'hfcbsv23z':-459.49008132, 'cc2cbsmp2v23z':-459.69408596, } ,
    'C':{'hfvdz':-37.68654462, 'hfvtz':-37.69156922, 'mp2vdz':-37.73830987, 'mp2vtz':-37.75848775, 'cc2vdz':-37.76037777, 'hfcbsv23z':-37.69320330, 'cc2cbsmp2v23z':-37.79104433, } ,
    'F':{'hfvdz':-99.37524017, 'hfvtz':-99.40552457, 'mp2vdz':-99.51584025, 'mp2vtz':-99.60536018, 'cc2vdz':-99.52757395, 'hfcbsv23z':-99.41537355, 'cc2cbsmp2v23z':-99.66155633, } ,
    'N':{'hfvdz':-54.39111466, 'hfvtz':-54.40068733, 'mp2vdz':-54.46185518, 'mp2vtz':-54.49661785, 'cc2vdz':-54.47849854, 'hfcbsv23z':-54.40380052, 'cc2cbsmp2v23z':-54.53109382, } ,
    'O':{'hfvdz':-74.79216566, 'hfvtz':-74.81175641, 'mp2vdz':-74.89413115, 'mp2vtz':-74.95490218, 'cc2vdz':-74.90995035, 'hfcbsv23z':-74.81812764, 'cc2cbsmp2v23z':-75.00115574, } ,
    'P':{'hfvdz':-340.70904065, 'hfvtz':-340.71630229, 'mp2vdz':-340.77255389, 'mp2vtz':-340.79957409, 'cc2vdz':-340.79131443, 'hfcbsv23z':-340.71866389, 'cc2cbsmp2v23z':-340.83224187, } ,
    'S':{'hfvdz':-397.49680223, 'hfvtz':-397.50926962, 'mp2vdz':-397.58060396, 'mp2vtz':-397.62580298, 'cc2vdz':-397.60055570, 'hfcbsv23z':-397.51332422, 'cc2cbsmp2v23z':-397.66893560, } ,
}}
adic.update(dct1)


dct1_0 = {'orca':{
'Br':{'hfvtz':-2572.44514552, 'hfvqz':-2572.44812188, 'mp2vtz':-2572.63951264, 'mp2vqz':-2572.71403372, 'cc2vtz':-2572.66124614, 'hfcbsv34z':-2572.44901867, 'cc2cbsmp2v34z':-2572.78759595, } ,
'Cl':{'hfvtz':-459.48543394, 'hfvqz':-459.48909542, 'mp2vtz':-459.64336259, 'mp2vqz':-459.66397505, 'cc2vtz':-459.67180840, 'hfcbsv34z':-459.49019864, 'cc2cbsmp2v34z':-459.70559129, } ,
'C':{'hfvtz':-37.69156947, 'hfvqz':-37.69330855, 'mp2vtz':-37.75848799, 'mp2vqz':-37.76536365, 'cc2vtz':-37.78076205, 'hfcbsv34z':-37.69383254, 'cc2cbsmp2v34z':-37.79181837, } ,
'F':{'hfvtz':-99.40552525, 'hfvqz':-99.41376992, 'mp2vtz':-99.60536083, 'mp2vqz':-99.63520602, 'cc2vtz':-99.62036110, 'hfcbsv34z':-99.41625408, 'cc2cbsmp2v34z':-99.66806762, } ,
'N':{'hfvtz':-54.40068622, 'hfvqz':-54.40371910, 'mp2vtz':-54.49661676, 'mp2vqz':-54.50759661, 'cc2vtz':-54.51470785, 'hfcbsv34z':-54.40463292, 'cc2cbsmp2v34z':-54.53225889, } ,
'O':{'hfvtz':-74.81175740, 'hfvqz':-74.81729435, 'mp2vtz':-74.95490314, 'mp2vqz':-74.97488676, 'cc2vtz':-74.97396227, 'hfcbsv34z':-74.81896265, 'cc2cbsmp2v34z':-75.00589863, } ,
'P':{'hfvtz':-340.71630580, 'hfvqz':-340.71872190, 'mp2vtz':-340.79957761, 'mp2vqz':-340.80758724, 'cc2vtz':-340.82102056, 'hfcbsv34z':-340.71944988, 'cc2cbsmp2v34z':-340.83374015, } ,
'S':{'hfvtz':-397.50926933, 'hfvqz':-397.51258085, 'mp2vtz':-397.62580280, 'mp2vqz':-397.63986008, 'cc2vtz':-397.65334885, 'hfcbsv34z':-397.51357862, 'cc2cbsmp2v34z':-397.67605370, } ,
}}
adic.update(dct1_0)


# orca: default setting on Grids
dct1_1 = {'orca':{
 'H': {'b3lypdef2tzvp': -0.498764220938, 'wb97xdef2tzvp': -0.501393616509},
 'C': {'b3lypdef2tzvp': -37.838155285315, 'wb97xdef2tzvp': -37.846231183856},
 'N': {'b3lypdef2tzvp': -54.579011674931, 'wb97xdef2tzvp': -54.591755729681},
 'O': {'b3lypdef2tzvp': -75.066885837081, 'wb97xdef2tzvp': -75.076743757959},
 'F': {'b3lypdef2tzvp': -99.736302688584, 'wb97xdef2tzvp': -99.747331673869},
 'P': {'b3lypdef2tzvp': -341.223918586333, 'wb97xdef2tzvp': -341.25262407453},
 'S': {'b3lypdef2tzvp': -398.071502872284, 'wb97xdef2tzvp': -398.107732865939},
 'Cl': {'b3lypdef2tzvp': -460.101732814945, 'wb97xdef2tzvp': -460.146375877741},
 'Br': {'b3lypdef2tzvp': -2574.001667573669, 'wb97xdef2tzvp': -2574.1387121629},
}}
adic.update(dct1_1)



# pyscf
dct3_0 = {'pyscf':{
    'H': {'pbedef2tzvp':-0.49876281, 'pbe0def2tzvp':-0.49876281, 'b3lypdef2tzvp':-0.49876281, 'wb97xdef2tzvp':-0.49876281, },
    'C': {'pbedef2tzvp':-37.83633343, 'pbe0def2tzvp':-37.83633339, 'b3lypdef2tzvp':-37.83633340, 'wb97xdef2tzvp':-37.83633340, },
    'N': {'pbedef2tzvp':-54.57729140, 'pbe0def2tzvp':-54.57729140, 'b3lypdef2tzvp':-54.57729140, 'wb97xdef2tzvp':-54.57729140, },
    'O': {'pbedef2tzvp':-75.06422091, 'pbe0def2tzvp':-75.06422087, 'b3lypdef2tzvp':-75.06422092, 'wb97xdef2tzvp':-75.06422073, },
    'F': {'pbedef2tzvp':-99.73440638, 'pbe0def2tzvp':-99.73440580, 'b3lypdef2tzvp':-99.73440591, 'wb97xdef2tzvp':-99.73440565, },
    'P': {'pbedef2tzvp':-341.22374520, 'pbe0def2tzvp':-341.22374520, 'b3lypdef2tzvp':-341.22374520, 'wb97xdef2tzvp':-341.22374520, },
    'S': {'pbedef2tzvp':-398.06969826, 'pbe0def2tzvp':-398.06969826, 'b3lypdef2tzvp':-398.06969826, 'wb97xdef2tzvp':-398.06969826, },
    'Cl': {'pbedef2tzvp':-460.09993275, 'pbe0def2tzvp':-460.09993275, 'b3lypdef2tzvp':-460.09993275, 'wb97xdef2tzvp':-460.09993275, },
    'Br': {'pbedef2tzvp':-2574.00024752, 'pbe0def2tzvp':-2574.00024752, 'b3lypdef2tzvp':-2574.00024752, 'wb97xdef2tzvp':-2574.00024752, },
}}
adic.update(dct3_0)



# g09
dct2 = { 'g09': {
 'H': {'b3lyp631g2dfp': -0.5002728,
  'b3lypdef2tzvp': -0.5021542,
  'b3lypvtz': -0.5021563,
  'wb97xdef2tzvp': -0.5013925},
 'C': {'b3lyp631g2dfp': -37.8467717,
  'b3lypdef2tzvp': -37.8594785,
  'b3lypvtz': -37.8585747,
  'wb97xdef2tzvp': -37.8459781},
 'N': {'b3lyp631g2dfp': -54.5838615,
  'b3lypdef2tzvp': -54.6039774,
  'b3lypvtz': -54.6017813,
  'wb97xdef2tzvp': -54.5915914},
 'O': {'b3lyp631g2dfp': -75.0645794,
  'b3lypdef2tzvp': -75.0962771,
  'b3lypvtz': -75.0918643,
  'wb97xdef2tzvp': -75.0768759},
 'F': {'b3lyp631g2dfp': -99.7187304,
  'b3lypdef2tzvp': -99.7701508,
  'b3lypvtz': -99.7628668,
  'wb97xdef2tzvp': -99.7471707},
 'P': {'b3lyp631g2dfp': -341.2575546,
  'b3lypdef2tzvp': -341.2805908,
  'b3lypvtz': -341.2858505,
  'wb97xdef2tzvp': -341.2530976},
 'S': {'b3lyp631g2dfp': -398.1057561,
  'b3lypdef2tzvp': -398.1323744,
  'b3lypvtz': -398.1386933,
  'wb97xdef2tzvp': -398.1079436},
 'Cl': {'b3lyp631g2dfp': -460.1366859,
  'b3lypdef2tzvp': -460.1669789,
  'b3lypvtz': -460.1746648,
  'wb97xdef2tzvp': -460.1467358},
 'Br': {'b3lyp631g2dfp': -2571.8235791,
  'b3lypdef2tzvp': -2574.1409432,
  'b3lypvtz': -2574.1876331,
  'wb97xdef2tzvp': -2574.1405094},
}}
adic.update(dct2)

dct2_1 = {'g09': {
'Br': {'hfvdz': -2572.3683737,
   'mp2vdz': -2572.473077,
   'cc2vdz': -2572.4871813,
   'hfvqz': -2572.4480199,
   'mp2vqz': -2572.7139306,
   'cc2vqz': -2572.7333367,
   'hfvtz': -2572.4450856,
   'mp2vtz': -2572.6394525,
   'cc2vtz': -2572.6611862},
  'C': {'hfvdz': -37.6865444,
   'mp2vdz': -37.7383097,
   'cc2vdz': -37.7603771,
   'hfvqz': -37.6933076,
   'mp2vqz': -37.7653628,
   'cc2vqz': -37.7865397,
   'hfvtz': -37.6915692,
   'mp2vtz': -37.7584877,
   'cc2vtz': -37.7807623},
  'Cl': {'hfvdz': -459.4711433,
   'mp2vdz': -459.5809419,
   'cc2vdz': -459.5988927,
   'hfvqz': -459.4890945,
   'mp2vqz': -459.6639741,
   'cc2vqz': -459.6933089,
   'hfvtz': -459.4854338,
   'mp2vtz': -459.6433625,
   'cc2vtz': -459.6718083},
  'F': {'hfvdz': -99.3752403,
   'mp2vdz': -99.5158404,
   'cc2vdz': -99.5275741,
   'hfvqz': -99.4137701,
   'mp2vqz': -99.6352062,
   'cc2vqz': -99.6502581,
   'hfvtz': -99.4055248,
   'mp2vtz': -99.6053603,
   'cc2vtz': -99.6203607},
  'H': {'hfvdz': -0.4992784,
   'mp2vdz': -0.4992784,
   'cc2vdz': -0.4992784,
   'hfvqz': -0.4999456,
   'mp2vqz': -0.4999456,
   'cc2vqz': -0.4999456,
   'hfvtz': -0.4998098,
   'mp2vtz': -0.4998098,
   'cc2vtz': -0.4998098},
  'N': {'hfvdz': -54.3911146,
   'mp2vdz': -54.4618551,
   'cc2vdz': -54.4785017,
   'hfvqz': -54.4037179,
   'mp2vqz': -54.5075955,
   'cc2vqz': -54.5248238,
   'hfvtz': -54.4006862,
   'mp2vtz': -54.4966167,
   'cc2vtz': -54.5147073},
  'O': {'hfvdz': -74.7921661,
   'mp2vdz': -74.8941316,
   'cc2vdz': -74.9099503,
   'hfvqz': -74.8172944,
   'mp2vqz': -74.9748868,
   'cc2vqz': -74.9935654,
   'hfvtz': -74.8117566,
   'mp2vtz': -74.9549023,
   'cc2vtz': -74.9739618},
  'P': {'hfvdz': -340.7090479,
   'mp2vdz': -340.7725612,
   'cc2vdz': -340.7913306,
   'hfvqz': -340.7187212,
   'mp2vqz': -340.8075865,
   'cc2vqz': -340.8279207,
   'hfvtz': -340.7163059,
   'mp2vtz': -340.7995777,
   'cc2vtz': -340.8210195},
  'S': {'hfvdz': -397.4968014,
   'mp2vdz': -397.5806033,
   'cc2vdz': -397.6005524,
   'hfvqz': -397.5125812,
   'mp2vqz': -397.6398605,
   'cc2vqz': -397.6671153,
   'hfvtz': -397.5092689,
   'mp2vtz': -397.6258023,
   'cc2vtz': -397.6533488}}}
adic.update(dct2_1)

dct2_2 = {'g09': {'Br': {'hfavtz': -2572.4454985, 'mp2avtz': -2572.6466854, 'cc2avtz': -2572.6684584}, 'Cl': {'hfavtz': -459.4859689, 'mp2avtz': -459.6473317, 'cc2avtz': -459.6762157}, 'C': {'hfavtz': -37.6918113, 'mp2avtz': -37.7595607, 'cc2avtz': -37.7818255}, 'F': {'hfavtz': -99.4068794, 'mp2avtz': -99.6121061, 'cc2avtz': -99.627827}, 'H': {'hfavtz': -0.4998212, 'mp2avtz': -0.4998212, 'cc2avtz': -0.4998212}, 'N': {'hfavtz': -54.4011622, 'mp2avtz': -54.4986473, 'cc2avtz':
    -54.5169239}, 'O': {'hfavtz': -74.8129822, 'mp2avtz': -74.9592941, 'cc2avtz': -74.9789523}, 'P': {'hfavtz': -340.7164915, 'mp2avtz': -340.800759, 'cc2avtz': -340.8222131}, 'S': {'hfavtz': -397.5098727, 'mp2avtz': -397.6283261, 'cc2avtz': -397.6561421}}}
adic.update(dct2_2)


# molpro
dct4_1 = {'molpro': {'Br': {'cc2f12avtz': -2572.62591871, 'cc2avtz': -2572.60619871}, 'Cl': {'cc2f12avtz': -459.69243724, 'cc2avtz': -459.67619145}, 'C': {'cc2f12avtz': -37.78834844, 'cc2avtz': -37.78172904}, 'F': {'cc2f12avtz': -99.6602009, 'cc2avtz': -99.62777119}, 'H': {'cc2f12avtz': -0.49985194, 'cc2avtz': -0.49982118}, 'N': {'cc2f12avtz': -54.52733662, 'cc2avtz': -54.5167142}, 'O': {'cc2f12avtz': -74.99980788, 'cc2avtz': -74.97882259}, 'P': {'cc2f12avtz': -340.82758694,
    'cc2avtz': -340.82212963}, 'S': {'cc2f12avtz': -397.66696535, 'cc2avtz': -397.65608397}}}
adic.update(dct4_1)


# qmc
dct5 = {'qmcpack': {
 'H': {'dmc_hf': -0.5,
  'dmc_hf_err': 0.0,
  'dmc_pbe': -0.5,
  'dmc_pbe_err': 0.0,
  'dmc_pbe0': -0.5,
  'dmc_pbe0_err': 0.0,
  'dmc_b3lyp': -0.5,
  'dmc_b3lyp_err': 0.0},
 'C': {'dmc_hf': -37.830807,
  'dmc_hf_err': 0.000246,
  'dmc_pbe': -37.829372,
  'dmc_pbe_err': 0.000305,
  'dmc_pbe0': -37.830114,
  'dmc_pbe0_err': 0.000268,
  'dmc_b3lyp': -37.829294,
  'dmc_b3lyp_err': 0.000227},
 'N': {'dmc_hf': -54.57468000000001,
  'dmc_hf_err': 0.000293,
  'dmc_pbe': -54.575637,
  'dmc_pbe_err': 0.000261,
  'dmc_pbe0': -54.575568000000004,
  'dmc_pbe0_err': 0.000269,
  'dmc_b3lyp': -54.575807,
  'dmc_b3lyp_err': 0.00028700000000000004},
 'O': {'dmc_hf': -75.051351,
  'dmc_hf_err': 0.000286,
  'dmc_pbe': -75.051498,
  'dmc_pbe_err': 0.0004969999999999999,
  'dmc_pbe0': -75.052266,
  'dmc_pbe0_err': 0.000366,
  'dmc_b3lyp': -75.05080699999999,
  'dmc_b3lyp_err': 0.000292},
 'F': {'dmc_hf': -99.716104,
  'dmc_hf_err': 0.000334,
  'dmc_pbe': -99.717548,
  'dmc_pbe_err': 0.00037,
  'dmc_pbe0': -99.71789100000001,
  'dmc_pbe0_err': 0.00035499999999999996,
  'dmc_b3lyp': -99.71677,
  'dmc_b3lyp_err': 0.000312}
}}
adic.update(dct5)


adic.update_ecbs()
dct = adic.dct


