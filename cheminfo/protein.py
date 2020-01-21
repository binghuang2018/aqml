
import cheminfo.RDKit as cio
import numpy as np


class protein(object):

    """
    purpose: eliminate charge, add H's
    """

    def __init__(self, pdb, option='exp'):

        """
        option: 'exp' -- experimental bond lengths, angles
                'b3lyp' -- b3lyp/6-31g(2df,p) geometry
        """

        self.option = option

        obj = cib.Mol('1HVI.pdb')

        icnt = 0
        iasN = []; iasO = []
        for ai in m.atoms:
            zi = ai.atomicnum
            vi = ai.valence
            vi2 = ai.implicitvalence
            ci = ai.formalcharge # somehow it's always 0, a bug of openbabel?
            if zi == 7:
                if vi == 4: # ai.implicitvalence is identical to ai.valence somehow
                    iasN.append( icnt )
            elif zi == 8:
                if vi == 1 and vi2 == 2: # [O-]
                    iasO.append( icnt )
            icnt += 1


