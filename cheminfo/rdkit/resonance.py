# -*- coding: utf-8 -*-
"""
molvs.resonance
~~~~~~~~~~~~~~~

Resonance (mesomeric) transformations.

"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

from rdkit import Chem


MAX_STRUCTURES = 1000


class ResonanceEnumerator(object):
    """Simple wrapper around RDKit ResonanceMolSupplier.

    """

    def __init__(self, mol, kekule_all=False, allow_incomplete_octets=False, unconstrained_cations=False,
                 unconstrained_anions=False, allow_charge_separation=False, max_structures=MAX_STRUCTURES):
        """
        :param mol: The input molecule.
        :type mol: rdkit.Chem.rdchem.Mol
        :param bool allow_incomplete_octets: include resonance structures whose octets are less complete than the the most octet-complete structure.
        :param bool allow_charge_separation: include resonance structures featuring charge separation also when uncharged resonance structures exist.
        :param bool kekule_all: enumerate all possible degenerate Kekule resonance structures (the default is to include just one).
        :param bool unconstrained_cations: if False positively charged atoms left and right of N with an incomplete octet are acceptable only if the conjugated group has a positive total formal charge.
        :param bool unconstrained_anions: if False, negatively charged atoms left of N are acceptable only if the conjugated group has a negative total formal charge.
        :param int max_structures: Maximum number of resonance forms.
        """
        self.mol = mol
        self.kekule_all = kekule_all
        self.allow_incomplete_octets = allow_incomplete_octets
        self.unconstrained_cations = unconstrained_cations
        self.unconstrained_anions = unconstrained_anions
        self.allow_charge_separation = allow_charge_separation
        self.max_structures = max_structures
        self.mesomers = None

    @property
    def nmesomers(self):
        """ number of mesomers, i.e., resonated structures """
        if not hasattr(self, '_n'):
            self._n = self.get_num_mesomers()
        return self._n

    def get_num_mesomers(self):
        if self.mesomers is None:
            self.get_mesomers()
        return len(self.mesomers)

    def get_mesomers(self):
        """Enumerate all possible resonance forms and return them as a list.

        :return: A list of all possible resonance forms of the molecule.
        :rtype: list of rdkit.Chem.rdchem.Mol
        """
        flags = 0
        if self.kekule_all:
            flags = flags | Chem.KEKULE_ALL
        if self.allow_incomplete_octets:
            flags = flags | Chem.ALLOW_INCOMPLETE_OCTETS
        if self.allow_charge_separation:
            flags = flags | Chem.ALLOW_CHARGE_SEPARATION
        if self.unconstrained_anions:
            flags = flags | Chem.UNCONSTRAINED_ANIONS
        if self.unconstrained_cations:
            flags = flags | Chem.UNCONSTRAINED_CATIONS
        results = []
        for result in Chem.ResonanceMolSupplier(self.mol, flags=flags, \
                              maxStructs=self.max_structures):
            # This seems necessary? ResonanceMolSupplier only does a partial sanitization
            Chem.SanitizeMol(result)
            results.append(result)
        self.mesomers = results
        # Potentially interesting: getNumConjGrps(), getBondConjGrpIdx() and getAtomConjGrpIdx()


def enumerate_resonance_smiles(smiles, isomeric=True):
    """Return a set of resonance forms as SMILES strings, given a SMILES string.

    :param smiles: A SMILES string.
    :returns: A set containing SMILES strings for every possible resonance form.
    :rtype: set of strings.
    """
    mol = Chem.MolFromSmiles(smiles)
    #Chem.SanitizeMol(mol)  # MolFromSmiles does Sanitize by default
    obj = ResonanceEnumerator(mol)
    obj.get_mesomers()
    return {Chem.MolToSmiles(m, isomericSmiles=isomeric) for m in obj.mesomers}


