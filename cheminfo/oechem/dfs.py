#!/usr/bin/env python

"""Enumerate subgraphs and canonical SMARTS from an OEChem molecule

For an explanation of the algorithm see
  http://dalkescientific.com/writings/diary/archive/2011/01/10/subgraph_enumeration.html

"""
from itertools import chain, product
import re

from openeye.oechem import *

#######

class Subgraph(object):
    def __init__(self, atoms, bonds):
        self.atoms = atoms
        self.bonds = bonds

def find_extensions(considered, new_atoms):
    # Find the extensions from the atoms in 'new_atoms'.
    # There are two types of extensions:
    #
    #  1. an "internal extension" is a bond which is not in 'considered'
    # which links two atoms in 'new_atoms'.
    #
    #  2. an "external extension" is a (bond, to_atom) pair where the
    # bond is not in 'considered' and it connects one of the atoms in
    # 'new_atoms' to the atom 'to_atom'.
    #
    # Return the internal extensions as a list of bonds and
    # return the external extensions as a list of (bond, to_atom) 2-ples.
    internal_extensions = set()
    external_extensions = []
    for atom in new_atoms:
        for outgoing_bond in atom.GetBonds():
            if outgoing_bond in considered:
                continue
            other_atom = outgoing_bond.GetNbr(atom)
            if other_atom in new_atoms:
                # This this is an unconsidered bond going to
                # another atom in the same subgraph. This will
                # come up twice, so prevent duplicates.
                internal_extensions.add(outgoing_bond)
            else:
                external_extensions.append( (outgoing_bond, other_atom) )

    return list(internal_extensions), external_extensions



def all_combinations(container):
    "Generate all 2**len(container) combinations of elements in the container"
    # This just sets up the underlying call
    return _all_combinations(container, len(container)-1, 0)

def _all_combinations(container, last, i):
    # This does the hard work recursively
    if i == last:
        yield []
        yield [container[i]]
    else:
        for subcombinations in _all_combinations(container, last, i+1):
            yield subcombinations
            yield [container[i]] + subcombinations

## I had an optimization that if limit >= len(external_extensions) then
## use this instead of the limited_external_combinations, but my timings
## suggest the result was slower, so I went for the simpler code.

#def all_external_combinations(container):
#    "Generate all 2**len(container) combinations of external extensions"
#    for external_combination in all_combinations(container):
#        # For each combination yield 2-ples containing
#        #   {the set of atoms in the combination}, [list of external extensions]
#        yield set((ext[1] for ext in external_combination)), external_combination

def limited_external_combinations(container, limit):
    "Generate all 2**len(container) combinations which do not have more than 'limit' atoms"
    return _limited_combinations(container, len(container)-1, 0, limit)

def _limited_combinations(container, last, i, limit):
    # Keep track of the set of current atoms as well as the list of extensions.
    # (An external extension doesn't always add an atom. Think of
    #   C1CC1 where the first "CC" adds two edges, both to the same atom.)
    if i == last:
        yield set(), []
        if limit >= 1:
            ext = container[i]
            yield set([ext[1]]), [ext]
    else:
        for subatoms, subcombinations in _limited_combinations(container, last, i+1, limit):
            assert len(subatoms) <= limit
            yield subatoms, subcombinations
            new_subatoms = subatoms.copy()
            ext = container[i]
            new_subatoms.add(ext[1])
            if len(new_subatoms) <= limit:
                yield new_subatoms, [ext] + subcombinations


def all_subgraph_extensions(subgraph, internal_extensions, external_extensions, k):
    # Generate the set of all subgraphs which can extend the input subgraph and
    # which have no more than 'k' atoms.
    assert len(subgraph.atoms) <= k

    if not external_extensions:
        # Only internal extensions (test case: "C1C2CCC2C1")
        it = all_combinations(internal_extensions)
        it.next()
        for internal_ext in it:
            # Make the new subgraphs
            bonds = frozenset(chain(subgraph.bonds, internal_ext))
            yield set(), Subgraph(subgraph.atoms, bonds)
        return

    limit = k - len(subgraph.atoms)

    if not internal_extensions:
        # Only external extensions
        # If we're at the limit then it's not possible to extend
        if limit == 0:
            return
        # We can extend by at least one atom.
        it = limited_external_combinations(external_extensions, limit)
        it.next()
        for new_atoms, external_ext in it:
            # Make the new subgraphs
            atoms = frozenset(chain(subgraph.atoms, new_atoms))
            bonds = frozenset(chain(subgraph.bonds, (ext[0] for ext in external_ext)))
            yield new_atoms, Subgraph(atoms, bonds)
        return

    # Mixture of internal and external (test case: "C1C2CCC2C1")
    external_it = limited_external_combinations(external_extensions, limit)
    it = product(all_combinations(internal_extensions), external_it)
    it.next()
    for (internal_ext, external) in it:
        # Make the new subgraphs
        new_atoms = external[0]
        atoms = frozenset(chain(subgraph.atoms, new_atoms))
        bonds = frozenset(chain(subgraph.bonds, internal_ext,
                                (ext[0] for ext in external[1])))
        yield new_atoms, Subgraph(atoms, bonds)
    return

def generate_subgraphs(mol, k=5):
    if k < 0:
        raise ValueError("k must be non-negative")

    # If you want nothing, you'll get nothing
    if k < 1:
        return

    # Generate all the subgraphs of size 1
    for atom in mol.GetAtoms():
        yield Subgraph(frozenset([atom]), frozenset())

    # If that's all you want then that's all you'll get
    if k == 1:
        return

    # Generate the intial seeds. Seed_i starts with bond_i and knows
    # that bond_0 .. bond_i will not need to be considered during any
    # growth of of the seed.
    # For each seed I also keep track of the possible ways to extend the seed.
    seeds = []
    considered = set()
    for bond in mol.GetBonds():
        considered.add(bond)
        subgraph = Subgraph(frozenset([bond.GetBgn(), bond.GetEnd()]),
                            frozenset([bond]))
        yield subgraph
        internal_extensions, external_extensions = find_extensions(considered,
                                                                   subgraph.atoms)
        # If it can't be extended then there's no reason to keep track of it
        if internal_extensions or external_extensions:
            seeds.append( (considered.copy(), subgraph,
                           internal_extensions, external_extensions) )

    # No need to search any further
    if k == 2:
        return

    # seeds = [(considered, subgraph, internal, external), ...]
    while seeds:
        considered, subgraph, internal_extensions, external_extensions = seeds.pop()

        # I'm going to handle all 2**n-1 ways to expand using these
        # sets of bonds, so there's no need to consider them during
        # any of the future expansions.
        new_considered = considered.copy()
        new_considered.update(internal_extensions)
        new_considered.update(ext[0] for ext in external_extensions)

        for new_atoms, new_subgraph in all_subgraph_extensions(
            subgraph, internal_extensions, external_extensions, k):

            assert len(new_subgraph.atoms) <= k
            yield new_subgraph

            # If no new atoms were added, and I've already examined
            # all of the ways to expand from the old atoms, then
            # there's no other way to expand and I'm done.
            if not new_atoms:
                continue

            # Start from the new atoms to find possible extensions
            # for the next iteration.
            new_internal, new_external = find_extensions(new_considered, new_atoms)
            if new_internal or new_external:
                seeds.append( (new_considered, new_subgraph, new_internal, new_external) )

#######

# I'm going to convert a substructure into a new molecule,
# canonicalize the new molecule, and use the canonical SMILES to
# create a canonical SMARTS for the original substructure.

# I'll preserve the aromaticity information by encoding (atomic
# number, is_aromatic) as (atomic number, isotope number) where the
# isotope == 1 for not aromatic and 2 for aromatic.  The corresponding
# SMILES will be either the form "[1*]" or "[2*]" where "*" is a valid
# atomic symbol.

# Given the SMILES I can reverse that with a string transformation.
# To make that easier I'll enumerate all possible reverse transformations.

organic_subset = set([5,6,7,8,9,15,16,17,35,53])

decode_mapping = {"[1*]": "*",  # Encode these manually
                  "[2*]": "*"}
for atomno in range(1, OEElemNo_MAXELEM):
    symbol = OEGetAtomicSymbol(atomno)
    if atomno in organic_subset:
        atom_smarts = symbol
    else:
        atom_smarts = "[%s]" % symbol
    decode_mapping["[1%s]" % symbol] = atom_smarts
    decode_mapping["[2%s]" % symbol] = atom_smarts.lower()

# Helper function to transform the SMILES back to the correct SMARTS
_atom_name_pat = re.compile(r"\[[12]\w+\]")
def replace_atom_terms(m, decode_mapping = decode_mapping):
    return decode_mapping[m.group()]


def generate_canonical_smarts(mol, k=5):
    "generate all canonial SMARTS in a molecule up to size 'k'"
    for seed in generate_subgraphs(mol, k):
        # Make a new molecule based on the substructure
        new_mol = OEGraphMol()
        new_atoms = {}
        for atom in seed.atoms:
            new_atom = new_mol.NewAtom(atom.GetAtomicNum())
            # 1 if not aromatic, 2 if aromatic
            new_atom.SetIsotope(atom.IsAromatic()+1)
            new_atoms[atom.GetIdx()] = new_atom

        for bond in seed.bonds:
            new_bond = new_mol.NewBond(new_atoms[bond.GetBgnIdx()],
                                       new_atoms[bond.GetEndIdx()],
                                       bond.GetOrder())
            # GetOrder() sets the Kekule form, but I want the aromatic
            # form.  If I don't do this I end up with terms like "c=c"
            # in my results. Setting it to a single bond works because
            # the output will always have "" as the bond (as in "CC",
            # "cC" or "cc"). The only way to get an incorrect result
            # is if the structure should be "c-c", which is rare. But
            # since the "" connection means "single or aromatic", then
            # the SMARTS won't miss it, so we're okay. As long as I'm
            # consistent.
            if bond.IsAromatic():
                new_bond.SetOrder(1)

        smiles = OECreateIsoSmiString(new_mol)
        smarts = _atom_name_pat.sub(replace_atom_terms, smiles)
        #print smiles, "->", smarts
        yield smarts

## A simple example program

if __name__ == "__main__":
    import sys
    if len(sys.argv) == 1:
        smiles = "BrC1=C2C3=C4C(C=C5C=NC(S3)=C45)=C2SC=C1"
#"CCC(C)C(C(=O)NCC(=O)NCC(=O)NC(Cc1ccccc1)C(=O)NC(C(C)CC)C(=O)NC(CCCCN)C(=O)NC(C(C)C)C(=O)NC(CCCNC(=N)N)C(=O)NC(CCC(=O)N)C(=O)NC(Cc2ccc(cc2)O)C(=O)NC(CC(=O)O)C(=O)NC(CCC(=O)N)C(=O)NC(C(C)CC)C(=O)NC(CC(C)C)C(=O)NC(C(C)CC)C(=O)NC(CCC(=O)O)C(=O)NC(C(C)CC)C(=O)NC(CS)C(=O)NCC(=O)NC(Cc3cnc[nH]3)C(=O)NC(CCCCN)C(=O)NC(C)C(=O)NC(C(C)CC)C(=O)NCC(=O)NC(C(C)O)C(=O)NC(C(C)C)C(=O)NC(CC(C)C)C(=O)NC(C(C)C)C(=O)NCC(=O)N4CCCC4C(=O)NC(C(C)O)C(=O)N5CCCC5C(=O)NC(C(C)C)C(=O)NC(CC(=O)N)C(=O)NC(C(C)CC)C(=O)NC(C(C)CC)C(=O)NCC(=O)NC(CCCNC(=N)N)C(=O)NC(CC(=O)N)C(=O)NC(CC(C)C)C(=O)NC(CC(C)C)C(=O)NC(C(C)O)C(=O)NC(CCC(=O)N)C(=O)NC(C(C)CC)C(=O)NCC(=O)NC(CS)C(=O)NC(C(C)O)C(=O)NC(CC(C)C)C(=O)NC(CC(=O)N)C(=O)NC(Cc6ccccc6)C(=O)O)NC(=O)CNC(=O)CNC(=O)C(C(C)CC)NC(=O)C(CCSC)NC(=O)C(CCCCN)NC(=O)C7CCCN7C(=O)C(CCCCN)NC(=O)C(Cc8c[nH]c9c8cccc9)NC(=O)C(CCCN=C(N)N)NC(=O)CNC(=O)C1CCCN1C(=O)C(CC(C)C)NC(=O)C(CO)NC(=O)C(CCSC)NC(=O)C(CCC(=O)O)NC(=O)C(CCC(=O)O)NC(=O)C(CC(C)C)NC(=O)C(C(C)C)NC(=O)C(C(C)O)NC(=O)C(CC(=O)O)NC(=O)C(CC(=O)O)NC(=O)C(C)NC(=O)CNC(=O)C(C(C)O)NC(=O)C(CC(=O)O)NC(=O)C(CC(C)C)NC(=O)C(CC(C)C)NC(=O)C(C)NC(=O)C(CCC(=O)O)NC(=O)C(CCCCN)NC(=O)C(CC(C)C)NC(=O)C(CCC(=O)N)NC(=O)CNC(=O)CNC(=O)C(C(C)CC)NC(=O)C(CCCCN)NC(=O)C(C(C)CC)NC(=O)C(C(C)O)NC(=O)C(C(C)C)NC(=O)C(CC(C)C)NC(=O)C1CCCN1C(=O)C(CCCNC(=N)N)NC(=O)C(CCC(=O)N)NC(=O)C(Cc1c[nH]c2c1cccc2)NC(=O)C(CC(C)C)NC(=O)C(C(C)O)NC(=O)C(C(C)CC)NC(=O)C(CCC(=O)N)NC(=O)C1CCCN1"
        k = 7
    elif len(sys.argv) == 2:
        smiles = sys.argv[1]
        k = 5
    elif len(sys.argv) == 3:
        smiles = sys.argv[1]
        k = int(sys.argv[2])
    else:
        raise SystemExit("""Usage: dfa_subgraph_enumeration.py <smiles> [<k>]
List all subgraphs of the given SMILES up to size k atoms (default k=5)
""")
    mol = OEGraphMol()
    assert OEParseSmiles(mol, smiles), "Could not parse input SMILES"
    OESuppressHydrogens(mol, False, False, False)
    for smarts in generate_canonical_smarts(mol, k):
        print smarts
