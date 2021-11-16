=========================
Molecular representations
=========================

This section details some of the main representations used in `aqml`.

Many-body representation in a nutshell
======================================
MBR is perhaps the most popular type of representation, among the many variants in literature.

BAML
====
[Todo]

atomic Spectrum of Axilrod-Teller-Muto potential (aSLATM)
=========================================================
Several versions of SLATM are available. The most concise \& vectorized version is obtained through `numpy` \& `itertools`. Take as an example the generation of aSLATM for an atom with atomic index `ia0`.

The prerequisite is to generate all possible types of many-body terms, each being characterized by a tuple of Z's. I.e., the unique 1-body terms are:
```python
zsu1 = [ (zi,) for zi in np.unique(zs0) ]
```
where `zs0` is the list of nuclear charges of the atoms in the molecule. Similarly, for 2- \& 3-body terms, the associated types are
```python
import itertools as ilt
zsu2 = list(itl.combinations_with_replacement(zsu1, 2))
zsu3 = list(itl.combinations_with_replacement(zsu1, 3))
```
E.g., for a molecule made up of `C` \& `H`, 
```python
zsu1 = [ (1,), (6,) ]
zsu2 = [(1, 1), (1, 6), (6, 6)]
zsu3 = [(1, 1, 1), (1, 1, 6), (1, 6, 6), (6, 6, 6)]
```
For aSLATM, things are a little different: since we've already got the type of the first (or central) atom, i.e., `zs[ia0]` (denoted as `Z_I`), we need only to know the the types of at most 2 remaining atoms, therefore, for the exemplified molecule `cxhy`, 2/3-body terms (`mbs2`/`mbs3`) are
```python
mbs2 = [ (1,), (6,) ]
mbs3 = [(1, 1), (1, 6), (6, 6)]
```
And the complete set of many-body terms associated with atom `ia0` are
```python
mbs2 = [ (Z_I, 1), (Z_I, 6) ]
mbs3 = [(Z_I, 1, 1), (Z_I, 1, 6), (Z_I, 6, 6)]
```
Apparently, the type `(Z_I, 1,6)` is equiv. to `(Z_J, 6,1)`.



### 2-body terms
```python
x = np.linspace(r0, rcut, ngx); 
xJ  = (zs0[ia0]*zs0[jas]) * (const/wdr)*np.exp(-0.5*( x[:, None]-ds_ij[None, :])**2/wdr**2) # of size (ngx, nb)
```
then sum up the same atomic pair indicated by (Z_I, Z_J):
```
x1new = np.zeros((len(zsu1), ngx)) # ngx: number of grids along the x-axis
for jz, (zj,) in enumerate(zsu1):
    x1new[jz] += np.einsum('xi->x', xJ[:, zj==zs[jas]])
```
Note that for the case of `zs[ia0]=6`, the 2-body term `(6,1)` is present, while the type `(1,6)` is not.

### 3-body terms
For three-body terms, we use as the scaling factor the so-called Axilrod-Teller-Muto potential (resulting from a third-order perturbation correction to the attractive London dispersion interactions, i.e., instantaneous induced dipole-induced dipole):

<img src="https://wikimedia.org/api/rest_v1/media/math/render/svg/2cdc6b041a7597e2ac209a26987f2123daeea921" class="mwe-math-fallback-image-inline" aria-hidden="true" style="vertical-align: -3.171ex; width:36.723ex; height:7.509ex;" alt="
V_{ijk}= E_{0} \left[
\frac{1 + 3 \cos\gamma_{i} \cos\gamma_{j} \cos\gamma_{k}}{\left( r_{ij} r_{jk} r_{ik} \right)^3}
\right]
"> 

where $r_{ij}$ is the distance between atoms `i` and `j`, and `\gamma_{i}` is the angle between the vectors `r_{ij}` and `r_{ik}`. 
The coefficient `E_{0}` is positive and of the order `V\alpha^{3}`, where `V` is 
the ionization energy and `\alpha` is the mean atomic polarizability; the exact value of `E_{0}` depends on the magnitudes of the dipole matrix elements and on the energies of the `p` orbitals ([from wikipedia] (https://en.wikipedia.org/wiki/Axilrod%E2%80%93Teller_potential)).

To vectorize the 3-body feature generation, we need to define a few arrays first:
```python
iats = np.arange(na) # na: total number of atoms of this molecule
atsr = iats[ np.logical_and(ds0[ia0]<=rcut, iats!=ia0) ]
atsp = np.array(list(itl.combinations(atsr, 2)), dtype=np.int32)
ias  =  np.array([ia0]*len(atsp), dtype=np.int32)
jas = atsp[:,0]; kas = atsp[:,1]
```
`(i|j|k)as` in particular.


```python
xa = np.linspace(-wa/2.0, np.pi+wa/2., nga)
znew = np.zeros((nab,nab, ngx,ngx,nga))
const = 1./(np.sqrt(2*np.pi))
xJ  = (const/wdr) * np.exp(-0.5*( x[:, None]-ds_ij[None, :])**2/wdr**2) # of size (ngx, nb)
yK  = (const/wdr) * np.exp(-0.5*( x[:, None]-ds_ik[None, :])**2/wdr**2) # of size (ngx, nb)
aJK = (const/wda) * np.exp(-0.5*(xa[:, None]-a_ijk[None, :])**2/wda**2) # size: (nga, nb)
x3 =  * np.einsum('xj,yj,aj->xyaj', xJ, yK, aJK) 
```
where,
```python
a_ijk = ang3[(ias, jas, kas)]
ca_ijk = cos3[(ias, jas, kas)]
ca_jik = cos3[(jas, ias, kas)]
ca_kij = cos3[(kas, ias, jas)]
ds_ij = rs0[(ias, jas)]
ds_ik = rs0[(ias, kas)]
ds_jk = rs0[(jas, kas)]
scale = 1.0 # in the original Axilrod–Teller potential, scale=3.0
prefactor = (zs0[ia0]*zs0[jas]*zs0[kas])
prefactor *= (1. + scale*ca_ijk*ca_jik*ca_kij)/(ds_ij*ds_jk*ds_ik)**3
```
At last, combine (sum up) the 3-body terms of the same type, as indicated by (Z_I, Z_J, Z_K),
```python
x3new = np.zeros((len(zsu2), ngx,ngx,nga))
for zj,zk in zsu2:
    ft = np.logical_or(np.logical_and(zs0[jas]==zj,zs0[kas]==zk), np.logical_and(zs0[jas]==zk,zs0[kas]==zj))
    x3new[ zsu2.index((zj,zk)) ] = np.einsum('xyaj->xya', x3[:,:,:, ft])
```
