
memory,13000,M

!file,1,01.int !save integrals file
!file,2,01.wfu !save wavefunctions to file, as well as geom & grad


geomtype=xyz
geometry = {
17

C 0.7807 1.297 -0.1388
C 2.1079 0.9544 -0.4163
C 2.5468 -0.3578 -0.2224
C 1.6475 -1.324 0.2419
C 0.3225 -0.9785 0.5174
C -0.1266 0.3399 0.3376
C -1.546 0.7312 0.6713
C -2.6526 0.0738 -0.1367
O -2.5125 -0.8135 -0.942
H 0.4444 2.3257 -0.2969
H 2.7988 1.7174 -0.7824
H 3.5829 -0.63 -0.4375
H 1.9841 -2.3521 0.3967
H -0.3721 -1.7404 0.8772
H -1.7869 0.5038 1.7305
H -1.6877 1.8237 0.5793
H -3.6707 0.4904 0.0972
}


basis={default=vtz

set,jkfit,context=jkfit
default=vqz

set,mp2fit,context=mp2fit
default=vqz

set,ri,context=jkfit
default=vqz
}


explicit,ri_basis=ri,df_basis=mp2fit,df_basis_exch=jkfit


hf
df-ccsd(t)-f12
e_f12a=energy(1)
e_f12b=energy(2)


---