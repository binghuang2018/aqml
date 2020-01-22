
Methods used:

- Geometry optimized at b3lyp/cc-pvdz by ORCA4, with keywords

```bash
! b3lyp TIGHTSCF cc-pvdz def2/J RIJCOSX Opt

%geom
maxiter 60
TolE 1e-4
TolRMSG 2e-3
TolMaxG 3e-3
TolRMSD 2e-2
TolMaxD 3e-2
end
```

- Based on the optimized geometry, polar and nmr calculation follow, using Gaussian 09

For NMR, the following keywords are used for calculation
```bash
   # nmr iop(3/76=1000002000, 3/77=0720008000, 3/78=0810010000) bv5lyp/cc-pVDZ
```

For polarizability,
```bash
   # polar iop(3/76=1000002000, 3/77=0720008000, 3/78=0810010000) bv5lyp/cc-pVDZ
```


