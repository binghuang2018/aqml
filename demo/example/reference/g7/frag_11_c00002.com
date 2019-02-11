%pal nprocs 2 end
%maxcore 1000
! b3lyp TIGHTSCF
! cc-pvdz def2/J RIJCOSX
! Opt

%geom
maxiter 60
TolE 1e-4
TolRMSG 2e-3
TolMaxG 3e-3
TolRMSD 2e-2
TolMaxD 3e-2
end

*xyz 0 1
C 0.8245 1.2316 -0.142
C 2.1142 1.0499 -0.4556
C 0.2268 -1.0549 0.5909
C -0.1051 0.2302 0.3712
C -1.5027 0.7253 0.6818
C -2.6053 0.1195 -0.1725
O -2.469 -0.7648 -1.0115
H 0.4204 2.2312 -0.3021
H 2.7046 1.8755 -0.8417
H -0.5039 -1.7657 0.9678
H -1.7456 0.5036 1.728
H -1.5816 1.8121 0.5679
H -3.6105 0.5328 0.03
H 2.6213 0.0984 -0.348
H 1.2114 -1.4578 0.382
*

