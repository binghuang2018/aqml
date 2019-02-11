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
O -2.4915 -0.8074 -0.9326
C -1.5507 0.7229 0.6627
C -2.6627 0.0945 -0.1191
H -1.7387 0.5805 1.7294
H -1.5127 1.7907 0.4352
H -3.6709 0.4954 0.0797
H -0.5971 0.2603 0.3966
*

