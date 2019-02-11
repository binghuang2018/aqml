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
C -1.546 0.7312 0.6713
H -1.7339 0.4732 1.7158
H -1.6419 1.8112 0.5401
H -0.5364 0.4191 0.3951
H -2.2718 0.2212 0.0341
*

