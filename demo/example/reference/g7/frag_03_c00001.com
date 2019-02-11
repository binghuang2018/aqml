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
O -2.5114 -0.8207 -0.9485
C -2.6536 0.08 -0.131
H -3.641 0.506 0.1082
H -1.8026 0.5185 0.4143
*

