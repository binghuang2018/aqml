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
C 0.8105 1.2893 -0.145
C 2.0781 0.9621 -0.4101
H 0.4518 2.3035 -0.2906
H 2.7822 1.7019 -0.7778
H 0.1064 0.5495 0.2227
H 2.4368 -0.0521 -0.2645
*

