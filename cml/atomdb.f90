
MODULE atomdb
  IMPLICIT NONE

!  PUBLIC :: radii, s2z, z2s, zstar

! INTERFACE ASSIGNMENT(=)
!   MODULE PROCEDURE copy_ions
! END INTERFACE


  CONTAINS

    character(len=2) function z2s(zi) 
 
        integer,intent(in) :: zi
        integer i

        integer, parameter :: namax = 7
        integer, dimension(namax), parameter :: Zs = (/1, 6, 7, 8, 9, 16, 17/)
        character(len=2), dimension(namax), parameter :: symbs = &
                                     (/'H ','C ','N ','O ','F ','S ', 'Cl'/)

        do i = 1, namax
            if (Zs(i) .eq. zi) then
                z2s = symbs(i)
            end if
        end do

    end function z2s

    real(8) function radii(z,opt)
        ! get the vdw radii of elements
        integer :: i
        integer,intent(in) :: z
        character(10),intent(in) :: opt
        integer, parameter :: namax = 7
        integer, dimension(namax), parameter :: Zs = (/1, 6, 7, 8, 9, 16, 17/)
        real(8), dimension(namax), parameter :: rsvdw = &
                                     (/1.20, 1.70, 1.55, 1.52, 1.47, 1.80, 1.75/) ! Angstrom
        real(8), dimension(namax), parameter :: rscov = &
                                     (/0.37, 0.77, 0.75, 0.73, 0.71, 1.02, 0.99/) ! Angstrom
        real(8), parameter :: a2b = 1.88972596
        real(8) :: rs(namax)

        if (trim(opt) .eq. 'vdw') then
            rs = rsvdw
        elseif (trim(opt) .eq. 'cov') then
            rs = rscov
        else
            stop "NO such option for radii"
        end if

        do i = 1, namax
            if (Zs(i) .eq. z) then
                radii = rs(i)*a2b
            end if
        end do

    end function radii

    integer function s2z(symb) ! nuclei charge value

        character(len=*), intent(in) :: symb
        integer :: ind, j

        integer, parameter :: namax = 7
        integer, dimension(namax), parameter :: Zs_all = (/1, 6, 7, 8, 9, 16, 17/)
        character(len=2), dimension(namax), parameter :: symbs_all = &
                                     (/'H ','C ','N ','O ','F ','S ', 'Cl'/)
        double precision, dimension(namax), parameter :: radius = &
                                     (/0.37, 0.77, 0.75, 0.73, 0.71, 1.02, 0.99/)
        ind = 0
        do j = 1, namax
            if (trim(symb) == trim(symbs_all(j))) then
                ind = j
            end if
        enddo

        if (ind .eq. 0) then
            print*, 'symb = ', "'",symb,"'", ' not supported'
            stop "please add more data in `db module (in FUNCTION `s2z)"
        else
            s2z = Zs_all(ind)
        end if

    end function s2z


    real(8) function zstar(z) !, zs2)

        !integer :: nz1
        !integer, dimension(:), intent(in) :: zs1
        integer, intent(in) :: z
        !real*8, dimension(size(zs1,1)), intent(out) :: zs2

        integer,parameter :: nz =  20
        integer,parameter,dimension(nz) :: zs = (/1,   2, &
                          3,  4,  5,  6,  7,  8,  9, 10, &
                         11, 12, 13, 14, 15, 16, 17, 18, &
                         35,  53/)
        ! data retrieved from https://en.wikipedia.org/wiki/Effective_nuclear_charge
        real*8,parameter,dimension(nz) :: zs_eff = (/1.0, 1.688, &
                1.279, 1.912, 2.421, 3.136, 3.834, 4.453, 5.100, 5.758, &
                2.507, 3.308, 4.066, 4.285, 4.886, 5.482, 6.116, 6.764, &
                9.028, 11.612/)

        integer j

        do j = 1,nz
            if (z .eq. zs(j)) then
                zstar = zs_eff(j)
            endif
        enddo

    end function zstar

end module

