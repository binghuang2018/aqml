

subroutine get_overlap(vs1, vs2, sigmas, nv, ifactor, ov)

    ! 
    ! calculate the overlap between two 
    ! gaussian distribution 
    !
    ! parameters
    ! =============================================
    ! 
    ! 

    implicit none

    integer, intent(in) :: nv
    logical, intent(in) :: ifactor
    real*8, intent(in) :: vs1(nv), vs2(nv), sigmas(nv)
    real*8, intent(out) :: ov
    integer :: i
    real*8, parameter :: pi = 3.1415926
    real*8 :: dv, factor

    ov = 1.0
    do i = 1,nv
        dv = vs1(i) - vs2(i)
        factor = 1.0
        if ( ifactor ) then 
            factor = 2.0 * sigmas(i) * sqrt(pi)
        endif
        ov = ov * exp( -dv*dv/(4.0*sigmas(i)*sigmas(i)) )/factor
    enddo

end subroutine get_overlap


subroutine get_overlapN(vss, n, nv, sigmas, ifactor, ov)

    ! 
    ! calculate the overlap between N
    ! gaussian distribution 
    !
    ! parameters
    ! =============================================
    ! nv        :  number of values assigned to 
    !              characterize each atom
    ! n         :  number of atoms
    ! vss(n,nv) :  ValueS for input atomS (state of atoms)
    !              each row characterizes an atom

    implicit none

    integer, intent(in) :: nv, n
    logical, intent(in) :: ifactor
    real*8, intent(in) :: vss(n,nv), sigmas(nv)
    real*8, intent(out) :: ov
    integer :: i
    real*8, parameter :: pi = 3.1415926
    real*8 :: const
    real*8 :: sigma, factor, dv, vsi(n), svsi

    const = 1.0/sqrt(dble(nv))
    ov = 1.0
    do i = 1,nv
        vsi = vss(:,i)
        svsi = sum(vsi)
        dv = sum(vsi*vsi) - svsi*svsi/dble(n)
        factor = 1.0
        sigma = sigmas(i)
        if ( ifactor ) then 
            factor = const *(sigma**2 * sqrt(pi) )**(0.5-0.5*dble(n))
        endif
        ov = ov * exp( -dv/(2.0*sigma*sigma) )/factor
    enddo
end subroutine get_overlapN


subroutine gdist(na1,na2,vcps1,vcps2,pls1,pls2, &
                 sigmas,ifactor,dcut,nv,power,d0,dist )

    ! 
    ! calculate the distancebetween two molecules
    !
    ! parameters
    ! =============================================
    ! dcut :  cutoff distance (i.e., Path Length), defaulted to 3
    !

    implicit none

    integer, intent(in) :: na1,na2,power,nv
    real*8, intent(in) :: vcps1(nv,na1),vcps2(nv,na2), &
                           pls1(na1,na1), pls2(na2,na2), &
                           sigmas(nv+1), dcut,d0
    logical, intent(in) :: ifactor
    real*8, intent(out) :: dist
    real*8 :: ov,ov2,l1,l2
    integer :: i,j,k,l
    real*8 :: vs1(nv),vs2(nv),vs3(nv),vs4(nv), &
              ds2(2,1), vss1(2,nv), vss2(4,nv), &
              sigmas_a(nv), sigma_r(1)

    sigmas_a = sigmas(1:nv)
    sigma_r(1) = sigmas(nv+1)

    dist = 0.
    do i = 1, na1
        vs1 = vcps1(:,i)
        vss1(1,:) = vs1
        vss1(2,:) = vs1
        call get_overlapN(vss1, 2, nv, sigmas_a, ifactor, ov)
        dist = dist + ov 
        do j = i+1, na1
            vs2 = vcps1(:,j)
            vss1(2,:) = vs2
            call get_overlapN(vss1, 2, nv, sigmas_a, ifactor, ov)
            dist = dist + ov*2.0
        enddo
    enddo

    do i = 1, na2
        vs1 = vcps2(:,i)
        vss1(1,:) = vs1
        vss1(2,:) = vs1
        call get_overlapN(vss1, 2, nv, sigmas_a, ifactor, ov)
        dist = dist + ov 
        do j = i+1, na2
            vs2 = vcps2(:,j)
            vss1(2,:) = vs2
            call get_overlapN(vss1, 2, nv, sigmas_a, ifactor, ov)
            dist = dist + ov*2
        enddo
    enddo

    do i = 1, na1
        do j = 1, na2
            vss1(1,:) = vcps1(:,i)
            vss1(2,:) = vcps2(:,j)
            call get_overlapN(vss1, 2, nv, sigmas_a, ifactor, ov)
            dist = dist - 2.0*ov
        enddo
    enddo


    ! 2-body terms 

    do i = 1, na1
        do j = i+1, na1
            l1 = pls1(i,j)
            if (l1 .gt. 0 .and. l1 .le. dcut) then
                vs1 = vcps1(:,i)
                vs2 = vcps1(:,j)
                do k = 1,na1
                    do l = k+1,na1
                        l2 = pls1(k,l) 
                        if (l2 .gt. 0 .and. l2 .le. dcut) then 
                            vs3 = vcps1(:,k)
                            vs4 = vcps1(:,l)
                            vss2(1,:) = vs1
                            vss2(2,:) = vs2
                            vss2(3,:) = vs3
                            vss2(4,:) = vs4
                            call get_overlapN(vss2, 4, nv, sigmas_a, ifactor, ov)
                            ds2(1,:) = l1
                            ds2(2,:) = l2
                            call get_overlapN(ds2, 2, 1, sigma_r, ifactor, ov2)
                            dist = dist + ov*ov2/((l1+d0)*(l2+d0))**power
                        endif
                    enddo
                enddo
            endif
        enddo
    enddo

    do i = 1, na2
        do j = i+1, na2
            l1 = pls2(i,j)
            if (l1 .gt. 0 .and. l1 .le. dcut) then
                vs1 = vcps2(:,i)
                vs2 = vcps2(:,j)
                do k = 1,na2
                    do l = k+1,na2
                        l2 = pls2(k,l) 
                        if (l2 .gt. 0 .and. l2 .le. dcut) then 
                            vs3 = vcps2(:,k)
                            vs4 = vcps2(:,l)
                            vss2(1,:) = vs1
                            vss2(2,:) = vs2
                            vss2(3,:) = vs3
                            vss2(4,:) = vs4
                            call get_overlapN(vss2, 4, nv, sigmas_a, ifactor, ov)
                            ds2(1,:) = l1
                            ds2(2,:) = l2
                            call get_overlapN(ds2, 2, 1, sigma_r, ifactor, ov2)
                            dist = dist + ov*ov2/((l1+d0)*(l2+d0))**power
                        endif
                    enddo
                enddo
            endif
        enddo
    enddo

    do i = 1, na1
        do j = i+1, na1
            l1 = pls1(i,j)
            if (l1 .gt. 0 .and. l1 .le. dcut) then
                vs1 = vcps1(:,i)
                vs2 = vcps1(:,j)
                do k = 1,na2
                    do l = k+1,na2
                        l2 = pls2(k,l) 
                        if (l2 .gt. 0 .and. l2 .le. dcut) then 
                            vs3 = vcps2(:,k)
                            vs4 = vcps2(:,l)
                            vss2(1,:) = vs1
                            vss2(2,:) = vs2
                            vss2(3,:) = vs3
                            vss2(4,:) = vs4
                            call get_overlapN(vss2, 4, nv, sigmas_a, ifactor, ov)
                            ds2(1,:) = l1
                            ds2(2,:) = l2
                            call get_overlapN(ds2, 2, 1, sigma_r, ifactor, ov2)
                            dist = dist - 2.*ov*ov2/((l1+d0)*(l2+d0))**power
                        endif
                    enddo
                enddo
            endif
        enddo
    enddo

    if (dist .le. 0 .and. dist .gt. -1.e-6) dist = 0.
    dist = sqrt(dist)

end subroutine gdist


subroutine gdist_matrix(nas,vcps,pls,sigmas, ifactor, &
                        dcut,power,n1,n2, d0, &
                        dm1,dm2,nm,namax,nat,nv)
    ! n1 : training set size
    ! n2 : test set size (default case: n2 = N - n1)
    ! dm1: distance matrix for training species

    implicit none 

    integer, intent(in) :: nv,nm,namax,nat,power, &
                           n1,n2
    real*8, intent(in) :: nas(nm)
    real*8, intent(in) :: vcps(nv,nat), pls(namax,namax,nm), &
                          sigmas(nv+1), d0,dcut
    logical, intent(in) :: ifactor
    real*8, intent(out) :: dm1(n1,n1), dm2(n1,n2)

    integer :: i,j,j0, ia1,ia2,ja1,ja2, &
               ias2(nm),ias1(nm), na1,na2
    real*8 :: dij,pls1(namax,namax),pls2(namax,namax)

    do i = 1,nm
        ias2(i) = sum(nas(:i))
    enddo
    ias1(1) = 1
    ias1(2:nm) = ias2(1:nm-1)

    dm1(:,:) = 0.

!$omp parallel do private(i,j,na1,na2,pls1,pls2,dij)
    do i = 1,n1
        do j = i+1,n1
            na1 = nas(i)
            na2 = nas(j)
            !vcps1 = vcps(:,ias1(i):ias2(i))
            !vcps2 = vcps(:,ias1(j):ias2(j))
            pls1 = pls(:,:,i)
            pls2 = pls(:,:,j)
            call gdist(na1,na2, &
                       vcps(:,ias1(i):ias2(i)), &
                       vcps(:,ias1(j):ias2(j)), &
                       pls1,pls2, &
                       sigmas,ifactor,dcut,nv,power,d0,dij)
            dm1(i,j) = dij
            dm1(j,i) = dij
        enddo
    enddo
!$omp end parallel do

!$omp parallel do private(i,j,j0,na1,na2,pls1,pls2,dij)
    do i = 1,n1 
        do j0 = 1,n2 
            j = nm-n2+j0
            na1 = nas(i)
            na2 = nas(j)
            !vcps1 = vcps(:,ias1(i):ias2(i))
            !vcps2 = vcps(:,ias1(j):ias2(j))
            pls1 = pls(:,:,i)
            pls2 = pls(:,:,j)
            call gdist(na1,na2, &
                 vcps(:,ias1(i):ias2(i)), &
                 vcps(:,ias1(j):ias2(j)), &
                 pls1,pls2, &
                 sigmas,ifactor,dcut,nv,power,d0,dij)
            dm2(i,j0) = dij
        enddo
    enddo
!$omp end parallel do

end subroutine gdist_matrix


subroutine ldist(na1,na2,vcps1,vcps2,pls1,pls2, &
                 sigmas,ifactor,dcut,nv,power,d0, &
                 debug, dist )

    ! 
    ! calculate the distance between two molecules
    ! using local VCP representation
    !
    ! parameters
    ! =============================================
    ! dcut :  cutoff distance (i.e., Path Length), defaulted to 3
    !

    implicit none

    integer, intent(in) :: na1,na2,power,nv
    real*8, intent(in) :: vcps1(nv,na1),vcps2(nv,na2), &
                           pls1(na1,na1), pls2(na2,na2), &
                           sigmas(nv+1), dcut,d0
    logical, intent(in) :: ifactor, debug
    real*8, intent(out) :: dist
    real*8 :: l1,l2,d1,d2, ov,ov1,ov2
    integer :: i,j,i2,i3,j2,j3,ia2,ia3,ja2,ja3, &
               icnt, nbmax, &
               inbs1((3**dcut-1)*2,na1), &
               inbs2((3**dcut-1)*2,na2)
    real*8 :: vs1(nv),vs2(nv),vs3(nv),vs4(nv), &
              ds2(2,1), vss1(2,nv), vss2(4,nv), &
              sigmas_a(nv), sigma_r(1), dt 

    sigmas_a = sigmas(1:nv)
    sigma_r(1) = sigmas(nv+1)

    ! get the idx of all neighboring atoms for each 
    ! atom in a molecule
    nbmax = 1
    inbs1(:,:) = 0
    do i = 1, na1
        icnt = 1
        do i2 = 1, na1
            l1 = pls1(i2,i)
            if (l1 .gt. 0 .and. l1 .le. dcut) then
                inbs1(icnt,i) = i2
!                write(*,*) ' -- ', i,i2
                icnt = icnt + 1
            endif
        enddo
        if (icnt-1 .gt. nbmax) nbmax = icnt-1
    enddo

    write(*,*) ' '
    inbs2(:,:) = 0
    do i = 1, na2
        icnt = 1
        do i2 = 1, na2
            l1 = pls2(i2,i)
            if (l1 .gt. 0 .and. l1 .le. dcut) then
                inbs2(icnt,i) = i2
!                write(*,*) ' -- ', i,i2
                icnt = icnt + 1
            endif
        enddo
        if (icnt-1 .gt. nbmax) nbmax = icnt-1
    enddo
    if (nbmax .gt. (3**dcut-1)*2) then
        stop "#ERROR: `nbmax .gt. 3^dcut-1)*2!!"
    endif

    dist = 0.
    do i = 1, na1
        do j = 1, na2
            ! 1-body terms 
            vss1(1,:) = vcps1(:,i)
            vss1(2,:) = vcps2(:,j)
            call get_overlapN(vss1, 2, nv, sigmas_a, ifactor, ov)
            vss1(1,:) = vcps1(:,i)
            vss1(2,:) = vcps1(:,i)
            call get_overlapN(vss1, 2, nv, sigmas_a, ifactor, ov1)
            vss1(1,:) = vcps2(:,i)
            vss1(2,:) = vcps2(:,i)
            call get_overlapN(vss1, 2, nv, sigmas_a, ifactor, ov2)
            d1 = ov1 + ov2 - 2*ov

            ! 2-body terms 
            d2 = 0.
            do i2 = 1, nbmax
                do i3 = 1, nbmax
                    ia2 = inbs1(i2,i)
                    ia3 = inbs1(i3,i)
                    if (ia2 .gt. 0 .and. ia3 .gt. 0) then
                        l1 = pls1(i,ia2)
                        l2 = pls1(i,ia3)
                        vss2(1,:) = vcps1(:,i)
                        vss2(2,:) = vcps1(:,ia2)
                        vss2(3,:) = vcps1(:,i)
                        vss2(4,:) = vcps1(:,ia3)
                        call get_overlapN(vss2, 4, nv, sigmas_a, ifactor, ov)
                        ds2(1,:) = l1
                        ds2(2,:) = l2
                        call get_overlapN(ds2, 2, 1, sigma_r, ifactor, ov2)
                        dt = ov*ov2/((l1+d0)*(l2+d0))**power
                        if (debug) write(*,*) ' ++ 1-1 ', i,ia2,i,ia3, dt
                        d2 = d2 + dt
                    endif
                enddo
            enddo

            do j2 = 1, nbmax
                do j3 = 1, nbmax
                    ja2 = inbs2(j2,j)
                    ja3 = inbs2(j3,j)
                    if (ja2 .gt. 0 .and. ja3 .gt. 0) then
                        l1 = pls2(j,ja2)
                        l2 = pls2(j,ja3)
                        vss2(1,:) = vcps2(:,j)
                        vss2(2,:) = vcps2(:,ja2)
                        vss2(3,:) = vcps2(:,j)
                        vss2(4,:) = vcps2(:,ja3)
                        call get_overlapN(vss2, 4, nv, sigmas_a, ifactor, ov)
                        ds2(1,:) = l1
                        ds2(2,:) = l2
                        call get_overlapN(ds2, 2, 1, sigma_r, ifactor, ov2)
                        dt = ov*ov2/((l1+d0)*(l2+d0))**power
                        if (debug) write(*,*) ' ++ 2-2 ', j,ja2,j,ja3, dt
                        d2 = d2 + dt
                    endif
                enddo
            enddo

            do i2 = 1, nbmax
                do j2 = 1, nbmax
                    ia2 = inbs1(i2,i)
                    ja2 = inbs2(j2,j)
                    if (ia2 .gt. 0 .and. ja2 .gt. 0) then
                        l1 = pls1(i,ia2)
                        l2 = pls2(j,ja2)
                        vss2(1,:) = vcps1(:,i)
                        vss2(2,:) = vcps1(:,ia2)
                        vss2(3,:) = vcps2(:,j)
                        vss2(4,:) = vcps2(:,ja2)
                        call get_overlapN(vss2, 4, nv, sigmas_a, ifactor, ov)
                        ds2(1,:) = l1
                        ds2(2,:) = l2
                        call get_overlapN(ds2, 2, 1, sigma_r, ifactor, ov2)
                        dt = -2.*ov*ov2/((l1+d0)*(l2+d0))**power
                        d2 = d2 + dt
                        if (debug) write(*,*) ' ++ 1-2 ', i,ia2,j,ja2, dt
                    endif
                enddo
            enddo
            if (debug) write(*,*) 'i,j,d1^2,d2^2 = ', i,j,d1,d2
            dist = dist + d1 + d2

        enddo
    enddo

    if (dist .le. 0.) then
        if (dist .le. -1.e-6) then
            stop "#ERROR: `dist .le. -1e-6??"
        else
            dist = 0.
        endif
    endif
    dist = sqrt(dist)

end subroutine ldist


subroutine lk_matrix(nas,vcps,pls,sigmas, ifactor, &
                        dcut,power,n1,n2, d0, &
                        k1,k2,nm,namax,nat,nv)
    !
    ! purpose:
    !   local kernel (lk) matrix calculation
    !
    ! variables
    ! ================================================
    ! n1 : training set size
    ! n2 : test set size (default case: n2 = N - n1)
    ! dm1: distance matrix for training species
    !

    implicit none 

    integer, intent(in) :: nv,nm,namax,nat,power, &
                           n1,n2
    real*8, intent(in) :: nas(nm)
    real*8, intent(in) :: vcps(nv,nat), pls(namax,namax,nm), &
                          sigmas(nv+1), d0,dcut
    logical, intent(in) :: ifactor
    real*8, intent(out) :: k1(n1,n1), k2(n1,n2)

    integer :: i,j,j0, ia1,ia2,ja1,ja2, &
               ias2(nm),ias1(nm), na1,na2
    real*8 :: dij,pls1(namax,namax),pls2(namax,namax)

    do i = 1,nm
        ias2(i) = sum(nas(:i))
    enddo
    ias1(1) = 1
    ias1(2:nm) = ias2(1:nm-1)

    k1(:,:) = 0.

!$omp parallel do private(i,j,na1,na2,pls1,pls2,dij)
    do i = 1,n1
        do j = i+1,n1
            na1 = nas(i)
            na2 = nas(j)
            !vcps1 = vcps(:,ias1(i):ias2(i))
            !vcps2 = vcps(:,ias1(j):ias2(j))
            pls1 = pls(:,:,i)
            pls2 = pls(:,:,j)
            call lk(na1,na2, &
                    vcps(:,ias1(i):ias2(i)), &
                    vcps(:,ias1(j):ias2(j)), &
                    pls1,pls2, &
                    sigmas,ifactor,dcut,nv,power,d0,kij)
            k1(i,j) = kij
            k1(j,i) = kij
        enddo
    enddo
!$omp end parallel do

!$omp parallel do private(i,j,j0,na1,na2,pls1,pls2,dij)
    do i = 1,n1 
        do j0 = 1,n2 
            j = nm-n2+j0
            na1 = nas(i)
            na2 = nas(j)
            !vcps1 = vcps(:,ias1(i):ias2(i))
            !vcps2 = vcps(:,ias1(j):ias2(j))
            pls1 = pls(:,:,i)
            pls2 = pls(:,:,j)
            call ldist(na1,na2, &
                 vcps(:,ias1(i):ias2(i)), &
                 vcps(:,ias1(j):ias2(j)), &
                 pls1,pls2, &
                 sigmas,ifactor,dcut,nv,power,d0,dij)
            dm2(i,j0) = dij
        enddo
    enddo
!$omp end parallel do

end subroutine lk_matrix


