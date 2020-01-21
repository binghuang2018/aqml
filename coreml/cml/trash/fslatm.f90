module fslatm

implicit none

real*8, parameter :: r0 = 0.1 
double precision, parameter :: eps = epsilon(0.0d0)
double precision, parameter :: pi = 4.0d0 * atan(1.0d0)


contains

function linspace(x0, x1, nx) result(xs)

    implicit none

    double precision, intent(in) :: x0,x1
    integer, intent(in) :: nx
    double precision, dimension(nx) :: xs
    integer :: i
    double precision :: step

    step = (x1 - x0) / (nx - 1)
    !!$OMP PARALLEL DO
    do i = 1, nx
        xs(i) = x0 + (i - 1) * step
    enddo
    !!$OMP END PARALLEL DO
end function linspace

function calc_angle(a, b, c) result(angle)
    implicit none
    double precision, intent(in), dimension(3) :: a
    double precision, intent(in), dimension(3) :: b
    double precision, intent(in), dimension(3) :: c
    double precision, dimension(3) :: v1
    double precision, dimension(3) :: v2
    double precision :: cos_angle
    double precision :: angle
    v1 = a - b
    v2 = c - b
    v1 = v1 / norm2(v1)
    v2 = v2 / norm2(v2)
    cos_angle = dot_product(v1,v2)
    ! Clipping
    if (cos_angle > 1.0d0) cos_angle = 1.0d0
    if (cos_angle < -1.0d0) cos_angle = -1.0d0
    angle = acos(cos_angle)
end function calc_angle


function calc_cos_angle(a, b, c) result(cos_angle)
    implicit none
    real*8, intent(in) :: a(3),b(3),c(3)
    real*8 :: v1(3),v2(3)
    real*8 :: cos_angle
    v1 = a - b
    !print *, 'v1 = ', v1
    v2 = c - b
    !print *, 'v2 = ', v2
    v1 = v1 / norm2(v1)
    v2 = v2 / norm2(v2)
    cos_angle = dot_product(v1,v2)
end function calc_cos_angle


function calc_dist(na, coords) result(ds)
    implicit none
    integer, intent(in) :: na
    real*8, intent(in) :: coords(3,na)
    real*8 :: ds(na,na)
    integer :: i,j 
    real*8 :: dij
    ds(:,:) = 0.0
    do i = 1,na
        do j = i+1,na
            dij = norm2(coords(:,i)-coords(:,j))
            ds(i,j) = dij
            ds(j,i) = dij
        enddo
    enddo
end function calc_dist

!end module fslatm_math


subroutine fget_sbot(na, coords, zs, z1, z2, z3, rscut, nx, dgrid, sigma, rpower3, ys)
    !use fslatm_math

    implicit none
    integer, intent(in) :: na
    double precision, dimension(3,na), intent(in) :: coords
    integer, dimension(:), intent(in) :: zs
    double precision, intent(in) :: rscut(2)
    integer, intent(in) :: nx
    double precision, intent(in) :: dgrid, sigma, rpower3
    double precision, dimension(nx), intent(out) :: ys
    integer, intent(in) :: z1, z2, z3
    integer, dimension(:), allocatable :: ias1, ias2, ias3
    integer :: nias1, nias2, nias3
    integer :: ia1, ia2, ia3
    double precision :: ds(na,na)
    double precision :: norm
    integer :: i,j,k
    double precision :: d2r,c0,ang,cak,cai,prefactor,r, a0,a1, inv_sigma
    real*8 :: xs(nx),cos_xs(nx)

    ds = calc_dist(na, coords)

    allocate(ias1(na))
    allocate(ias2(na))
    allocate(ias3(na))

    ias1 = 0
    ias2 = 0
    ias3 = 0

    nias1 = 0
    do i = 1, na
        if (int(zs(i)).eq.z1) then
            nias1 = nias1 + 1
            ias1(nias1) = i
        endif
    enddo

    nias2 = 0
    do i = 1, na
        if (int(zs(i)).eq.z2) then
            nias2 = nias2 + 1
            ias2(nias2) = i
        endif
    enddo 

    nias3 = 0
    do i = 1, na
        if (int(zs(i)).eq.z3) then
            nias3 = nias3 + 1
            ias3(nias3) = i
        endif
    enddo

    d2r = pi/180.0d0
    a0 = -20.0d0 * d2r
    a1 = pi + 20.0d0 * d2r
    xs = linspace(a0, a1, nx)
    prefactor = 1.0d0 / 3.0d0
    c0 = prefactor * (mod(z1,1000)*mod(z2,1000)*mod(z3,1000)) * dgrid
    c0 = c0 / sqrt(2*sigma**2*pi) ! !use normalized gaussian

    ys = 0.0d0
    inv_sigma = -1.0d0 / (2*sigma**2)

    !!$OMP PARALLEL DO
    do i = 1, nx
        cos_xs(i) = cos(xs(i)) * c0
    enddo
    !!$OMP END PARALLEL do

    !!$OMP PARALLEL DO PRIVATE(i,j,k,ang,cai,cak,r) REDUCTION(+:ys) SCHEDULE(DYNAMIC)
    do ia1 = 1, nias1
        do ia2 = 1, nias2
            if (ias2(ia2).eq.ias1(ia1)) cycle
            if (.not. ((ds(ias2(ia2),ias1(ia1)) > eps) .and. &
                     & (ds(ias2(ia2),ias1(ia1)) <= rscut(1)))) cycle
            do ia3 = 1, nias3
                !if (ds(ias1(ia1),ias3(ia3)) > rscut(1)) cycle
                if (ias2(ia2).eq.ias3(ia3) .or. ias1(ia1).eq.ias3(ia3)) cycle
                if (.not. ((ds(ias2(ia2),ias3(ia3)) > eps) .and. &
                         & (ds(ias2(ia2),ias3(ia3)) <= rscut(1)))) cycle

                    i = ias1(ia1)
                    j = ias2(ia2)
                    k = ias3(ia3)

                    ang = calc_angle(coords(:, i), coords(:, j), coords(:, k))
                    cak = calc_cos_angle(coords(:, i), coords(:, k), coords(:, j))
                    cai = calc_cos_angle(coords(:, k), coords(:, i), coords(:, j))
                    r = ds(i,j) * ds(i,k) * ds(k,j)
                    ys = ys + (c0 + cos_xs*cak*cai)/(r**rpower3) * ( exp((xs-ang)**2 * inv_sigma) )

            enddo
        enddo
    enddo
    !!$OMP END PARALLEL do

    deallocate(ias1)
    deallocate(ias2)
    deallocate(ias3)
end subroutine fget_sbot


subroutine fget_sbot_local(na,coords,zs,ia,ib,z1,z2,z3,rscut,nx,dgrid,sigma,w3,rpower3, ys)
    !use fslatm_math
    implicit none
    integer, intent(in) :: na
    double precision, dimension(3,na), intent(in) :: coords
    integer, dimension(:), intent(in) :: zs
    double precision, intent(in) :: rscut(2)
    integer, intent(in) :: nx,ia,ib
    double precision, intent(in) :: dgrid,sigma, w3, rpower3
    double precision, dimension(nx), intent(out) :: ys
    integer, intent(in) :: z1, z2, z3
    integer, dimension(:), allocatable :: ias1, ias3
    integer :: nias1, nias3
    integer :: ia1, ia2, ia3
    double precision :: ds(na,na)
    double precision :: norm
    integer :: i,j,k
    double precision :: d2r,c0,ang,cak,cai,prefactor,r, a0,a1, inv_sigma
    double precision :: xs(nx),cos_xs(nx)
    !logical :: stop_flag

    ys = 0.0d0
    if (zs(ia).ne.z2) return
    if (ib.gt.0 .and. zs(ib).ne.z1) return ! `ib is defaulted to 0

    ds = calc_dist(na, coords)

    allocate(ias1(na))
    allocate(ias3(na))

    ias1 = 0
    ias3 = 0


    nias1 = 0
    if (ib .gt. 0) then
        if (ib .eq. ia) return
        if (int(zs(ib)).eq.z1) then
            nias1 = nias1 + 1
            ias1(nias1) = ib
        endif
    else
        do i = 1, na
            if (zs(i).eq.z1 .and. i.ne.ia) then
                nias1 = nias1 + 1
                ias1(nias1) = i
            endif
        enddo
    endif

    nias3 = 0
    do i = 1, na
        if (int(zs(i)).eq.z3 .and. i.ne.ia) then
            nias3 = nias3 + 1
            ias3(nias3) = i
        endif
    enddo

    d2r = pi/180.0d0
    a0 = -20.0d0 * d2r
    a1 = pi + 20.0d0 * d2r
    xs = linspace(a0, a1, nx)
    prefactor = 1.0d0 ! / 3.0d0
    c0 = prefactor * (mod(z1,1000)*mod(z2,1000)*mod(z3,1000)) * dgrid
    c0 = w3 * c0 / sqrt(2*sigma**2*pi) ! !use normalized gaussian
    inv_sigma = -1.0d0 / (2*sigma**2)

    !!$OMP PARALLEL DO
    do i = 1, nx
        cos_xs(i) = cos(xs(i)) * c0
    enddo
    !!$OMP END PARALLEL do

    !!$OMP PARALLEL DO PRIVATE(i,j,k,ang,cai,cak,r) REDUCTION(+:ys) SCHEDULE(DYNAMIC)
    do ia1 = 1, nias1
        if (ib.gt.0) then
            if (.not. ((ds(ias1(ia1),ia) > eps) .and. &
                     & (ds(ias1(ia1),ia) <= rscut(2)))) cycle
        else
            if (.not. ((ds(ias1(ia1),ia) > eps) .and. &
                     & (ds(ias1(ia1),ia) <= rscut(1)))) cycle
        endif
        do ia3 = 1, nias3
            if (ia.eq.ias3(ia3) .or. ias1(ia1).eq.ias3(ia3)) cycle
            !if (ds(ias1(ia1),ias3(ia3)) > rscut(1)) cycle
            if (.not. ((ds(ia,ias3(ia3)) > eps) .and. &
                     & (ds(ia,ias3(ia3)) <= rscut(1)))) cycle

                ! note that each pair of (i,k) in [i,j,k] can be
                ! considered only once!!
                i = ias1(ia1)
                j = ia
                k = ias3(ia3)

                ang = calc_angle(coords(:, i), coords(:, j), coords(:, k))
                cak = calc_cos_angle(coords(:, i), coords(:, k), coords(:, j))
                cai = calc_cos_angle(coords(:, k), coords(:, i), coords(:, j))
                r = ds(i,j) * ds(i,k) * ds(k,j)
                ys = ys + (c0 + cos_xs*cak*cai)/(r**rpower3) * ( exp((xs-ang)**2 * inv_sigma) )

                !print *, ' in sbot_local (z1!=z3), I,J,K = ', i,j,k, '##', c0, cak, cai, r
                !call check_nan1(nx,ys)

        enddo
    enddo
    !!$OMP END PARALLEL do

    deallocate(ias1)
    deallocate(ias3)

end subroutine fget_sbot_local


subroutine fget_sbop(na, coords, zs, z1, z2, rscut, nx, dgrid, sigma, rpower2, ys)
    !use fslatm_math
    implicit none
    integer, intent(in) :: na
    double precision, dimension(3,na), intent(in) :: coords
    integer, dimension(:), intent(in) :: zs
    double precision, intent(in) :: rscut(2)
    integer, intent(in) :: nx
    double precision, intent(in) :: dgrid,sigma,rpower2
    double precision, dimension(nx), intent(out) :: ys
    integer, intent(in) :: z1, z2
    double precision :: r,racut2
    integer :: i
    integer, dimension(:), allocatable :: ias1, ias2
    integer :: nias1, nias2, ia1,ia2
    double precision, dimension(nx) :: xs
    double precision :: c0
    double precision :: inv_sigma
    double precision, dimension(nx) :: xs0

    allocate(ias1(na))
    allocate(ias2(na))

    ias1 = 0
    ias2 = 0

    nias1 = 0
    do i = 1, na
        if (int(zs(i)).eq.z1) then
            nias1 = nias1 + 1
            ias1(nias1) = i
        endif
    enddo

    nias2 = 0
    do i = 1, na
        if (int(zs(i)).eq.z2) then
            nias2 = nias2 + 1
            ias2(nias2) = i
        endif
    enddo

    xs = linspace(r0, rscut(1), nx)
    ys = 0.0d0

    c0 = mod(z1,1000)*mod(z2,1000)
    c0 = c0 / sqrt(2*sigma**2*pi) ! !use normalized gaussian
    inv_sigma = -0.5d0 / sigma**2 
    xs0 = c0/(xs**rpower2) * dgrid
    racut2 = rscut(1)**2

    !!$OMP PARALLEL DO REDUCTION(+:ys)
    do ia1 = 1, nias1
        do ia2 = 1, nias2
            if (ias1(ia1).eq.ias2(ia2)) cycle
            r = sum((coords(:,ias1(ia1)) - coords(:,ias2(ia2)))**2)
            if (r < racut2) ys = ys + xs0 * exp( inv_sigma * (xs - sqrt(r))**2 )

        enddo
    enddo
    !!$OMP END PARALLEL DO

    deallocate(ias1)
    deallocate(ias2)

end subroutine fget_sbop


subroutine fget_sbop_local(coords, zs, ia, ib, z1, z2, rscut, nx, dgrid, sigma, w2,rpower2, ys)
    !use fslatm_math
    implicit none
    double precision, dimension(:,:), intent(in) :: coords
    integer, dimension(:), intent(in) :: zs
    double precision, intent(in) :: rscut(2)
    integer, intent(in) :: nx
    integer, intent(in) :: ia, ib
    double precision, intent(in) :: dgrid
    double precision, intent(in) :: sigma
    double precision, intent(in) :: w2,rpower2
    double precision, dimension(nx), intent(out) :: ys
    integer, intent(in) :: z1, z2
    double precision :: r
    double precision :: racut,racut2
    integer :: i
    integer :: na
    integer, dimension(:), allocatable :: ias2
    integer :: ia2, nias2
    double precision, dimension(nx) :: xs
    double precision :: c0
    double precision :: inv_sigma
    double precision, dimension(nx) :: xs0

    ys = 0.0d0
    if (zs(ia).ne.z1) return
    if (ib.gt.0) then
        if (ib.eq.ia .or. zs(ib).ne.z2) return
    endif

    allocate(ias2(na))
    ias2 = 0
    nias2 = 0
    if (ib.gt.0) then
        nias2 = 1
        ias2(1) = ib
        racut = rscut(2)
    else
        do i = 1, na
            if (int(zs(i)).eq.z2 .and. i.ne.ia) then
                nias2 = nias2 + 1
                ias2(nias2) = i
            endif
        enddo
        racut = rscut(1)
    endif
    xs = linspace(r0, rscut(1), nx)

    c0 = (mod(z1,1000)*mod(z2,1000)) !* (1./2.)  ! added "*(1./2.)" @Dec.2, 2018
    c0 = c0 / sqrt(2*sigma**2*pi) ! !use normalized gaussian
    inv_sigma = -0.5d0 / sigma**2 
    xs0 = w2 * c0/(xs**rpower2) * dgrid
    racut2 = racut**2

    !!$OMP PARALLEL DO REDUCTION(+:ys)
    do ia2 = 1, nias2
        r = sum((coords(:,ia) - coords(:,ias2(ia2)))**2)
        if (r < racut2) ys = ys + xs0 * exp( inv_sigma * (xs - sqrt(r))**2 )
    enddo
    !!$OMP END PARALLEL DO

    deallocate(ias2)
end subroutine fget_sbop_local


subroutine fget_spectrum(na0,zs,coords,n1,n2,n3,mbs1,mbs2,mbs3, &
                      n, nsx,rscut,dgrid,sigma,rpower2,rpower3, ys)
    implicit none
    integer, intent(in) :: na0
    integer, intent(in) :: zs(na0), n1,n2,n3, n,nsx(3)
    real*8, intent(in) :: coords(3,na0), rscut(2),dgrid,sigma
    real*8, intent(in) :: rpower2,rpower3
    integer, intent(in) :: mbs1(n1),mbs2(2,n2),mbs3(3,n3)
    real*8, intent(out) :: ys(n)
    integer :: imb1,imb2,imb3,z1,z2,z3, na
    integer :: jb,je, ja

    ys(:) = 0.0
    do imb1 = 1,n1
        z1 = mbs1(imb1)
        do ja = 1,na0
            if (z1.eq.zs(ja)) then
                ys(imb1) = ys(imb1) + z1
            endif
        enddo
    enddo
    do imb2 = 1,n2
        z1 = mbs2(1,imb2)
        z2 = mbs2(2,imb2)
        jb = n1 + (imb2-1)*nsx(1)
        je = n1 + imb2*nsx(1)
        call fget_sbop(na0,coords,zs,z1,z2,rscut,nsx(1),dgrid,sigma,rpower2,ys(jb:je))
    enddo
    do imb3 = 1,n3
        z1 = mbs3(1,imb3)
        z2 = mbs3(2,imb3)
        z3 = mbs3(3,imb3)
        jb = n1 + n2*nsx(1) + (imb3-1)*nsx(3)
        je = n1 + n2*nsx(1) + imb3*nsx(3)
        call fget_sbot(na0,coords,zs,z1,z2,z3,rscut,nsx(3),dgrid,sigma,rpower3,ys(jb:je))
    enddo
end subroutine


subroutine check_nan1(n1,x)
    implicit none
    integer, intent(in) :: n1
    real*8, intent(in):: x(n1)
    integer :: i
    do i = 1,n1
        if (isnan(x(i))) then
            print *, 'x(i)=',i,'NaN'
            stop 
        endif
    enddo
end



subroutine check_nan2(n1,n2,x)
    implicit none
    integer, intent(in) :: n1,n2
    real*8, intent(in):: x(n1,n2)
    integer :: i,j
    do i = 1,n1
      do j = 1,n2
        if (isnan(x(i,j))) then
            print *, 'x(i,j)=',i,j,'NaN'
            stop 
        endif
      enddo
    enddo
end


subroutine check_nan3(n1,n2,n3,x)
    implicit none
    integer, intent(in) :: n1,n2,n3
    real*8, intent(in):: x(n1,n2,n3)
    integer :: i,j,k
    do i = 1,n1
      do j = 1,n2
        do k = 1,n3
          if (isnan(x(i,j,k))) then
            print *, 'x(i,j,k)=',i,j,k,'NaN'
            stop 
          endif
        enddo
      enddo
    enddo
end


subroutine fget_local_spectrum(na0,zs,coords,nbmax,nbrs,n,nu,n1,n2,n3,mbs1,mbs2,mbs3, &
                               nsx,rscut,dgrid,sigma,w2,rpower2,w3,rpower3, ia0,ib0, ys,ys2)
    ! 
    ! Get aSLATM for a molecule consisting of `na0 atoms
    !
    ! nbmax: the maximal number of neighbors within a radius of `rscut
    !         for all atoms
    ! 
    implicit none
    integer, intent(in) :: na0
    integer, intent(in) :: ia0,ib0
    integer, intent(in) :: zs(na0), nu,n1,n2,n3, n,nbmax,nsx(3)
    real*8, intent(in) :: coords(3,na0), rscut(2),dgrid,sigma
    real*8, intent(in) :: rpower2,rpower3, w2,w3
    integer, intent(in) :: mbs1(n1),mbs2(2,n2),mbs3(3,n3), nbrs(nbmax,na0)
    real*8, intent(out) :: ys(n,na0),ys2(nu,nbmax,na0)
    integer :: imb1,imb2,imb3,z1,z2,z3, na1,na2
    integer :: i,j,ia,ja,zi,ibgn,iend, ias1(na0),ias2(nbmax,na0),nas2(na0)

    ias1(:) = 0
    if (ia0.gt.0) then
        na1 = 1
        ias1(1) = ia0
    elseif (ia0.eq.0) then
        na1 = na0
        do ia = 1,na1
            ias1(ia) = ia
        enddo
    endif

    ias2(:,:) = -1
    nas2 = 0
    na2 = 0
    if (ib0.eq.0) then
        do ia=1,na0
            ias2(:,ia) = nbrs(:,ia)
            nas2(ia) = count(nbrs(:,ia)>-1)
        enddo
        !print *, ' ** nas2 = ', nas2
        na2 = sum(nas2)
    elseif (ib0.gt.0) then
        nas2(ia0) = 1
        na2 = 1
        ias2(1,ia0) = ib0
    endif
    !print *, ' -- nbmax = ', nbmax
    !print *, ' ----- ias1 = ', ias1
    !print *, ' ----- ias2 = ', ias2

    ys(:,:) = 0.0
    !print *, ' #####1'
    do i = 1,na1
        ia = ias1(i)
        zi = zs(ia)
        do imb1 = 1,n1
            z1 = mbs1(imb1)
            if (z1.eq.zi) then
                ys(imb1,ia) = z1
            endif
            !print *, ' 1body: ', ys(1:n1, ia)
        enddo
        do imb2 = 1,n2
            z1 = mbs2(1,imb2)
            z2 = mbs2(2,imb2)
            ibgn = n1 + (imb2-1)*nsx(1) + 1
            iend = n1 + imb2*nsx(1)
            !print *, ' 2body idx: ', ibgn, iend
            call fget_sbop_local(coords,zs,ia,0,z1,z2,rscut,nsx(1),dgrid,sigma,w2,rpower2,ys(ibgn:iend,ia))
            !call check_nan2(n,na0,ys)
        enddo
        do imb3 = 1,n3
            z1 = mbs3(1,imb3)
            z2 = mbs3(2,imb3)
            z3 = mbs3(3,imb3)
            ibgn = n1 + n2*nsx(1) + (imb3-1)*nsx(3) + 1
            iend = n1 + n2*nsx(1) + imb3*nsx(3)
            !print *, ' 3body idx: ', ibgn,iend
            call fget_sbot_local(na0,coords,zs,ia,0,z1,z2,z3,rscut,nsx(3),dgrid,sigma,w3,rpower3,ys(ibgn:iend,ia))
            !call check_nan2(n,na0,ys)
        enddo
    enddo
    !print *, '#####2'
    ys2(:,:,:) = 0.0 ! for bond representation
    if (na2.gt.0) then
        !print *, ' ++++++++ ias1 = ', ias1
        do i = 1,na1
            !print *, ' *** ias2(:,i) = ', ias2(:,i)
            do j = 1,nas2(i)
                ia = ias1(i)  !!! `ia already starts from 1!!
                ja = ias2(j,i) + 1  !!! convert python idx (starts from 0) to fortran (starts from 1)
                if (ja.eq.ia .or. ja.le.0) cycle
                !print *, '  +++++ i,j,ia,ja = ', i,j,ia,ja
                do imb2 = 1,n2
                    z1 = mbs2(1,imb2)
                    z2 = mbs2(2,imb2)
                    ibgn = n1 + (imb2-1)*nsx(2) + 1
                    iend = n1 + imb2*nsx(2)
                    !print *, ' B(i->j) 2body idx: ', ibgn, iend
                    call fget_sbop_local(coords,zs,ia,ja,z1,z2,rscut,nsx(2),dgrid,sigma,w2,rpower2,ys2(ibgn:iend,j,ia))
                    !call check_nan3(n,nbmax,na0,ys2)
                enddo
                do imb3 = 1,n3
                    z1 = mbs3(1,imb3)
                    z2 = mbs3(2,imb3)
                    z3 = mbs3(3,imb3)
                    ibgn = n1 + n2*nsx(2) + (imb3-1)*nsx(3) + 1
                    iend = n1 + n2*nsx(2) + imb3*nsx(3)
                    !print *, ' B(i->j) 3body idx: ', ibgn, iend, ', j=',j, ', ys2.shape', n,nbmax,na0
                    call fget_sbot_local(na0,coords,zs,ia,ja,z1,z2,z3,rscut,nsx(3),dgrid,sigma,w3,rpower3,ys2(ibgn:iend,j,ia))
                    !print *, ' DONE!'
                    !call check_nan3(n,nbmax,na0,ys2)
                enddo
            enddo
        enddo
    endif

end subroutine


subroutine fget_all_local_spectrum(nm,na,nas,zs,coords,nbmax,nbrs, &
                    n,nu,n1,n2,n3,nsx, mbs1,mbs2,mbs3, rscut,dgrid,sigma, &
                    w2,rpower2,w3,rpower3, ia0,ib0, ys,ys2)
    integer, intent(in) :: nm,na
    integer, intent(in) :: nas(nm),zs(na)
    integer, intent(in) :: n, nbmax, nu,n1,n2,n3, nsx(3), ia0,ib0
    real*8, intent(in) :: coords(3,na), rscut(2),dgrid,sigma
    integer, intent(in) :: mbs1(n1),mbs2(2,n2),mbs3(3,n3), nbrs(nbmax,na)
    real*8, intent(in) :: rpower2,rpower3, w2,w3
    real*8, intent(out) :: ys(n,na), ys2(n,nbmax,na)
    integer :: i, ib,ie, ias1(nm),ias2(nm)

    ! get ias1,ias2
    do i = 1,nm
        ias2(i) = sum(nas(1:i))
    enddo
    ias1(1) = 1
    ias1(2:nm) = ias2(1:nm-1)+1

    !$OMP parallel do private(i,ib,ie)
    do i = 1,nm
        ib = ias1(i)
        ie = ias2(i)
        call fget_local_spectrum(nas(i),zs(ib:ie),coords(:,ib:ie), &
                          nbmax,nbrs(:,ib:ie), n,nu,n1,n2,n3,mbs1,mbs2,mbs3, &
                          nsx,rscut,dgrid,sigma,w2,rpower2, w3,rpower3, &
                          ia0,ib0, ys(:,ib:ie),ys2(:,:,ib:ie))
    enddo
    !$OMP end parallel do
end subroutine


subroutine fget_local_dij(na1,na2,n,x1,x2,dsij)
    implicit none
    integer, intent(in) :: na1,na2,n
    real*8, intent(in) :: x1(n,na1),x2(n,na2)
    !character*1, intent(in) :: kernel
    real*8, intent(out) :: dsij(na1,na2)
    integer :: i,j
    do i = 1,na2
        do j = 1,na1
            !if (kernel.eq.'l') then
            !    t = t + exp( - sum(x1(:,i)-x2(:,j))/kwds(ic) )
            !elseif (kernel .eq. 'g') then
            dsij(j,i) = norm2(x1(:,j)-x2(:,i))
            !endif
        enddo
    enddo
end subroutine


subroutine fget_local_kij(na1,na2,n,x1,x2,nco,kwds,kijs)
    implicit none
    integer, intent(in) :: na1,na2,n, nco
    real*8, intent(in) :: x1(n,na1),x2(n,na2)
    real*8, intent(in) :: kwds(nco)
    !character*1, intent(in) :: kernel
    real*8, intent(out) :: kijs(nco)
    integer :: ic,i,j
    real*8 :: t

    do ic = 1,nco
        t = 0.0d0
        do i = 1,na1
            do j = 1,na2
                !if (kernel.eq.'l') then
                !    t = t + exp( - sum(x1(:,i)-x2(:,j))/kwds(ic) )
                !elseif (kernel .eq. 'g') then
                    t = t + exp( - (norm2(x1(:,i)-x2(:,j)))**2/(2.0*kwds(ic)**2) )
                !endif
            enddo
        enddo
        kijs(ic) = t
    enddo
end subroutine


subroutine fget_kij(n,x1,x2,nco,kwds,kijs)
    implicit none
    integer, intent(in) :: n, nco
    real*8, intent(in) :: x1(n),x2(n)
    real*8, intent(in) :: kwds(nco)
    !character*1, intent(in) :: kernel
    real*8, intent(out) :: kijs(nco)
    integer :: ic

    do ic = 1,nco
        !if (kernel.eq.'l') then
        !    kijs(ic) = exp( - sum(x1-x2)/kwds(ic) )
        !elseif (kernel .eq. 'g') then
            kijs(ic) = exp( - (norm2(x1-x2))**2/(2.0*kwds(ic)**2) )
        !endif
    enddo
end subroutine


subroutine fget_mk1(local,nm,na,nas,zs,coords,nbmax,nbrs,n1,n2,n3,mbs1,mbs2,mbs3,&
                    rscut,dgrid,sigma, w2,rpower2,w3,rpower3, nco,kwds, k1)
    ! 
    ! K for training set
    !
    ! Calculate molecular kernel matrix K (upper case), to be distinguished with
    ! lower case k, characterizing similarity between atoms.
    ! The i-th and j-th entry of K is 
    !        K(i,j) = \sum_{I \in i} \sum_{J \in j} k(I,J)
    ! 
    logical, intent(in) :: local    
    integer, intent(in) :: nm,na, nbmax
    integer, intent(in) :: nas(nm),zs(na)
    integer, intent(in) :: n1,n2,n3, nbrs(nbmax,na)
    real*8, intent(in) :: coords(3,na), rscut(2),dgrid,sigma
    integer, intent(in) :: mbs1(n1),mbs2(2,n2),mbs3(3,n3), nco
    real*8, intent(in) :: kwds(nco), w2,rpower2,w3,rpower3
    !character*1, intent(in) :: kernel
    real*8, intent(out) :: k1(nco,nm,nm)
    real*8, allocatable :: xs1(:,:),xs2(:,:),x1(:),x2(:), nul(:,:,:)
    integer :: i,j, iar, im,iab,iae, namax, nsx(3)
    integer :: n,nu
    integer :: pair(2),iasr(2), amap(na), ias1(nm),ias2(nm)
    real*8 :: dmax,kijs(nco)

    ! get ias1,ias2
    do i = 1,nm
        ias2(i) = sum(nas(1:i))
    enddo
    ias1(1) = 1
    ias1(2:nm) = ias2(1:nm-1)+1

    ! get amap: atomic idx --> mol idx
    !print *, '##1'
    amap(:) = 0
    do i = 1,nm
        amap(ias1(i):ias2(i)) = i
    enddo

    ! setup grid
    nsx = 0
    nsx(1) = int((rscut(1) - r0)/dgrid) + 1
    !d2r = pi/dble(180) ! degree to rad
    !a0 = -20.0*d2r
    !a1 = pi + 20.0*d2r
    nsx(3) = int((pi + 40.0*(pi/dble(180)))/dgrid) + 1
    n = n1 + n2*nsx(1) + n3*nsx(3)
    nu = n1 + n2*nsx(2) + n3*nsx(3)
    namax = maxval(nas)
    !print *, '##2'
    allocate( xs1(n,namax),xs2(n,namax), x1(n),x2(n), nul(nu,nbmax,namax) )
    xs1(:,:) = 0.0
    xs2(:,:) = 0.0
    x1(:) = 0.0
    x2(:) = 0.0
    !print *, '##3'
    if (local) then
        !$OMP parallel do private(i,j,xs1,xs2,kijs)
        do i = 1,nm
            do j = i,nm
                !print *, ' I,J = ', I,J
                call fget_local_spectrum(nas(i),zs(ias1(i):ias2(i)),coords(:,ias1(i):ias2(i)), &
                                  nbmax,nbrs(:,ias1(i):ias2(i)), n,nu,n1,n2,n3,mbs1,mbs2,mbs3, nsx, &
                                  rscut,dgrid,sigma,w2,rpower2,w3,rpower3, 0, -1, xs1(:,1:nas(i)), nul(:,:,1:nas(i)))
                call fget_local_spectrum(nas(j),zs(ias1(j):ias2(j)),coords(:,ias1(j):ias2(j)), &
                                  nbmax,nbrs(:,ias1(i):ias2(i)), n,nu,n1,n2,n3,mbs1,mbs2,mbs3, nsx, &
                                  rscut,dgrid,sigma,w2,rpower2,w3,rpower3, 0, -1, xs2(:,1:nas(j)), nul(:,:,1:nas(i)))
                call fget_local_kij(nas(i),nas(j),n,xs1(:,1:nas(i)),xs2(:,1:nas(j)), nco,kwds, kijs)
                k1(:,i,j) = kijs
                k1(:,j,i) = kijs
            enddo
        enddo
        !$OMP end parallel do
    else
        !$OMP parallel do private(i,j,x1,x2,kijs)
        do i = 1,nm
            do j = i,nm
                call fget_spectrum(nas(i),zs(ias1(i):ias2(i)),coords(:,ias1(i):ias2(i)), n1,n2,n3,&
                                  mbs1,mbs2,mbs3, n,nsx,rscut,dgrid,sigma,rpower2,rpower3,x1)
                call fget_spectrum(nas(j),zs(ias1(j):ias2(j)),coords(:,ias1(j):ias2(j)), n1,n2,n3, &
                                  mbs1,mbs2,mbs3, n,nsx,rscut,dgrid,sigma,rpower2,rpower3,x2)
                call fget_kij(n,x1,x2, nco,kwds, kijs)
                k1(:,i,j) = kijs
                k1(:,j,i) = kijs
            enddo
        enddo
        !$OMP end parallel do
    endif
    deallocate( xs1,xs2, x1,x2, nul )
end subroutine


subroutine fget_mk2(local,nm1,nm,na,nas,zs,coords,nbmax,nbrs,n1,n2,n3,mbs1,mbs2,mbs3,&
                    rscut,dgrid,sigma, w2,rpower2,w3,rpower3, nco,kwds, k2)
    ! 
    ! K between test & training molecules
    ! 
    logical, intent(in) :: local    
    integer, intent(in) :: nm1,nm,na, nbmax
    integer, intent(in) :: nas(nm),zs(na), nbrs(nbmax,na)
    integer, intent(in) :: n1,n2,n3
    real*8, intent(in) :: coords(3,na), rscut(2),dgrid,sigma, kwds(nco)
    integer, intent(in) :: mbs1(n1),mbs2(2,n2),mbs3(3,n3), nco
    real*8, intent(in) :: rpower2,rpower3,w2,w3
    !character*1, intent(in) :: kernel  !! only gaussian kernel is allowed
    real*8, intent(out) :: k2(nco,nm-nm1,nm1)
    real*8, allocatable :: xs1(:,:),xs2(:,:),x1(:),x2(:), nul(:,:,:)
    integer :: i,j, n,nu, im, namax, nsx(3)
    integer :: amap(na), ias1(nm),ias2(nm)
    real*8 :: kijs(nco)

    ! get ias1,ias2
    do i = 1,nm
        ias2(i) = sum(nas(1:i))
    enddo
    ias1(1) = 1
    ias1(2:nm) = ias2(1:nm-1)+1

    ! get amap: atomic idx --> mol idx
    amap(:) = 0
    do i = 1,nm
        amap(ias1(i):ias2(i)) = i
    enddo
    print *, '##1'

    ! setup grid
    nsx = 0
    nsx(1) = int((rscut(1) - r0)/dgrid) + 1
    !d2r = pi/dble(180) ! degree to rad
    !a0 = -20.0*d2r
    !a1 = pi + 20.0*d2r
    nsx(3) = int((pi + 40.0*(pi/dble(180)))/dgrid) + 1
    n = n1 + n2*nsx(1) + n3*nsx(3)
    nu = n1 + n2*nsx(2) + n3*nsx(3)

    namax = maxval(nas)
    allocate( xs1(n,namax),xs2(n,namax), x1(n),x2(n), nul(nu,nbmax,namax) )

    xs1(:,:) = 0.0
    xs2(:,:) = 0.0
    x1(:) = 0.0
    x2(:) = 0.0
    print *, '##2'
    if (local) then
        !!$OMP parallel do private(i,j,kijs,xs1,xs2)
        do i = 1,nm1
            do j = nm1+1,nm
                print *, ' i,j = ', i,j
                call fget_local_spectrum(nas(i),zs(ias1(i):ias2(i)),coords(:,ias1(i):ias2(i)), &
                                  nbmax, nbrs(:,ias1(i):ias2(i)), n,nu,n1,n2,n3,mbs1,mbs2,mbs3, nsx,&
                                  rscut,dgrid,sigma,w2,rpower2,w3,rpower3, 0, -1, xs1(:,1:nas(i)), nul(:,:,1:nas(i)))
                print *, '++2'
                call fget_local_spectrum(nas(j),zs(ias1(j):ias2(j)),coords(:,ias1(j):ias2(j)), &
                                  nbmax, nbrs(:,ias1(j):ias2(j)), n,nu,n1,n2,n3,mbs1,mbs2,mbs3, nsx,&
                                  rscut,dgrid,sigma,w2,rpower2,w3,rpower3, 0, -1, xs2(:,1:nas(j)), nul(:,:,1:nas(i)))
                print *, '++3'
                call fget_local_kij(nas(i),nas(j),n,xs1(:,1:nas(i)),xs2(:,1:nas(j)), nco,kwds, kijs)
                print *, '++4'
                k2(:,j,i) = kijs
            enddo
        enddo
        !!$OMP end parallel do
    else
        !$OMP parallel do private(i,j,kijs,x1,x2)
        do i = 1,nm1
            do j = nm1+1,nm
                call fget_spectrum(nas(i),zs(ias1(i):ias2(i)),coords(:,ias1(i):ias2(i)), &
                            n1,n2,n3,mbs1,mbs2,mbs3, n,nsx,rscut,dgrid,sigma,rpower2,rpower3,x1)
                call fget_spectrum(nas(j),zs(ias1(j):ias2(j)),coords(:,ias1(j):ias2(j)), &
                                  n1,n2,n3,mbs1,mbs2,mbs3, n,nsx,rscut,dgrid,sigma,rpower2,rpower3,x2)
                call fget_kij(n,x1,x2, nco,kwds, kijs)
                k2(:,j,i) = kijs
            enddo
        enddo
        !$OMP end parallel do
    endif
    deallocate( xs1,xs2, x1,x2, nul )
end subroutine


subroutine fget_dij_max(local,nm,na,nas,zs,coords,nbmax,nbrs,n1,n2,n3,mbs1,mbs2,mbs3,&
                    rscut,dgrid,sigma, w2,rpower2,w3,rpower3, dmax)
    ! 
    ! K for training set
    !
    ! Calculate molecular kernel matrix K (upper case), to be distinguished with
    ! lower case k, characterizing similarity between atoms.
    ! The i-th and j-th entry of K is 
    !        K(i,j) = \sum_{I \in i} \sum_{J \in j} k(I,J)
    ! 
    logical, intent(in) :: local    
    integer, intent(in) :: nm,na, nbmax
    integer, intent(in) :: nas(nm),zs(na)
    integer, intent(in) :: n1,n2,n3, nbrs(nbmax,na)
    real*8, intent(in) :: coords(3,na), rscut(2),dgrid,sigma
    integer, intent(in) :: mbs1(n1),mbs2(2,n2),mbs3(3,n3)
    real*8, intent(in) :: w2,rpower2,w3,rpower3
    !character*1, intent(in) :: kernel
    real*8, intent(out) :: dmax
    real*8, allocatable :: xs1(:,:),xs2(:,:),x1(:),x2(:), &
                           ds(:,:),dsij(:,:), nul(:,:,:)
    integer :: i,j, iar, im,iab,iae, namax, nsx(3)
    integer :: n,nu
    integer :: pair(2),iasr(2), amap(na), ias1(nm),ias2(nm)

    ! get ias1,ias2
    do i = 1,nm
        ias2(i) = sum(nas(1:i))
    enddo
    ias1(1) = 1
    ias1(2:nm) = ias2(1:nm-1)+1

    ! get amap: atomic idx --> mol idx
    amap(:) = 0
    do i = 1,nm
        amap(ias1(i):ias2(i)) = i
    enddo

    ! setup grid
    nsx = 0
    nsx(1) = int((rscut(1) - r0)/dgrid) + 1
    !d2r = pi/dble(180) ! degree to rad
    !a0 = -20.0*d2r
    !a1 = pi + 20.0*d2r
    nsx(3) = int((pi + 40.0*(pi/dble(180)))/dgrid) + 1
    n = n1 + n2*nsx(1) + n3*nsx(3)
    nu = n1 + n2*nsx(2) + n3*nsx(3)

    namax = maxval(nas)
    allocate( xs1(n,namax),xs2(n,namax), x1(n),x2(n), nul(nu,nbmax,namax) )
    allocate( ds(nm,nm),dsij(namax,namax) )
    xs1(:,:) = 0.0
    xs2(:,:) = 0.0
    x1(:) = 0.0
    x2(:) = 0.0
    if (local) then
        !$OMP parallel do private(i,j,xs1,xs2,dsij)
        do i = 1,nm
            do j = i,nm
                call fget_local_spectrum(nas(i),zs(ias1(i):ias2(i)),coords(:,ias1(i):ias2(i)), &
                                  nbmax,nbrs(:,ias1(i):ias2(i)), n,nu,n1,n2,n3,mbs1,mbs2,mbs3, nsx, &
                                  rscut,dgrid,sigma,w2,rpower2,w3,rpower3, 0, -1, xs1(:,1:nas(i)), nul(:,:,1:nas(i)))
                call fget_local_spectrum(nas(j),zs(ias1(j):ias2(j)),coords(:,ias1(j):ias2(j)), &
                                  nbmax,nbrs(:,ias1(i):ias2(i)), n,nu,n1,n2,n3,mbs1,mbs2,mbs3, nsx, &
                                  rscut,dgrid,sigma,w2,rpower2,w3,rpower3, 0, -1, xs2(:,1:nas(j)), nul(:,:,1:nas(i)))
                call fget_local_dij(nas(i),nas(j),n,xs1(:,1:nas(i)),xs2(:,1:nas(j)), dsij(1:nas(i),1:nas(j)))
                ds(i,j) = maxval(dsij)
            enddo
        enddo
        !$OMP end parallel do
    else
        !$OMP parallel do private(i,j,x1,x2)
        do i = 1,nm
            do j = i,nm
                call fget_spectrum(nas(i),zs(ias1(i):ias2(i)),coords(:,ias1(i):ias2(i)), n1,n2,n3,&
                                  mbs1,mbs2,mbs3, n,nsx,rscut,dgrid,sigma,rpower2,rpower3,x1)
                call fget_spectrum(nas(j),zs(ias1(j):ias2(j)),coords(:,ias1(j):ias2(j)), n1,n2,n3, &
                                  mbs1,mbs2,mbs3, n,nsx,rscut,dgrid,sigma,rpower2,rpower3,x2)
                ds(i,j) = norm2(x1-x2)
            enddo
        enddo
        !$OMP end parallel do
    endif
    dmax = maxval(ds)
    deallocate( xs1,xs2, x1,x2, nul, dsij,ds )
end subroutine


end module


