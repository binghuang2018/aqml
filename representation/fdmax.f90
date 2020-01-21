!
! get the maximal value of distance matrix
! for a given representation
!

subroutine fget_ij_dmax(local,nm,na,nas,zs,coords,rpower2,pair)
    ! 
    ! get the two atomic idx I, J whose distance is maximal among all
    ! possible pairs of atoms
    !
    implicit none
    logical, intent(in) :: local
    integer, intent(in) :: nm,na
    integer, intent(in) :: nas(nm),zs(na)
    real*8, intent(in) :: coords(3,na),rpower2
    integer, intent(out) :: pair(2) ! final pair of atomic/molecular idx
    integer :: idxf,jdxf 
    integer :: i,j
    integer, allocatable :: zsu(:) ! unique zs
    integer :: zmax, nzu, ias1(nm),ias2(nm), idx1(nm),idx2(nm)
    !integer :: amap(na)
    real*8 :: v1min,v2max,v1s(nm),v2s(nm)

    ! get ias1,ias2
    do i = 1,nm
        ias2(i) = sum(nas(1:i))
    enddo
    ias1(1) = 1
    ias1(2:nm) = ias2(1:nm-1)+1

    zmax = maxval(zs)
    nzu = 0
    allocate( zsu(zmax) )
    do i = 1,zmax
        if (any(i .eq. zs)) then
            zsu(nzu+1) = i
            nzu = nzu + 1
        endif
    enddo

    ! get atom I (or molecule I) with \sum_J 1/R_IJ**6 being the smallest
    !$OMP parallel do private(i)
    do i = 1,nm
        call fget_l2(local,i,ias1(i),nas(i),zs(ias1(i):ias2(i)),coords(:,ias1(i):ias2(i)), &
                      rpower2, v1s(i),idx1(i),v2s(i),idx2(i))
    enddo
    !$OMP end parallel do

    v1min = minval(v1s)
    do i = 1,nm
        if (v1min .eq. v1s(i)) then
            idxf = idx1(i)
            exit
        endif
    enddo
    v2max = maxval(v2s)
    do j = 1,nm
        if (v2max .eq. v2s(j)) then
            jdxf = idx2(j)
            exit
        endif
    enddo
    pair(1) = idxf
    pair(2) = jdxf
end subroutine


subroutine fget_l2(local,im0,ia0,na,zs,coords,rpower2,vmin,imin,vmax,imax)
    implicit none
    integer, intent(in) :: im0,ia0,na
    integer, intent(in) :: zs(na)
    real*8, intent(in) :: coords(3,na), rpower2
    integer, intent(out) :: imin,imax
    real*8, intent(out) :: vmin,vmax
    logical, intent(in) :: local
    real*8 :: vs(na),v, ds(na,na)
    integer :: i,j

    ds(:,:) = 0.0
    ds = calc_dist(na,coords)

    if (local) then
        do i = 1,na
            vs(i) = zs(i)
            do j = 1,na
                if (j.ne.i) vs(i) = vs(i) + zs(i)*zs(j)/ds(i,j)**rpower2
            enddo
        enddo
        vmin = minval(vs)
        vmax = maxval(vs)
        do i = 1,na
            if (vmin .eq. vs(i)) then
                imin = i+ia0
                exit
            endif
        enddo
        do j = 1,na
            if (vmax .eq. vs(j)) then
                imax = j+ia0
                exit
            endif
        enddo
    else
        v = 0.0
        do i = 1,na
            do j = 1,na
                if (i.eq.j) then
                    v = v + zs(i)
                else
                    v = v + zs(i)*zs(j)/ds(i,j)**rpower2
                endif
            enddo
        enddo
        vmin = v
        vmax = v
        imin = im0
        imax = im0
    endif
end subroutine


