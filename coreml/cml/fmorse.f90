
subroutine calc_grids(x0, x1, nx, xs)

    implicit none
    double precision, intent(in) :: x0,x1
    integer, intent(in) :: nx
    double precision, intent(out), dimension(nx) :: xs
    integer :: i
    double precision :: step

    step = (x1 - x0) / (nx - 1)
    !!$OMP PARALLEL DO
    do i = 1, nx
        xs(i) = x0 + (i - 1) * step
    enddo
    !!$OMP END PARALLEL DO
end subroutine


subroutine calc_eij_morse(rij, param, eij)

    implicit none
    real*8, dimension(3), intent(in) :: param
    real*8, intent(in) :: rij
    real*8, intent(out) :: eij
    real*8 :: t
    t = exp(-param(2)*(rij-param(3)))
    eij = param(1) * (t*t - 2*t)
end subroutine


subroutine calc_total_eij_morse(nm, nb, nbt, x, y, param, dys)

    implicit none
    integer, intent(in) :: nm, nb, nbt
    real*8, intent(in) :: x(nb,nm), y(nm), param(3,nbt)
    real*8, intent(out) :: dys(nm) ! total 2-body Morse energies
    integer :: i,j,k,jb,je, nav
    real*8 :: e2, t

    nav = nb/nbt
    if (nb - nav*nbt .ne. 0) then
        stop "#ERROR: nb != nbt*nav"
    endif

    dys(:) = 0.0
    do i = 1, nm
        e2 = 0.0
        do j = 1, nbt
            jb = (j-1)*nav + 1
            je = j*nav
            do k = jb, je
                t = exp(-param(2,j)*(x(k,i)-param(3,j)))
                e2 = e2 + param(1,j)*(t*t - 2*t)
            enddo
        enddo
        dys(i) = e2 - y(i)
    enddo

end subroutine



subroutine calc_ediff_morse(nm, nb, nbt, x, y, param, mae, rmse)

    implicit none
    integer, intent(in) :: nm, nb, nbt
    real*8, intent(in) :: x(nb,nm), y(nm), param(3,nbt)
    real*8, intent(out) :: mae, rmse
    integer :: i,j,k,jb,je, nav
    real*8 :: e2, t, dys(nm)

    nav = nb/nbt
    if (nb - nav*nbt .ne. 0) then
        stop "#ERROR: nb != nbt*nav"
    endif

    dys(:) = 0.0
    do i = 1, nm
        e2 = 0.0
        do j = 1, nbt
            jb = (j-1)*nav + 1
            je = j*nav
            do k = jb, je
                t = exp(-param(2,j)*(x(k,i)-param(3,j)))
                e2 = e2 + param(1,j)*(t*t - 2*t)
            enddo
        enddo
        dys(i) = e2 - y(i)
    enddo
    rmse = sqrt(sum(dys*dys))
    mae = sum(abs(dys))/nm

end subroutine



subroutine fscan_pramas_morse(nm, nb, nbt, x, y, npts, ranges, cs)
    implicit none
    integer, intent(in) :: nm, nb, nbt
    integer, intent(in) :: npts(3) ! npt of a, b, c
    real*8, intent(in) :: x(nb,nm), y(nm), ranges(2,3), cs(nbt)
    real*8 :: es2(nm) ! total 2-body Morse energies
    integer :: i,j,k,ib,ie, idx(4), nav, ia1,ia2,ib1,ib2
    real*8 :: e2, t, a0,a1, b0,b1, c0,c1, param(3,nbt), mae,rmse
    real*8, allocatable :: xsa(:),xsb(:),xsc(:),maes(:,:,:,:),rmses(:,:,:,:)
    
    nav = nb/nbt
    if (nb - nav*nbt .ne. 0) then
        stop "#ERROR: nb != nbt*nav"
    endif

    allocate( xsa(npts(1)), xsb(npts(2)), xsc(npts(3)), &
                maes(npts(2),npts(1),npts(2),npts(1)), &
                rmses(npts(2),npts(1),npts(2),npts(1)) )
    a0 = ranges(1,1) !1.0
    a1 = ranges(2,1) !300.0
    call calc_grids(a0,a1,npts(1),xsa) 
    b0 = ranges(1,2) !1.0
    b1 = ranges(2,2) !300.0
    call calc_grids(b0,b1,npts(2),xsb) 

    if ( npts(3) .eq. 1 ) then
        if ( all(cs .eq. 0) ) then
            stop "#ERROR: please specify non-zero param `cs"
        endif
    else
        c0 = ranges(1,3) !1.0
        c1 = ranges(2,3) !300.0
        call calc_grids(c0,c1,npts(3),xsc) 
    endif


    if ( nbt .eq. 1 ) then
        write(*,*) 'todo: '
    elseif ( nbt .eq. 2 ) then        ! reduction(max: rmse) 
        !$OMP PARALLEL DO PRIVATE(param,mae,rmse)
        do ia1 = 1, npts(1)
            do ib1 = 1, npts(2)
                do ia2 = 1, npts(1)
                    do ib2 = 1, npts(2)
                        param = reshape( (/ xsa(ia1),xsb(ib1),cs(1), xsa(ia2),xsb(ib2),cs(2) /), shape(param))
                        call calc_ediff_morse(nm, nb, nbt, x, y, param, mae, rmse)
                        maes(ib2,ia2,ib1,ia1) = mae
                        rmses(ib2,ia2,ib1,ia1) = rmse
                    enddo
                enddo
            enddo
        enddo
        !$OMP END PARALLEL DO
        idx = minloc(rmses)
        write(*,*) 'a1,b1,a2,b2 = ', xsa(idx(4)),xsb(idx(3)),xsa(idx(2)),xsb(idx(1))
    else
        stop "#ERROR: todo"
    endif
    deallocate( xsa,xsb,xsc, maes,rmses )

end subroutine



