
subroutine iargsort(N, a, b, option)
    ! Returns the indices that would sort an array.
    integer, intent(in):: N, option
    integer, intent(in) :: a(N)  ! array of numbers
    integer, intent(out) :: b(N) ! indices into the array 'a' that sort it
    !
    ! Example
    ! iargsort([10, 9, 8, 7, 6])   ! Returns [5, 4, 3, 2, 1]

    integer :: i,imin                      ! indices: i, i of smallest
    integer :: temp                        ! temporary
    integer :: n1, a2(N), t(N), ic
    a2 = a
    do i = 1, N
        b(i) = i
    end do
    do i = 1, N-1
        ! find ith smallest in 'a'
        imin = minloc(a2(i:),1) + i - 1

        ! swap to position i in 'a' and 'b', if not already there
        if (imin /= i) then
            temp = a2(i)
            a2(i) = a2(imin)
            a2(imin) = temp
            temp = b(i)
            b(i) = b(imin)
            b(imin) = temp
        end if
    end do
    
    ! 0: do nothing
    ! 1: ascending & (-1 at end), 2: descending & (-1 pushed to end)
    if (option .eq. 2) then
        t(:) = b(:)
        do i = 0,N-1
            b(i+1) = t(N-i)
        enddo
    endif
end subroutine


subroutine get_next(n, g, ias0, nn)
    !
    ! get next shell of atoms connected to atoms in `ias0
    !
    implicit none
    integer, intent(in) :: n
    integer, intent(in) :: g(n,n)
    integer, intent(inout) :: ias0(n)
    integer, intent(out) :: nn
    integer :: i,i0,j, na0, ias(n)
    if ( all(ias0 .lt. 0) ) stop "#ERROR: empty `ias0?"
    na0 = count( ias0.gt.0 )
    if ( any(ias0(1:na0).lt.0) ) stop "#ERROR: positive value shown elsewhere?"
    !print *, '     na0 = ', na0
    !print *, 'g = '
    !do i0 = 1,n; print *, g(i0,:); enddo
    ias(:) = -1
    nn = na0
    do i0 = 1,na0
        i = ias0(i0)
        do j = 1,n
            if ( g(i,j).gt.0 .and. all(ias0.ne.j) ) then
                nn = nn + 1
                ias0(nn) = j
                !print *, '  ** i,j, nn = ', i,j, nn
                !print *, '     ias0 = ', ias0( 1:nn )
            endif
        enddo
    enddo
    !print *, ' -- ias0 = ', ias0( 1:count(ias0.gt.0) )
end subroutine

subroutine connected_components(n0, nrmax, g, nc, iasc, icon)
    !
    ! get num_components & nodes of each component
    !
    implicit none 
    integer, intent(in) :: n0, nrmax
    integer, intent(in) :: g(n0,n0)
    logical, intent(in) :: icon
    ! assume at most 6 components
    integer, intent(out) :: nc,iasc(n0,nrmax)

    integer :: iast(n0),ia,i,j, n,n1, nn,nn1,nn2, ias0(n0)
    integer, allocatable :: t(:)
    logical :: iok
    iasc(:,:) = -1
    iast(:) = -1
    nc = 0
    ia = 0
    iok = .false.

    ! calc the num of atoms `n (if `n0 > `n, some of `g is filled w -1)
    n = 0
    do i = 1,n0
        if ( any(g(i,:).ge.0) ) n = n + 1
    enddo
    !n = int( sqrt( real(count(g.ge.0)) ) )
    !print *, ' n = ', n

    do i = 1,n
        ! `i may have been visited
        !print *, ' i, n = ', i, n
        !print *, '       iast = ', iast
        if (any(i.eq.iast)) cycle

        ! re-initialize `ias0 as it's only used temperarily
        ias0(:) = -1
        ! re-initialize the first element
        ias0(1) = i

        ! the num of elements in `ias0
        nn1 = 1
        !print *, ''
        !print *, ' a) ias0 = ', ias0
        ! update `nn, i.e., nn2
        !print *, '     *** g2 = '
        !do j=1,n0; print *, g(j,:); enddo
        call get_next(n0,g,ias0,nn2)
        !print *, '         nn2 = ', nn2
        !print *, ' b) ias0 = ', ias0

        ! if no new neighbors can be found, nn2 .eq. nn1
        ! and stop the do while loop
        do while (nn1.lt.nn2)
            nn1 = nn2
            call get_next(n0,g,ias0,nn2)
            !print *, 'ias0 = ', ias0
        enddo
        !print *, 'ias0 = ', ias0

        ! update num of components
        nc = nc + 1

        ! assign temp `ias0
        ! 1: ascending, 2: descending
        n1 = count(ias0.gt.-1)
        allocate( t(n1) )
        call iargsort(n1, ias0(1:n1), t, 1) 
        do j = 1,n1
            iasc(j,nc) = ias0(t(j))
        enddo
        deallocate( t )
        !print *, ' nc, iasc(:,nc) = ', nc, iasc(:,nc)

        ! update `iast
        nn = count( ias0.gt.0 )
        iast(ia+1:ia+nn) = ias0(1:nn)
        ia = ia + nn
        !print *, ' _____ i,n,ia = ',i, n, ia
        !print *, ' _____ iast = ', iast
        if ( ia.eq.n ) then
            iok = .true.
            exit
        endif
    enddo

! you may encounter case like H2C-C(-O)-CH2,
! which has two components after rationalization.
! Even in this case, u have `ia .eq. n
    if ( .not.iok ) then
        !print *, 'ia, n = ', ia, n
        stop "#ERROR: not all atoms exhausted?"
    endif
    !print *, '  done!'
end subroutine


subroutine subarray(n,ias,mask)
    !
    ! retrieve a subset of elements from an 
    ! array based on conditional var `mask
    !
    implicit none
    integer, intent(in) :: n
    integer, intent(inout) :: ias(n)
    logical, intent(in) :: mask(n)
    integer :: i, icnt, t(n)
    t(:) = -1
    icnt = 0
    do i = 1,n
        if ( mask(i) .and. ias(i).gt.0 ) then
            icnt = icnt + 1
            t(icnt) = ias(i)
        endif
    enddo
    ias(:) = t(:)
end subroutine


subroutine subarray2d(na, g, ias, g2)
    !
    ! retrieve a subset of elements from an 
    ! array based on idxs specified by `ias
    !
    implicit none
    integer, intent(in) :: na
    integer, intent(in) :: g(na,na), ias(na)
    integer, intent(inout) :: g2(na,na)
    integer :: i,j,boij, na2
    na2 = count(ias.gt.0)
    do i = 1,na2-1
        do j = i+1,na2
            boij = g(ias(i),ias(j))
            g2(i,j) = boij
            g2(j,i) = boij
        enddo
    enddo
end subroutine


subroutine remove(n, ias, ia)
    implicit none 
    integer, intent(in) :: ia, n
    integer, intent(inout) :: ias(n)
    integer :: i, icnt, t(size(ias,1))
    !if ( count(ias.eq.ia) .eq. 0 ) stop "#ERROR: `ia not in `ias?"
    t(:) = -1
    icnt = 0
    do i = 1,n
        if ( ia.ne.ias(i) .and. ias(i).gt.0 ) then
            icnt = icnt+1
            t(icnt) = ias(i)
        endif
    enddo
    ias(:) = t(:)
end subroutine


subroutine remove2d(n, ips, ip)
    implicit none 
    integer, intent(in) :: n
    integer, intent(inout) :: ips(2,n)
    integer, intent(in) :: ip(2)
    integer :: i, icnt, t(2,n)
    t(:,:) = -1
    icnt = 0
    do i = 1,n
        if ( any(ip.ne.ips(:,i)) .and. all(ips(:,i).gt.0) ) then
            icnt = icnt+1
            t(:,icnt) = ips(:,i)
        endif
    enddo
    if (icnt .eq. n) stop "#ERROR: `ip not in `ips??"
    ips = t
end subroutine


subroutine update_bom(n, nrmax, nbmax, zs, tvs, g0, iok, bom, icon)

    ! heavy atoms only
    ! notes:
    !   bom -- input & output
    !   pls -- path length matrix

    implicit none 
    integer, intent(in) :: nrmax, nbmax ! max_num_conjugated_subg, max_num_bonds
    integer, intent(in) :: n
    integer, intent(in) :: zs(n), tvs(n)
    integer, intent(in) :: g0(n,n)
    integer, intent(out) :: bom(n,n)

! for debug purpose, i.e., if some criterio is met in
! the python script, then we print relevant info below
    logical, intent(in) :: icon

    logical, intent(out) :: iok ! is the update sucessful? It may be a radical
    integer :: i,j,k, bo1,bo2, dvi,zi, ias(n), iasp(n),vec(n), &
               vs(n), dvs(n)

    integer :: cni,ia2,ja2, na1,ias1(n),na2,ias2(n), ie
    integer :: cns0(n), g2(n,n), cns2(n), ias2c_rel(n,nrmax), &
               ias2c(n), ias_compl(n), &
               ias_compl_copy(n)
    integer :: nc2,na2c, ias1c(n),na1c, iasc(n), icnt, icount
    integer :: ia,ia1,ja,l,ial,nal,nal2,nalc, iax,ka1,ka2, iac,nac
    logical :: icontinue, hasa1

    integer :: nclique,dv,BO,ias_dv(n),na_dv
    logical :: mask(n)
    integer :: nc_dv ! num_components_dv
    integer :: cliques_dv(n,nrmax), & 
               ipss(2,nbmax,nrmax), ipss_ir(2,nbmax,nrmax), g2c(n,n)
    integer :: nr, np,np0,ip, ic, kt, boij
    integer :: ats1(n),n1,ats2(n),n2, cnti,j0, t(n)

    iok = .true. ! assume ok for all

    ! coord_num
    cns0 = sum(g0,dim=1)

    bom(:,:) = g0(:,:)
    do i = 1,n
        ias(i) = i
    enddo

    vs = sum(bom, dim=1)
    dvs = tvs - vs

!   ! fix =O, =S
!   do i = 1,n
!       dvi = dvs(i)
!       zi = zs(i)
!       cni = cns0(i)
!       ! for =O, =S
!       if (dvi.eq.1 .and. cni.eq.1 .and. (zi.eq.8 .or. zi.eq.16)) then
!           do j = 1,n
!               if (g0(i,j) .gt. 0) then
!                   bom(i,j) = 2
!                   bom(j,i) = 2
!                   exit ! exit the inner-most do loop
!               endif
!           enddo
!       endif
!   enddo
!   dvs = tvs - sum(bom,dim=1)
!   if ( all(dvs.eq.0) ) return

!   ! fix -C#N, -C#P, -C=N#N, -C=P#P
!   do i = 1,n
!       dvi = dvs(i)
!       zi = zs(i)
!       cni = cns0(i)
!       ! for #N
!       if (dvi.eq.2 .and. cni.eq.1 .and. (zi.eq.7 .or. zi.eq.15)) then
!           do j = 1,n
!               if (g0(i,j) .gt. 0) then
!                   bom(i,j) = 3
!                   bom(j,i) = 3
!                   exit
!               endif
!           enddo
!       endif
!   enddo
!   dvs = tvs - sum(bom,dim=1)
!   if ( all(dvs.eq.0) ) return

    ! fix -N$C
    do i = 1,n
        dvi = dvs(i)
        zi = zs(i)
        cni = cns0(i)
        ! for $C
        if (dvi.eq.3 .and. cni.eq.1 .and. zi.eq.6) then
            do j = 1,n
                if (g0(i,j) .gt. 0) then
                    bom(i,j) = 4
                    bom(j,i) = 4
                    exit
                endif                        
            enddo
        endif
    enddo
    dvs = tvs - sum(bom,dim=1)
    vs = sum(bom, dim=1) 
    if ( all(dvs.eq.0) ) return

!   if (icon) print *, '  ------------ vs = ', vs

    ! fix P2 in  R-C1#P2=P3-R
!   vec(:) = -1
!   do i = 1,n
!       dvi = dvs(i)
!       zi = zs(i)
!       cni = cns0(i)
!       if ( dvi.eq.3 .and. zi.eq.15 .and. cni.eq.2 ) then
!           ! if there are two neighbors with dvi=2 & 3, 
!           ! respectively, then we r in.
!           iasP = pack(ias, mask= g0(:,i).gt.0, vector=vec) !dvs*merge(1,0,g0(:,i)>0)
!           !if (icon) print *, '   *** i, iasP = ', i, ',', iasP
!           if (dvs(iasP(1)).eq.1 .and. dvs(iasP(2)).eq.2) then
!               bom(i,iasP(1)) = 2
!               bom(iasP(1),i) = 2
!               bom(i,iasP(2)) = 3
!               bom(iasP(2),i) = 3
!           elseif (dvs(iasP(1)).eq.2 .and. dvs(iasP(2)).eq.1) then
!               bom(i,iasP(1)) = 3
!               bom(iasP(1),i) = 3
!               bom(i,iasP(2)) = 2
!               bom(iasP(2),i) = 2
!           endif
!           vs = sum(bom,dim=1)
!           !if (icon) print *, '   *** vs = ', vs
!           dvs = tvs - vs
!           !if (icon) print *, '   *** dvs = ', dvs 
!       endif
!   enddo
!   if ( all(dvs.eq.0) ) return


! test
    if (icon) print *, '  ##1'


    if (icon) then 
       i = count(zs.gt.1)
       print *, ' -- zs = ', zs(1:i)
       print *, ' --  dvs = ', dvs(1:i)
       print *, ' -- vs = ', vs(1:i)
    endif

    if ( all(dvs.eq.0) ) return


! *************************************************************************
!        New algorithm: tackle standalone atoms with dvi = 2
! *************************************************************************

    do dvi = 1, 2

        vec(:) = -1
        ats2 = pack(ias, dvs.eq.dvi, vec)
        n1 = count(ats2.gt.0)

        ats1(:) = -1

        do while (.true.)

            if ( n1.eq.0 .or. all(ats1.eq.ats2) ) exit

            ! find atoms satisfying
            ! 1) dvi = 1 (or 2) 
            ! 2) there exists only one neighbor with dvi >= 1
            do i = 1, n
                if (dvs(i) .eq. dvi) then
                    cnti = 0
                    j0 = 0
                    do j = 1, n
                        if (g0(i,j).gt.0 .and. dvs(j).ge.dvi) then
                            cnti = cnti + 1
                            j0 = j
                        endif
                    enddo
                    if (cnti .eq. 1) then
                        bom(i,j0) = dvi+1
                        bom(j0,i) = dvi+1
                    endif
                endif
                vs = sum(bom,dim=1)
                dvs = tvs - vs
            enddo

            vec(:) = -1
            t = pack(ias, dvs.eq.dvi, vec)
            ats1(:) = ats2(:)
            ats2 = t
            n1 = count(ats2.gt.0)
        enddo
    enddo

    if ( all(dvs.eq.0) ) return
!   print *, ' KO'

! **************************************************************************
!    assign triple bond to standalone atom (and its neighbor) with dvi=2
!    By `standalone, I mean that the subgraph consisting of atoms all with
!    dvi = 2
! **************************************************************************
!   na1 = 0
!   na2 = 0
!   ias1(:) = -1
!   ias2(:) = -1
!   do i = 1,n
!       dvi = dvs(i)
!       if ( dvi .eq. 1 ) then
!           na1 = na1 + 1
!           ias1(na1) = ias(i)
!       elseif ( dvi.eq.2) then
!           na2 = na2 + 1
!           ias2(na2) = ias(i)
!       endif
!   enddo
!   !if (icon) print *, ' #1'

!   !test 
!   !if (icon) then
!   !    print *, '    ias2 = ', ias2
!   !endif

!   ! ** added on Aug 13, 2018 @start_2
!   ! in the case >C1-C2-S3(-H)(-C7-C8H)-C4-C5<, 
!   ! ias2 = [2,3,4,5,7,8], which don't form a
!   ! chain of double bonds; This is due to the 
!   ! fact that the two atoms in -C7-C8H with dvi=2
!   ! have no neighboring atoms with dvi=1. Thus, we
!   ! process these two atoms first.
!   cns2(:) = -1
!   if ( na2 .gt. 0 ) then
!       g2(:,:) = -1
!       g2(1:na2,1:na2) = 0
!       call subarray2d(n,g0,ias2,g2)
!       cns2(1:na2) = sum(g2(1:na2,1:na2),dim=1)

!       !if (icon) then
!       !    print *, ' cns2 = ', cns2(1:na2)
!       !endif

!       ! add triple bond to any standalone atom (and its neighbor) with
!       ! dvi = 2, e.g., C3 & C4 in ">C1=C2(-R)-C3#C4-R"
!       do ia2 = 1,na2
!           if (cns2(ia2).eq.1) then   
!               ! find out if there exists a neighboring atom with dvi .eq. 1
!               hasa1 = .false.
!               do ia1 = 1,na1
!                   if ( g0(ias2(ia2),ias1(ia1)).eq.1 ) then
!                       hasa1 = .true.
!                       exit
!                   endif
!               enddo
!               !if (icon) print *, ' hasa1 = ', hasa1, ', na1,na2 = ', na1,na2
!               if (hasa1.eqv..false.) then
!                   do ja2 = 1,na2
!                       i = ias2(ia2)
!                       j = ias2(ja2)
!                       !if (icon) print *, '               i,j  = ', i,j
!                       if ( g0(i,j).gt.0 ) then
!                           if (icon) print *, ' ++ Gotcha, ', i, j
!                           bom(i,j) = 3
!                           bom(j,i) = 3
!                       endif
!                   enddo
!               endif
!           endif
!       enddo
!   endif
!   dvs = tvs - sum(bom,dim=1)
!   vs = sum(bom, dim=1) 
!   if ( all(dvs.eq.0) ) return


! **************************************************************************
!      work around structure like >C-C-C-C<, i.e., dvs = [1,2,...2,1]
! **************************************************************************
!   na1 = 0
!   na2 = 0
!   ias1(:) = -1
!   ias2(:) = -1
!   do i = 1,n
!       dvi = dvs(i)
!       if ( dvi .eq. 1 ) then
!           na1 = na1 + 1
!           ias1(na1) = ias(i)
!       elseif ( dvi.eq.2) then
!           na2 = na2 + 1
!           ias2(na2) = ias(i)
!       endif
!   enddo


!   ! na2 -- number of atoms with dvi .eq. 2
!   cns2(:) = -1
!   if ( na2 .gt. 0 ) then
!       g2(:,:) = -1
!       g2(1:na2,1:na2) = 0
!       ias2c_rel(:,:) = -1

!       ias_compl(:) = -1
!       ias_compl_copy(:) = -1
!       call subarray2d(n,g0,ias2,g2)
!       cns2(1:na2) = sum(g2(1:na2,1:na2),dim=1)

!       !print *, ' #2.-5'
!       !do i = 1,na2; print *, g2(i,1:na2); enddo         ! #####
!       if (na2.eq.1) then
!           nc2 = 1
!           ias2c_rel(1,1) = 1
!       else
!           call connected_components(n, nrmax, g2, nc2, ias2c_rel, icon) 
!       endif

!       ! ** added on Aug 13, 2018 @start_3
!       ! in the case >C1-C2-S3(-C6X3)(-C7X)-C4-C5<, 
!       ! ias2 = [2,3,7,4], within such a subgraph `g2,
!       ! the coord_num of S3 is 3, cannot form a linear
!       ! chain and thus cannot be converted to conjugated
!       ! chain of double bonds --> abandon
!       do ia2 = 1,na2
!         if ( cns2(ia2) .gt. 2 ) then
!           iok = .false.
!           exit
!         endif
!       enddo
!       if (.not. iok) return
!       ! ** added on Aug 13, 2018 @end_3


!       !print *, ' #2.-4'
!       !print *, 'nc2, ias2c_rel = ', nc2, ',', ias2c_rel         ! #####
!       do ic = 1,nc2
!           !print *, ' ic = ', ic                            ! #####
!           !print *, ' ias2c_rel(:,ic) = ', ias2c_rel(:,ic)
!           na2c = count( ias2c_rel(:,ic).gt.0 )
!           !print *, ' na2c = ', na2c

!           ! initialize `ias2c
!           ias2c(:) = -1
!           ias2c(1:na2c) = ias2(ias2c_rel(1:na2c,ic)) ! now absolute idx
!           !print *, '  ** na2c, ias2c(1:na2c) = ', na2c,',', ias2c(1:na2c)
!           ias1c(:) = -1 ! indices of atoms with dvi=1
!           na1c = 0
!           ! get neighboring atoms with BO=1
!           do i = 1, na2c
!               ia = ias2c(i)
!               do j = 1, na1
!                   ja = ias1(j)
!                   if ( bom(ia,ja) .eq. 1 ) then
!                       na1c = na1c+1
!                       ias1c(na1c) = ja
!                   endif
!               enddo
!           enddo
!           !print *, '  ** na1c, ias1c(1:na1c) = ', na1c, ',', ias1c(1:na1c)  ! #####
!           !print *, ' #2.-3'
!           if ( na1c .eq. 2 ) then 
!               iasc(:) = -1
!               ! now sort atoms to form a linear chain!
!               iasc(1) = ias1c(1)
!               icnt = 1
!               ias_compl(:) = ias2c(:)

!               nal = count(ias_compl.gt.0) 
!               nal2 = 0
!               !print *, ' #2.-3.-5'
!               do while (nal.gt.0) ! .and. nal.gt.nal2)
!                   !!!! explanation of the requirement: `nal.gt.nal2
!                   !!!! e.g., in the case of a subgraph C1-C7-C6-C2-C3-C4-C5
!                   !!!!                      with dvs = 1, 2, 2, 1, 2, 2, 2
!                   !!!! thus, when ias2c = [7,6], ias1c = [1,2], which is ok
!                   !!!! 
!                   !!!!print *, 'nal = ', nal         ! #####
!                   !!!! you need a copy of `ias_compl here
!                   !!!! because `ias_compl is being changed
!                   !!!! dynamically. 
!                   ias_compl_copy(:) = ias2c(:)
!                   nalc = count(ias_compl_copy.gt.0)
!                   do l = 1,nalc
!                       ial = ias_compl_copy(l)
!                       !print *, ' __ l = ', l
!                       !print *, ' __ ial = ', ial
!                       if (bom(ial,iasc(icnt)).eq.1 .and. all(iasc.ne.ial)) then
!                           iasc(icnt+1) = ial
!                           !print *, ' __ iasc = ', iasc
!                           icnt = icnt+1
!                           if (any(ias_compl.eq.ial)) then
!                               call remove(n,ias_compl,ial)
!                           endif
!                           nal = count(ias_compl.gt.0)
!                       endif
!                   enddo
!                   !nal = nal2
!                   !nal2 = count(ias_compl.gt.0)
!                   nal = count(ias_compl.gt.0)
!               enddo
!               !print *, ' #2.-3.-4'
!               nac = icnt + 1
!               iasc(nac) = ias1c(2)
!               !print *, 'nac = ', nac         ! #####
!               !print *, 'iasc = ', iasc         ! #####

!               ! now check if the two end atoms along this Line
!               ! are not connected to another atom with `dv=1,
!               ! # e.g., >C-C(-X)-C-C-C(-X)-C<
!               icontinue = .true.
!               do ia = 1,na1 
!                   !print *, 'ia,na1 = ', ia,na1          ! #####
!                   iax = ias1(ia)
!                   if ( count(ias1c.eq.iax).eq.0 ) then
!                       !print *, ' * yeah 1'         ! #####
!                       if ( any(bom(iax,ias1c).eq.1) ) then
!                           !print *, ' * yeah 2'
!                           icontinue = .false.
!                           exit
!                       endif
!                   endif
!               enddo
!               !print *, ' icontinue = ', icontinue         ! #####
!               if (icontinue) then
!                   do ia = 1,nac-1
!                       ka1 = iasc(ia)
!                       ka2 = iasc(ia+1)
!                       !print *, ' ####### ka1,ka2 = ', ka1,ka2         ! #####
!                       bom(ka1,ka2) = 2
!                       bom(ka2,ka1) = 2
!                   enddo
!               else
!                   !print *, ' ** exit the innermost loop'         ! #####
!                   exit
!               endif
!           !elseif ( na1c .eq. 1 ) then
!           !   ! if there are some atoms with `dv = 2, but only 1
!           !   ! corresponding neighboring heav atom with dv = 1, then
!           !   ! this subgraph is still valid if one of the heav atoms
!           !   ! is at the end of a molecule
!           !   ! e.g., C=CC#C, 
!           !   iok = .false.
!           !   !print *, '       oh no, `na1c = 1'
!           !   !exit
!           endif
!           !print *, ' #2.-2'
!       enddo
!   endif

    vs = sum(bom,dim=1)
    dvs = tvs - vs
    if ( all(dvs.eq.0) ) return

    !if (icon) then
    !    print *, ' -- 2) vs = ', vs           ! #####
    !    print *, ' -- 2)dvs = ', dvs           ! #####
    !endif

    !if (icon) print *, ' #2'

    nclique = 0
    do dv = 1,2
        !print *, ' _______________________________ dv = ', dv
        ! for C_sp2, dv = 4-3 = 1, N_sp2, dv = 3-2 = 1;
        ! for C_sp, dv = 4-2 = 2
        ! for O_sp2, dv = 2-1 = 1
        BO = dv + 1
        ! atoms to be re-bonded after double/triple bonds
        ! are assigned

        mask(:) = .false.
        do i = 1,n
            if ( dvs(i).eq.dv) mask(i) = .true.
        enddo
        if (count(mask).eq.0) cycle

        ias_dv(:) = ias(:)
        call subarray(n,ias_dv,mask)
        na_dv = count( ias_dv.gt.0 )
        !print *, ' ** dv, na_dv, na_dv%2 = ', dv, na_dv, mod(na_dv,2)

        if (mod(na_dv,2) .eq. 1) then
            !print *, '  ++ now exit'
            iok = .false.
            exit
        endif

        !if (.not. allocated(g2)) allocate( g2(na_dv,na_dv) )
        !if (.not. allocated(cliques_dv)) allocate( cliques_dv(na_dv,nrmax) )
        !if ( allocated(g2) ) deallocate(g2, cliques_dv )
        !allocate( g2(na_dv,na_dv), cliques_dv(na_dv,nrmax) )

        g2(:,:) = -1
        do i = 1,na_dv; g2(i,i) = 0; enddo
        call subarray2d(n,g0,ias_dv,g2)
        !print *, 'ias_dv = ', ias_dv
        !print *, ' ## g2 = '
        !do i = 1,na_dv; print *, g2(i,1:na_dv); enddo

        np0 = count(g2.gt.0)/2
        if (np0 .eq. 0) then
            iok = .false.
            exit
        endif
        !print *, 'np0 = ', np0
        !print *, ' ** '

        cliques_dv(:,:) = -1
        call connected_components(n, nrmax, g2, nc_dv, cliques_dv, icon) 
        !print *, 'na_dv, nc_dv = ', na_dv, ',', nc_dv
        !print *, 'cliques_dv = ', cliques_dv(:,1:nc_dv)
        !allocate( ipss(2,np0,nrmax), ipss_ir(2,np0,1) )
        ipss(:,:,:) = -1
        ip = 0
        do i = 1, nc_dv
            nac = count( cliques_dv(:,i).gt.0 )
            if (icon) then
                print *, ''
                print *, ' ======================= Now we are dealing with subg', i, '/', nc_dv
                print *, ' nodes = ', cliques_dv(1:nac,i)
            endif

            !print *, 'nac = ', nac
            if ( mod(nac,2) .eq. 1) then
                iok = .false.
                exit ! `exit only affect the innermost loop
            else
                !print *, ' +++++ gotcha!'
                if ( nac .eq. 2) then
                    ip = ip + 1
                    ipss(1,ip,1) = cliques_dv(1,i)
                    ipss(2,ip,1) = cliques_dv(2,i) 
                    !print *, ' ***** '
                else
                    !allocate( g2c(nac,nac) )
                    g2c(:,:) = -1
                    do j = 1,nac; g2c(j,j) = 0; enddo
                    call subarray2d(n,g2,cliques_dv(:,i),g2c)

                    !print *, ' __ g2c = '
                    !do j = 1,nac; print *, ' --- ', g2c(j,1:nac); enddo

                    ipss_ir(:,:,:) = -1
                    if (icon) print *, 'now try to locate double bonds...'
                    call locate_double_bonds(n,g2c,nbmax,nrmax,ipss_ir,nr,icon)
                    if (icon) print *, 'double bonds found!'
                    if (icon) print *, 'nr = ', nr
                    if ( nr .eq. 0 ) then
                        iok = .false.
                        !deallocate( g2c )
                        exit
                    elseif ( nr .eq. 1 ) then
                        np = count(ipss_ir(:,:,1).gt.0)/2
                        !print *, 'ipss = '
                        do j = 1,np
                            ip = ip+1
                            do k = 1,2
                                kt = ipss_ir(k,j,1)
                                ipss(k,ip,1) = cliques_dv(kt,i)
                            enddo
                        enddo
                        !print *, ipss
                    else
                        stop "#ERROR: `nr > 1??"
                    endif
                    !deallocate( g2c )
                endif
            endif
        enddo
        if (.not. iok) then
            !deallocate( g2, cliques_dv, ipss, ipss_ir )
            exit
        endif

        do i = 1,ip
            !print *, '   a1,a2,bo = ', ipss(1,i,1),ipss(2,i,1),BO
            ia1 = ias_dv( ipss(1,i,1) ) ! to absolute idx
            ia2 = ias_dv( ipss(2,i,1) )
            bom(ia1,ia2) = BO
            bom(ia2,ia1) = BO
        enddo
        !deallocate( g2, cliques_dv, ipss, ipss_ir )

        ! update `vs
        vs = sum(bom,dim=1)
    enddo
    if (icon) print *, ' #3'

    ! check for the last time
    if (iok) then
        !vs = sum(bom,dim=1)
        dvs = tvs - vs
        if ( any(dvs.ne.0) ) then
            iok = .false.
!           print *, '+ zs = ', zs 
!           print *, '+ vs = ', vs
!           print *, '+tvs = ', tvs
        endif
    endif
    !print *, ' '
end subroutine


subroutine dijkstra(n,g,pls)
    implicit none
    integer, parameter :: INF = 2**20 ! for molecular graph, it's huge
    integer, intent(in) :: n
    integer, intent(in) :: g(n,n)
    integer, intent(out) :: pls(n,n)
    integer :: i,ia,j,ja,k,ka, pl, ni,nj, nv, &
               iast(n),ias(n),jas(n)
    logical :: mask(n),visited(n)

    do i = 1,n
        iast(i) = i
    enddo

    pls(:,:) = INF
    do i = 1,n
        pls(i,i) = 0
        do j = 1,n
            if (g(i,j).gt.0) pls(i,j) = 1
        enddo
    enddo

    ! Dijkstra's algorithm
    do ia = 1,n
        visited(:) = .false.
        visited(ia) = .true.
        nv = count(visited)
        do while (nv.lt.n)
            ias(:) = iast(:)
            call get_neibs(n,g,visited,ias,ni)
            !if (any(ias.eq.ia)) stop "#ERROR: `ia still in `ias?"
            !print *, ' ia,ias = ', ia, ',', ias(1:ni)
            do j = 1,ni
                ja = ias(j)

                jas(:) = iast(:)
                mask = g(ja,:).gt.0 .and. (.not.visited)
                nj = count(mask)
                call subarray(n,jas,mask)
                !print *, '  |__ ja, jas = ', ja,jas(1:nj)
                do k = 1,nj
                    ka = jas(k)
                    pl = pls(ia,ja)+1
                    !print *, '      |_ pl, ia,ka, pl0 = ', pl,ia,ka,pls(ia,ka)
                    if (pl.lt.pls(ia,ka)) then
                        pls(ia,ka) = pl
                    endif
                enddo
                visited(ja) = .true.
            enddo
            nv = count(visited)
        enddo
    enddo
end subroutine


subroutine get_neibs(na,g,visited,ias,nnb)
    implicit none
    integer, intent(in) :: na
    integer, intent(in) :: g(na,na)
    logical, intent(in) :: visited(na)
    integer, intent(inout) :: ias(na)
    integer, intent(out) :: nnb 
    integer :: i,j
    nnb = 0
    do i = 1,na
        if (visited(i)) then
            do j = 1,na
                if (g(i,j).gt.0 .and. (.not.visited(j))) then
                    nnb = nnb+1
                    ias(nnb) = j
                endif
            enddo
        endif
    enddo
end subroutine


subroutine ishare_node(edge, nb, edges, iok)
    implicit none 
    integer, intent(in) :: nb
    integer, intent(in) :: edge(2),edges(2,nb)
    logical, intent(out) :: iok
    integer :: e1,e2
    e1 = edge(1)
    e2 = edge(2)
    iok = .false.
    if ( any(edges.eq.e1) .or. any(edges.eq.e2) ) then
        iok = .true.
    endif
end subroutine


subroutine comba(nb,edges,na,vs,nv)
    implicit none
    integer, intent(in) :: nb,na
    integer, intent(in) :: edges(2,nb)
    integer, intent(out) :: vs(na),nv
    integer :: i,j,eij
    vs(:) = -1
    nv = 0
    do i = 1,nb
        do j = 1,2
            eij = edges(j,i)
            if (all(vs.ne.eij) .and. eij.gt.0) then
                nv = nv + 1
                vs(nv) = edges(j,i)
            endif
        enddo
    enddo
end subroutine


subroutine locate_double_bonds(na,g,np,nr0,ipss,nr,icon)
    implicit none
    integer, intent(in) :: na,np,nr0
    logical, intent(in) :: icon
    integer, intent(in) :: g(na,na)
    integer, intent(inout) :: ipss(2,np,nr0)
    integer, intent(out) :: nr
    integer :: i,iu,j,ju,k, ib0,ib1,ibc,ib,nb,nb0,nb1,nb2, &
               i0,j0,dsg(na,na), ds(np)
    integer :: edges0(2,np), edges(2,np), t(2,np), sg(na,na) 
    integer, allocatable :: dsgt(:,:)
    integer :: iasv(na),iasr(na),map(na), nodes0(na), &
               g2(na,na), edge_i(2), edge(2), &
               ipss_ir(2,np), ipss_ir0(2,np), vs(na)
    logical :: iok, iok2, iko, ifd ,istat, iexist
    integer :: ir, nv, n,n2, cns(na),cns2(na)

    ! num atom
    n = count(g(1,:).ge.0)
    nb = n/2
    if (n.lt.4) stop "#ERROR: na < 4?"

    !do i = 1,na; print *, g(i,:); enddo
    nb0 = count( g.gt.0 )/2

    ! work out the double bonds that can be easily identified,
    ! for those atoms having only one neighbor
    ipss(:,:,:) = -1
    ipss_ir0(:,:) = -1
    ibc = 0
    ib0 = nb ! initialization
    ! determine all double bonds that are not in a ring
    iasv(:) = -1 ! visited indices of atoms
    iasr(:) = -1 ! remaining indices of atoms
    n2 = n
    map(:) = -1
    do i = 1,n2
        map(i) = i
    enddo
    g2(:,:) = g(:,:)
    cns2(:) = -1
    cns2(1:n) = sum(g2(1:n,1:n),dim=1)
    cns2(n+1:na) = -1
    !print *, '    cns2 = ', cns2
    iko = .false. ! proceed; otherwise if `iko = .true.
    do while (ib0 .gt. 0) ! exit when there is no non-ring double bond
        ! initialize the num of standalone bonds (contain atom with cn=1)
        ! to be 0, such that if no more standalone bond can be identified,
        ! exist the while loop (due to reassignment of `ib1 to `ib0 at the end)
        ib1 = 0

        ! identify all bonds with one atom having CN=1 (coordination num)
        ! 
        ! vars need for explanation
        ! ===============================
        ! iasv: all visited unique atom idxs associated with the bonds
        ! ib1,ibc: identical, records the num of bonds visited so far
        do i0 = 1,n2
            i = map(i0)
            if (cns2(i0).eq.1) then
                do j0 = 1,n2
                    j = map(j0)
                    if (g2(i0,j0) .gt. 0) then
                        exit
                    endif
                enddo
                iu = min(i,j)
                ju = max(i,j)
                iexist = .false.
                do k = 1,ibc
                    ! unique bonds: no repetition!
                    if (ipss_ir0(1,k).eq.iu .and. ipss_ir0(2,k).eq.ju) then
                        iexist = .true.
                        exit
                    endif
                enddo
                if (.not. iexist) then
                    ib1 = ib1 + 1
                    ibc = ibc + 1
                    ipss_ir0(1,ibc) = iu
                    ipss_ir0(2,ibc) = ju
                    if ( all(iasv.ne.iu) ) iasv(count(iasv.gt.0)+1) = iu
                    if ( all(iasv.ne.ju) ) iasv(count(iasv.gt.0)+1) = ju
                endif
            endif
        enddo
        if (icon) print *, '  ***** ib1 = ', ib1

        ! vs: unique vertexes (atoms)
        ! nv: number of vertexes (size(vs)=na, size(vs>-1)=nv)
        call comba(np, ipss_ir0, na, vs, nv)
        if (icon) print *, ' * ibc, nb0 = ', ibc,nb0
        if (nv .lt. ibc*2) then
            ! e.g., for [CX3][CX3]([CX3])([CX3]), ibc = 3 while 
            ! nb0 = 2 (i.e., n/2), i.e., some edges share the same node,
            ! thus such subg is not amon; Now we r safe to exit
            nr = 0
            iko = .true.
            exit
        elseif (nv .eq. ibc*2) then 
            if (n .eq. nv) then ! (ibc .eq. nb0) then 
                ! now all double bonds located, exit the subroutine
                nr = 1
                ipss(:,:,1) = ipss_ir0(:,:)
                iko = .true.
                exit
            endif
        endif

        ! if either 1)  or 2), then proceed to 
        ! 
        if ( ib1 .gt. 0 ) then
            do i = 1,n2
                if ( all(map(i).ne.iasv) ) iasr(count(iasr.gt.0)+1) = map(i)
            enddo
            n2 = count(iasr.gt.0)
            !print *, ' ** n2, iasr = ', n2,iasr(1:n2)

            map(1:n2) = iasr(1:n2)
            map(n2+1:na) = -1
            g2(:,:) = -1
            do i0 = 1,n2
                i = map(i0)
                g2(i0,i0) = 0
                do j0 = i0+1,n2
                    j = map(j0)
                    g2(i0,j0) = g(i,j)
                    g2(j0,i0) = g(i,j)
                enddo
            enddo
            cns2(1:n2) = sum(g2(1:n2,1:n2),dim=1)
            cns2(n2+1:na) = -1
        endif
        ib0 = ib1
    enddo

    if (icon) then
        print *, ' ++ iko = ', iko
        print *, ' ++ identified double bonds:'
        do j = 1,ibc; print *, '   ', ipss_ir0(:,j); enddo
        print *, '   iasr = ', iasr(1:count(iasr.gt.0))
    endif

    ! ring exists
    if (.not. iko) then
        !print *, ' @@ nb0 = ', nb0
        !allocate( edges0(2,nb0), edges(2,nb0-1) )
        edges0(:,:) = -1
        ib = 0
        do i = 1,n-1
            do j = i+1,n
                if (g(i,j) .gt. 0) then
                    ib = ib+1
                    edges0(1,ib) = i
                    edges0(2,ib) = j
                endif
            enddo
        enddo
        if ( ib .ne. nb0 ) then
            !print *, 'ib,nb0 = ', ib,nb0
            stop "#ERROR: `ib != `nb0??"
        endif

        ! remove all standalone double bonds (e.g, 
        ! there exists one atom with cn=1)
        do i = 1,ibc
            call remove2d(np,edges0,ipss_ir0(:,i))
        enddo
        nb1 = nb0 - ibc
        if (icon) then
            print *, ' edges0 = '
            do i = 1,nb0; print *, ' @@     ', edges0(:,i); enddo
        endif

        allocate(dsgt(n,n))
        dsgt(:,:) = 0
        call dijkstra(n,g(1:n,1:n),dsgt)
        dsg(:,:) = -1
        dsg(1:n,1:n) = dsgt
        deallocate( dsgt )

        do i=1,na; nodes0(i) = i; enddo

        ifd = .false.
        ir = 0

200 FORMAT(2I2, ';')
201 FORMAT(A12)

        do i = 1, nb1

            ib = ibc
            ipss_ir(:,:) = ipss_ir0(:,:) ! -1

! ====================================
! new code: concise & efficient _start
            edges(:,:) = edges0(:,:)
            if (ibc .eq. 0) then
                edge_i = edges0(:,i)
                ipss_ir(:,1) = edge_i
                call remove2d(np,edges,edge_i)
                ib = ib + 1
            endif

            call comba(np, edges, na, vs, nv)
            sg(:,:) = -1
            do j = 1,nv; sg(j,j) = 0; enddo 
            call subarray2d(na,g,vs(1:nv),sg)
            nb2 = count(edges.gt.0)/2
            do while (nv.gt.0 .or. (.not. all(sg(1:nv,1:nv).eq.0)))
                if (icon) then
                    write(*,201,advance='no') 'edges = '
                    do j = 1,nb2
                        write(*,200,advance='no') edges(:,j)!,edges(2,j)
                    enddo
                    print *, ''
                    write(*,201,advance='no') 'ipss_ir = '
                    do j = 1,count(ipss_ir.gt.0)/2
                        write(*,200,advance='no') ipss_ir(:,j)!,edges(2,j)
                    enddo
                    print *, ''
                endif

                edge = edges(:,1)
                call ishare_node(edge,np,ipss_ir,istat)
                if (istat) then
                    call remove2d(np,edges,edge)
                else
                    if (icon) print *, '        ++ now we r in'
                    call is_adjacent_edge(edge, np, ipss_ir, na, dsg, iok)
                    if (iok) then
                        ib = ib + 1
                        call remove2d(np, edges, edge)
                        if (icon) print *, ' Gotcha!!   edge = ', edge
                        ipss_ir(:,ib) = edge
                        do k = 1,nb1
                            !print *, '                k, edges(:,k)', k, ',', edges(:,k)
                            call ishare_node(edges0(:,k),np,ipss_ir,istat)
                            if (istat) then
                                ! remove `edge_k from `edges
                                call remove2d(np,edges,edges0(:,k))
                            endif
                        enddo
                    else
                        ! shift this edge to the end of sequence
                        nb2 = count(edges.gt.0)/2
                        t(:,:) = edges(:,:)
                        edges(:,1:nb2-1) = t(:, 2:nb2)
                        edges(:,nb2) = edge
                    endif
                endif

                nb2 = count(edges.gt.0)/2
                if ( nb2 .gt. 0 ) then
                    call comba(np, edges, na, vs, nv)
                    if (nv.gt.0) then
                        call subarray2d(na,g,vs(1:nv),sg)
                    endif
                else
                    nv = 0
                    sg(:,:) = 0
                endif
            enddo
            if ( ib.eq.nb .and. nv.eq.0) then
                ifd = .true.
                ir = ir + 1
                ipss(:,:,ir) = ipss_ir
                exit ! no need to proceed
            endif
            !print *, ' ## done ', i
        enddo

        !if (.not.ifd) print *, " * warning: conj env w all nodes not found"
        nr = ir
    endif

! new code: concise & efficient  _end
! ====================================



!           edge_i = edges0(:,i)
!           edges(:,:) = -1
!           do j = 1,2
!               edges(j, 1:i-1) = edges0(j, 1:i-1)
!               edges(j, i:nb0-1) = edges0(j, i+1:nb0)
!           enddo

!           if (ibc .gt. 0) then
!               ! check if edge_i and atoms in `ipss_ir share 
!               ! the same atoms
!               call ishare_node(edge_i,np,ipss_ir,istat)
!               if (istat) then
!                   ! remove `edge_i from `edges
!                   call remove2d(np,edges,edge_i)
!               endif
!           endif

!           ipss_ir(:,1) = edge_i
!           ib = ib + 1

!           if (ib .eq. nb0) then
!               ipss(:,:,1) = ipss_ir(:,:)
!               nr = 1
!               exit
!           endif

!           !call sort_edges(np,edges,edge_i,na,dsg,ds)
!           !if (all(ds.le.0)) cycle

!           !print *, ' ** edges = '
!           !do j = 1, nb0; print *, '     ', edges(:,j); enddo

!           call comba(np, edges, na, vs, nv)
!           !allocate( sg(nv,nv) )
!           sg(:,:) = -1
!           do j = 1,nv; sg(j,j) = 0; enddo 
!           call subarray2d(na,g,vs(1:nv),sg)
!           nb2 = count(edges.gt.0)/2
!           do while (nv.gt.0 .or. (.not. all(sg(1:nv,1:nv).eq.0)))
!               !print *, ' ** a) edges = '
!               !do j = 1, nb2; print *, '     ', edges(:,j); enddo
!               edge = edges(:,1)
!               call is_adjacent_edge(edge, np, ipss_ir, na, dsg, iok)
!               if (iok) then
!                   ib = ib + 1
!                   !print *, ' Gotcha!!   edge = ', edge
!                   ipss_ir(:,ib) = edge
!                   do k = 2,nb2
!                       !print *, '                k, edges(:,k)', k, ',', edges(:,k)
!                       call ishare_node(edges(:,k),np,ipss_ir,istat)
!                       if (istat) then
!                           ! remove `edge_k from `edges
!                           call remove2d(np,edges,edges(:,k))
!                       endif
!                   enddo
!               endif
!               call remove2d(np, edges, edge)
!               nb2 = count(edges.gt.0)/2
!               !print *, ' ** b) edges = '
!               !do j = 1, nb2; print *, '     ', edges(:,j); enddo
!               !print *, ' ** nb2 = ', nb2
!               !print *, ' ib,nb = ', ib,nb 
!               if ( nb2 .gt. 0 ) then
!                   call comba(np, edges, na, vs, nv)
!                   !deallocate( sg )
!                   if (nv.gt.0) then
!                       !allocate( sg(nv,nv) )
!                       call subarray2d(na,g,vs(1:nv),sg)
!                   endif
!               else
!                   nv = 0
!                   sg(:,:) = 0
!               endif
!           enddo
!           !if (allocated(sg)) deallocate( sg )
!           if ( ib.eq.nb .and. nv.eq.0) then
!               ifd = .true.
!               ir = ir + 1
!               ipss(:,:,ir) = ipss_ir
!               exit ! no need to proceed
!           endif
!           !print *, ' ## done ', i
!       enddo

!       !if (.not.ifd) print *, " * warning: conj env w all nodes not found"
!       nr = ir
!   endif
!   !deallocate( edges0, edges )

end subroutine locate_double_bonds


subroutine sort_edges(nb0, edges, edge, na, dsg, ds)
    !
    ! sort the edges by the relative distance to nodes in `edge_i
    !
    implicit none
    integer, intent(in) :: nb0,na
    integer, intent(inout) :: edges(2,nb0)
    integer, intent(out) :: ds(nb0)
    integer, intent(in) :: dsg(na,na),edge(2)
    integer :: n1, nb, ib, i,j,k,l, &
               dsi(4), t(2,nb0), dst(nb0)
    integer, allocatable :: seq(:)

    ds(:) = -1
    nb = count( edges .gt. 0 )/2
    k = edge(1)
    l = edge(2)
    !print *, ' + edges = ', edges
    do ib = 1,nb
        i = edges(1,ib)
        j = edges(2,ib)
        dsi(1) = dsg(i,k)
        dsi(2) = dsg(j,l)
        dsi(3) = dsg(i,l)
        dsi(4) = dsg(j,k)
        ds(ib) = minval(dsi)
    enddo
    !print *, ' * ds = ', ds
    allocate( seq(nb) )
    call iargsort(nb,ds(1:nb),seq,1) ! 2: descending, 1: ascending
    !print *, ' * seq = ', seq
    t(:,:) = -1
    dst(:) = -1
    do ib = 1,nb
        t(:,ib) = edges(:,seq(ib))
        dst(ib) = ds(seq(ib))
    enddo
    edges = t
    ds = dst
    deallocate( seq )
    !print *, ' ++ edges = ', edges
end subroutine


subroutine is_adjacent_edge(edge, nb0, edges, na, dsg, iok)
    implicit none
    integer, intent(in) :: nb0, na
    integer, intent(in) :: edge(2), edges(2,nb0), dsg(na,na)
    logical, intent(out) :: iok
!   logical :: istat
    integer :: ib,nb,i1,i2,j1,j2, dsi(4)
    i1 = edge(1)
    i2 = edge(2)
    iok = .false.
    nb = count(edges.gt.0)/2 !size(edges,2)
    if (nb .eq. 0) stop "#ERROR: empty `edges?"
    !call ishare_node(edge,nb,edges,istat)
    !if (.not. istat) then
    do ib = 1,nb
        j1 = edges(1,ib)
        j2 = edges(2,ib)
        dsi(1) = dsg(i1,j1)
        dsi(2) = dsg(i2,j2)
        dsi(3) = dsg(i1,j2)
        dsi(4) = dsg(i2,j1)
        if ( all(dsi.ge.1) .and. any(dsi.eq.1) ) then
            iok = .true.
            exit
        endif
    enddo
    !endif
end subroutine


subroutine g2triu1d(n,g, nr,gr)
    !
    ! molecular graph to triu(g) concatenated to 1d
    !
    implicit none 
    integer, intent(in) :: n, nr
    integer, intent(in) :: g(n,n)
    integer, intent(out) :: gr(nr)
    integer :: i,j,ic
    ic = 0
    do i = 1,n-1
        do j = i+1,n
            ic = ic + 1
            gr(ic) = g(i,j)
        enddo
    enddo
end subroutine


subroutine triu1d2g(nr,gr, n,g)
    !
    ! 1d triu(g) to g
    ! 
    implicit none
    integer, intent(in) :: nr, n
    integer, intent(in) :: gr(nr)
    integer, intent(out) :: g(n,n)
    integer :: i,j,ic
    ic = 0
    do i = 1,n-1
        do j = i+1,n
            ic = ic + 1
            g(i,j) = gr(ic)
            g(j,i) = gr(ic)
        enddo
    enddo
end subroutine


