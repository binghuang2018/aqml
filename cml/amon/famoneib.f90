module famoneib
    
    ! fortran module to compute the adjacency matrix of amons
    ! in a query molecule and then applied to do combinations
    ! to get legitimate molecule complexes representing the vdw
    ! interaction in query molecule
    
!    use sorting, only :: iargsort
!    use itertools, only :: combinations

    implicit none

    contains
 
        function check_rij(n1,n2,coords1,coords2,dminVDW) result(iok)
            !
            ! check if any `r .lt. dminVDW
            !
            implicit none

            integer, intent(in) :: n1, n2
            real, intent(in) :: coords1(3,n1), coords2(3,n2), dminVDW
            integer :: i,j
            logical :: iok
            real :: vec(3), rij

            iok = .true.
            do i = 1,n1
                do j = 1,n2
                    vec = coords1(:,i)-coords2(:,j)
                    rij = sqrt( vec(1)*vec(1)+vec(2)*vec(2)+vec(3)*vec(3) ) 
                    if (rij .le. dminVDW) then
                        iok = .false.
                        exit
                    endif
                enddo
            enddo

        end function check_rij


        function is_vdw_neib(k,nb,na1,na2,ias1,ias2,ncbs) result(iok)
            ! 
            ! subgraphs may have different num_heav_atoms, to
            ! make the relevant `ias index of the same size,
            ! pad by -1
            !
            implicit none 
            integer, intent(in) :: k,nb,na1,na2
            integer, intent(in) :: ias1(k),ias2(k),ncbs(2,nb)
            integer :: i,j,ia,ja,ib,tval
            logical :: iok
 
            iok = .false.
            do i = 1,na1
                do j = 1,na2
                    ia = ias1(i)
                    ja = ias2(j)
                    if (ia .gt. ja) then
                        tval = ia; ia = ja; ja = tval
                    endif
                    do ib = 1,nb
                        if ( ncbs(1,ib) .eq. ia .and. &
                              & ncbs(2,ib) .eq. ja ) then
                            iok = .true.
                            exit
                        endif
                    enddo
                enddo
            enddo
 
        end function is_vdw_neib
 

        function is_subvec(k2,n,ias,iass) result(iok)
            ! 
            ! check if a vec is a sub vector of an array
            !
            implicit none
            integer, intent(in) :: k2,n
            integer, intent(in) :: ias(k2),iass(k2,n)
            integer i, t(k2)
            logical :: iok
            iok = .false.
            do i = 1,n
                if ( all(ias .eq. iass(:,i)) ) iok = .true.; exit
            enddo
        end function is_subvec

end module famoneib


subroutine get_amon_adjacency(k,k2,nm,nas,nat,nasv,iass,coords, &
                       nb,ncbs,dminVDW,gv,gc) !result(g)
    !
    ! get the adjacency matrix of amons
    ! 
    ! nasv -- num_atoms_heav
    ! gv   -- vdW graph connectivity between amons (if two parts
    !         are connected by vdW bond, then assign to 1; otherwise, 0)
    ! gc   -- covalent graph connectivity between amons. Assign the
    !         value to 1 when one amon is part of another or these two
    !         amons are in close proximity.
    !
    use famoneib

    implicit none
    integer, intent(in) :: k,k2,nm,nat,nb
    integer, intent(in) :: nas(nm), nasv(nm), &
                         & iass(k,nm), ncbs(2,nb)
    real, intent(in) :: coords(3,nat),dminVDW
    integer, intent(out) :: gv(nm,nm), gc(nm,nm)

    integer :: i,j, nv1,nv2, ia1,ia2,ja1,ja2, &
               & na1,na2, ias(1+nm)

    ias(1) = 0
    do i = 2,nm+1
        ias(i) = ias(i-1) + nas(i-1)
    enddo

    gv(:,:) = 0; gc(:,:) = 0
    !$omp parallel do private(i,j,nv1,nv2, ia1,ia2,ja1,ja2,na1,na2)
    do i = 1,nm-1
        do j = i+1,nm
            nv1 = nasv(i); nv2 = nasv(j)
            if (nv1 + nv2 .le. k2) then
                ia1 = ias(i)+1; ia2 = ias(i+1); na1 = ia2-ia1+1
                ja1 = ias(j)+1; ja2 = ias(j+1); na2 = ja2-ja1+1

                if ( check_rij(na1,na2,coords(:,ia1:ia2), &
                               & coords(:,ja1:ja2), dminVDW) ) then
                    if ( is_vdw_neib(k,nb,nv1,nv2,iass(:,i),iass(:,j),ncbs) ) then
                        gv(i,j) = 1; gv(j,i) = 1
                    endif
                else
                    gc(i,j) = 1; gc(j,i) = 1
                endif
            endif
       enddo
    enddo
    !$omp end parallel do
    
end subroutine get_amon_adjacency

