

subroutine fl1_distance(n1,n2,nc,A, B, D)

    implicit none
    integer, intent(in) :: n1,n2,nc
    double precision, dimension(nc,n1), intent(in) :: A
    double precision, dimension(nc,n2), intent(in) :: B
    double precision, dimension(n1,n2), intent(out) :: D
    integer :: i, j
    real*8 :: t(nc)

!$OMP PARALLEL DO
    do i = 1, n2
        do j = 1, n1
            D(j,i) = sum(abs(a(:,j) - b(:,i)))
        enddo
    enddo
!$OMP END PARALLEL DO


end subroutine fl1_distance


subroutine fl2_distance(n1,n2,nc,A, B, D)

    implicit none
    integer, intent(in) :: n1,n2,nc
    double precision, dimension(nc,n1), intent(in) :: A
    double precision, dimension(nc,n2), intent(in) :: B
    double precision, dimension(n1,n2), intent(out) :: D
    real*8 :: t(nc)
    integer :: i, j

!$OMP PARALLEL DO PRIVATE(t)
    do i = 1, n2
        do j = 1, n1
            t = A(:,j) - B(:,i)
            D(j,i) = sqrt(sum(t*t))
        enddo
    enddo
!$OMP END PARALLEL DO

end subroutine fl2_distance


subroutine fp_distance_double(A, B, D, p)

    implicit none

    double precision, dimension(:,:), intent(in) :: A
    double precision, dimension(:,:), intent(in) :: B
    double precision, dimension(:,:), intent(inout) :: D
    double precision, intent(in) :: p

    integer :: na, nb, nv
    integer :: i, j

    double precision, allocatable, dimension(:) :: temp
    double precision :: inv_p

    nv = size(A, dim=1)

    na = size(A, dim=2)
    nb = size(B, dim=2)

    inv_p = 1.0d0 / p

    allocate(temp(nv))

!$OMP PARALLEL DO PRIVATE(temp)
    do i = 1, nb
        do j = 1, na
            temp(:) = abs(A(:,j) - B(:,i))
            D(j,i) = (sum(temp**p))**inv_p
        enddo
    enddo
!$OMP END PARALLEL DO

    deallocate(temp)

end subroutine fp_distance_double

subroutine fp_distance_integer(A, B, D, p)

    implicit none

    double precision, dimension(:,:), intent(in) :: A
    double precision, dimension(:,:), intent(in) :: B
    double precision, dimension(:,:), intent(inout) :: D
    integer, intent(in) :: p

    integer :: na, nb, nv
    integer :: i, j

    double precision, allocatable, dimension(:) :: temp
    double precision :: inv_p

    nv = size(A, dim=1)

    na = size(A, dim=2)
    nb = size(B, dim=2)

    inv_p = 1.0d0 / dble(p)

    allocate(temp(nv))

!$OMP PARALLEL DO PRIVATE(temp)
    do i = 1, nb
        do j = 1, na
            temp(:) = abs(A(:,j) - B(:,i))
            D(j,i) = (sum(temp**p))**inv_p
        enddo
    enddo
!$OMP END PARALLEL DO

    deallocate(temp)

end subroutine fp_distance_integer
