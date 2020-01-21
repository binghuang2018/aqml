! MIT License
!
! Copyright (c) 2016 Anders Steen Christensen, Lars A. Bratholm, Felix A. Faber
!
! Permission is hereby granted, free of charge, to any person obtaining a copy
! of this software and associated documentation files (the "Software"), to deal
! in the Software without restriction, including without limitation the rights
! to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
! copies of the Software, and to permit persons to whom the Software is
! furnished to do so, subject to the following conditions:
!
! The above copyright notice and this permission notice shall be included in all
! copies or substantial portions of the Software.
!
! THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
! IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
! FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
! AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
! LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
! OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
! SOFTWARE.

subroutine calc_lk_gaussian(x1, x2, zs1, zs2, nas1, nas2, zmax, iab, &
                                sigmas, nm1, nm2, nsigmas, kernels)

    implicit none

    real(8), dimension(:,:), intent(in) :: x1, x2
    integer, dimension(:), intent(in) :: zs1, zs2
    integer, intent(in) :: nm1, nm2 ! Number of molecules
    integer, dimension(:), intent(in) :: nas1, nas2 ! number of atoms in each mol
    integer, intent(in) :: zmax
    logical, intent(in) :: iab ! if set to .true., dij=0 if i and j are disimilar atoms
    real(8), dimension(:,:), intent(in) :: sigmas ! list of kernel widths
    integer, intent(in) :: nsigmas
    real(8), dimension(nsigmas,zmax) :: inv_sigma2
    real(8), dimension(nsigmas,nm1,nm2), intent(out) :: kernels
    integer :: a, b, i, j, k, ni, nj, ia,ja
    real(8), allocatable, dimension(:,:) :: dsij
    real(8), allocatable, dimension(:,:,:) :: sgijs
    integer, allocatable, dimension(:) :: ias, jas

    allocate(ias(nm1), jas(nm2))

    !$OMP PARALLEL DO
    do i = 1, nm1
        ias(i) = sum(nas1(:i)) - nas1(i)
    enddo
    !$OMP END PARALLEL DO

    !$OMP PARALLEL DO
    do j = 1, nm2
        jas(j) = sum(nas2(:j)) - nas2(j)
    enddo
    !$OMP END PARALLEL DO

    inv_sigma2 = -0.5d0/sigmas**2
    kernels(:,:,:) = 0.0d0

    allocate(dsij(maxval(nas1), maxval(nas2)))
    allocate( sgijs(nsigmas, maxval(nas1), maxval(nas2)) )
    !dsij(:,:) = 0.0d0

    !$OMP PARALLEL DO PRIVATE(dsij,sgijs,ni,nj,ia,ja)
     do a = 1, nm1
        ni = nas1(a)
        do b = 1, nm2
            nj = nas2(b)
            dsij(:,:) = 0.0d0
            sgijs(:,:,:) = 0.0d0
            do i = 1, ni
                do j = 1, nj
                    dsij(i, j) = 1.e9
                    ia = i + ias(a)
                    ja = j + jas(b)
                    sgijs(:,i,j) = inv_sigma2(:,zs1(ia))
                    if ( (iab .eqv. .true.) .or. ( zs1(ia) .eq. zs2(ja) ) ) then
                        dsij(i, j) = sum((x1(:,ia) - x2(:,ja))**2)
                    endif
                enddo
            enddo
            do k = 1, nsigmas
                kernels(k, a, b) =  sum(exp(dsij(:ni,:nj)*sgijs(k,:,:)))
            enddo
        enddo
    enddo
    !$OMP END PARALLEL DO

    deallocate(dsij,sgijs)
    deallocate(ias)
    deallocate(jas)

end subroutine calc_lk_gaussian



subroutine calc_lk_laplacian(x1, x2, nas1, nas2, sigmas, &
        & nm1, nm2, nsigmas, kernels)

    implicit none

    real(8), dimension(:,:), intent(in) :: x1
    real(8), dimension(:,:), intent(in) :: x2

    ! List of numbers of atoms in each molecule
    integer, dimension(:), intent(in) :: nas1
    integer, dimension(:), intent(in) :: nas2

    ! Sigma in the Gaussian kernel
    real(8), dimension(:), intent(in) :: sigmas

    ! Number of molecules
    integer, intent(in) :: nm1
    integer, intent(in) :: nm2

    ! Number of sigmas
    integer, intent(in) :: nsigmas

    ! -1.0 / sigma^2 for use in the kernel
    real(8), dimension(nsigmas) :: inv_sigma2

    ! Resulting alpha vector
    real(8), dimension(nsigmas,nm1,nm2), intent(out) :: kernels

    ! Internal counters
    integer :: a, b, i, j, k, ni, nj

    ! Temporary variables necessary for parallelization
    real(8), allocatable, dimension(:,:) :: dsij

    integer, allocatable, dimension(:) :: ias
    integer, allocatable, dimension(:) :: jas

    allocate(ias(nm1))
    allocate(jas(nm2))

    !$OMP PARALLEL DO
    do i = 1, nm1
        ias(i) = sum(nas1(:i)) - nas1(i)
    enddo
    !$OMP END PARALLEL DO

    !$OMP PARALLEL DO
    do j = 1, nm2
        jas(j) = sum(nas2(:j)) - nas2(j)
    enddo
    !$OMP END PARALLEL DO

    inv_sigma2(:) = -1.0d0 / sigmas(:)
    kernels(:,:,:) = 0.0d0

    allocate(dsij(maxval(nas1), maxval(nas2)))
    dsij(:,:) = 0.0d0

    !$OMP PARALLEL DO PRIVATE(dsij,ni,nj)
     do a = 1, nm1
        ni = nas1(a)
        do b = 1, nm2
            nj = nas2(b)

            dsij(:,:) = 0.0d0
            do i = 1, ni
                do j = 1, nj

                    dsij(i, j) = sum(abs(x1(:,i + ias(a)) - x2(:,j + jas(b))))

                enddo
            enddo


            do k = 1, nsigmas
                kernels(k, a, b) =  sum(exp(dsij(:ni,:nj) * inv_sigma2(k)))
            enddo

        enddo
    enddo
    !$OMP END PARALLEL DO

    deallocate(dsij)
    deallocate(ias)
    deallocate(jas)

end subroutine calc_lk_laplacian

subroutine calc_vector_kernels_laplacian(x1, x2, nas1, nas2, sigmas, &
        & nm1, nm2, nsigmas, kernels)

    implicit none

    ! Descriptors for the training set
    real(8), dimension(:,:,:), intent(in) :: x1
    real(8), dimension(:,:,:), intent(in) :: x2

    ! List of numbers of atoms in each molecule
    integer, dimension(:), intent(in) :: nas1
    integer, dimension(:), intent(in) :: nas2

    ! Sigma in the Gaussian kernel
    real(8), dimension(:), intent(in) :: sigmas

    ! Number of molecules
    integer, intent(in) :: nm1
    integer, intent(in) :: nm2

    ! Number of sigmas
    integer, intent(in) :: nsigmas

    ! -1.0 / sigma^2 for use in the kernel
    real(8), dimension(nsigmas) :: inv_sigma

    ! Resulting alpha vector
    real(8), dimension(nsigmas,nm1,nm2), intent(out) :: kernels

    ! Internal counters
    integer :: i, j, k, ni, nj, ia, ja

    ! Temporary variables necessary for parallelization
    real(8), allocatable, dimension(:,:) :: dsij

    inv_sigma(:) = -1.0d0 / sigmas(:)

    kernels(:,:,:) = 0.0d0

    allocate(dsij(maxval(nas1), maxval(nas2)))
    dsij(:,:) = 0.0d0

    !$OMP PARALLEL DO PRIVATE(dsij,ni,nj)
    do j = 1, nm2
        nj = nas2(j)
        do i = 1, nm1
            ni = nas1(i)

            dsij(:,:) = 0.0d0

            do ja = 1, nj
                do ia = 1, ni

                    dsij(ia,ja) = sum(abs(x1(:,ia,i) - x2(:,ja,j)))

                enddo
            enddo

            do k = 1, nsigmas
                kernels(k, i, j) =  sum(exp(dsij(:ni,:nj) * inv_sigma(k)))
            enddo

        enddo
    enddo
    !$OMP END PARALLEL DO

    deallocate(dsij)

end subroutine calc_vector_kernels_laplacian

subroutine calc_vector_kernels_gaussian(x1, x2, nas1, nas2, sigmas, &
        & nm1, nm2, nsigmas, kernels)

    implicit none

    ! ARAD descriptors for the training set, format (i,j_1,5,m_1)
    real(8), dimension(:,:,:), intent(in) :: x1
    real(8), dimension(:,:,:), intent(in) :: x2

    ! List of numbers of atoms in each molecule
    integer, dimension(:), intent(in) :: nas1
    integer, dimension(:), intent(in) :: nas2

    ! Sigma in the Gaussian kernel
    real(8), dimension(:), intent(in) :: sigmas

    ! Number of molecules
    integer, intent(in) :: nm1
    integer, intent(in) :: nm2

    ! Number of sigmas
    integer, intent(in) :: nsigmas

    ! -1.0 / sigma^2 for use in the kernel
    real(8), dimension(nsigmas) :: inv_sigma2

    ! Resulting alpha vector
    real(8), dimension(nsigmas,nm1,nm2), intent(out) :: kernels

    ! Internal counters
    integer :: i, j, k, ni, nj, ia, ja

    ! Temporary variables necessary for parallelization
    real(8), allocatable, dimension(:,:) :: dsij

    inv_sigma2(:) = -0.5d0 / (sigmas(:))**2


    kernels(:,:,:) = 0.0d0

    allocate(dsij(maxval(nas1), maxval(nas2)))
    dsij(:,:) = 0.0d0

    !$OMP PARALLEL DO PRIVATE(dsij,ni,nj,ja,ia)
    do j = 1, nm2
        nj = nas2(j)
        do i = 1, nm1
            ni = nas1(i)

            dsij(:,:) = 0.0d0

            do ja = 1, nj
                do ia = 1, ni

                    dsij(ia,ja) = sum((x1(:,ia,i) - x2(:,ja,j))**2)

                enddo
            enddo

            do k = 1, nsigmas
                kernels(k, i, j) =  sum(exp(dsij(:ni,:nj) * inv_sigma2(k)))
            enddo

        enddo
    enddo
    !$OMP END PARALLEL DO

    deallocate(dsij)

end subroutine calc_vector_kernels_gaussian


subroutine fgaussian_kernel(a, na, b, nb, k, sigma)

    implicit none

    real(8), dimension(:,:), intent(in) :: a
    real(8), dimension(:,:), intent(in) :: b

    integer, intent(in) :: na, nb

    real(8), dimension(:,:), intent(inout) :: k
    real(8), intent(in) :: sigma

    real(8), allocatable, dimension(:) :: temp

    real(8) :: inv_sigma
    integer :: i, j

    inv_sigma = -0.5d0 / (sigma*sigma)

    allocate(temp(size(a, dim=1)))

!$OMP PARALLEL DO PRIVATE(temp)
    do i = 1, nb
        do j = 1, na
            temp(:) = a(:,j) - b(:,i)
            k(j,i) = exp(inv_sigma * sum(temp*temp))
        enddo
    enddo
!$OMP END PARALLEL DO

    deallocate(temp)

end subroutine fgaussian_kernel

subroutine flaplacian_kernel(a, na, b, nb, k, sigma)

    implicit none

    real(8), dimension(:,:), intent(in) :: a
    real(8), dimension(:,:), intent(in) :: b

    integer, intent(in) :: na, nb

    real(8), dimension(:,:), intent(inout) :: k
    real(8), intent(in) :: sigma

    real(8) :: inv_sigma

    integer :: i, j

    inv_sigma = -1.0d0 / sigma

!$OMP PARALLEL DO
    do i = 1, nb
        do j = 1, na
            k(j,i) = exp(inv_sigma * sum(abs(a(:,j) - b(:,i))))
        enddo
    enddo
!$OMP END PARALLEL DO

end subroutine flaplacian_kernel


subroutine flinear_kernel(a, na, b, nb, k)

    implicit none

    real(8), dimension(:,:), intent(in) :: a
    real(8), dimension(:,:), intent(in) :: b

    integer, intent(in) :: na, nb

    real(8), dimension(:,:), intent(inout) :: k

    integer :: i, j

!$OMP PARALLEL DO
    do i = 1, nb
        do j = 1, na
            k(j,i) = dot_product(a(:,j), b(:,i))
        enddo
    enddo
!$OMP END PARALLEL DO

end subroutine flinear_kernel


subroutine fmatern_kernel_l2(a, na, b, nb, k, sigma, order)

    implicit none

    real(8), dimension(:,:), intent(in) :: a
    real(8), dimension(:,:), intent(in) :: b

    integer, intent(in) :: na, nb

    real(8), dimension(:,:), intent(inout) :: k
    real(8), intent(in) :: sigma
    integer, intent(in) :: order

    real(8), allocatable, dimension(:) :: temp

    real(8) :: inv_sigma, inv_sigma2, d, d2
    integer :: i, j

    allocate(temp(size(a, dim=1)))

    if (order == 0) then
        inv_sigma = - 1.0d0 / sigma

        !$OMP PARALLEL DO PRIVATE(temp)
            do i = 1, nb
                do j = 1, na
                    temp(:) = a(:,j) - b(:,i)
                    k(j,i) = exp(inv_sigma * sqrt(sum(temp*temp)))
                enddo
            enddo
        !$OMP END PARALLEL DO
    else if (order == 1) then
        inv_sigma = - sqrt(3.0d0) / sigma

        !$OMP PARALLEL DO PRIVATE(temp, d)
            do i = 1, nb
                do j = 1, na
                    temp(:) = a(:,j) - b(:,i)
                    d = sqrt(sum(temp*temp))
                    k(j,i) = exp(inv_sigma * d) * (1.0d0 - inv_sigma * d)
                enddo
            enddo
        !$OMP END PARALLEL DO
    else
        inv_sigma = - sqrt(5.0d0) / sigma
        inv_sigma2 = 5.0d0 / (3.0d0 * sigma * sigma)

        !$OMP PARALLEL DO PRIVATE(temp, d, d2)
            do i = 1, nb
                do j = 1, na
                    temp(:) = a(:,j) - b(:,i)
                    d2 = sum(temp*temp)
                    d = sqrt(d2)
                    k(j,i) = exp(inv_sigma * d) * (1.0d0 - inv_sigma * d + inv_sigma2 * d2)
                enddo
            enddo
        !$OMP END PARALLEL DO
    end if

    deallocate(temp)

end subroutine fmatern_kernel_l2


subroutine fsargan_kernel(a, na, b, nb, k, sigma, gammas, ng)

    implicit none

    real(8), dimension(:,:), intent(in) :: a
    real(8), dimension(:,:), intent(in) :: b
    real(8), dimension(:), intent(in) :: gammas

    integer, intent(in) :: na, nb, ng

    real(8), dimension(:,:), intent(inout) :: k
    real(8), intent(in) :: sigma

    real(8), allocatable, dimension(:) :: prefactor
    real(8) :: inv_sigma
    real(8) :: d

    integer :: i, j, m

    inv_sigma = -1.0d0 / sigma

    ! Allocate temporary
    allocate(prefactor(ng))


    !$OMP PARALLEL DO PRIVATE(d, prefactor)
        do i = 1, nb
            do j = 1, na
                d = sum(abs(a(:,j) - b(:,i)))
                do m = 1, ng
                    prefactor(m) = gammas(m) * (- inv_sigma * d) ** m
                enddo
                k(j,i) = exp(inv_sigma * d) * (1 + sum(prefactor(:)))
            enddo
        enddo
    !$OMP END PARALLEL DO

    ! Clean up
    deallocate(prefactor)

end subroutine fsargan_kernel
