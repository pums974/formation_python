program fconvolve
use fortranmodule
implicit none

integer(kind=8),parameter :: vmax = 2000;
integer(kind=8),parameter :: wmax = 2000;
integer(kind=8),parameter :: smax = 9;
integer(kind=8),parameter :: tmax = 9;

integer(kind=8),allocatable :: f(:,:),h(:,:),g(:,:)
integer(kind=8) :: err
integer :: i,j
INTEGER(kind=8) :: t1,t2, count_rate,elap
integer :: s,ms,us,ns

allocate(f(vmax,wmax), h(vmax,wmax))
allocate(g(smax,tmax))
  
    
  do j = 1,wmax
  do i = 1,vmax
      f(i,j) = i*wmax + j
  enddo
  enddo

  do j = 1,tmax
  do i = 1,smax
      g(i,j) = i*wmax + j
  enddo
  enddo
  
  CALL SYSTEM_CLOCK(t1, count_rate)
  do i=1,70
    call convolve_fortran_pure(f, g, vmax, wmax, smax, tmax, h, err)
  enddo
  CALL SYSTEM_CLOCK(t2)

  elap = (t2-t1)/70
  s = elap/count_rate
  ms = (elap - s*count_rate)*1000/count_rate
  us = ((elap - s*count_rate)*1000 - ms*count_rate)*1000/count_rate
  ns = (((elap - s*count_rate)*1000 - ms*count_rate)*1000 - us*count_rate)*1000/count_rate

  write(*,'(A,4(I4,A2))') "Time :",s,"s ",ms,"ms ", us,"us ", ns,"ns"

deallocate(f,g,h)

end program fconvolve
