!cccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccc
subroutine logspace(n,x)

! Computes a logarithmic equally spaced grid
! Modified Flomin to enable interfacing with python

! Input :
!   - x0 :: First grid point
!   - xu :: Final grid point
!   - n  :: Number of points on grid
! 
! Output :
!   - x  :: output logarithmic frequency vector

   implicit none

   integer (kind = 4), intent(in)  :: n
   integer (kind = 4)              :: i
   real    (kind = 4)  :: x0, xu 
   real    (kind = 4), intent(out) :: x(n)
   real    (kind = 4) :: dx

! Initialize values
   x0 = alog10(0.1)
   xu = alog10(100.0)

! Begin computation

   dx = (xu-x0) / (n-1)
   do i=1,n
      x(i) = 10.0**(x0 + dx*(i-1))
   end do

   return
end subroutine logspace
!cccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccc

!cccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccc
subroutine freq2per(x,n,xo)

! Computes period from frequency
!
! Inputs :
!   - x :: frequency vector
!   - n :: number points

   implicit none

   integer (kind = 4), intent(in)  ::  n
   integer (kind = 4) :: i
   real    (kind = 4), intent(in) :: x(n)
   real    (kind = 4), intent(out) :: xo(n)

   do i=1,n
      if (x(i) .ne. 0.0) then
         xo(i) = 1.0/x(i)
      else
         stop 'null frequency'
      end if
   end do

   return
end subroutine freq2per

!cccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccc

SUBROUTINE RSPS(A,T,N,M,DT,D,SA)
!cccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccc
!   !! Modified by Flomin (03/05/2019)
!   I edited the subroutine so that it supports binding with python   
!
!   Subroutine RSPS was published by Li Dahua and Di Qingyan in
!   "Earthquake Research in China",Vol.7,N.4,1993(Allerton Press, NY)
!
!   RSPS:  Compute maximum response.
!   On entry --
!     N = number of values given in the time series.
!     M = number of values in the output response spectrum
!     A()= acceleration time series, cm/sec/sec.
!     T()= array of oscillator periods
!     D   = damping fraction.
!     DT  = sampling interval, seconds.
!
!   On return --
!     SD()= maximum relative displacement response spectrum, cm.
!     SV  =   "        "     velocity       "        " , cm/sec.
!     SA  =   "     absolute acceleration   "     ", cm/sec/sec.
!
      implicit none

      integer, intent(in)          :: N
      integer, intent(in)         :: M
      real (kind = 4), intent(in) :: A(N)      
      real (kind = 4), intent(in) :: DT
      real (kind = 4), intent(in) :: D
      real (kind = 4), intent(in) :: T(M)
      real (kind = 4), intent(out) :: SA(M)
      real (kind = 4) :: SD(M)
      real (kind = 4) :: SV(M)

!      dimension  a(N), sd(M), sv(M), sa(M), t(M)
!
      real (kind = 4 ) :: SQD, AMAX, VMAX, DMAX, DP, DLT, W, W2, W2D, WSQD
      real (kind = 4 ) :: Z, XT, SXT, DSXT, CXT, A11, A12, A21, A22, GA1, DW
      real (kind = 4 ) :: A0, V0, DX, DXWD, XA1, SA1, SV1, SD1, V1, VDXWD, A1
      integer (kind = 4) :: J, L, K, I

      SQD=SQRT(1.-D*D)
      DO 2 J=1,M
      AMAX=0.0
      VMAX=0.0
      DMAX=0.0
      DP=T(J)/10.
      L=1
      IF(DP.LT.DT) L=INT(DT/DP+1.0-0.00001)
      DLT=DT/L
      W=6.283185308/T(J)
      DW=2.*D*W
      W2=W*W
      W2D=W2*DLT
      WSQD=W*SQD
      Z=EXP(-D*W*DLT)
      XT=WSQD*DLT
      SXT=SIN(XT)
      DSXT=D*SXT/SQD
      CXT=COS(XT)
      A11=Z*(DSXT+CXT)
      A12=Z*SXT/WSQD
      A21=-A12*W2
      A22=Z*(-DSXT+CXT)
      GA1=A(1)
      V0=0.0
      A0=-GA1
      DO 1 I=1,N-1
      DX=(A(I+1)-A(I))/L
      DXWD=DX/W2D
      DO 11 K=1,L
      GA1=GA1+DX
      VDXWD=V0+DXWD
      V1=A11*VDXWD+A12*A0-DXWD
      A1=A21*VDXWD+A22*A0
      XA1=GA1+A1
      SA1=ABS(XA1)
      SV1=ABS(V1)
      SD1=ABS(XA1+DW*V1)/W2
      IF(SA1.GT.AMAX) AMAX=SA1
      IF(SV1.GT.VMAX) VMAX=SV1
      IF(SD1.GT.DMAX) DMAX=SD1
      V0=V1
11     A0=A1
1     continue
      SA(J)=AMAX
      SV(J)=VMAX
2     SD(J)=DMAX

      RETURN
END SUBROUTINE RSPS
!cccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccc

