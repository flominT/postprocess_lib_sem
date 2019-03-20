#!/usr/bin/env python3

import numpy as np
import matplotlib.pyplot as plt
import ipdb as db

def gen_2dsig(x,y):
  """
  Generate an arbitrary 2d function
  """
  #f = 10.0
  z = np.zeros( (len(x),len(y)) )
  z = (x[:,np.newaxis] + 2* y[np.newaxis,:])  + 3
  return z

def fft_sym(A,B):
  """
  Define 2d symetries for ifft2
  """
  Nx,Nz = A.shape
  Nx2,Nz2 = int(Nx/2), int(Nz/2)
  if ( (Nx % 2) != 1 ) and ( (Nz % 2) != 1 ) : # if both are even
    A[:Nx2+1,:Nz2+1] = B
    A[Nx2+1:,:Nz2+1] = np.conj( np.flip(B[1:Nx2,:Nz2+1],axis=0) )
    A[Nx2+1:,Nz2+1:] = np.flip( A[Nx2+1:,:Nz2-1],axis=1 )
    A[:Nx2+1,Nz2+1:] = np.conj( np.flip(B[:Nx2+1,1:Nz2],axis=1) )

  elif ( (Nx % 2) == 1 ) and ( (Nz % 2) == 1 ) : # if both are odd
    A[:Nx2+1,:Nz2+1] = B
    A[Nx2+1:,:Nz2+1] = np.conj( np.flip(B[1:Nx2+1,:Nz2+1],axis=0) )
    A[Nx2+1:,Nz2+1:] = np.flip( A[Nx2+1:,1:Nz2+1],axis=1 )
    A[:Nx2+1,Nz2+1:] = np.conj( np.flip(B[:Nx2+1,1:Nz2+1],axis=1) )

  elif ( (Nx % 2) == 1 ) and ( (Nz % 2) != 1 ) : # if x is odd
    A[:Nx2+1,:Nz2+1] = B
    A[Nx2+1:,:Nz2+1] = np.conj( np.flip(B[1:Nx2+1,:Nz2+1],axis=0) )
    A[Nx2+1:,Nz2+1:] = np.flip( A[Nx2+1:,1:Nz2],axis=1 )
    A[:Nx2+1,Nz2+1:] = np.conj( np.flip(B[:Nx2+1,1:Nz2],axis=1) )

  elif ( (Nx % 2) != 1 ) and ( (Nz % 2) == 1 ) : # if z is odd
    A[:Nx2+1,:Nz2+1] = B
    A[Nx2+1:,:Nz2+1] = np.conj( np.flip(B[1:Nx2,:Nz2+1],axis=0) )
    A[Nx2+1:,Nz2+1:] = np.flip( A[Nx2+1:,1:Nz2+1],axis=1 )
    A[:Nx2+1,Nz2+1:] = np.conj( np.flip(B[:Nx2+1,1:Nz2+2],axis=1) )

  return A


if __name__ == '__main__':
  x1 = np.arange(200)*1e-5   #np.fft.fftfreq(200,d=0.0001)
  y1 = np.arange(200)*0.25
  z1 = gen_2dsig(x1[:101],y1[:101])
  z2= gen_2dsig(x1,y1)

  z1_fft = np.fft.fft2(z1)
  z2_fft = np.zeros(z2.shape,dtype=np.complex_)
  z2_fft = fft_sym(z2_fft,z1_fft)

  z2_inv = np.real(np.fft.ifft2(z2_fft))

  print(np.allclose(z2_inv,z2,rtol=1e-15))

  db.set_trace()


