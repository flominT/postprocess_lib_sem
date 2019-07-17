#!/usr/bin/env python3

"""
Module to generate source time functions.

"""

import warnings
import numpy as np
import matplotlib.pyplot as plt
import ipdb as db
from obspy.signal.invsim import cosine_taper
from houches_fb import *

def ricker(f=10, length=0.5, dt=0.002, peak_loc=0.25, write=True,plot=True, **kwargs):
  """ricker creates a shifted causal ricker wavelet (Maxican hat).
    :param f: center frequency of Ricker wavelet (default 10)
    :param length: float
    :type length: signal length in unit of second (default 0.5 sec)
    :param dt: float
    :type dt: time sampling interval in unit of second (default 0.002 sec)
    :param peak_loc: float
    :type peak_loc: location of wavelet peak in unit of second (default 0.25
    sec)
    :return: shifted Ricker wavelet starting from t=0.
    :rtype: np.ndarray
    Note that the returned signal always starts at t=0. For a different
    starting point, it can be achieved by shifting the time vector instead.
  """
  # Input check
  if f <= 0:
    raise ValueError("Center frequency (f) needs to be positive.")

  if length <= 0:
    raise ValueError("Signal length (len) needs to be positive.")

  if dt <= 0:
    raise ValueError("Time interval (dt) needs to be positive.")

  if length < peak_loc:
    warnings.warn("The peak location is outside the signal range. All "
                      "zero output will be provided.")
    return np.zeros(int(length / dt))
  else:
    # Generate time sequence based on sample frequency/period, signal length
    # and peak location
    t = np.linspace(-peak_loc, length - peak_loc - dt, int(length / dt))

    # Shift time to the correct location
    t_out = t + peak_loc  # time shift Ricker wavelet based on peak_loc

    # Generate Ricker wavelet signal based on reference
    y = (1 - 2 * np.pi ** 2 * f ** 2 * t ** 2) * np.exp(
            -np.pi ** 2 * f ** 2 * t ** 2)

    data = np.column_stack((t_out,y,np.zeros(y.shape)))

    # write data in a txt file
    if write:
      if 'directory' in kwargs:
        filename = directory + 'ricker_wavelet_fc_' + str(f)
        np.savetxt(filename,data,delimiter='\t')
      else:
        filename = 'ricker_wavelet_fc_' + str(f)
        np.savetxt(filename,data,delimiter='\t',fmt='%10.5f %10.5f %10.5f')

    return y, t_out

def plot_fft(y,t,plot=True):
    dt = t[1] - t[0]
    # FFT transform
    nf   = int( len(y)/2 )
    df   = (1/(2.0 * dt)) / nf
    freq = np.arange(nf) * df

    fft  = np.abs(np.fft.fft(y))[:nf]

    if plot:
      fig, ax = plt.subplots(1,2)
      ax[0].plot(t,y)
      ax[1].plot(freq,fft)
      ax[1].set_xlim(0,50)
      plt.show()

def gabor(f=10, length=0.5, dt=0.002, peak_loc=0.25, write=True, **kwargs):
  """
  gabor creates a shifted causal gabor wavelet.

  ** input params:
    - f [float]        : central frequency of the wavelet
    - length [float]   : duration or length of the wavelet
    - dt [float]       : time sampling interval in units of seconds
    - peak_loc [float] : location of central peak
    - write [bool]     : Boolean to activate saving mode

  ** returns:
    - t_out : output time vector
    - w     : gabor wavelet
  """

  # Check input
  if f <= 0:
    raise ValueError('Wavelet central frequency must be positive')
  if length <= 0:
    raise ValueError('Length or duration must be positive')
  if dt <= 0:
    raise ValueError('time sampling step must be positive')
  if psi < 0 or psi > np.pi:
    raise ValueError('psi must be between 0 and pi')

  if length < peak_loc:
    warnings.warn('The peak location is outside the signal range.'
                   'Zeros will be return as values')
    return np.zeros(int(length/dt))
  else:
    # Generate time vector
    t = np.linspace( -peak_loc, length - peak_loc -dt ,int(length/dt) )
    t_out = t + peak_loc  # shift time to correct location

    # Generate gabor wavelet
    if 'norm' in kwargs:
      A = kwargs["norm"]
    else: A = 1
    y = A * np.cos( 2*np.pi*f*t) * np.exp( -1 * t**2 )

    data = np.column_stack((t_out,y,np.zeros(y.shape)))

    # write data in a txt file
    if write:
      if 'directory' in kwargs:
        filename = directory + 'gabor_wavelet_fc_' + str(f)
        np.savetxt(filename,data,delimiter='\t')
      else:
        filename = 'gabor_wavelet_fc_' + str(f)
        np.savetxt(filename,data,delimiter='\t',fmt='%10.5f %10.5f %10.5f')

    if "plot" in kwargs:
      import matplotlib.pyplot as plt
      plt.figure()
      plt.plot(t_out,y,c='r')
      plt.xlabel('time [s]')
      plt.ylabel('amplitude')
      plt.title('gabor wavelet')
      plt.show()

  return y ,  t_out


def write(t,y,name):
  data = np.column_stack((t,y,np.zeros(y.shape)))

  filename = name
  np.savetxt(filename,data,delimiter='\t',fmt='%10.6f %10.6f %10.6f')

def gtrunc(pga, fn, dt, N, Ns):
  #taper = 0.5 * ( 1 + np.cos(np.linspace(np.pi , 2 * np.pi, N-Ns)))
  t = np.arange(N) * dt
  a = np.zeros(N)
  w = 2 * np.pi * fn
  cos = np.cos(w * t[:-Ns] + np.pi/2)
  sin = np.sin(w * t[:-Ns] + np.pi/2)
  gauss = np.exp( -1 * w * t[:-Ns]**2 )
  x = sin  * gauss #* cosine_taper(N-Ns,0.01)

  #plt.figure()
  #plt.plot(cos,'r')
  #plt.plot(gauss,'b')
  #plt.plot(x,'g')
  #plt.plot(expo,'k')
  #plt.show()

  a[Ns:] = pga * (x / np.abs(x).max())

  print(x.max(),x.min())
  print(a.max(),a.min())

  return t, a

def fftd1(x, dt, pp, n):
  nt = len(x)
  x  = x - np.mean(x)
  # Tapering the data

  if pp > 0.0:
    x = taper(x, pp)

  # FFT
  df = 1.0 / ((nt-1) * dt)
  nf = nt // 2 + 1
  s  = np.fft.fft(x)[:nf]
  f  = np.arange( nf ) * df

  # mulomega

  w = 2.0 * np.pi * f * 1j
  s = s * w**n

  # IFFT

  if np.remainder(nt,2) != 0: # odd
    y = s[1:nf]
  else:                       # even
    y = s[1:nf-1]

  y  = np.conj( y[::-1] )
  zz = np.real(np.fft.ifft( np.append(s,y) ))

  return zz

def gab(fn, dt, N, Ns):
  t  = np.arange(N) * dt
  a  = np.zeros( len(t) )
  w  = 2.0 * np.pi * fn
  g  = 2.0
  ts = 1.0
  x  = np.exp(-(w*(t[:-Ns]-ts)/g)**2) * np.cos(w*(t[:-Ns]-ts) + np.pi/2)

  a[Ns:] = x

  return t, a

"""
if __name__ == '__main__':
  #############
  # Main code #
  #############

  dt   = 0.001
  N    = 10000
  Ns   = 1000
  f0   = 1.0
  fu   = 20.0

  t, a = gab(100, dt, N, Ns)

  # Derivative in the frequency domain (better than in the time domain)

  a = fftd1(a, dt, 0.025, 2)

  # Filtrage (you use obspy, and I use SAC)
  # Be careful, acausal filtering (one pass only!!!)

  #a = sac(a, 'BU', 0.0, 0.0, 10, 'LP', f0, fu, dt, 1)
  a = 100 * a / np.abs(a).max()
  print(max(a),min(a))
  # plot
  plt.figure()
  plt.plot(t,a)
  plt.show()

"""

if __name__ == '__main__':
  # Read source time
  sf = '/Users/flomin/Desktop/thesis/simulations/Nice/plane_wave/elast_sh/source_bis'
  src = np.genfromtxt(sf)
  mx = np.max(np.abs(src[:,1]))
  mt = np.max(src[:,0])
  dt = src[1,0] - src[0,0]
  npts = src.shape[0]
  t1, y = gtrunc(mx, 6.5, dt, npts, 2500)
  sf_fft = np.abs(np.fft.fft(src[:,1])) * dt
  print(np.allclose(t1,src[:,0]))
  # Plot
  plt.figure()
  plt.plot(src[:,0],src[:,1],'r')
  plt.plot(t1,y)
  plt.show()

  #write(t1,y,'source6')
  plot_fft(y,t1)

