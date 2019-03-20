#!/usr/bin/env python3

"""
Module to generate source time functions.

"""

import warnings
import numpy as np

def ricker(f=10, length=0.5, dt=0.002, peak_loc=0.25, write=True, **kwargs):
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

def gabor(f=10, length=0.5, dt=0.002, peak_loc=0.25, psi=np.pi/4, write=True, **kwargs):
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
    y = A * np.cos( 2*np.pi*f*t + psi ) * np.exp( -1 * t**2 )

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

if __name__ == '__main__':
  gabor(plot=True)
  pass
