#!/usr/bin/env python3

import sys
import os
import ipdb as db

def make_dir(directory):
  if os.path.isdir(directory):
      print("directory alread exists")
  else :
      os.makedirs(directory)

def set_plot_param(option='tick',**kwargs):
  """ Set plot font parameters

         Input  :
           - option : [tick,label,c_tick,]
                  c_tickk for colorbar tick
         Return :
           - Dictionary of parameters set
  """

  assert option in ['tick','label','c_tick']

  if 'fontsize' in kwargs:
    fontsize = kwargs['fontsize']
  else:
    fontsize = 12


  if 'fontweight' in kwargs:
    fontweight = kwargs['fontweight']
  else:
    fontweight = 10


  if option == 'tick':
    tick_param = { 'axis'       : 'both',
                   'color'      : 'black',
                   'labelsize'  : fontsize,
                   'labelcolor' : 'black',
                  }
    return tick_param

  elif option == 'label':

    label_param = {   'fontname' : 'serif',
                   'fontsize'   : fontsize,
                   'color'      : 'black',
                   'fontweight' : fontweight
                   }
    return label_param

  elif option == 'c_tick':
    c_param ={ 'color'      : 'black',
               'labelsize'  : fontsize,
               'labelcolor' : 'black',
             }
    return c_param

def set_rcParams():
  import matplotlib.pyplot as plt

  plt.rcParams['axes.labelsize'] = 14
  plt.rcParams['axes.linewidth'] = 0.8
  plt.rcParams['axes.titlesize'] = 16
  plt.rcParams['axes.xmargin']   = 0.02
  plt.rcParams['axes.titlepad']  = 4
  plt.rcParams['figure.figsize'] = [8, 6]
  plt.rcParams['figure.subplot.bottom'] = 0.1
  plt.rcParams['figure.subplot.top']    = 0.93
  plt.rcParams['figure.subplot.hspace'] = 0.3
  plt.rcParams['figure.subplot.wspace'] = 0.2
  plt.rcParams['figure.titlesize']      = 18
  plt.rcParams['figure.titleweight']    = 'bold'
  #plt.rcParams['grid.color']            = '#101010'
  plt.rcParams['grid.alpha']             = 0.5
  plt.rcParams['grid.linewidth']        = 1
  plt.rcParams['legend.fontsize']       = 12
  #plt.rcParams['lines.markersize']      = 12
  plt.rcParams['lines.color']           = 'k'
  plt.rcParams['xtick.labelsize']       = 12
  plt.rcParams['ytick.labelsize']       = 12
  plt.rcParams['xtick.major.size']      = 4
  plt.rcParams['xtick.major.width']     = 1.15
  plt.rcParams['ytick.major.size']      = 4
  plt.rcParams['ytick.major.width']     = 1.15
  plt.rcParams['xtick.minor.width']     = 0.7
  plt.rcParams['xtick.minor.visible']   = True
  plt.rcParams['ytick.minor.visible']   = True
  #plt.rcParams['font.sans-serif']       = 'Arial'
  plt.rcParams['font.family']           = 'serif'
  plt.rcParams['patch.linewidth']       = 0.5
  plt.rcParams['text.usetex']           = True

def fourier(sig,dt,detrend=False):
    import numpy as np
    if detrend:
      detrend = np.subtract(sig,np.mean(sig))
    else:
      detrend = sig
    s = np.abs(np.fft.fft(detrend,axis=0))

    n = detrend.shape[0]
    f = np.fft.fftfreq(n,dt)
    
    if n%2:
      nf = int((n+1)/2)
    else:
      nf = int(n/2)
    
    fft_sig = s[:nf]
    fft_freq  = f[:nf]

    return fft_sig, fft_freq


