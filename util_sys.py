#!/usr/bin/env python3

import sys
import os
import ipdb as db
import numpy as np
import scipy
import resampy
import matplotlib.pyplot as plt

def make_dir(directory):
  if os.path.isdir(directory):
      print("directory alread exists")
  else :
      os.makedirs(directory)

def file_exist(filename):
  return os.path.isfile(filename)


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

  plt.rcParams['axes.labelsize'] = 13
  plt.rcParams['axes.linewidth'] = 0.8
  plt.rcParams['axes.titlesize'] = 15
  plt.rcParams['axes.xmargin']   = 0.02
  plt.rcParams['axes.titlepad']  = 4
  plt.rcParams['figure.figsize'] = [8, 6]
  plt.rcParams['figure.subplot.bottom'] = 0.15
  plt.rcParams['figure.subplot.top']    = 0.9
  plt.rcParams['figure.subplot.hspace'] = 0.1
  plt.rcParams['figure.subplot.wspace'] = 0.25
  plt.rcParams['figure.titlesize']      = 17
  plt.rcParams['figure.titleweight']    = 'bold'
  #plt.rcParams['grid.color']            = '#101010'
  plt.rcParams['grid.alpha']             = 0.5
  plt.rcParams['grid.linewidth']        = 1
  plt.rcParams['legend.fontsize']       = 12
  #plt.rcParams['lines.markersize']      = 12
  plt.rcParams['lines.color']           = 'k'
  plt.rcParams['lines.linewidth']       = 1
  plt.rcParams['xtick.labelsize']       = 10
  plt.rcParams['ytick.labelsize']       = 10
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


def fourier(sig,dt,detrend=True, axis=None):
    ndim = sig.ndim
    assert (ndim <= 2), "Data must be a 1D or 2D numpy array"
    axis = axis or 0
    
    if detrend:
        if ndim == 2:
            detrend = np.subtract(sig,np.mean(sig,axis=axis)[np.newaxis,:])
        elif ndim == 1:
            detrend = np.subtract(sig,np.mean(sig))
    else:
      detrend = sig
    
    s = np.abs(dt*np.fft.fft(detrend,axis=axis))

    n = detrend.shape[axis]
    f = np.fft.fftfreq(n,dt)

    if n%2:
      nf = int((n+1)/2)
    else:
      nf = int(n/2)

    if ndim == 2:
        return s[:nf,:], f[:nf]
    else:
        return s[:nf], f[:nf]

def fix_length(data, size, axis=-1, **kwargs):
      kwargs.setdefault('mode', 'constant')

      n = data.shape[axis]

      if n > size:
          slices = [slice(None)] * data.ndim
          slices[axis] = slice(0, size)
          return data[tuple(slices)]

      elif n < size:
          lengths = [(0, 0)] * data.ndim
          lengths[axis] = (0, size - n)
          return np.pad(data, lengths, **kwargs)

      return data

def resample_sig(y, orig_sr, target_sr, res_type='kaiser_best', fix=False, scale=False, **kwargs):
    # sampling rate in Hz
    if orig_sr == target_sr:
        return y

    ratio = float(target_sr) / orig_sr
    n_samples = int(np.ceil(y.shape[-1] * ratio))
  
    if res_type in ('scipy', 'fft'):
        y_hat = scipy.signal.resample(y, n_samples, axis=-1)
    elif res_type == 'polyphase':
        if int(orig_sr) != orig_sr or int(target_sr) != target_sr:
            raise Exception('polyphase resampling is only supported for integer-valued sampling rates.')

        # For polyphase resampling, we need up- and down-sampling ratios
        # We can get those from the greatest common divisor of the rates
        # as long as the rates are integrable
        orig_sr = int(orig_sr)
        target_sr = int(target_sr)
        gcd = np.gcd(orig_sr, target_sr)
        y_hat = scipy.signal.resample_poly(y, target_sr // gcd, orig_sr // gcd, axis=-1)
    else:
        y_hat = resampy.resample(y, orig_sr, target_sr, filter=res_type, axis=-1)

    if fix:
        y_hat = fix_length(y_hat, n_samples, **kwargs)

    if scale:
        y_hat /= np.sqrt(ratio)
    return y_hat

def plot_ssr(data,xmin,xmax,clim,title='',key='',suptitle='',**kwargs):
  set_rcParams()
  cmap = p_colormap()
  fig , ax = plt.subplots(figsize=(6,5))
  fig.subplots_adjust(top=0.9)
  im = ax.imshow(data.T,origin='lower',extent=[xmin,xmax,0,50], interpolation='bilinear',
        aspect='auto',vmin=clim[0],vmax=clim[1],cmap=cmap)
  cb = fig.colorbar(im)
  if 'ctitle' in kwargs.keys():
    ctitle = kwargs['ctitle']
  else:
    ctitle = 'Amplification'
  cb.set_label(ctitle, rotation=90, fontsize=14)
  cb.ax.minorticks_off()
  cb.ax.tick_params(axis='y',labelsize=10)
  #ax.tick_params(axis='x', which='minor', bottom=False)
  ax.tick_params(axis='both',labelsize=12)
  ax.set_ylim(0.2,12)
  ax.set_xlabel('Distance along the profile [m]',fontsize=15)
  ax.set_ylabel('Frequency [Hz]',fontsize=15)
  ax.set_title(title,fontsize=16)
  fig.suptitle(suptitle, x = 0.44, y=0.99, fontsize=16)
  return fig

def p_colormap():
  from matplotlib import colors
  filename = '/Users/flomin/Desktop/thesis/MyScripts/python/modules/syl.dat'
  name     = 'wsyl'
  palette = open(filename)
  lines = palette.readlines()
  carray = np.zeros([len(lines), 3])

  for num, line in enumerate(lines):
      carray[num, :] = [float(val) for val in line.strip().split()]

  cmap = colors.ListedColormap(carray, name=name)

  return cmap

def make_fig_legend(fig,color,label,ncol=1,loc=None):
  from matplotlib.lines import Line2D
  assert(len(color) == len(label))
  lines = []
  for c in color:
    lines.append(Line2D([0],[0], color=c))
  fig.legend(lines,label,ncol=ncol,loc=loc, fontsize=10)

if __name__ == '__main__':
    val = file_exist('README.md')
    print(val)

