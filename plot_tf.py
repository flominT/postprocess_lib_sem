#!/usr/bin/env python
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
from matplotlib import style
import matplotlib as mpl
style.use('ggplot')

def plot_tf(tf,f,rcoord,clim=None,fmax=10,savefile=None,cmap='jet',ctitle='Amplification',**kwargs):
  #sns.set_style('whitegrid')
  #cmap = mpl.cm.get_cmap(cmap)
  #cmaplist = [cmap(i) for i in range(cmap.N)]
  #cmaplist[0] = (1,1,1,1.0)
  #cmap = mpl.colors.LinearSegmentedColormap.from_list('Custom',cmaplist,cmap.N)
  #bounds = np.linspace(1.05,7,256)
  #norm = mpl.colors.BoundaryNorm(bounds,cmap.N)
  fig = plt.figure(figsize=(8,6))
  fig.subplots_adjust(hspace=0.35)
  ax = fig.add_subplot(111)
  xcoord = rcoord
  if clim :
    cmin = clim[0]
    cmax = clim[1]
    im = ax.imshow(tf, cmap=cmap, aspect='auto', interpolation='bilinear',vmin=cmin, vmax=cmax, \
                    origin='lower', extent=[min(xcoord),max(xcoord), min(f),max(f)])
    #im = ax.imshow(tf, cmap=cmap, aspect='auto', interpolation='bilinear',norm=norm, \
    #                origin='lower', extent=[min(xcoord),max(xcoord), min(f),max(f)])
  else :
    im = ax.imshow(tf, cmap=cmap, aspect='auto', interpolation='bilinear',\
           origin='lower', extent=[min(xcoord),max(xcoord), min(f),max(f)])

  if 'xlim' in kwargs   :
    ax.set_xlim(kwargs['xlim'][0], kwargs['xlim'][1])
  else:
    ax.set_xlim(0,max(xcoord))

  if 'ylim' in kwargs   :
    ax.set_ylim(kwargs['ylim'][0], fmax)
  else :
    ax.set_ylim(0.1,fmax)

  if 'ylabel' in kwargs :
    ax.set_ylabel(kwargs['ylabel'], fontsize=16)
  else :
    ax.set_ylabel('frequency [Hz]', fontsize=16)
  if 'xlabel' in kwargs :
    ax.set_xlabel(kwargs['xlabel'], fontsize=16)
  else:
    ax.set_xlabel('Horizontal profile [m]', fontsize=16)
  if 'title' in kwargs  :
    ax.set_title(kwargs['title'],fontsize=18)
  else :
    ax.set_title('Transfer function',fontsize=18)

  if clim :
    cb = fig.colorbar(im, shrink=0.5, aspect=10, pad=0.01, ticks=np.linspace(cmin,cmax,cmax+1), \
                  boundaries=np.linspace(cmin,cmax,cmax+1))
  else :
    cb = fig.colorbar(im, shrink=0.5, aspect=10, pad=0.01)
  cb.set_label(ctitle, labelpad=20, y=0.5, rotation=90, fontsize=14)

  if savefile != None:fig.savefig(savefile,dpi=300)

  plt.show()

