#!/usr/bin/env python

"""
Post processing script
"""

from sem2d import sem2dpack,tf_error
import sys 
sys.path.append('/Users/flomin/Desktop/thesis/MyScripts/python/modules')
import functions as fnc
import matplotlib.pyplot as plt
import numpy as np
import ipdb as db
import wiggle as wig
from plot_tf import plot_tf
import tf_misfit as mtf
import pickle



# Directories
#---------
dir_visla = '/Volumes/flomin/test/shVisla/'
dir_elast = '/Volumes/flomin/test/psvElast/'
save_dir  = '/Users/flomin/Desktop/thesis/figures/'

save = False 
plot = False 
wiggle = False
TF  = False
TFdiff = True 
source = False
nsurf, niter, dt, fmax = 420, 1500, 1e-2, 10
# Save Filenames 
#---------
visla_wiggle_filename = save_dir + 'visla_layered_psvWiggle.png'
elast_wiggle_filename  = save_dir + 'elast_layered_psvWiggle.png'

visla_tf_filename = save_dir + 'visla_layered_SHTF.png'
elast_tf_filename  = save_dir + 'elast_layered_SHTF.png'

diff_filename = save_dir + 'TF_diff.png'

srcfilename  = save_dir + 'source.png'

# Plot parameters
#---------
title_wiggle_elast  = 'Elast layered SH Wiggle'
title_wiggle_visla  = 'Visla layered SH Wiggle'

title_tf_elast  = 'SEM2D Elastic SH transfer function'
title_tf_visla  = 'SEM2D Viscoelastic SH transfer function'
title_tf_vislafd = 'Finite difference Viscoelastic transfer function'
title_tf_vislagd = 'Discontinuous Galerkin Viscoelastic transfer function'

xlabel = 'horizontal distance along profile [m]'
ylabel = 'frequency [Hz]'

wiggle_xlim   = [0,420]
wiggle_ylim   = [0,4]

tf_xlim = [250,1750]

# Create sem2d objects
#----------
vislapsv = sem2dpack(dir_visla)
elastpsv = sem2dpack(dir_elast)

# Plot source spectrum
#---------
if source: vislapsv.plot_source()

# Plot Wiggle
#---------
if wiggle:
  elastpsv.plot_wiggle(420,compo='x',sf=0.2,save=save,save_dir=elast_wiggle_filename,\
                    ylim=wiggle_ylim,title=title_wiggle_elast)
  vislapsv.plot_wiggle(420,compo='x',sf=0.2,save=save,save_dir=visla_wiggle_filename,xlim=wiggle_xlim,\
                    ylim=wiggle_ylim,title=title_wiggle_visla)
  plt.show()
# Compute TF
#---------
if TF:
  elast_tf = elastpsv.compute_tf(420,(450,1650),fmax=12,compo='y')
  visla_tf = vislapsv.compute_tf(420,(450,1650),fmax=fmax,compo='y')
  
  # Plot TF
  #---------
  if plot:
    elastpsv.plot_tf(fmax=10,savefile=elast_tf_filename,xlabel=xlabel,\
                  ylabel=ylabel,title=title_tf_elast)
    vislapsv.plot_tf(fmax=10,savefile=visla_tf_filename,xlabel=xlabel,\
                 ylabel=ylabel,title=title_tf_visla)

# Compute TF diff
#-----------

if TFdiff:
  tf_diff = tf_error(visla_tf,fd_spec)

  # Plot TF diff
  #-----------
  xcoord = elastpsv.rcoord[:,0]
  figd = plt.figure()
  ax = figd.add_subplot(111)
  im = ax.imshow(tf_diff, origin='lower',aspect='auto',interpolation='bilinear',\
               cmap='jet',extent=[min(xcoord),max(xcoord),0,50])
  ax.set_ylim(0.1,12)
  cb = figd.colorbar(im, shrink=0.5, aspect=10, pad=0.01)
  ax.set_xlabel('Distance [m]')
  ax.set_ylabel('frequency [Hz]')
  ax.set_title('Amplitude difference between Elast and Visla Layered TF')
  if save : plt.savefig(diff_filename,dpi=300)
  plt.show()






