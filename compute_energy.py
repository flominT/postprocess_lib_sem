#!/usr/bin/env python

"""
Compute Energy of signal
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
from scipy.signal import welch
import ipdb as db

dir_visla = '/Users/flomin/Desktop/thesis/simulations/Nice/layered/vislaPSV/'

obj = sem2dpack(dir_visla)

obj.read_seismo()
energy = [welch(obj.velocity[:,i],fs=1./obj.dt,nperseg=50)[1] for i in range(211)]
energy = np.array(energy)
plt.figure()
im = plt.imshow(energy.T,aspect='auto',origin='lower',\
                cmap='jet')
cb = plt.colorbar(im)
plt.ylim([0,20])
plt.show()

