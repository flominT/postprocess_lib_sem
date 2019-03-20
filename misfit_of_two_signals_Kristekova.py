# ------------------------------------------------------------------
#  Filename: misfit_of_two_signals_Kristekova.py
#  Purpose : Goodness of fit between two signals 
#  by Kristekova's criteria
#  Author  : Elif ORAL
#
# --------------------------------------------------------------------

# Librairies necessaires
import numpy 			   as     np
from   matplotlib.colors   import LinearSegmentedColormap
import matplotlib.colors   as     mcolors
# pour les filtres
from numpy                 import array, where, fft
from scipy.fftpack         import hilbert
from scipy.signal          import iirfilter, lfilter, remez, convolve, get_window
# pour le misfit
from   tf_misfit_elif      import *



# Filters #
# Copyright (C) 2009 Tobias Megies, Moritz Beyreuther, Yannik Behr #

# Low-pass filter of Butterworth
def lowpass(data, freq, df=200, corners=4, zerophase=False):
    """
    Butterworth-Lowpass Filter.
 
    Filter data removing data over certain frequency freq using corners 
    corners.
 
    :param data: Data to filter, type numpy.ndarray.
    :param freq: Filter corner frequency.
    :param df: Sampling rate in Hz; Default 200.
    :param corners: Filter corners. Note: This is twice the value of PITSA's 
        filter sections
    :param zerophase: If True, apply filter once forwards and once backwards.
        This results in twice the number of corners but zero phase shift in
        the resulting filtered trace.
    :return: Filtered data.
    """
    fe = 0.5 * df
    [b, a] = iirfilter(corners, freq / fe, btype='lowpass', ftype='butter',
                       output='ba')
    if zerophase:
        firstpass = lfilter(b, a, data)
        return lfilter(b, a, firstpass[::-1])[::-1]
    else:
        return lfilter(b, a, data)
#



# High-pass filter of Butterworth
def highpass(data, freq, df=200, corners=4, zerophase=False):
    """
    Butterworth-Highpass Filter.
 
    Filter data removing data below certain frequency freq using corners.
 
    :param data: Data to filter, type numpy.ndarray.
    :param freq: Filter corner frequency.
    :param df: Sampling rate in Hz; Default 200.
    :param corners: Filter corners. Note: This is twice the value of PITSA's 
        filter sections
    :param zerophase: If True, apply filter once forwards and once backwards.
        This results in twice the number of corners but zero phase shift in
        the resulting filtered trace.
    :return: Filtered data.
    """
    fe = 0.5 * df
    [b, a] = iirfilter(corners, freq / fe, btype='highpass', ftype='butter',
                       output='ba')
    if zerophase:
        firstpass = lfilter(b, a, data)
        return lfilter(b, a, firstpass[::-1])[::-1]
    else:
        return lfilter(b, a, data)
# 

# Colormap of Fabian

def make_colormap(seq):
    """Return a LinearSegmentedColormap
    seq: a sequence of floats and RGB-tuples. The floats should be increasing
    and in the interval (0,1).
    """
    seq = [(None,) * 3, 0.0] + list(seq) + [1.0, (None,) * 3]
    cdict = {'red': [], 'green': [], 'blue': []}
    for i, item in enumerate(seq):
        if isinstance(item, float):
            r1, g1, b1 = seq[i - 1]
            r2, g2, b2 = seq[i + 1]
            cdict['red'].append([item, r1, r2])
            cdict['green'].append([item, g1, g2])
            cdict['blue'].append([item, b1, b2])
    return mcolors.LinearSegmentedColormap('CustomMap', cdict)


def fabian_cmap ():
	# Making a colormap
	c    = mcolors.ColorConverter().to_rgb
	# cc = ['#ffffff', '#dfe6ff', '#a5b5da', '#516b8e', '#c5ce9b',
	#       '#fcab6c', '#f50000']
	cc = ['#f50000', '#fcab6c', '#c5ce9b', '#516b8e', '#a5b5da',
	      '#dfe6ff', '#ffffff']


	# cc1 = np.logspace(np.log10(0.25),np.log10(0.95),6)
	cc1 = np.linspace(0.75,1,6)

	cmap = make_colormap([c(cc[0]), c(cc[1]), cc1[0],
	                      c(cc[1]), c(cc[2]), cc1[1],
	                      c(cc[2]), c(cc[3]), cc1[2],
	                      c(cc[3]), c(cc[4]), cc1[3],
	                      c(cc[4]), c(cc[5]), cc1[4],
	                      c(cc[5]), c(cc[6]), cc1[5],
	                      c(cc[6])])
	return cmap	
#




###############
### PROGRAM ###
###############


# Name of the reference-signal file 
file_refer  = 'velocyz001'

# Name of the file of the signal to be compared to reference
file_input  = 'XREC1'




# Reading the files 
time = np.genfromtxt(file_input,usecols=0)

st1 = np.genfromtxt(file_input,usecols=1)
st2 = np.genfromtxt(file_refer,usecols=1) ###



# # Filtering
# # ATTENTION to dt (here assuming that both signals have same dt)
# st1= lowpass (st1, 10.0,  df=1/(time[1]-time[0]), corners=2, zerophase=True)
# st2= lowpass (st2, 10.0,  df=1/(time[1]-time[0]), corners=2, zerophase=True)



# Computation and Plotting
fabiancm = fabian_cmap()
print 'Attention to time step before entering into plot_tf_gofs'
print 'Check the frequency band used for filtering'

figure   =  plot_tf_gofs(st1, st2, dt=time[1]-time[0], t0=0.0, fmin=0.1, fmax=10.0, nf=100, w0=6,
                 norm='global', st2_isref=True, a=10., k=1., left=0.1,
                 bottom=0.1, h_1=0.2, h_2=0.125, h_3=0.2, w_1=0.2, w_2=0.6,
                 w_cb=0.01, d_cb=0.0, show=False, plot_args=['k', 'r', 'b'],
                 ylim=0., clim=0., cmap=fabiancm)



# Saving the figure
figure.savefig('blabla.png', dpi=300)


#