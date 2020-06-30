#!/usr/bin/env python

"""
@Author :: Flomin T.

Class for manipulating SEM2DPACK output files.
  see user manual for more about SEM2DPACK code.
"""

import sys
import numpy as np
import time
import matplotlib.pyplot as plt
import glob
from scipy.interpolate import griddata as gd
import scipy.signal as sp
import matplotlib.animation as anim
import multiprocessing as mp
import os
import ipdb as db
import wiggle as wig
from filters import bandpass
import pandas as pd
from stockwell import st
import warnings
from util_sys import *
import fcode as fc
from dtw import accelerated_dtw as dtw
from scipy.spatial.distance import euclidean
from obspy.signal.konnoohmachismoothing import konno_ohmachi_smoothing as konno
from mpl_toolkits.axes_grid1 import make_axes_locatable
warnings.filterwarnings("ignore",category=DeprecationWarning)


class sem2dpack(object):

  """
    Class to create fortran code (SEM2DPACK) simulation objects.
    It defines a set of instance attribute and instance methods for the post-processing and visualizing
    each simulation outputs.

    Attributes ::
    -------------

    -- directory : the simulation directory
    -- mdict     : dictionary containing spectral element grid infos
    -- dt        : simulation time step
    -- npts      : Number of points in record, npts * dt gives
	                   the simulation time
    -- nsta      : number of reciever stations
    -- velocity  : velocity traces
    -- tvec      : time vector (0:dt:npts*dt)
    -- fmax      : maximum frequency of simulation
    -- tf        : transfer function in case of sedimentary basins
    -- fft_freq  : frequecy vector
    -- rcoord    : reciever stations coordinates

    Instance methods (short description) ::
    ----------------


    Static methods ::
    -----------------


  """

  def __init__(self,directory,freqs=[0.1,12],component='x'):
    self.directory = directory
    self.mdict = {}
    self.dt = 0.0
    self.npts = 0
    self.nsta = 0
    self.tvec = np.array([])
    self.fmax = 0.0
    self.interpolated_tf = np.array([])
    self.interpolated_f = np.array([])
    self.rcoord = np.array([])
    self.x_interp = np.array([])
    self.vs_int = np.array([])
    self._component = ''
    self._freqs = freqs
    self._component = component

    try:
      self.__read_Specgrid()
      self.__read_header()
    except:
      print(self.directory)
      raise Exception('Not a sem2dpack simulation directory')

  def __read_Specgrid(self):
    """
      Read the properties of the spectral element grid and store them in a dictionary "self.mdict"
    """
    #read mesh information
    filename = self.directory + 'grid_sem2d.hdr'
    g=np.genfromtxt(filename,dtype=int)
    nel,npgeo,ngnod,npt,ngll = g[1,:]

    #read spectral element grid coordinates
    filename = self.directory + 'coord_sem2d.tab'
    g=np.genfromtxt(filename,skip_header=1)
    coord = g[:,1:]

    #read ibool file
    filename = self.directory + 'ibool_sem2d.dat'
    with open(filename,'rb') as f:
      ibool=np.fromfile(f,np.int32).reshape((ngll,ngll,nel),order='F')

    #read gll information
    filename = self.directory + 'gll_sem2d.tab'
    g=np.genfromtxt(filename)
    x, w, h = g[0,:], g[1,:], g[2:,:]
    self.mdict ={"nel" : nel,  # Number of elements in mesh
            "npgeo" : npgeo,   # Number of global nodes
            "ngnod" : ngnod,   # Polynomial order
            "npt" : npt,       # Number of points in spectral mesh
            "ngll" : ngll,     # Number of gll points
            "coord" : coord,   # Coordinates of all global nodes points
            "ibool" : ibool,   # Array for global to local mapping (node number of each element [ngll,ngll,nel])
            "x" : x,  # GLL coordinates on the reference element [-1,1]
            "w" : w,  # weights of GLL polynomials
            "h" : h,  # derivatives of Lagrange polynomials

            }

  def __read_header(self):
    """
      Read seismic header file of SEM2DPACK simulation.
      The method broadcasts the simulation parameters and
      receiver coordinates instances.

      Upon exit, the method updates the following instances:

        self.dt : simulation time step
        self.npts : number of points in recoord
        self.nsta : number of receivers in the simulation
        self.rcoord : coordinates of receivers
        self.x_rcoord : if extra receivers are present (e.g receivers which store strain & strain)

    """

    filename = self.directory + 'SeisHeader_sem2d.hdr'
    try :
      f = open(filename, 'r')
    except:
      msg = 'No Header file <SeisHeader_sem2d.hdr> in directory'
      print(msg)
      answer = input("Do you want to continue [Y/N] : ")
      if answer.upper() == 'Y':
        return
      else:
        sys.exit()

    f.readline()
    string = f.readline()
    header_line   = string.rstrip(" ").split()

    self.dt    = float(header_line[0])
    self.npts = int(header_line[1])
    self.nsta = int(header_line[2])

    # Seismos
    f.readline()
    self.rcoord  = np.zeros((self.nsta,2))
    for reciever in np.arange(self.nsta):
      string = f.readline()
      reciever_line   = string.rstrip(" ").split()
      # x-coord
      self.rcoord[reciever,0] = float(reciever_line[0])
      # z-coord
      self.rcoord[reciever,1] = float(reciever_line[1])

    #extra station
    try:
      xsta = int(f.readline())
      self.xsta = xsta
      f.readline()
      self.x_rcoord = np.zeros((xsta,2))

      for ex_reciever in range(xsta):
        xtra = f.readline()
        x_reciever_line = xtra.rstrip(" ").split()
        self.x_rcoord[ex_reciever,0] = float(x_reciever_line[0])
        self.x_rcoord[ex_reciever,1] = float(x_reciever_line[0])
    except :
      print("No Extra recievers")
      self.x_rcoord = None

    f.close()
    return self.dt, self.npts, self.nsta, self.rcoord, self.x_rcoord

  @staticmethod
  def readField(fname):
    """
      Staticmethod which reads the snapshots binary files of a simulation.
    """
    with open(fname,'rb') as f:
      field = np.fromfile(f,np.float32)
    return field

  def read_seismo(self,filter_s=False,freqs=None,scale=False,verbose=False):
    """
       Reads the seismograms or traces the simulations

       Parameters ::
       -------------
         -- filter_s [dtype:bool] : if True seismograms are bandpassed filtered between freqs range
         -- freqs [dytpe:list]    : limits of frequency range for filtering.

       Upon exit, the method updates the following instances:
         -- self.velocity :: velocity traces
         -- self.tvec     :: time vector

    """

    if self._component == 'x':   # PSV case
      filename_x   = self.directory + 'Ux_sem2d.dat'
      filename_z = self.directory + 'Uz_sem2d.dat'

      try :
        with open(filename_x, 'rb') as fid:
          veloc_array_x = np.fromfile(fid,np.float32)
        with open(filename_z, 'rb') as fid:
          veloc_array_z = np.fromfile(fid,np.float32)
      except :
        raise Exception('No velocity file in {:s}'.format(self.directory))

      l = len(veloc_array_x)
      self.velocity = np.zeros((self.npts,self.nsta))
      self.velocity_z = np.zeros((self.npts,self.nsta))

      for i in range(int(l/self.nsta)):
        limit1 = i*self.nsta
        limit2 = (i+1)*self.nsta
        self.velocity[i,:] = veloc_array_x[limit1:limit2]
        self.velocity_z[i,:] = veloc_array_z[limit1:limit2]

      self.tvec = np.arange(self.npts) * self.dt

      if filter_s :
        freqs = freqs or self._freqs
        if verbose:
          print('*** Filtering between {} and {} Hz'.format(freqs[0],freqs[1]))
        self.velocity = self.filter_seismo(self.velocity,freqs=freqs,ftype='bandpass',dt=self.dt)
        self.velocity_z = self.filter_seismo(self.velocity_z,freqs=freqs,ftype='bandpass',dt=self.dt)
        return self.velocity
    elif self._component == 'y': # SH case
      filename_y   = self.directory + 'Uy_sem2d.dat'
      try :
        with open(filename_y, 'rb') as fid:
          veloc_array = np.fromfile(fid,np.float32)
      except :
        raise Exception('No velocity file in {:s}'.format(self.directory))

      l = len(veloc_array)
      self.velocity = np.zeros((self.npts,self.nsta))

      for i in range(int(l/self.nsta)):
        limit1 = i*self.nsta
        limit2 = (i+1)*self.nsta
        self.velocity[i,:] = veloc_array[limit1:limit2]

      self.tvec = np.arange(self.npts) * self.dt

      if filter_s :
        freqs = freqs or self._freqs
        if verbose:
          print('*** Filtering between {} and {} Hz'.format(freqs[0],freqs[1]))
        self.velocity = self.filter_seismo(self.velocity,freqs=freqs,ftype='bandpass',dt=self.dt)
        return self.velocity

    if scale:
      self.velocity /= scale
    return self.velocity

  def read_stress_strain(self):
    """
      Reads stress and strain information.

      Defines the followinginstances:
        -- self.stress
        -- self.strain
    """

    stress_file = self.directory + 'EXTRA_stress_sem2d.dat'
    strain_file = self.directory + 'EXTRA_strain_sem2d.dat'

    if os.path.isfile(stress_file) :
      with open(stress_file, 'rb') as sid :
        stress = np.fromfile(sid,np.float32)
      with open(strain_file, 'rb') as sid :
        strain = np.fromfile(sid,np.float32)

      l = len(stress)
      assert self.npts == (l/self.xsta)

      self.stress = np.zeros( (self.npts,self.xsta) )
      self.strain = np.zeros( (self.npts,self.xsta) )

      for i in range(int(l/self.xsta)):
        limit1 = i * self.xsta
        limit2 = (i+1) * self.xsta
        self.strain[i,:] = strain[limit1:limit2]
        self.stress[i,:] = stress[limit1:limit2]

      return self.stress, self.strain
    else:
      print("No stress and strain files were found")

  def read_iai_param(self):
    """
      Reads shear modolus, deviatoric stress, and S paramaters of the
      Iai model.

      Defines the following instances :
        -- self.shear_mod
        -- self.deviatoric_stress
        -- self.s_param
    """

    shear_mod_file  = self.directory + 'EXTRA_current_shear_modulus_sem2d.dat'
    dev_stress_file = self.directory + 'EXTRA_deviatoric_stress_sem2d.dat'
    s_param_file    = self.directory + 'EXTRA_S_parameter_sem2d.dat'

    if os.path.isfile(shear_mod_file):
      with open(shear_mod_file, 'rb') as sid:
        shear_mod = np.fromfile(sid,np.float32)
      with open(dev_stress_file, 'rb') as sid:
        deviatoric_stress = np.fromfile(sid, np.float32)
      with open(s_param_file, 'rb') as sid:
        s_param = np.fromfile(sid,np.float32)

      l = len(shear_mod)

      assert self.npts == (l/self.xsta), 'Recording error'

      self.shear_mod = np.zeros( (self.npts,self.xsta) )
      self.deviatoric_stress = np.zeros( (self.npts,self.xsta) )
      self.s_param = np.zeros( (self.npts,self.xsta) )

      for i in range( int(l/self.xsta) ):
        limit1 = i * self.xsta
        limit2 = (i+1) * self.xsta

        self.shear_mod[i,:] = shear_mod[limit1:limit2]
        self.deviatoric_stress[i,:] = deviatoric_stress[limit1:limit2]
        self.s_param[i,:] = s_param[limit1:limit2]

    else :
      print('No Iai model parameter files found')

  def decimate_sig(self,q=4,filter_s=True):
    """
      Decimate velocity traces by a factor q,

    """

    self.read_seismo(filter_s=filter_s)
    fe = 1/self.dt
    deci_fe = fe / q
    deci_dt = 1/deci_fe
    self.decimated_veloc = sp.decimate(self.velocity,q,n=4,axis=0)
    n = self.decimated_veloc.shape[0]
    self.deci_tvec = np.arange(n) * deci_dt
    return


  def compute_fft(self,filt=True,freqs=[0.1,10.0],axis=0):
    """
      Compute the Fourier of all the traces.

      Parameters
      ----------
        -- field ['V','D','A'] :: field on which to compute fft
      Defines the following instances :
        -- self.fft_sig
        -- self.fft_freq

    """

    if hasattr(self,'velocity') :
      veloc = self.velocity
    else :
      if filt:
        veloc = self.read_seismo(filter_s=True)
      else:
        veloc = self.read_seismo(filter_s=False)


    detrend = np.subtract(veloc,np.mean(veloc,axis=axis)[np.newaxis,:])
    s = np.abs(self.dt*np.fft.fft(detrend,axis=axis))

    n = detrend.shape[0]
    f = np.fft.fftfreq(n,self.dt)

    if n%2:
      nf = int((n+1)/2)
    else:
      nf = int(n/2)

    self.fft_sig = s[:nf,:]
    self.fft_freq  = f[:nf]

    return s[:nf,:],f[:nf]


  @staticmethod
  def interp(field,coord):
    """
    Interpolates field over a meshgrid.
    The meshgrid's size depends on the argument coord.
    """

    xcoord = coord[:,0]
    zcoord = coord[:,1]
    ext = [min(xcoord), max(xcoord), min(zcoord), max(zcoord)]
    x,z = np.meshgrid(np.linspace(ext[0],ext[1],1000),np.linspace(ext[2],ext[3],1000),sparse=True)
    y = gd((xcoord,zcoord),field,(x,z),method='linear')
    y =np.flipud(y)

    return y

  def animate(self,savefile=None,cmap='jet',interval=1500,repeat_delay=1000,duration=1, background=True):
    """
    Animates SEM2DPACK snapshots
    """

    # Plot parameters

    filename = "v"+self._component+"_*_sem2d.dat"
    coord  = self.mdict["coord"]
    xcoord = coord[:,0]
    zcoord = coord[:,1]
    frames = sorted(glob.iglob(self.directory + filename))
    nframe = int(len(frames)/4)
    ext = [min(xcoord), max(xcoord), -1 * max(zcoord), min(zcoord)]
    ims = []
    field =[]

    for i in range(nframe):
      f = self.readField(frames[i])
      field.append(f)

    pool = mp.Pool(processes=os.cpu_count()) # Initializes a pool of processes

    results = [pool.apply_async(self.interp,args=(x,coord)) for x in field]  # run the processes
    output  = [p.get() for p in results]  # retrieve processes information

    duration = duration or (self.npts * self.dt)

    fig, ax = plt.subplots(figsize=(10,4))

    if background:
      vsfile = self.directory + 'Cs_gll_sem2d.tab'
      with open(vsfile,'r') as v:
        vs_int = pd.read_csv(v,sep='\s+',names=['vs','x','z'])
      ax.scatter(vs_int['x'], vs_int['z'], c=vs_int['vs'], s=20, cmap='jet')

    Writer = anim.writers['ffmpeg']
    writer = Writer(fps=2,metadata=dict(artist='Flomin'))
    vmin , vmax = -5e-10, 5e-10

    for i in range(nframe):
      frametitle = 'Snapshot at time = {:.1f} secs'.format((i/nframe)*duration)
      ttl = ax.text(0.5, 1.01, frametitle, ha='center', \
                     va='bottom', transform=ax.transAxes,fontsize=18)
      im = ax.imshow(output[i],extent=ext,cmap=cmap,\
                      aspect="auto",animated=True,vmin=vmin,vmax=vmax)

      im.set_clim(vmin,vmax)
      ims.append([im,ttl])
      ims.append([im,])

    ani = anim.ArtistAnimation(fig,ims,interval=interval,blit=False,
                              repeat_delay=repeat_delay)

    ax.set_xlabel('Length [m]')
    ax.set_ylabel('Depth [m]')
    ax.minorticks_off()
    ax.invert_yaxis()

    c= plt.colorbar(im, fraction=0.1,pad=0.08,shrink=0.8)
    c.set_label('particle velocity $[ms^{-1}]$')

    if savefile : ani.save(savefile,writer=writer,savefig_kwargs={'pad_inches':0.01})
    plt.show(block=True)

    return


  def plot_snapshot(self,filename,savefile=None,cmap='jet'):
    """
      Plot the snapshot a particular time
    """
    frame_names = "v"+self._component+"_*_sem2d.dat"
    nframe = len(sorted(glob.iglob(self.directory + frame_names))) - 1
    duration = self.dt * self.npts

    if not isinstance(filename,str) :
      raise Exception('TypeError : filename must be string ')
    else :
      frame_number = int(filename.split('_')[1])
      filename = self.directory + filename

    field = self.readField(filename)
    coord = self.mdict["coord"]
    xcoord = coord[:,0] ; zcoord = coord[:,1]
    y = self.interp(field,coord)
    vmin = np.nanmin(y)
    vmax = np.nanmax(y)
    a_ratio = (np.max(zcoord) - np.min(zcoord)) / (np.max(xcoord) - np.min(xcoord)) # aspect ratio

    fig, ax = plt.subplots()
    im = ax.imshow(y,extent=[min(xcoord)/1e3, max(xcoord)/1e3, min(zcoord), max(zcoord)],cmap=cmap,
                  vmin=vmin,vmax=vmax, aspect=a_ratio)
    plt.tight_layout
    c=plt.colorbar(im,format='%.0e', fraction=0.046, pad=0.06, shrink=0.4)
    plt.ylabel('Depth [m]')
    plt.xlabel('Length [m]')
    c.set_clim(vmin,vmax)
    c.set_label('Particle velocity $ms^{-1}}$')
    plt.title('Snapshot at t = {:.3f} $sec$'.format( (frame_number/nframe)*duration) )
    if savefile : plt.savefig(savefile,dpi=300)
    plt.show()


  def plot_wiggle(self,ssta=None,sf=None,savefile=None,stride=1,axis=None,**kwargs):
    if isinstance(ssta,int):
      begin = 0
      end   = ssta -1
    elif isinstance(ssta,(list,tuple)):
      begin = ssta[0]
      end   = ssta[1]-1
    else:
      begin, end = 0, -1

    xx = self.rcoord[begin:end:stride,0]
    if not hasattr(self,'velocity'):
      print("Re-reading traces")
      self.read_seismo(filter_s=True)

    if sf != None:
      axis = wig.wiggle(self.velocity[:,begin:end:stride],self.tvec,sf=sf,axis=axis)
    else :
      axis = wig.wiggle(self.velocity[:,begin:end:stride],self.tvec,axis=axis)


    return axis


  def plot_trace(self,trace_number=0):
    if not self.velocity.size:
      print("Re-reading traces")
      self.read_seismo(filter_s=True)

    plt.figure()
    plt.plot(self.tvec,self.velocity[:,trace_number])
    plt.xlabel('Time [s]')
    plt.ylabel('Velocity [ms]')
    plt.title('Trace number {} at x = {}'.format(trace_number,self.rcoord[trace_number,0]))
    plt.show()


  def compute_tf(self, nsurface, blim, smooth=True, filt=False,
                 saveBr=False, useBr=False, brockName=None,bd=40):
    """
     Computes the 2D transfer function of a sedimentary basin.

     -- parameters --
     * nsurface (int) :: number of surface recieves
     * bmin (float)  :: x-coordinate of the leftwards (lower) limit between the
                      sedimentary basin and the bed rock
     * bmax (float)  :: x-coordinate of the rightwards limit between the
     		      sedimentary basin and the bed rock
     * smooth (bool) :: To apply a konno-Ohmachi smoothing to the
                      signal's spectra

     * brockName :: Bedrock spectrum file name to load if useBr=True

     * bd        :: band width of Konno-Ohmachi smoothing function

    -- return --
     * Initializes self.raw_ssr variable which contains the transfer functions

    """

    if not hasattr(self,'velocity'):
      print("Reading velocity traces")
      if filt:
        self.read_seismo(filter_s=True)
      else:
        self.read_seismo(filter_s=False)

    # Get rock station x-coordinates
    nt, nx = self.velocity.shape
    xcoord = self.rcoord[:,0]
    xmin = np.where(xcoord[:nsurface]<blim[0])
    xmax = np.where(xcoord[:nsurface]>blim[1])

    br_sta_coord = np.append(xmin[0],xmax[0])   # bed rock station coordinates

    # Compute the tranfer function on displacements

    if filt :
      self.compute_fft()
    else:
      self.compute_fft(filt=False)

    if not useBr :

      br_fft = self.fft_sig[:,br_sta_coord]
      br_fft = br_fft.mean(axis=1)
      if saveBr: np.save(brockName,br_fft)

    else :

      br_fft = np.load(brockName)

    # Smoothing the spectrum
    if smooth :
      br_fft = konno(br_fft,self.fft_freq,normalize=True)
      basin_fft = konno(self.fft_sig.T,self.fft_freq,normalize=True)
    else:
      basin_fft = self.fft_sig.T

    raw_ssr = basin_fft[:nsurface,:]/br_fft[np.newaxis,:]


    self.raw_ssr = raw_ssr

    return self.raw_ssr


  def plot_tf(self,savefile=None,cmap='jet',**kwargs):

    if 'raw_ssr' in kwargs.keys():
      raw_ssr = kwargs['raw_ssr'].T
      xcoord = kwargs['xcoord']
      fft_freq = kwargs['freq']
    else:
      if not hasattr(self,'raw_ssr'):
        print('Must run compute_tf first !! \n Existing ...')
        sys.exit()
      else:
        raw_ssr = self.raw_ssr.T

      xcoord = self.rcoord[:,0]
      fft_freq = self.fft_freq

    # Interpolated array on 2d meshgrid
    xmin , xmax = np.min(xcoord) , np.max(xcoord)

    #set_rcParams()

    #-- Plot
    if 'axis' in kwargs.keys():
      ax = kwargs['axis']
    else:
      fig, ax = plt.subplots(figsize=(8,6))

    if 'clim' in kwargs:
      cmin = kwargs['clim'][0]
      cmax = kwargs['clim'][1]
    else :
      cmin = 0
      cmax = 7

    im = ax.imshow(raw_ssr, cmap=cmap, aspect='auto', interpolation='bilinear',vmin=cmin, vmax=cmax, \
     origin='lower', extent=[xmin,xmax,0,max(fft_freq)])

    #-- Set plot parameters ----
    if 'xlim' in kwargs   :
      ax.set_xlim(kwargs['xlim'][0], kwargs['xlim'][1])
    else:
      ax.set_xlim(xmin,xmax)

    if 'ylim' in kwargs   :
      ax.set_ylim(kwargs['ylim'][0], kwargs['ylim'][1])
    else : ax.set_ylim(0.1,self._freqs[1])

    if 'axis' in kwargs.keys():
      divider = make_axes_locatable(ax)
      cax     = divider.append_axes('right', size='3%', pad=0.2)
      c  = plt.colorbar(im, cax=cax, fraction=0.046, shrink=0.6)
      im.set_clim(0,7)
      c.set_label('Amplification')
      c.ax.minorticks_off()
      ax.xaxis.set_visible(False)

      if 'ylabel' in kwargs :
        ax.set_ylabel(kwargs['ylabel'])
      else :
        ax.set_ylabel('Frequency [Hz]')

      if 'xlabel' in kwargs :
        ax.set_xlabel(kwargs['xlabel'])
      else :
        ax.set_xlabel('Horizontal profile [m]')

      if 'title' in kwargs  : ax.set_title(kwargs['title'])

      return ax
    else:

      if 'ylabel' in kwargs :
        ax.set_ylabel(kwargs['ylabel'])
      else :
        ax.set_ylabel('Frequency [Hz]',fontsize=6)

      if 'xlabel' in kwargs :
        ax.set_xlabel(kwargs['xlabel'])
      else :
        ax.set_xlabel('Horizontal profile [m]',fontsize=6)

      if 'title' in kwargs  : ax.set_title(kwargs['title'])

      cb = fig.colorbar(im, shrink=0.6, aspect=10, pad=0.02, ticks=np.linspace(cmin,cmax,cmax+1))
      cb.set_label('Amplification', labelpad=15, y=0.5, rotation=90)
      cb.minorticks_off()

      if savefile != None:
        fig.savefig(savefile,dpi=300)

    plt.show()



  def plot_source(self,savefile=None,source_name=None):
    #if not isinstance(source_name,str):
    #  print('source file name must be str object')
    if source_name:
      source_name = source_name
    else:
      source_name = 'SourcesTime_sem2d.tab'

    source_file = self.directory + source_name
    amp = np.genfromtxt(source_file)
    # plot spectra
    dt = amp[1,0]-amp[0,0]

    n = amp.shape[0]
    if n%2:
      nf = int(n+1)/2
    else :
      nf = int(n)/2

    spec = np.fft.fft(amp[:,1])[:nf]
    f = np.fft.fftfreq(n,dt)[:nf]

    fig = plt.figure(figsize=(8,5))
    fig.subplots_adjust(wspace=0.3)
    ax1  = fig.add_subplot(121)
    ax2  = fig.add_subplot(122)
    ax1.plot(amp[:,0],amp[:,1])
    ax2.plot(f,spec)
    ax1.ticklabel_format(style='sci',scilimits=(0,0),axis='y')
    ax2.ticklabel_format(style='sci',scilimits=(0,0),axis='y')
    ax1.set_xlabel('time [s]',fontsize=14) ; ax1.set_ylabel('velocity [$ms^{-1}$]',fontsize=14)
    ax2.set_xlabel('frequency [Hz]',fontsize=14) ; ax2.set_ylabel('amplitude',fontsize=14)
    ax1.set_title('Source time function',fontsize=16)
    ax2.set_title('Source spectrum',fontsize=16)
    ax2.set_xlim(0,15)
    ax1.set_xlim(0,2)
    #plt.tight_layout
    if savefile : plt.savefig(savefile)
    plt.show()


  @staticmethod
  def plot_im(matrix,vmin,vmax,cmin,cmax,**kwargs):
    fig = plt.figure(figsize=(8,6))
    ax = fig.add_subplot(111)
    im = ax.imshow(matrix,cmap='jet',aspect='auto',interpolation='bilinear', \
                   vmin=vmin, vmax=vmax, origin='lower', extent=extent)
    if 'xlim' in kwargs   : ax.set_xlim(kwargs['xlim'][0], kwargs['xlim'][1])
    if 'ylim' in kwargs   :
      ax.set_ylim(kwargs['ylim'][0], kwargs['ylim'][1])
    else : ax.set_ylim(0.1,fmax)

    if 'ylabel' in kwargs : ax.set_ylabel(kwargs['ylabel'], fontsize=16)
    if 'xlabel' in kwargs : ax.set_xlabel(kwargs['xlabel'], fontsize=16)
    if 'title' in kwargs  : ax.set_title(kwargs['title'],fontsize=18)

    # colorbar
    cb = fig.colorbar(im, shrink=0.5, aspect=10, pad=0.01,\
                     ticks=np.linspace(cmin,cmax,cmax+1), \
                     boundaries=np.linspace(cmin,cmax,cmax+1))
    cb.set_label('Amplification', labelpad=20, y=0.5, rotation=90, fontsize=15)
    plt.show()

  @staticmethod
  def filter_seismo(sismo,freqs=[0.1,10],ftype='bandpass',dt=0.01):
    """
    filter seismograms.
    Inputs:
      -freqs[tuple][(0.1,10)] : corner frequencies of filter response
      -ftype[str][default=bandpass] : filter type
    Return:
      -Updates self.velocity array.
    """
    filtered_s = np.zeros(sismo.shape)
    if ftype == 'bandpass':
      for i in range(sismo.shape[-1]):
        filtered_s[:,i] = bandpass(sismo[:,i],freqs[0],freqs[1],dt=dt,
                                   corners=4,zerophase=True)

    return filtered_s

  def plot_Vs(self,vs_br=1000,cmap='jet',axis=None,clim=None, size='2%'):
    """
       Makes a scatter plot of the velocities

       :Params
       -------
       -  vs_br : Bedrock velocity to be remove from the scatter plot
    """

    #from scipy.spatial.distance import pdist
    vsfile = self.directory + 'Cs_gll_sem2d.tab'
    with open(vsfile,'r') as v:
      vs_int = pd.read_csv(v,sep='\s+',names=['vs','x','z'])
    tmp = vs_int.drop_duplicates()
    vs_int = tmp.drop(tmp[tmp['vs']==vs_br].index)
    self.gll_vs = vs_int
    min_vs , max_vs = np.min(vs_int['vs']), np.max(vs_int['vs'])

    if axis :
      divider = make_axes_locatable(axis)
      cax     = divider.append_axes('right', size=size, pad=0.2)
      x = self.rcoord[:,0]
      #axis.fill_between(x,np.ones(x.shape)*-34,y[7,:],facecolor='#b26400')
      im = axis.scatter(vs_int['x'], vs_int['z'], c=vs_int['vs'], s=20, cmap='jet')
      c  = plt.colorbar(im, cax=cax, fraction=0.046, pad=0.06, shrink=0.4)
      if clim: im.set_clim(*clim)
      c.ax.tick_params(labelsize=8)
      c.set_label('Velocity [$ms^{-1}$]')
      axis.set_xlabel('Horizontal profile [m]')
      axis.set_ylabel('Depth [m]')

      return axis

    else:
      # Figure
      fig , ax = plt.subplots(figsize=(12,4))
      ax.scatter(vs_int['x'], vs_int['z'], c=vs_int['vs'], s=20, cmap=cmap)
      plt.show(block=True)

    return self.gll_vs


  def compute_st(self,frmin=None,fnyq=None):
    """
    Compute the stockwell transform
    """

    if not self.velocity.size :
      print("Reading velocity traces")
      self.read_seismo(filter_s=True)

    frmin = frmin or 0
    fnyq  = fnyq or (1./(2.*self.dt))
    df    = 1. / (self.npts * self.dt)
    fnpt  = int(fnyq/df)
    stock = []

    for i in range(self.nsta):
      trace = self.velocity[:,i]
      tmp = st.st(trace,frmin,fnpt)
      stock.append(np.abs(tmp))
      #stock.append(tmp)

    self.stockwell = np.array(stock)
    return self.stockwell, (int(frmin),int(fnyq))

  def plot_stockwel(self,sta_number,frmin=0.0,fnyq=10,cmap='seismic',**kwargs):
    set_rcParams()

    """
      Plot the Stockwell transform of a seismogram.

      :Inputs
      -------

       - sta_number [int] :: receiver indice to plot
       - frmin    [float] :: minimum frequency of stockwell transform
       - fnyq     [float] :: maximum frequency of stockwell transform
    """

    # Check input parameters
    if not isinstance(sta_number,int):
      try :
        sta_number = int(sta_number)
      except :
        msg = 'Station number must of an integer or of type convertable to an integer'
        raise Exception(msg)

    # Check if the compute_st method has been called
    if hasattr(self,'stockwell'):
      pass
    else :
      self.compute_st(frmin=frmin,fnyq=fnyq)

    dx = self.rcoord[1,0] - self.rcoord[0,0]

    if 'label' in kwargs.keys():
      label = kwargs['label']
    else:
      label = None


    fig , ax = plt.subplots(2,1,figsize=(10,7),sharex='col')
    ax[0].plot(self.tvec,self.velocity[:,sta_number],'k',label=label)
    ax[0].set_ylabel('Velocity [m$s^{-1}$]')
    ax[0].set_title('Velocity trace at {} m'.format(str(dx * sta_number)))
    ax[1].imshow(self.stockwell[sta_number,:,:],aspect='auto',extent=[0,np.max(self.tvec),frmin,fnyq],
                  cmap=cmap,origin='lower')
    ax[1].set_title('S transform')
    ax[1].set_xlabel('Time [s]')
    ax[1].set_ylabel('Frequency [Hz]')
    if 'key'in kwargs.keys():
      fig.suptitle(kwargs['key'])
    try:
      ax[0].legend()
    except:
      pass
    if 'savefile' in kwargs.keys():
      fig.savefig(kwargs['savefile'])

    plt.show(block=False)

  def compute_pv(self,op='pgv',component=None,freqs=None,n_surf=None):
    component = component or self._component
    n_surf = n_surf or self.nsta
    if not hasattr(self,'velocity'):
      print("Reading velocity traces")
      veloc = self.read_seismo(filter_s=True,freqs=freqs)[:,:n_surf]
    else:
      veloc = self.velocity[:,:n_surf]

    if op == 'pgv':
      maxs     = np.max(np.abs(veloc),axis=0)
      max_args = np.argmax(np.abs(veloc),axis=0) * self.dt
      self.pgv = maxs

      return self.pgv, max_args
    elif op == 'pga':
      # differentiate the velocities
      dt = self.dt
      dadx = np.gradient(veloc,dt,axis=0)
      maxs = np.max(np.abs(dadx),axis=0)
      self.pga = maxs

      return self.pga

  def peak_response(self,T,dr):
    """
    RESPONSE SPECTRUM
    Computes the peak dynamic response of a single-degree-of-freedom systems
    (mass-spring-dashpot) using the Newmark Method for linear systems

    Inputs
     - a  : input accelorgram (in ms-2)
     - T  : fundametal period (in s)
     - dr : damping ratio = damping coeficient / critical damping coefficient

    Outputs
     - maxA : peak acceleration response
     - maxV : peak velocity response
     - maxD : peak displacement response

    """

    if not self.velocity.size:
      print("Reading velocity traces")
      veloc = self.read_seismo(filter_s=True)
    else :
      veloc = self.velocity

    # Integrate the velocities to get acceleration
    accel = np.gradient(veloc,self.dt,axis=0)

    if isinstance(T,(int,float,list,tuple)):
      T = np.array(T)
    l = T.size
    a = np.zeros( (l,*accel.shape) )
    v = np.zeros( (l,*accel.shape) )
    d = np.zeros( (l,*accel.shape) )

    # Parameters defined for the average acceleration method (Page 177 of Chopra's bok')
    gamma = 0.5
    beta  = 0.25

    # Properties of the SDOF
    w = 2 * np.pi / T      # Angular frequency
    w = w[:,np.newaxis]
    m = 1                  # Mass
    k = m * w**2           # stifness
    dc = 2.0 * dr * m * w

    # Initial calculations
    a[:,0,:] = ( (-1 * m * accel[0,:]) - (dc * v[:,0,:]) - (k * d[:,0,:]) ) / m

    kk     = k + ( (gamma * dc) / (beta * self.dt) ) + ( m / ( beta * self.dt**2 ) )

    var_a  = ( m / (beta / self.dt) ) + ( (gamma * dc) / beta )

    var_b  = ( m / (2.0 * beta) ) + ( self.dt * dc * ( (gamma / (2 * beta) ) - 1) )

    # Iteration
    for j in range(1,accel.shape[0]):
      dp = (-1 * m  * ( accel[j,:] - accel[j-1,:]) ) + (var_a * v[:,j-1,:]) \
                      + (var_b * a[:,j-1,:])
      du = dp / kk

      dv = ( (gamma * du) / (beta / self.dt) ) - ( gamma * v[:,j-1,:] / beta ) + \
                ( self.dt *  a[:,j-1,:] * ( 1 - (gamma/(2.0 * beta)) ) )

      da = ( du / (beta * self.dt**2) ) - ( v[:,j-1,:] / (beta * self.dt) ) - ( a[:,j-1,:] / (2.0 * beta) )

      d[:,j,:] = d[:,j-1,:] + du
      v[:,j,:] = v[:,j-1,:] + dv
      a[:,j,:] = a[:,j-1,:] + da

    self.maxA = np.max(np.abs(a+accel),axis=1)
    self.maxV = np.max(np.abs(v),axis=1)
    self.maxD = np.max(np.abs(d),axis=1)

    return self.maxA


  def psa_sac(self,T=[2,6],dr=0.05,**kwargs):
    """
    Computes the acceleration spectral response of an accerelation time series in cm/s/s

    !! uses fortran wrapper subroutine fcode

    """

    if not hasattr(self,'velocity'):
      print("Reading velocity traces")
      veloc = self.read_seismo(filter_s=True)
    else :
      veloc = self.velocity

    # Integrate the velocities to get acceleration
    accel = np.gradient(veloc,self.dt,axis=0)  # m/s/s
    maxA = []

    # Pseudo spectral acceleration parameters
    Nrsp  = 200
    damp =  dr  # damping ratio

    log_fr  = fc.logspace(Nrsp)
    freq    = fc.freq2per(log_fr,Nrsp)
    N       = accel.shape[0]
    M       = len(freq)
    for i in range(self.nsta):
      array = np.zeros((accel.shape[0],2))
      array[:,0]  = self.tvec
      array[:,1]  = accel[:,i] * 1e2
      peak_r = fc.rsps(array[:,1],freq,self.dt,damp,N,M)
      maxA.append(peak_r/1e2)
    maxA = np.array(maxA)

    # get indice of corresponding period
    ind = []
    for i in T:
      toto = np.abs(log_fr - i)
      tind  = np.where(toto == np.min(toto))[0][0]
      ind.append(tind)

    peak_a = maxA[:,ind].T
    if 'n_surf' in kwargs.keys():
      return peak_a[:,:int(kwargs['n_surf'])]
    return peak_a

  def select_1d_vs(self,x):
    """
    Select the 1d vertical velocity profile on a spectral element grid

    -- Input parameters:
      x  :: array of station numbers

    -- Output parameters:
      sorted_df['z']  :: Sorted depth in ascending order
      sorted_df['vs'] :: Corresponding shear velocity profiles.
    """
    x = np.array(x,dtype=np.int)

    vsfile = self.directory + 'Cs_gll_sem2d.tab'
    with open(vsfile,'r') as v:
      vs_int = pd.read_csv(v,sep='\s+',names=['vs','x','z'])
    tmp = vs_int.drop_duplicates()

    # Get station postion
    dx          = self.rcoord[1][0] - self.rcoord[0][0]

    station_pos = self.rcoord[x,0]

    x_coord     = tmp['x']

    # Select points with x = station_pos
    x_index     = [ np.where( np.abs((x_coord - i)) <= dx/2 )[0] for i in station_pos ]
    tmp_bis     = [ tmp.iloc[i] for i in x_index ]

    # Sort points according to depth (z)
    sorted_df      = [ i.sort_values('z',ascending=True) for i in tmp_bis ]
    sorted_z       = [ i['z'].values for i in sorted_df ]
    sorted_vs      = [ i['vs'].values for i in sorted_df ]
    #print(tmp['z'].max() - tmp['z'].min())

    return sorted_z, sorted_vs

  def energy(self,period=None,option='pow',nsurface=None):

    if not hasattr(self,'velocity'):
      print("Reading velocity traces")
      self.read_seismo(filter_s=True)

    nsta = nsurface or self.velocity.shape[1]
    if period:
      period_pts = np.int( period / self.dt)
      n_interval = np.ceil( self.npts / period_pts ).astype(int)
      pad_npts   = period_pts * n_interval
      pad_array  = np.zeros((pad_npts,nsta))
      pad_array[:self.npts,:] = self.velocity

      # Power
      xsig = np.reshape(pad_array,(n_interval,-1,nsta))
      if option == 'pow':
        power = np.sum(xsig*xsig,axis=1) / (self.npts * self.dt)
      elif option == 'eng':
        power = np.sum(xsig*xsig,axis=1)

      p_tvec = np.arange( len(power) ) * period

      self.power = power
      self.p_tvec = p_tvec

      return self.power, self.p_tvec

    else:
      if option == 'pow':
        power = np.sum(self.velocity[:,:nsta],axis=0) / (self.npts * self.dt)
      elif option == 'eng':
        power = np.sum(self.velocity[:,:nsta],axis=0)
      self.power = power

      return self.power




  def cc_matrix(self,n_surf=None):
    """
     Computes the cross-correlation matrix between all pairs of surface receivers

        :Inputs
          -- n_surf [dtype:int] :: number of surface receivers

        :Return
          -- cc [dtype:np.ndarray] :: 2d array of number cross-correlation arrays
                                      cc[i,j] = cross-correlation between signal_i and signal_j
    """
    if not self.velocity.size:
      print("Reading velocity traces")
      self.read_seismo(filter_s=True)

    shape = n_surf or self.velocity.shape[1]
    cc = np.zeros((shape,shape),dtype=object)  # cross correlation array
    deltat = np.zeros(cc.shape)                # array of lag time between signal pairs
    tri_indices = np.triu_indices_from(cc)

    maxlag = int( (self.npts * 2 - 1) / 2 )
    tcc = np.arange(-maxlag,maxlag+1) * self.dt

    for i, j in zip(tri_indices[0],tri_indices[1]):
      tmp = sp.correlate(self.velocity[:,i],self.velocity[:,j])
      indmax = np.where(tmp==np.max(tmp))[0][0]
      deltat[i,j] = np.abs(tcc[indmax])
      cc[i,j] = tmp

    assert tcc.shape == cc[0,0].shape

    # bind the variables to the class
    self.cross_corr = cc
    self.corr_time  = tcc
    self.lag_time   = deltat
    return cc, tcc, deltat


  def pairwise_dtw(self,q=4,sta_range=[115,315]):
    """
      Compute the pair-wise DTW for a range of stations.
      The DTW distance and path are stored in an array suitable for
      resistivity-like plots.
    """

    if not hasattr(self,'decimated_veloc'):
      self.decimate_sig(q=q,filter_s=True)

    lb, rb = sta_range
    traces = self.decimated_veloc[:,lb:rb]

    n = traces.shape[1]
    self.pair_dtw_dist = np.zeros((n-1,2*n-3))
    self.pair_dtw_path = np.zeros((n-1,2*n-3),dtype=object)

    for j in range(n-1):
      for i in range(n-1-j):
        dtw_dist , _ , _ , dtw_path = dtw( traces[:,i], traces[:,i+j+1] , dist=euclidean )
        self.pair_dtw_dist[j,j+(i*2)] = dtw_dist
        self.pair_dtw_path = dtw_path

    self.dtw_dist_path = {'dist':self.pair_dtw_dist, 'path':self.pair_dtw_path}

    return self.dtw_dist_path


# =================================================
#
#    TOP LEVEL FUNCTIONS
#
# =================================================

def cc_matrix_static(signal1,signal2,dt):
  """
  Computes the cross correlation matrix of 2 arrays

  :Inputs
     -- signal1 [dtype:np.ndarray] :: 1st array of shape (n,m1) to cross-correlate with signal2
     -- signal2 [dtype:np.ndarray] :: 2nd array of shape (n,m2) to cross-correlate with signal1
     -- dt [dtype:np.float]  :: time step of the signal in signal1 and signal2

  :Outputs
     -- cc [dtype:np.ndarray] :: output matrix (m1,m2) containing the cross-correlation values
     -- tc
  """

  shape1 = signal1.shape
  shape2 = signal2.shape

  cc = np.zeros((shape1[1],shape2[1]),dtype=object)  # cross correlation array
  deltat = np.zeros(cc.shape)                # array of lag time between signal pairs
  tri_indices = np.triu_indices_from(cc)

  maxlag = int( ( shape1[0] * 2 - 1) / 2 )
  tcc = np.arange(-maxlag,maxlag+1) * dt

  tic = time.time()
  for i, j in zip(tri_indices[0],tri_indices[1]):
    tmp = sp.correlate(signal1[:,i],signal2[:,j])
    indmax = np.where(tmp==np.max(tmp))[0][0]
    deltat[i,j] = np.abs(tcc[indmax])
    cc[i,j] = tmp
  print('cross-correlation matrix computed in {:.2f} secs'.format(time.time() - tic))
  assert tcc.shape == cc[0,0].shape


  return cc, tcc, deltat


def plot_tf(tf,freq,xcoord,fmax,savefile=None,cmap='jet',**kwargs):

    # Interpolated array on 2d meshgrid
    xmin , xmax = np.min(xcoord) , np.max(xcoord)
    ymax = np.max(freq)
    dy   = freq[1] - freq[0]
    x1,y1 = np.meshgrid(xcoord[:tf.shape[0]],freq)
    xi, yi = np.linspace(xmin,xmax,int(2*xmax)), np.linspace(0,ymax,int(ymax/dy))
    X,Y = np.meshgrid(xi,yi)
    zi = gd((x1.ravel(order='F'),y1.ravel(order='F')),tf.ravel(),(X,Y))

    # Plot
    fig = plt.figure(figsize=(8,6))
    if 'clim' in kwargs:
      cmin = kwargs['clim'][0]
      cmax = kwargs['clim'][1]
    else :
      cmin = 0
      cmax = 7
    ax = fig.add_subplot(111)

    im = ax.imshow(zi, cmap=cmap, aspect='auto', interpolation='bilinear',vmin=cmin, vmax=cmax, \
     origin='lower', extent=[xmin,xmax, min(freq),ymax])

    if 'xlim' in kwargs   : ax.set_xlim(kwargs['xlim'][0], kwargs['xlim'][1])
    if 'ylim' in kwargs   :
      ax.set_ylim(kwargs['ylim'][0], kwargs['ylim'][1])
    else : ax.set_ylim(0.1,fmax)

    if 'ylabel' in kwargs :
      ax.set_ylabel(kwargs['ylabel'], fontsize=16,color='black')
    else :
      ax.set_ylabel('Frequency [Hz]', fontsize=16,color='black')
    if 'xlabel' in kwargs :
      ax.set_xlabel(kwargs['xlabel'], fontsize=16,color='black')
    else :
      ax.set_xlabel('Horizontal profile [m]', fontsize=16,color='black')
    if 'title' in kwargs  : ax.set_title(kwargs['title'],fontsize=20,color='black')

    ax.tick_params(axis='both',labelsize=12,color='black')

    cb = fig.colorbar(im, shrink=0.6, aspect=10, pad=0.02, ticks=np.linspace(cmin,cmax,cmax+1), \
                  boundaries=np.linspace(cmin,cmax,cmax+1))
    cb.set_label('Amplification', labelpad=15, y=0.5, rotation=90, fontsize=16, color='black')
    cb.ax.yaxis.set_tick_params(color='black',labelsize=12)

    if savefile != None:
      fig.savefig(savefile,dpi=300)

    plt.show(block=False)

def test_res_fig(directory):
  obj = sem2dpack(directory,component='x')
  obj.decimate_sig(q=4,filter_s=True)
  traces = obj.decimated_veloc[:,115:135]

  n = traces.shape[1]
  out_array = np.zeros((n-1,2*n-3))

  for j in range(n-1):
    for i in range(n-1-j):
      out_array[j,j+(i*2)] = dtw( traces[:,i], traces[:,i+j+1] , dist=euclidean )[0]


  db.set_trace()

if __name__ == '__main__':
  dir2 = '/Users/flomin/Desktop/thesis/simulations/Nice/plane_wave/elast_psv/'
  obj2 = sem2dpack(dir2,component='x')
  obj2.pairwise_dtw(sta_range=[115,125])


