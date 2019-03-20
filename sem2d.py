#!/usr/bin/env python3

"""
@Author :: Flomin T.

Class for manipulating SEM2DPACK output files.
  see user manual for more about SEM2DPACK code.
"""

import sys
sys.path.append('/Users/flomin/Desktop/thesis/MyScripts/python/modules')

import numpy as np
import time as t
from matplotlib.path import Path
import matplotlib.patches as pt
import matplotlib.pyplot as plt
from matplotlib.mlab import griddata
import matplotlib as mp
import seaborn as sns
from houches_fb import *
import glob
import fnmatch
from scipy.interpolate import griddata as gd
import matplotlib.animation as anim
import multiprocessing as mp
import os
import ipdb as db
import wiggle as wig
from scipy.signal import welch
import scipy.signal as sp
from filters import *
import pandas as pd
from stockwell import st
import warnings
warnings.filterwarnings("ignore",category=DeprecationWarning)

class sem2dpack(object):
  """
    Class to postprocess SEM2DPACK simulation outputs.
    It implements both object methods and static methods
    for easy handling and visulazing the data outputs.
    It has the following instances:
      - directory :: the simulation directory
      - mdict     :: dictionary containing spectral element grid infos
      - dt        :: simulation time step
      - npts      :: Number of points in record, npts * dt gives
	             the simulation time
      - nsta      :: number of reciever stations
      - velocity  :: velocity traces
      - tvec      :: time vector (0:dt:npts*dt)
      - fmax      :: maximum frequency of simulation
      - tf        :: transfer function in case of sedimentary basins
      - f         :: frequecy vector
      - rcoord    :: reciever stations coordinates
  """

  def __init__(self,directory,freqs=[0.1,10],component='x'):
    self.directory = directory
    self.mdict = {}
    self.dt = 0.0
    self.npts = 0
    self.nsta = 0
    self.velocity = np.array([])
    self.tvec = np.array([])
    self.fmax = 0.0
    self.interpolated_tf = np.array([])
    self.interpolated_f = np.array([])
    self.freq = np.array([])
    self.rcoord = np.array([])
    self.x_interp = np.array([])
    self.vs_int = np.array([])
    self._component = ''
    self._freqs = freqs
    self._component = component

    try:
      self.__readSpecgrid()
      self.__read_header()
    except:
      raise Exception('Not a sem2dpack simulation directory')

  def __readSpecgrid(self):
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
    x,w,h=g[0,:],g[1,:],g[2:,:]
    self.mdict ={"nel" : nel,  # Number of elements in mesh
            "npgeo" : npgeo,   # Number of global nodes
            "ngnod" : ngnod,   # Polynomial order
            "npt" : npt,       # Number of points in spectral mesh
            "ngll" : ngll,     # Number of gll points
            "coord" : coord,   # Coordinates of gll points
            "ibool" : ibool,   # Array for global to local mapping
            "x" : x,
            "w" : w,
            "h" : h,

            }

  def __read_header(self,extra=False):
    """
    Read seismic header file of SEM2DPACK simulation.
    The method broadcasts the simulation parameters and
    receiver coordinates instances.
    """
    filename = self.directory + 'SeisHeader_sem2d.hdr'
    f = open(filename, 'r')
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
    if extra :
      xsta = f.readline()
      xsta = int(xsta)
      f.readline()
      self.xsta_coord = np.zeros((xsta,2))

      for ex_reciever in range(nxsta):
        xtra = f.readline()
        x_reciever_line = xtra.rstrip(" ").split()
        xstat_coord[ex_reciever,0] = float(x_reciever_line[0])
        xstat_coord[ex_reciever,1] = float(x_reciever_line[0])
    else:
      xstat_coord=None
    f.close()
    return self.dt, self.npts, self.nsta, self.rcoord, xstat_coord

  @staticmethod
  def readField(fname):
    with open(fname,'rb') as f:
      field = np.fromfile(f,np.float32)

    return field

  def read_seismo(self,filter_s=False):
    if self._component == 'z': filename = self.directory + 'Uz_sem2d.dat'
    elif self._component == 'x': filename = self.directory + 'Ux_sem2d.dat'
    elif self._component == 'y': filename = self.directory + 'Uy_sem2d.dat'

    try :
      with open(filename, 'rb') as fid:
        veloc_array = np.fromfile(fid,np.float32)
    except : raise Exception('Velocity file does not exist')

    l = len(veloc_array)
    self.velocity = np.zeros((self.npts,self.nsta))

    for i in range(int(l/self.nsta)):
      limit1 = i*self.nsta
      limit2 = (i+1)*self.nsta
      self.velocity[i,:] = veloc_array[limit1:limit2]

    self.tvec = np.arange(self.npts) * self.dt

    if filter_s :
      self.velocity = self.filter_seismo(self.velocity,freqs=self._freqs,ftype='bandpass',dt=self.dt)
      return self.velocity
    return self.velocity

  def compute_fft(self,filt=True,freqs=[0.1,10.0],axis=0):
    if not self.velocity.size :
      pass
    else :
      if filt:
        veloc = self.read_seismo(component='x',filter_s=True)
      else:
        veloc = self.read_seismo(component='x')

    detrend = np.subtract(veloc,np.mean(veloc,axis=axis)[np.newaxis,:])
    s = np.abs(self.dt*np.fft.fft(detrend,axis=axis))
    df = 1.0 / ((detrend.shape[0]-1)*self.dt)
    nf = int(np.ceil( detrend.shape[0]/2.0) ) +1
    f = np.arange( nf ) * df
    self.fft = s[:nf,:]
    self.fs  = f[:nf]
    return s[:nf,:],f[:nf]

  @staticmethod
  def interp(field,coord):
    """
    Interpolates argument field over a meshgrid.
    Meshgrid size depends on argument coord.
    """
    xcoord = coord[:,0]
    zcoord = coord[:,1]
    nbx = len(xcoord)
    nbz = len(zcoord)
    ext = [min(xcoord), max(xcoord), min(zcoord), max(zcoord)]
    x,z=np.meshgrid(np.linspace(ext[0],ext[1],1000),np.linspace(ext[2],ext[3],1000),sparse=True)
    y = gd((xcoord,zcoord),field,(x,z),method='linear')
    y =np.flipud(y)

    return y

  def animate(self,save=False,savefile='im.flv',interval=1000,repeat_delay=1):
    """
    Animates SEM2DPACK snapshots
    """
    filename = "v"+self._component+"_*_sem2d.dat"
    coord  = self.mdict["coord"]
    xcoord = coord[:,0]
    zcoord = coord[:,1]
    frames = sorted(glob.iglob(self.directory + filename))
    nframe = int(len(frames))
    ext = [min(xcoord), max(xcoord), min(zcoord), max(zcoord)]
    ims = []
    field =[]
    pool = mp.Pool(processes=os.cpu_count()) # set the number of processes
    for i in range(nframe):
      f = self.readField(frames[i])
      field.append(f)

    results = [pool.apply_async(self.interp,args=(x,coord)) for x in field]  # run the processes
    output = [p.get() for p in results]  # retrieve processes information

    duration = self.npts * self.dt

    fig, ax = plt.subplots()
    xlabel = 'Length [m]'
    ylabel = 'Depth [m]'
    Writer = anim.writers['ffmpeg']
    writer = Writer(metadata=dict(artist='Me'))
    for i in range(int(nframe)):
      frametitle = 'snapshot at time = ' + str(round((i/nframe)*duration,1)) + ' secs'
      ttl = plt.text(0.5, 1.01, frametitle, ha='center', \
                     va='bottom', transform=ax.transAxes,fontsize="large")
      im = plt.imshow(output[i],extent=ext,cmap='seismic',\
                      aspect="auto",animated=True)
      ims.append([im,ttl])

    ani = anim.ArtistAnimation(fig,ims,interval=interval,blit=False,
                              repeat_delay=repeat_delay)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    #plt.gca().invert_yaxis()
    c=plt.colorbar(fraction=0.046,pad=0.04,shrink=0.4)
    c.set_label('particle velocity $[ms^{-1}]$')
    if save : ani.save(savefile,writer=writer,dpi=300)
    plt.show()
    return


  def plot_snapshot(self,filename,save=False,outdir='./'):
    if not isinstance(filename,str) :
      raise Exception('TypeError : filename must be string ')
    else :
      filename = self.directory + filename
    field = self.readField(filename)
    coord = self.mdict["coord"]
    xcoord = coord[:,0] ; zcoord = coord[:,1]
    nbx = len(xcoord)/4 ; nbz = len(zcoord)/4
    ext = [min(xcoord), max(xcoord), min(zcoord), max(zcoord)]
    x,z=np.meshgrid(np.linspace(ext[0],ext[1],1000),np.linspace(ext[2],ext[3],1000))
    y = gd((xcoord,zcoord),field,(x,z),method='linear')
    y =np.flipud(y)
    fig = plt.figure()
    fig.subplots_adjust(hspace=0.35)
    ax = fig.add_subplot(111)
    im = ax.imshow(y,extent=[min(xcoord), max(xcoord)/1e3, min(zcoord), max(zcoord)/1e3],cmap='jet')
    plt.tight_layout
    vmin = -0.4e-10
    vmax = 0.4e-10
    c=plt.colorbar(im,fraction=0.046, \
    pad=0.04,shrink=0.4)
    plt.ylabel('Depth [km]')
    plt.xlabel('Length [km]')
    c.set_clim(vmin,vmax)
    c.set_label('amplitude')
    plt.title('Snapshot at t = $1.6sec$')
    if save : plt.savefig(outdir+filename+'.png',dpi=300)
    plt.show()


  def plot_wiggle(self,ssta=None,sf=None,component='x',save_dir=None,stride=1,**kwargs):
    ssta = ssta or self.nsta
    xx = self.rcoord[:,0]
    if not self.velocity.size:
      print("Re-reading traces")
      self.read_seismo(compo=component,filter_s=True)
    if sf!=None:
      wig.wiggle(self.velocity[:,:ssta:stride],self.tvec,xx=xx,sf=sf)
    else :
      wig.wiggle(self.velocity[:,:ssta:stride],self.tvec,xx=xx)
    plt.xlabel('horizontal profile of reciever stations [m]',fontsize=20,color='black')
    plt.ylabel('time [s]',fontsize=20,color='black')
    if "xlim" in kwargs:
      xlim = kwargs["xlim"]
      plt.xlim(xlim[0],xlim[1])
    #else:
    #  plt.xlim([0,max(xx)+2])

    if "ylim" in kwargs:
      ylim = kwargs["ylim"]
      plt.ylim(ylim[1],ylim[0])
    else:
      plt.ylim([max(self.tvec),0])

    if "title" in kwargs:
      title = kwargs["title"]
      plt.title(title,fontsize=22)
    plt.xticks(fontsize=18,color='black')
    plt.yticks(fontsize=18,color='black')
    if save_dir:
      plt.savefig(save_dir,dpi=300)
    plt.show()


  def compute_tf(self, nsurface, blim, smooth=False, proc=2,
                 saveBr=False, useBr=False, brockName=None,):
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
     * proc (int)  :: To run the code in parallel on `proc` number of
                      processors. Define proc=1 to run the code
		      sequentially
     * option (str) :: [default:bedrock] Reference signal to compute standard
                      spectral ratio,
                      - 'BR' : to use bedrock signal
                      - 'SRC'  : to use source signal spectrum
    -- return --
    + self.interpolated_tf (ndarray) :: transfer function

    """

    if not self.velocity.size:
      print("Reading velocity traces")
      self.read_seismo(filter_s=True)

    # Get rock station x-coordinates
    nt, nx = self.velocity.shape
    xcoord = self.rcoord[:,0]
    zcoord = self.rcoord[:,1]
    xmin = np.where(xcoord[:nsurface]<blim[0])
    xmax = np.where(xcoord[:nsurface]>blim[1])
    br_sta_coord = np.append(xmin[0],xmax[0])   # bed rock station coordinates

    if proc <= 1:                    # run code sequentially
      br_fft = []
      for i in br_sta_coord:
        sig_fft , freq = fourier(self.velocity[:,i], self.dt, 0.025)
        br_fft.append( sig_fft )

      # mean bedrock trace
      br_fft = np.array(br_fft)
      br_mean_fft = np.mean(br_fft,axis=0)

      if smooth:
        df    = freq[1] - freq[0]
        br_mean_fft = ko2(br_mean_fft,freq)

      basin_fft = []
      for i in range(nsurface):
        sig_fft,freq = fourier( self.velocity[:,i], self.dt, 0.025)
        if smooth:
          amp = ko2(amp,f)
        basin_fft.append( amp / br_mean_fft )
      basing = np.array(basin_fft)

    else:

      # Initialize pool of processors
      pool = mp.Pool(processes=proc)
      if not useBr :

        results = [pool.apply_async(fourier,args=(self.velocity[:,i],self.dt,0.025)) \
                   for i in br_sta_coord]
        br_fft = [p.get()[0] for p in results]
        freq = results[0].get()[1]
        del results

        br_fft = np.array(br_fft)
        br_fft = np.mean(br_fft,axis=0)

        if saveBr: np.save(brockName,br_fft)

      else :

        br_fft = np.load(brockName)

      self.ref_fft = br_fft

      # Smoothing the spectrum
      if smooth :
        br_fft = ko2( br_fft , freq )

      # Computing spectral ratio all over the basin
      results = [pool.apply_async(fourier,args=[self.velocity[:,i],self.dt,0.025]) for i \
                in range(nsurface)]

      if useBr :
        freq = results[0].get()[1]

      basin_fft = [ np.divide(p.get()[0],br_fft) for p in results ]
      basin_fft = np.array(basin_fft)

      self.raw_ssr = basin_fft

    self.freq  = freq

    return self.raw_ssr, self.freq


  def plot_tf(self,savefile=None,cmap='jet',**kwargs):

    # Interpolated array on 2d meshgrid
    xcoord = self.rcoord[:,0]
    xmin , xmax = np.min(xcoord) , np.max(xcoord)
    ymax = np.max(self.freq)
    dy = self.freq[1]-self.freq[0]
    x1,y1 = np.meshgrid(xcoord,self.freq)
    xi, yi = np.linspace(xmin,xmax,int(2*xmax)), np.linspace(0,ymax,int(ymax/dy))
    X,Y = np.meshgrid(xi,yi)
    zi = gd((x1.ravel(order='F'),y1.ravel(order='F')),self.raw_ssr.ravel(),(X,Y))

    # Plot
    sns.set_style('whitegrid')
    fig = plt.figure(figsize=(8,6))

    if 'clim' in kwargs:
      cmin = kwargs['clim'][0]
      cmax = kwargs['clim'][1]
    else :
      cmin = 0
      cmax = 7
    ax = fig.add_subplot(111)

    im = ax.imshow(zi, cmap=cmap, aspect='auto', interpolation='bilinear',vmin=cmin, vmax=cmax, \
     origin='lower', extent=[xmin,xmax, min(self.freq),ymax])

    if 'xlim' in kwargs   : ax.set_xlim(kwargs['xlim'][0], kwargs['xlim'][1])
    if 'ylim' in kwargs   :
      ax.set_ylim(kwargs['ylim'][0], kwargs['ylim'][1])
    else : ax.set_ylim(0.1,self._freqs[1])

    if 'ylabel' in kwargs :
      ax.set_ylabel(kwargs['ylabel'], fontsize=16,color='black')
    else :
      ax.set_ylabel('Frequency [Hz]', fontsize=16,color='black')
    if 'xlabel' in kwargs :
      ax.set_xlabel(kwargs['xlabel'], fontsize=16,color='black')
    else :
      ax.set_xlabel('Horizontal profile [m]', fontsize=16,color='black')
    if 'title' in kwargs  : ax.set_title(kwargs['title'],fontsize=18,color='black')

    ax.tick_params(axis='both',labelsize=12,color='black')

    cb = fig.colorbar(im, shrink=0.6, aspect=10, pad=0.02, ticks=np.linspace(cmin,cmax,cmax+1), \
                  boundaries=np.linspace(cmin,cmax,cmax+1))
    cb.set_label('Amplification', labelpad=15, y=0.5, rotation=90, fontsize=16, color='black')
    cb.ax.yaxis.set_tick_params(color='black',labelsize=12)

    if savefile != None:
      fig.savefig(savefile,dpi=300)

    plt.show()

  def plot_meshnode(self):
    filename = self.directory + 'MeshNodesCoord_sem2d.tab'
    nel = self.mdict["nel"]
    n
    with open(filename,'r') as f:
      nodes = np.genfromtxt(f)

  def plot_source(self,savefile=None,source_name=None):
    #if not isinstance(source_name,str):
    #  print('source file name must be str object')
    if source_name:
      source_name = source_name
    else:
      source_name = 'SourcesTime_sem2d.tab'

    source_file = self.directory + source_name
    with open(source_file,'r') as src:
      amp = np.genfromtxt(src)
    # plot spectra
    dt = amp[1,0]-amp[0,0]
    spec,f = fourier(amp[:,1],dt,0.025)
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
    sns.set_style('whitegrid')
    fig = plt.figure(figsize=(8,6))
    ax.add_subplot(111)
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

  def plot_Vs(self,vs_br):
    from scipy.spatial.distance import pdist
    vsfile = self.directory + 'Cs_gll_sem2d.tab'
    with open(vsfile,'r') as v:
      vs_int = pd.read_csv(v,sep='\s+',names=['vs','x','z'])
    tmp = vs_int.drop_duplicates()
    self.vs_int = tmp.drop(tmp[tmp['vs']==vs_br].index)
    #dx = pdist(self.vs_int['x'][:,None],'cityblock')
    #dx = np.min(dx[np.nonzero(dx)])
    #dz = pdist(self.vs_int['z'][:,None],'cityblock')
    #dz = np.min(dz[np.nonzero(dz)])
    minx,maxx = np.min(self.vs_int['x']),np.max(self.vs_int['x'])
    minz,maxz = np.min(self.vs_int['z']),np.max(self.vs_int['z'])
    l = len(self.vs_int['x'])
    xi,zi = np.linspace(minx,maxx,l), np.linspace(minz,maxz,l)
    Xi,Zi = np.meshgrid(xi,zi)
    #plt.scatter(self.vs_int['x'],self.vs_int['z'],c=self.vs_int['vs'],cmap='jet')
    #plt.show()
    x = self.vs_int['x'].values
    z = self.vs_int['z'].values
    vs = self.vs_int['vs'].values
    y = gd((x,z),vs,(Xi,Zi),method='nearest')
    plt.figure()
    plt.imshow(y,cmap='jet',aspect='auto')
    plt.show()
    db.set_trace()

  def compute_st(self,frmin=None,fnyq=None,component='x'):
    """
    Compute the stockwell transform
    """
    frmin = frmin or 0.0
    fnyq  = fnyq or (1./(2.*self.dt))
    df    = 1. / (self.npts * self.dt)
    fnpt  = int(fnyq/df)

    if not self.velocity.size :
      print("Reading velocity traces")
      self.read_seismo(component=component,filter_s=True)

    stock = []

    for i in range(self.nsta):
      trace = self.velocity[:,i]
      tmp = st.st(trace,frmin,fnpt)
      stock.append(np.abs(tmp))

    self.stockwell = np.array(stock)
    return self.stockwell, (frmin,fnyq)

  def compute_pv(self,op='pgv',component=None,freqs=None):
    component = component or self._component
    if not self.velocity.size:
      print("Reading velocity traces")
      veloc = self.read_seismo(filter_s=True)
    else:
      veloc = self.velocity

    if op == 'pgv':
      maxs = np.max(np.abs(veloc),axis=0)
      self.pgv = maxs
    elif op == 'pga':
      # differentiate the velocities
      dt = self.dt
      dadx = np.gradient(veloc,dt,axis=0)
      maxs = np.max(np.abs(dadx),axis=0)
      self.pga = maxs

      return self.pga

    return self.pgv

  def peak_response(self,T,dr):
    """
    RESPONSE SPECTRUM
    Computes the peak dynamic response of a single-degree-of-freedom systems
    (mass-spring-dashpot) using the Newmark Method for linear systems

    Inputs
     - a  : input accelorgram (in ms-2)
     - T  : fundametal period (in s)
     - dr : damping ratio = damping coeficient / critical damping coefficient

    Output
     - maxA : peak acceleration response
     - maxV : peak velocity response
     - maxD : peak displacement response

    """

    if not self.velocity.size:
      print("Reading velocity traces")
      veloc = self.read_seismo(filter_s=True)

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



def energy(signal,dt=0.01):
  fpsd,Spsd = welch(signal,fs=1./dt)
  f_fft = np.fft.fftfreq(len(signal),d=dt)
  dfft = f_fft[1] - f_fft[0]
  dfpsd = fpsd[1] - fpsd[0]
  Spsd_int = np.sum(Spsd)

  E = (1./dt)*(dfpsd/dfft)*Spsd_int
  return Spsd,fpsd,E

def tf_error(vec1,vec2):
  """
  computes the relative error between two arrays.
  Normalizes with respect to the average of both arrays.
  """
  if vec1.shape != vec2.shape:
    raise Exception('Input arrays must be of the same shape')
  abs_diff = np.abs(vec1-vec2)/1.5
  maxs = np.maximum(vec1,vec2)
  #diff = np.array([(abs_diff[i,:]/maxs[i])*100.0 for i in \
  #                range(ref.shape[0])])
  #diff = abs_diff/mean
  return (abs_diff/maxs)*100

def plot_wiggle(traces,tvec,ssta,save=False,save_dir=None,**kwargs):
  wig.wiggle(traces[:,:ssta],tvec)
  plt.xlabel('horizontal profile of reciever stations',fontsize=16,color='black')
  plt.ylabel('time [s]',fontsize=16,color='black')
  if "xlim" in kwargs:
    xlim = kwargs["xlim"]
    plt.xlim(xlim[0],xlim[1])
  if "ylim" in kwargs:
    ylim = kwargs["ylim"]
    plt.ylim(ylim[1],ylim[0])
  if "title" in kwargs:
    title = kwargs["title"]
    plt.title(title,fontsize=18)
  plt.xticks(fontsize=12,color='black')
  plt.yticks(fontsize=12,color='black')
  if save:
    plt.savefig(save_dir,dpi=300)
  plt.show()

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
    sns.set_style('whitegrid')
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
    if 'title' in kwargs  : ax.set_title(kwargs['title'],fontsize=18,color='black')

    ax.tick_params(axis='both',labelsize=12,color='black')

    cb = fig.colorbar(im, shrink=0.6, aspect=10, pad=0.02, ticks=np.linspace(cmin,cmax,cmax+1), \
                  boundaries=np.linspace(cmin,cmax,cmax+1))
    cb.set_label('Amplification', labelpad=15, y=0.5, rotation=90, fontsize=16, color='black')
    cb.ax.yaxis.set_tick_params(color='black',labelsize=12)

    if savefile != None:
      fig.savefig(savefile,dpi=300)

    plt.show()


def correlation(trace1, trace2, t, maxlag=100, plot=False):
  """
      This function computes the correlation function of trace1 and trace2 as a function of time.
      The maximum time shift, maxlag, is the maximum number of index values by which the two discrete time series
      are shifted with respect to each other. If we only want to determine differential traveltimes, maxlag
      can be chosen pretty small.
  """


  import numpy as np
  import matplotlib.pylab as plt



  #- Initialisations. -------------------------------------------------------------------------

  time_index=np.arange(-maxlag,maxlag+1)
  tcc=time_index*(t[1]-t[0])
  cc=np.zeros(len(tcc))

  nt=len(t)

  #- Compute correlation function. ------------------------------------------------------------

  for k in time_index:
    if k>0:
      cc[k+maxlag]=np.sum(trace1[k:nt]*trace2[0:nt-k])
    else:
      cc[k+maxlag]=np.sum(trace1[0:nt+k]*trace2[-k:nt])

  #- Differential travel time -----------------------------------------------------------------

  ind_max = np.where( cc == np.max(cc) )[0][0]
  deltat  = np.abs(tcc[ind_max])

  #- Plot if wanted. --------------------------------------------------------------------------

  if plot==True:
    plt.plot(tcc,cc)
    plt.show()

  #- Return output. ---------------------------------------------------------------------------

  return cc, tcc, deltat

if __name__ == '__main__':
  test_dir = '/Users/flomin/Desktop/thesis/simulations/Nice/plane_wave/elastic/SH/'
  test_obj = sem2dpack(test_dir,component='y')
  test_obj.peak_response(4,5)
  test_obj.compute_pv(op='pga')
  db.set_trace()

