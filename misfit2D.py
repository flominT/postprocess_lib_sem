#!/usr/bin/env python3

"""
 @Author :: Flomin T.
 @Date   :: 17 sept 2018

 Plot 2D misfit of signals

"""

import sys
sys.path.append('/Users/flomin/Desktop/thesis/MyScripts/python/modules')
import matplotlib.pyplot as plt
import numpy as np
import ipdb as db
import pickle
from sem2d import sem2dpack
from tf_misfit import *
from obspy.imaging.cm import obspy_divergent, obspy_sequential
import multiprocessing as mp
from util_sys import *

def read_fd(filename,niter,nsurf):
  with open(filename,'r') as fd:
    fdx = np.fromfile(fd,np.float32)

  fd_sismo = np.zeros((niter,nsurf))
  for i in range(nsurf):
    limit1 = i*niter
    limit2 = (i+1)*niter
    fd_sismo[:,i] = fdx[limit1:limit2]
  return fd_sismo




def FEM2D(sig1,sig2,dt,fmin,fmax,nf,w0,norm,option='FEM',savefile=None):
  """
   Frequecy enveloppe misfit (FEM) of a 2D signals
  """
  pool = mp.Pool(processes=2)
  if option == 'FEM':
    tmp = [pool.apply_async(fem,args=(sig1[:,i], sig2[:,i],dt,fmin,\
             fmax, nf, w0, norm, True)) for i in range(sig1.shape[1])]
    fem2d = [p.get() for p in tmp]
  elif option == 'FPM':
    tmp = [pool.apply_async(fpm,args=(sig1[:,i], sig2[:,i],dt,fmin,\
             fmax, nf, w0, norm, True)) for i in range(sig1.shape[1])]
    fem2d = [p.get() for p in tmp]
  elif option == 'TEM':
    tmp = [pool.apply_async(tem,args=(sig1[:,i], sig2[:,i],dt,fmin,\
             fmax, nf, w0, norm, True)) for i in range(sig1.shape[1])]
    fem2d = [p.get() for p in tmp]
  elif option == 'TPM':
    tmp = [pool.apply_async(tpm,args=(sig1[:,i], sig2[:,i],dt,fmin,\
             fmax, nf, w0, norm, True)) for i in range(sig1.shape[1])]
    fem2d = [p.get() for p in tmp]
  fem2d = np.array(fem2d)

  npts = sig1.shape[0]

  #----
  # Plot
  fig = plt.figure()
  ax = fig.add_subplot(111)
  if option == 'FEM':
    im = ax.imshow(fem2d.T,cmap=obspy_divergent,aspect='auto',\
                   extent=[0,max(xcoord),fmin,fmax])
    ax.set_ylim(fmin,fmax)
    ax.set_xlabel('horizontal distance along profile [m]')
    ax.set_ylabel('frequency [Hz]')
    ax.set_title('Frequency Envelope Misfit')
    cb = fig.colorbar(im,shrink=0.5,aspect=10,pad=0.01)
    cb.set_label(option)
    if savefile : plt.savefig(savefile,dpi=300)
  elif option == 'FPM':
    im = ax.imshow(fem2d.T,cmap=obspy_divergent,aspect='auto',\
                   extent=[0,max(xcoord),fmin,fmax])
    ax.set_ylim(fmin,fmax)
    ax.set_xlabel('horizontal distance along profile [m]')
    ax.set_ylabel('frequency [Hz]')
    ax.set_title('Frequency Phase Misfit')
    cb = fig.colorbar(im,shrink=0.5,aspect=10,pad=0.01)
    cb.set_label(option)
    if savefile : plt.savefig(savefile,dpi=300)
  elif option == 'TEM':
    im = ax.imshow(fem2d.T,cmap=obspy_divergent,aspect='auto',\
                   extent=[0,max(xcoord),0,npts*dt],vmin=-0.005,vmax=0.005)
    ax.set_ylim(0,npts*dt)
    ax.set_xlabel('horizontal distance along profile [m]')
    ax.set_ylabel('time [s]')
    ax.set_title('Time Envelope Misfit')
    cb = fig.colorbar(im,shrink=0.5,aspect=10,pad=0.01)
    cb.set_label(option)
    if savefile : plt.savefig(savefile,dpi=300)
  elif option == 'TPM':
    im = ax.imshow(fem2d.T,cmap=obspy_divergent,aspect='auto',\
                   extent=[0,max(xcoord),0,npts*dt],vmin=-0.005,vmax=0.005)
    ax.set_ylim(0,npts*dt)
    ax.set_xlabel('horizontal distance along profile [m]')
    ax.set_ylabel('time [s]')
    ax.set_title('Time Phase Misfit')
    cb = fig.colorbar(im,shrink=0.5,aspect=10,pad=0.01)
    cb.set_label(option)
    if savefile : plt.savefig(savefile,dpi=300)
  plt.show()
  return

def plt_misfit(st1, st2, dt=0.01, t0=0., fmin=1., fmax=10., nf=100, w0=6,
                norm='global', st2_isref=True, left=0.1, bottom=0.1,
                h_1=0.3, h_2=0.125, h_3=0.2, w_1=0.2, w_2=0.6, w_cb=0.01,
                d_cb=0.0, show=True, plot_args=['k', 'r', 'b'], ylim=0.,
                clim=0., cmap=obspy_divergent,savefig=None,label1='sig1',label2='sig2'):
    import matplotlib.pyplot as plt
    from matplotlib.ticker import NullFormatter

    npts = st1.shape[-1]
    tmax = (npts - 1) * dt
    t = np.linspace(0., tmax, npts) + t0
    f = np.logspace(np.log10(fmin), np.log10(fmax), nf)

    # compute time frequency misfits
    _tfem = tfem(st1, st2, dt=dt, fmin=fmin, fmax=fmax, nf=nf, w0=w0,
                 norm=norm, st2_isref=st2_isref)

    _tem = tem(st1, st2, dt=dt, fmin=fmin, fmax=fmax, nf=nf, w0=w0, norm=norm,
               st2_isref=st2_isref)

    _fem = fem(st1, st2, dt=dt, fmin=fmin, fmax=fmax, nf=nf, w0=w0, norm=norm,
               st2_isref=st2_isref)

    _em = em(st1, st2, dt=dt, fmin=fmin, fmax=fmax, nf=nf, w0=w0, norm=norm,
             st2_isref=st2_isref)

    _tfpm = tfpm(st1, st2, dt=dt, fmin=fmin, fmax=fmax, nf=nf, w0=w0,
                 norm=norm, st2_isref=st2_isref)

    _tpm = tpm(st1, st2, dt=dt, fmin=fmin, fmax=fmax, nf=nf, w0=w0, norm=norm,
               st2_isref=st2_isref)

    _fpm = fpm(st1, st2, dt=dt, fmin=fmin, fmax=fmax, nf=nf, w0=w0, norm=norm,
               st2_isref=st2_isref)

    _pm = pm(st1, st2, dt=dt, fmin=fmin, fmax=fmax, nf=nf, w0=w0, norm=norm,
             st2_isref=st2_isref)

    if len(st1.shape) == 1:
        _tfem = _tfem.reshape((1, nf, npts))
        _tem = _tem.reshape((1, npts))
        _fem = _fem.reshape((1, nf))
        _em = _em.reshape((1, 1))
        _tfpm = _tfpm.reshape((1, nf, npts))
        _tpm = _tpm.reshape((1, npts))
        _fpm = _fpm.reshape((1, nf))
        _pm = _pm.reshape((1, 1))
        st1 = st1.reshape((1, npts))
        st2 = st2.reshape((1, npts))
        ntr = 1
    else:
        ntr = st1.shape[0]

    figs = []

    for itr in np.arange(ntr):
        set_rcParams()
        fig = plt.figure(figsize=(13,9))

        # plot signals
        ax_sig = fig.add_axes([left + w_1, bottom + h_2 + h_3+0.003, w_2, h_1])
        ax_sig.plot(t, st1[itr], plot_args[0],label=label1)
        ax_sig.plot(t, st2[itr], plot_args[1],label=label2)
        ax_sig.legend()
        # plot TEM
        ax_tem = fig.add_axes([left + w_1, bottom + h_1 + h_2 + h_3+0.002, w_2, h_2])
        ax_tem.plot(t, _tem[itr], plot_args[2])

        # plot TFEM
        ax_tfem = fig.add_axes([left + w_1, bottom + h_1 + 2 * h_2 + h_3, w_2,
                                h_3])

        x, y = np.meshgrid(
            t, np.logspace(np.log10(fmin), np.log10(fmax),
                           _tfem[itr].shape[0]))
        img_tfem = ax_tfem.pcolormesh(x, y, _tfem[itr], cmap=cmap)
        img_tfem.set_rasterized(True)
        ax_tfem.set_yscale("log")
        ax_tfem.set_ylim(fmin, fmax)

        # plot FEM
        ax_fem = fig.add_axes([left, bottom + h_1 + 2 * h_2 + h_3, w_1, h_3])
        ax_fem.semilogy(_fem[itr], f, plot_args[2])
        ax_fem.set_ylim(fmin, fmax)
        ax_fem.tick_params(axis='both',color='black',labelsize=10,labelcolor='black')

        # plot TPM
        ax_tpm = fig.add_axes([left + w_1, bottom, w_2, h_2])
        ax_tpm.plot(t, _tpm[itr], plot_args[2])

        # plot TFPM
        ax_tfpm = fig.add_axes([left + w_1, bottom + h_2, w_2, h_3])

        x, y = np.meshgrid(t, f)
        img_tfpm = ax_tfpm.pcolormesh(x, y, _tfpm[itr], cmap=cmap)
        img_tfpm.set_rasterized(True)
        ax_tfpm.set_yscale("log")
        ax_tfpm.set_ylim(f[0], f[-1])

        # add colorbars
        ax_cb_tfpm = fig.add_axes([left + w_1 + w_2 + d_cb + w_cb, bottom,
                                   w_cb, h_2 + h_3])
        fig.colorbar(img_tfpm, cax=ax_cb_tfpm)

        # plot FPM
        ax_fpm = fig.add_axes([left, bottom + h_2, w_1, h_3])
        ax_fpm.semilogy(_fpm[itr], f, plot_args[2])
        ax_fpm.set_ylim(fmin, fmax)
        ax_fpm.tick_params(axis='both',color='black',labelsize=10,labelcolor='black')

        # set limits
        ylim_sig = np.max([np.abs(st1).max(), np.abs(st2).max()]) * 1.1
        ax_sig.set_ylim(-ylim_sig, ylim_sig)

        if ylim == 0.:
            ylim = np.max([np.abs(_tem).max(), np.abs(_tpm).max(),
                           np.abs(_fem).max(), np.abs(_fpm).max()]) * 1.1

        ax_tem.set_ylim(-ylim, ylim)
        ax_fem.set_xlim(-ylim, ylim)
        ax_tpm.set_ylim(-ylim, ylim)
        ax_fpm.set_xlim(-ylim, ylim)

        ax_sig.set_xlim(t[0], t[-1])
        ax_tem.set_xlim(t[0], t[-1])
        ax_tpm.set_xlim(t[0], t[-1])

        if clim == 0.:
            clim = np.max([np.abs(_tfem).max(), np.abs(_tfpm).max()])

        img_tfpm.set_clim(-clim, clim)
        img_tfem.set_clim(-clim, clim)

        # add text box for EM + PM
        textstr = 'EM = %.2f\nPM = %.2f' % (_em[itr], _pm[itr])
        props = dict(boxstyle='round', facecolor='white',edgecolor='red')
        ax_sig.text(-0.4, 0.5, textstr, transform=ax_sig.transAxes,
                    verticalalignment='center', horizontalalignment='left',
                    bbox=props)

        ax_tpm.set_xlabel('Time (s)',color='k',fontsize='14')
        ax_fem.set_ylabel('Frequency (Hz)',color='k',fontsize='14')
        ax_fpm.set_ylabel('Frequency (Hz)',color='k',fontsize='14')

        # add text boxes
        props = dict(boxstyle='round', facecolor='white', alpha=0.5)
        ax_tfem.text(0.95, 0.85, 'TFEM', transform=ax_tfem.transAxes,
                     verticalalignment='top', horizontalalignment='right',
                     bbox=props)
        ax_tfpm.text(0.95, 0.85, 'TFPM', transform=ax_tfpm.transAxes,
                     verticalalignment='top', horizontalalignment='right',
                     bbox=props)
        ax_tem.text(0.95, 0.75, 'TEM', transform=ax_tem.transAxes,
                    verticalalignment='top', horizontalalignment='right',
                    bbox=props)
        ax_tpm.text(0.95, 0.75, 'TPM', transform=ax_tpm.transAxes,
                    verticalalignment='top', horizontalalignment='right',
                    bbox=props)
        ax_fem.text(0.9, 0.85, 'FEM', transform=ax_fem.transAxes,
                    verticalalignment='top', horizontalalignment='right',
                    bbox=props)
        ax_fpm.text(0.9, 0.85, 'FPM', transform=ax_fpm.transAxes,
                    verticalalignment='top', horizontalalignment='right',
                    bbox=props)

        fig.suptitle('Envelope (top) and Phase (bottom) Misfits',color='k',fontsize=16)
        # remove axis labels
        ax_tfpm.xaxis.set_major_formatter(NullFormatter())
        ax_tfem.xaxis.set_major_formatter(NullFormatter())
        ax_tem.xaxis.set_major_formatter(NullFormatter())
        ax_sig.xaxis.set_major_formatter(NullFormatter())
        ax_tfpm.yaxis.set_major_formatter(NullFormatter())
        ax_tfem.yaxis.set_major_formatter(NullFormatter())


        ax_sig.set_ylabel('Velocity [$ms^{-1}$]',color='k',fontsize=12)
        figs.append(fig)

    if show:
        if savefig : plt.savefig(savefig,dpi=300)
        plt.show()
    else:
        if ntr == 1:
            return figs[0]
        else:
            return figs

if __name__ == '__main__':
  # Directories
  #---------
  sem_dir = '/Users/flomin/Desktop/thesis/simulations/Nice/plane_wave/visla_psv_15s/'
  fd_dir = "/Users/flomin/Desktop/thesis/simulations/other_results/vislaFD/fsismos_x"
  savefile = "/Users/flomin/Desktop/thesis/figures/sismoDiff/visla_sem_fd_misfit.png"
  nsurf, niter, dt, fmax = 420, 1500, 1e-2, 10

  sem_obj = sem2dpack(sem_dir)
  sem_obj.read_seismo()
  sem_sismo = sem_obj.velocity[:-1,:-1]
  fd_sismo  = read_fd(fd_dir,niter,nsurf)
  fd_sismo = np.roll(fd_sismo,1,axis=0)
  xcoord = sem_obj.rcoord[:,0]
  
  #FEM2D(sem_sismo,fd_sismo,dt,fmin=0.5,fmax=10,nf=100,w0=6,norm='global',option='TEM',savefile=savefile)
  st1 = sem_sismo[:,210]
  st2 = fd_sismo[:,210]

  plt_misfit(st1, st2, dt=0.01, t0=0., fmin=1., fmax=10., nf=100, w0=6,
                    norm='global', st2_isref=True, left=0.1, bottom=0.1,
                    h_1=0.2, h_2=0.125, h_3=0.2, w_1=0.2, w_2=0.6, w_cb=0.01,
                    d_cb=0.0, show=True, plot_args=['k', 'r', 'b'], ylim=0.,
                    clim=0., cmap=obspy_divergent,savefig=savefile,label1='SEM',label2='FD')


