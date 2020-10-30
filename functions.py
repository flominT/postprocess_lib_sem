#!/usr/bin/env python3

import sys
sys.path.append('/Users/flomin/Desktop/thesis/MyScripts/python/modules')
import ipdb as db            # Debugger (Only neccessary for developpement mode)
import time
from sem2d import sem2dpack
import pickle
from util_sys import *
import os
import glob
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import make_axes_locatable
sys.path.append('/Users/flomin/Desktop/thesis/cubit/Nice/scripts')
from test_bis import plot_nice, read_profile
import numpy as np
from tf_misfit import pm, em
from dtaidistance import dtw
#from dtw import accelerated_dtw as dtw
#from scipy.spatial.distance import euclidean
from scipy.stats import gmean

CASES          = [ "HETE", # Heterogeneous simulations
                   "HOMO", # Reference simulations
                  ]

DB_dir         = "/Users/flomin/Desktop/thesis/DataBase/"

def create_obj(case="HOMO", freqs=[0.1,10], load=False, debug=False, verbose=True, param_dict={}):

    """
       Function to create, pickle and load sem2dpack objects depending on the case
       parameter.
       Creates an object database in the $DB_dir directory

       Parameters :: [type:default] for input variables
       -------------

         -- case [str:HOMO]       : simulation type

         -- freqs [list:[0.1,10]] : frequency intervals for bandpass filtering the seismograms
                                    The values are used to initialize the state of the sem2dpack.freqs attribute.
                                    No filtering is applied till the sem2dpack.read_seismo method is called with
                                    filter_s = True. freqs values can be updated afterwards.

         -- load [bool:False]     : If load is True, Directly load the saved pickled objects (time saving)
                                       else if load is False, create object.

         -- debug [bool:False]    : If debug=True and load=False, does not save the objects.

         -- param_dict  [dict]    : Input dictionary of parameters of the database.
                    Necessary keys are:

                    - PROJ_NAME  : project name
                    - REF_SIMUL_TYPE [tuple] : Simulation types (visla_sh, visla_psv, elast_sh or elast_psv)
                    - HETE_SIMUL_TYPE [Dictionary] : Stochastic simulation types {'sim_type':'cor_len',}
                    - REF_DIR    : Reference simulation directory
                    - HETE_DIR   : Stochastic simulation directories
                    - N_SIMUL    : Number of simulations per stochastic simulation

    """

    # mandatory parameters keys
    param_keys = set(['PROJ_NAME','REF_SIMUL_TYPE','HETE_SIMUL_TYPE','REF_DIR','HETE_DIR'])
    assert param_keys.issubset(param_dict.keys())
    assert case in CASES, "Wrong case parameter given : valid cases are {:s}".format( \
           ','.join([c for c in CASES]) )

    tic  = time.time()

    # create project directory if it doesn't exist
    proj_dir = DB_dir + param_dict['PROJ_NAME'] + "/"
    make_dir(proj_dir)

    save_file_dir   =  proj_dir + case
    make_dir(save_file_dir)

    if not load :

        dict_out = {}

        if case == "HOMO":
            #------- Reference directories ---------------------------------------
            # Check if directory exist
            
            save_file_name = save_file_dir + '/' + param_dict['PROJ_NAME'] + '_REF_objects.pk'

            if file_exist(save_file_name):
                overwrite = input("Do you want to overwrite {} [Y/N]?".format(param_dict['PROJ_NAME'] + '_REF_objects.pk'))
            else :
                overwrite = 'Y'

            if overwrite.upper() == 'Y':
                for key in param_dict["REF_SIMUL_TYPE"]:
                    directory   = param_dict["REF_DIR"] + key + '/'

                    if not os.path.isdir(directory):
                        msg = '{} does not exist !! \n Skipping ....'.format(directory)
                        print(msg)
                        pass
                    else:
                        #------- Create objects --------------------------------------------
                        compo = get_compo(key)
                        dict_out[key] = sem2dpack(directory,freqs=freqs,component=compo)

                #------- Save -------
                if not debug :
                    with open(save_file_name, 'wb') as f:
                        pickle.dump(dict_out,f,protocol=pickle.HIGHEST_PROTOCOL)

        else :

            for key in param_dict["HETE_SIMUL_TYPE"].keys():
                save_file_name = save_file_dir + '/' + key + '_object.pk'
                if file_exist(save_file_name):
                    overwrite = input("Do you want to overwrite {} [Y/N]?".format(key + '_objects.pk'))
                else :
                    overwrite = 'Y'

                if overwrite.upper() == 'Y':  
                    dict_out[key] = {}
                    compo = get_compo(key)
                    for cl in param_dict["HETE_SIMUL_TYPE"][key]:

                        if verbose:
                          print('{} : Creating objects for {}'.format(key,cl))

                        tmp_dir = param_dict["HETE_DIR"] + key + '/' + cl

                        if not os.path.isdir(tmp_dir):
                            msg = '{} does not exist !! \n Skipping ....'.format(tmp_dir)
                            print(msg)
                            pass
                        else:
                            dict_out[key][cl] = []

                            # Check the number of simulations and redefine N_SIMUL correspondinly

                            sim_dir = glob.glob(tmp_dir + '/n[0-9]*')
                            sim_num = len(sim_dir)
                            
                            if sim_num != param_dict["N_SIMUL"]:
                                print(" {} contains {} simulations \n".format(tmp_dir,sim_num))
                                print("Redefining global N_SIMUL to {}".format(sim_num))

                            for direc in sim_dir:

                                directory = direc + '/'

                                dict_out[key][cl].append(sem2dpack(directory,freqs=freqs,component=compo))

                    if not debug:
                        if bool(dict_out[key]):
                            if overwrite.upper() == 'Y':
                                with open(save_file_name, 'wb') as f:
                                    pickle.dump(dict_out[key],f,protocol=pickle.HIGHEST_PROTOCOL)
    else:

      #-- Load pickles objects --------------------------------------------

      dict_out = {}

      if case == 'HOMO':
          save_file_dir  = proj_dir + case
          save_file_name = save_file_dir + '/' + param_dict['PROJ_NAME'] + '_REF_objects.pk'
          try:
              with open(save_file_name, 'rb') as f:
                  dict_out = pickle.load(f)
          except:
              print("No SEM2DPACK object file for {} in database".format(key.upper()))
              pass 
      else:
          sim_type = param_dict["HETE_SIMUL_TYPE"].keys()

          #-- Load pickles objects --------------------------------------------
          for key in sim_type :
            save_file_dir  = proj_dir + case
            save_file_name = save_file_dir + '/' + key + '_object.pk'

            try:
              with open(save_file_name, 'rb') as f:
                  loaded_obj = pickle.load(f)
              dict_out[key] = loaded_obj

            except:
              print("No SEM2DPACK object file for {} in database".format(key.upper()))
              pass


    print('Objects created or loaded in {:.3f} secs'.format(time.time() - tic))
    return dict_out

def get_compo(key):
    if key[-2:] == 'sh':
        compo = 'y'
    else :
        compo = 'x'
    return compo

def run(load=False,cases=None):
    local_cases = cases or CASES
    try:
        assert isinstance(cases,(list,tuple))
    except:
        local_cases = [cases,]
    obj = {}
    for case in local_cases:
        print(case)
        obj[case] = create_obj(case=case,load=load)



class process_sim(object):

  """
  Class to process Nice simulation objects loaded or created using create_obj function.

  Attributes  ::
  -------------

  The class has no binded attributes

  Instance methods (short description) ::
  -------------------

   -- compute_ref_tf : compute homogeneous simulations transfer functions
   -- compute_hete_tf : compute heterogeneous simulations transfer functions
   -- plot_1d_tf     : plot 1d transfer functions
   -- compare_peak_values   : compute and plot peak values
   -- compute_psa : compute and plot the pseudo-spectral acceleration
   -- compute_energy : compute and plot the energy of the signal.
   -- plot_1d_vs     : plot 1d velocity profile
   -- wiggle_plot    : plot and save the time traces as wiggles
   -- plot_all_cases_tf : plot 1d transfer functions for all cases
   -- plot_stockwell   : plot the stockwell transform of each simulation
   -- plot_cc_matrix  : plot the cross correlation between traces.
   -- map_simulation  : map simulations plot to simulation number (for heterogeneous simulations)

  Static methods ::
  -----------------

  -- load
  -- plot_config_1dtf
  -- plot_config_pv
  -- plot_config_psa
  -- apply_method
  -- compute_stats
  -- compute_ratio

  """

  def __init__(self, ref_dict=None, hete_dict=None, pickle_dir=None, save_dir="./"):
    self.ref_obj    = ref_dict
    self.hete_obj   = hete_dict
    self.pickle_dir = pickle_dir
    self.save_dir   = save_dir

  def time_traces(self,key=None,scale=None,freqs=None):
    try :
        assert isinstance(key,(list,tuple))
    except:
        if key:
            keys = [key,]
        else:
            keys = list(self.ref_obj.keys())
            
    ref_trace  = {k : self.ref_obj[k].read_seismo(filter_s=True,scale=scale,freqs=freqs) for k in keys}
    hete_trace = self.apply_method(self.hete_obj,'read_seismo',params={'filter_s':True,'scale':scale,'freqs':freqs})

    if key :
        hete_trace = {k : hete_trace[k] for k in keys}

    self.hete_trace = hete_trace
    self.ref_trace  = ref_trace
    return ref_trace, hete_trace

  def plot_ref_wiggle(self, nsta, ftype='ximage', stride=3, sf=0.15,key=None, norm=None, naxis=False,
                      save_dir=None,plot_func=None,axis_title='',**kwargs):
    set_rcParams()

    if naxis:
        # Reference wiggle
  
        ref_fig, ax1, ax2 = self.subplot_2ax()
        divider = make_axes_locatable(ax1)
        cax     = divider.append_axes('right',size='3%',pad=0.2)
        cax.remove()
        if ftype == 'wig':
          ax1 = self.ref_obj[key].plot_wiggle(ssta=nsta,stride=stride,sf=sf, norm=norm, axis=ax1)
        elif ftype == 'ximage':
          xcoord  = self.ref_obj[key].rcoord[:,0]
          tvec    = self.ref_obj[key].tvec
          ext     = [min(xcoord),max(xcoord),max(tvec),min(tvec)]
          if 'vrange' in kwargs.keys():
              vmin, vmax = kwargs['vrange'][0], kwargs['vrange'][1]
          else:
              vmin, vmax = None, None
          if ('compo' in kwargs.keys()) and (kwargs['compo']==True):
              ax1.imshow(self.ref_obj[key].velocity_z[:,:nsta],aspect='auto',cmap='gray',
                    extent=ext, vmin=vmin, vmax=vmax)
              axis_title = axis_title + '- z component'
          else:
              ax1.imshow(self.ref_obj[key].velocity[:,:nsta],aspect='auto',cmap='gray',
                    extent=ext, vmin=vmin, vmax=vmax)

        if plot_func:
          plot_func(axis=ax2)
        else:
          if 'init' in kwargs.keys():
            init = kwargs['init']
          else:
            init = True
          if key[6:8] == 'ho':
              plot_nice(op='homo',axis=ax2, init=init)
          elif key[6:8] == 'ta':
              plot_nice(op='tabular',axis=ax2, init=init)
          else:
              plot_nice(op='default',axis=ax2, init=init)
          ax2.set_ylim([-34,39])

        if 'ylim' in kwargs.keys():
            ax1.set_ylim(kwargs['ylim'][0],kwargs['ylim'][1])
        else:
            ax1.set_ylim(5,0)
            
        ax1.set_ylabel('Time [s]',fontsize=14)
        ax2.set_xlabel('Distance along the profile [m]',fontsize=14)
        ax2.set_ylabel('Depth [m]', fontsize=14)
        ax1.tick_params(axis='both', which='major', labelsize=12, labelbottom=False)
        ax1.yaxis.set_label_coords(-0.1,0.5)
        ax2.yaxis.set_label_coords(-0.1,0.5)
        ax1.set_title(axis_title, fontsize=17, pad=10)
        if save_dir :
            if ('compo' in kwargs.keys()) and (kwargs['compo']==True):
              save_file  = save_dir + 'ref_' + key + '_wiggle_z.png'
            else:
              save_file  = save_dir + 'ref_' + key + '_wiggle.png'
            ref_fig.savefig(save_file, dpi=150, bbox_inches='tight', pad_inches=0.05)
      
    else:
        self.ref_obj[key].plot_wiggle(ssta=nsta,stride=stride)

        return

  def plot_hete_wiggle(self, nsta, ftype='ximage', stride=3, sf=0.15, key=None, norm=None, naxis=False,
                      save_dir=None, plot_func=None, s=2,
                      axis_title='', nreal=7,clim=None,**kwargs):
    set_rcParams()
    if 'init' in kwargs.keys():
      init = kwargs['init']
    else:
      init = True
    x,y = read_profile(init=init)

    for cor_l in self.hete_obj[key].keys():
        if naxis:
            hete_fig , ax1, ax2 = self.subplot_2ax()
            divider = make_axes_locatable(ax1)
            cax     = divider.append_axes('right',size='3%',pad=0.2)
            cax.remove()

            if ftype == 'wig':
              ax1 = self.hete_obj[key][cor_l][nreal].plot_wiggle(ssta=nsta, sf=sf, norm=norm, stride=stride,axis=ax1)
            elif ftype == 'ximage':
              xcoord  = self.ref_obj[key].rcoord[:,0]
              tvec    = self.ref_obj[key].tvec
              ext     = [min(xcoord),max(xcoord),max(tvec),min(tvec)]
              if 'vrange' in kwargs.keys():
                  vmin, vmax = kwargs['vrange'][0], kwargs['vrange'][1]
              else:
                  vmin, vmax = None, None

              if ('compo' in kwargs.keys()) and (kwargs['compo']==True):
                  ax1.imshow(self.hete_obj[key][cor_l][nreal].velocity_z[:,:nsta],aspect='auto',cmap='gray',
                        extent=ext, vmin=vmin, vmax=vmax)
                  title = axis_title[cor_l] + ' - z component'
              else:
                  ax1.imshow(self.hete_obj[key][cor_l][nreal].velocity[:,:nsta],aspect='auto',cmap='gray',
                        extent=ext, vmin=vmin, vmax=vmax)
                  title = axis_title[cor_l]

            ax2 = self.hete_obj[key][cor_l][nreal].plot_Vs(vs_br=1000,cmap='jet',axis=ax2,clim=clim, size='3%',s=s)

            if plot_func:
                plot_func(ax2,option='BR')
            else:
                ax2.fill_between(x,np.ones(x.shape)*-34,y[-1,:],facecolor='#b26400')
                ax2.set_ylim([-34,39])
                ax2.set_xlim(min(x),max(x))

            if 'ylim' in kwargs.keys():
                ax1.set_ylim(kwargs['ylim'][0],kwargs['ylim'][1])
            else:
                ax1.set_ylim(5,0)
            ax1.set_ylabel('Time [s]',fontsize=14)
            ax1.tick_params(axis='both', which='major', labelsize=12, labelbottom=False)
            ax1.set_title(title,fontsize=17,pad=10)
            ax2.spines['right'].set_visible(False)
            ax2.spines['top'].set_visible(False)
            ax1.yaxis.set_label_coords(-0.1,0.5)
            ax2.yaxis.set_label_coords(-0.1,0.5)
            if save_dir:
                if ('compo' in kwargs.keys()) and (kwargs['compo']==True):
                    save_file = save_dir + key + '_' + cor_l + '_z.png'
                else:
                    save_file = save_dir + key + '_' + cor_l + '.png'
                hete_fig.savefig(save_file, dpi=150, bbox_inches='tight', pad_inches=0.05)

        else:
            self.hete_obj[k][cor_l][nreal].plot_wiggle(ssta=nsta,stride=stride)

    plt.show(block=True)


  def em_pm(self, nsta, key=None, pickle_load=False):
    try:
        assert isinstance(key,(list,tuple))
    except:
        if key:
            key = [key,]
        else:
            key = self.hete_obj.keys()

    make_dir(self.pickle_dir + 'EP_misfit/')
    pickle_name = self.pickle_dir + 'EP_misfit/em_pm.pk'

    dt = self.ref_obj[key[0]].dt
    Em = {}
    Pm = {}

    if not pickle_load:
        
        if hasattr(self,'hete_trace'): # Check if time_trace method has been called
            pass
        else: # call the method
            self.time_traces(key=None,scale=False)

        for k in key:
            Em[k] = {}
            Pm[k] = {}
            for cor_l in self.hete_trace[k].keys():
                Em[k][cor_l] = [em(self.ref_trace[k][:,:nsta].T,self.hete_trace[k][cor_l][i][:,:nsta].T,dt=dt,st2_isref=False) 
                                  for i in range(len(self.hete_trace[k][cor_l]))]
                Pm[k][cor_l] = [pm(self.ref_trace[k][:,:nsta].T,self.hete_trace[k][cor_l][i][:,:nsta].T,dt=dt,st2_isref=False) 
                                  for i in range(len(self.hete_trace[k][cor_l]))]

        pickle_ep = {'Em':Em, 'Pm':Pm}

        with open(pickle_name,'wb') as f:
            pickle.dump(pickle_ep, f, protocol=pickle.HIGHEST_PROTOCOL)
    else:
        with open(pickle_name,'rb') as f:
            pickle_ep = pickle.load(f)
        Em = pickle_ep['Em']
        Pm = pickle_ep['Pm']

    Em_stats = self.compute_stats(Em)
    Pm_stats = self.compute_stats(Pm)
    return Em, Pm, Em_stats, Pm_stats


  def TF(self, key=None, save=False, tf_param=None, plot_param=None):
    """
    computes the reference transfer functions for the homogeneous simulations
    """
    try:
        assert isinstance(key,(list,tuple))
    except:
        if key:
            key = [key,]
        else:
            key = self.hete_obj.keys()

    ref_tf  = {}
    hete_tf = {}

    bedrock_dir = self.pickle_dir + 'BR_TF/'
    make_dir(bedrock_dir)

    # Reference TF
    for k in key:
      tf_param.update({'brockName': bedrock_dir + k + '_br_ssr'})
      ref_tf[k]  = self.ref_obj[k].compute_tf(**tf_param,saveBr=True)

    # Heterogeneous TF
    for k in key:
      hete_tf[k] = {}
      tf_param.update({'brockName': bedrock_dir + k + '_br_ssr.npy'})
      for cor_l in self.hete_obj[k].keys():
        hete_tf[k][cor_l] = [obj.compute_tf(**tf_param,useBr=True) for obj in self.hete_obj[k][cor_l]]

    # Statistics
    hete_stats = self.compute_stats(hete_tf)
        
    return ref_tf, hete_tf, hete_stats

  def plot_1d_tf(self, n_sta=240, key='visla_sh',save=False):
    """
    Plot the 1D transfer function of a particular station.

    Input :
      -- n_stat :: station indice
      -- key    :: simulation type [default :: 'visla_sh']
      -- case   ::
    """
    cl = ['a10_s5','a10_s30','a50_s5','a50_s30']
    # if key[-2:] == 'sh':
    #   key_hete = 'visla_sh'
    # else:
    #   key_hete = 'visla_psv'

    #-------  Load objects and compute transfer functions ---------------
    hete_obj, hete_tf  = self.compute_hete_tf(key=key,plot=False,CL=cl,smooth=True)
    ref_obj, ref_tf   = self.compute_ref_tf(key=key,plot=False,smooth=True)


    #------- Compute statistics for heterogeneous simulations----

    hete_stats = self.compute_stats(hete_tf,option='mean')

    #--- plot ----------------------

    #-- Initialize save directory --

    if save:
      save_dir = SAVE_DIR['hete'] + 'TF_1d/' + key + '/'
      make_dir(save_dir)
    else:
      save_dir = None

    self.make_tf_figures(n_sta, ref_obj, hete_obj, ref_tf, hete_stats,
                       colors='rbgc', key=key, save_dir = save_dir  )


  def duration(self, key=None, atype='ABI', freqs=None, n_surf=None):

    try:
        assert isinstance(key,(list,tuple))
    except:
        if key:
            key = [key,]
        else:
            key = self.hete_obj.keys()

    ref_duration = {}
    for k in key:
      ref_duration[k] = self.ref_obj[k].compute_ai(freqs=freqs,n_surf=n_surf)

    hete_duration = self.apply_method(self.hete_obj,'compute_ai',params={'freqs':freqs, 'n_surf':n_surf,'atype':atype})
    duration_stats = self.compute_stats(hete_duration)

    return ref_duration, hete_duration, duration_stats

  def compare_peak_values(self, key=None, pv_type='pgv',n_surf=None):
    """
    Plot and compute the difference between peak ground motion indicators (PGV/PGA)
    of a Homogenoeus and Heterogeneous medium.
    """

    try:
        assert isinstance(key,(list,tuple))
    except:
        if key:
            key = [key,]
        else:
            key = self.hete_obj.keys()

    freqs = [0.1,10]

    #------- Get peak values -------------------------------
    # 1) Reference peak values
    ref_pgv = {}
    for k in key:
      ref_pgv[k] = self.ref_obj[k].compute_pv(op=pv_type,freqs=freqs,n_surf=n_surf,component='x')

    data = self.select_key(self.hete_obj,key)
    hete_pgv = self.apply_method(data,'compute_pv',params={'op':pv_type,'n_surf':n_surf,'freqs':freqs,'component':'x'})

    #---- Compute statistics on the heterogeneous simulations --
    hete_stats = self.compute_stats(hete_pgv)

    return ref_pgv, hete_pgv, hete_stats


  def compute_psa(self, key='elast_sh', T=None, save=None, plot_op = 'default', pickle_load=True, n_surf=None, corl=None):
    """
      Computes the pseudo-spectral acceleration (PSA) as a function of frequency.

      Input :
       -- key :: Simulation type
       -- T   :: Frequencies at which to compute the PSA
    """
    
    make_dir(self.pickle_dir + 'PSA/')
    pickle_name = self.pickle_dir +  'PSA/peak_response.pk'
    n_surf = n_surf or 421

    if isinstance(T,(list,tuple)):
      period_indice = [i-1 for i in T]
    elif isinstance(T,int):
      period_indice = T - 1
      
    if not pickle_load:
      T  = [1,2,3,4,5,6,7,8]

      #-- ref psa
      ref_pr = {}
      for k in self.ref_obj:
        ref_pr[k] = self.ref_obj[k].psa_sac(T,n_surf=n_surf)

      hete_pr = self.apply_method(self.hete_obj,'psa_sac',params={'T':T,'n_surf':n_surf})

      pickle_pr = {'ref_pr':ref_pr, 'hete_pr':hete_pr}

      # Save peak responses

      with open(pickle_name,'wb') as f:
        pickle.dump(pickle_pr, f, protocol=pickle.HIGHEST_PROTOCOL)

    else :
      with open(pickle_name,'rb') as f:
        pickle_pr = pickle.load(f)

      ref_pr  = pickle_pr['ref_pr']
      hete_pr = pickle_pr['hete_pr']

    if corl:
      hete_pr = self.select_key(hete_pr[key],corl)
    else:
      hete_pr = self.select_key(hete_pr[key],self.hete_obj[key].keys())
    
    # Select periods
    ref_pr = { k : ref_pr[k][period_indice,:] for k in ref_pr.keys()}
    hete_pr = { cl : [hete_pr[cl][i][period_indice,:] for i in range(len(hete_pr[cl]))] for cl in hete_pr.keys()}

    #---- Compute statistics on the heterogeneous simulations --
    hete_stats = self.compute_stats(hete_pr)

    # Compute ratio
    # hete_r = self.compute_ratio(ref_pr,hete_max)
    
    # Plot spectral response values
    # xcoord   = self.ref_obj[key].rcoord[:,0]

    # if save:
    #   save_dir = SAVE_DIR['hete'] + 'peak_response/' + key + '/'
    #   save_dir_ref = SAVE_DIR['hete'] + 'peak_response/reference/'
    #   make_dir(save_dir)
    #   make_dir(save_dir_ref)
    # else :
    #   save_dir = None

    # if plot_op == 'maps':
    #   self.make_psa_figures(ref_pr[key], hete_pr, xcoord, T, save_dir=save_dir,
    #                     colors=colors, plot_op=plot_op, key=key, hete_obj=hete_obj)
    # elif plot_op == 'ref':
    #   self.make_psa_figures(ref_pr, hete_pr, xcoord, T, save_dir=save_dir_ref,
    #                     colors=colors, plot_op=plot_op, key=key)
    # elif plot_op == 'freq':
    #   self.make_psa_figures(ref_pr, hete_stats, xcoord, T, save_dir=save_dir,
    #                     colors=colors, plot_op=plot_op, key=key)
    # else:
    #   self.make_psa_figures(ref_pr, hete_stats, xcoord, T, save_dir=save_dir,
    #                     colors=colors, plot_op=plot_op, key=key)

    return ref_pr, hete_pr , hete_stats

  def plot_1d_vs(self,key='elast_sh',sta_number=240):
    """
        Plot the 1d velocity profile of the 2D velocity model based on GLL points.

        -- Input
          ** key :: [str] simulation type
          ** sta_number :: [int] Reciever station number (used to determine x-coordinate of mesh)
    """

    #-- Load objects ---
    ref_obj  = self.load(case='HOMO')
    hete_obj = self.load(case='HETE')

    #-- Select simulation objects --
    simul_ref   = ref_obj[key]
    simul_obj   = hete_obj[key]

    #-- Loop over the cases to plot profiles --
    z_r, vs_r = simul_ref.select_1d_vs(sta_number) # reference 1d vs
    color = cycle('rgbyc')

    for k in simul_obj.keys():
      fig,ax = plt.subplots()
      ax.plot(vs_r,z_r,'k.',label='ref')
      n=0
      for cor_l in simul_obj[k]:
        z,vs = cor_l.select_1d_vs(sta_number)
        ax.plot(vs,z,'.',c=next(color),label='n{}'.format(n+1))
        ax.set_title('{0:s} {1:s}'.format(key,k))
        n += 1

        ax.set_xlabel('Velocity [ms$^{-1}$]')
        ax.set_ylabel('Depth [m]')
        ax.legend()
      plt.show()

    return



  def plot_energy(self,key='visla_sh',period=None,save=True,plot_op='stats'):
    #-- Load objects ---
    ref_obj  = self.load(case='HOMO')
    hete_obj = self.load(case='HETE')[key]

    hete_energy         = self.apply_method(hete_obj,'energy',{'period':period,'option':'pow','nsurface':421})
    hete_energy         = self.select_key(hete_energy,CL)
    ref_energy          = {dic_key : ref_obj[dic_key].energy(period=period,option='pow',nsurface=421) for dic_key in ref_obj}

    # statistic
    hete_stats   = self.compute_stats(hete_energy)

    if save:
      save_dir = SAVE_DIR['hete'] + 'Energy/'
      make_dir(save_dir)

    plot_param = {'xlabel':'Energy', 'plot_type':'energy'}
    xcoord = ref_obj[key].rcoord[:,0]
    if plot_op == 'all':
      self.make_grid_figure1d(ref_energy,hete_energy,hete_obj,xcoord,key=key, save_dir=save_dir, plot_param = plot_param)
    elif plot_op == 'stats':
      cl = ['a10_s5', 'a50_s5', 'a10_s30', 'a50_s30']
      self.make_stats_fig1d(ref_energy,hete_stats,xcoord,CL=cl,colors='gcrb',save_dir=save_dir,plot_param=plot_param,key=key)

  def plot_all_cases_tf(self,n_sta,case='HOMO',cor_l='a10',save=False):

    colors = lambda n : ['#' + ''.join([random.choice('0123456789ABCDEF') for j in range(6)]) for i in range(n)]
    color = cycle('rbgk')

    set_rcParams()

    if case == 'HOMO':
      self.compute_ref_tf()

      freq = self.ref_obj['elast_sh'].freq
      #tvec = self.ref_obj['elast_sh'].tvec

      fig1,ax1 = plt.subplots()

      for key in sorted(self.ref_tf.keys()):
        if key == 'elast_12_sh':
          pass
        else:
          ax1.plot(freq,self.ref_tf[key][n_sta,:],c=next(color),label=key)
      ax1.legend()
      ax1.set_xlim([0.1,10])
      ax1.set_ylim([0,8])
      ax1.set_xlabel('Frequecy [Hz]')
      ax1.set_ylabel('Amplification')
      ax1.set_title('Homogeneous simulation TFs for station at x = {} m'.format(int(n_sta*5)))
      ax1.grid()

      if save:
        save_dir = '/Users/flomin/Desktop/thesis/postprocess/Nice/figures/plane_wave/ref_tf/'
        make_dir(save_dir)
        fig1.savefig(save_dir+'ref_tf_station_x_{}m'.format(int(n_sta*5)),dpi=300)
      plt.show()

    else:
      hete_obj = self.load(case=case)
      hete_tf  = {}
      for key in sorted(hete_obj.keys()):
        bed_rock_name = {'brockName' : pickle_dir + key + '.npy'}
        h_tf_param.update(bed_rock_name)
        hete_tf[key]  = self.apply_method(hete_obj[key],'compute_tf',h_tf_param)

      tf_stats = self.compute_stats(hete_tf,option='mean')

      # plot vectors
      freq = hete_obj['elast_sh'][cor_l][1].freq

      # plot
      fig, ax = plt.subplots()
      for key in sorted(tf_stats.keys()):
        ax.plot(freq,tf_stats[key][cor_l][0][n_sta,:],c=next(color),label=key)
        ax.legend()
      ax.set_xlim([0.1,10])
      ax.set_ylim([0,10])
      ax.set_title('Average transfer functions at station x = {0:d} m \n medium {1:s}'.format(n_sta * 5,cor_l))
      ax.set_xlabel('Frequency [Hz]')
      ax.set_ylabel('Amplification')

      if save:
        save_dir = '/Users/flomin/Desktop/thesis/postprocess/Nice/figures/plane_wave/hete_tf_1d/'
        make_dir(save_dir)
        fig.savefig(save_dir+'hete_tf_station_x_{}m'.format(int(n_sta*5)),dpi=300)

      plt.show()

  def plot_stockwell(self,key='visla_sh',n_sta=[50,140,240,250],load=False):
    ref_obj  = self.load(case='HOMO')
    hete_obj = self.load(case='HETE')


    stockwell_dir = pickle_dir + 'STOCKWELL/'
    make_dir(stockwell_dir)

    ref_filename  = stockwell_dir  + key + '_ref_stockwell.pk'
    hete_filename = stockwell_dir  + key + '_hete_stockwell.pk'

    if not load :

      #ref_st , flim  = ref_obj[key].compute_st()

      # figure file names
      save_dir = SAVE_DIR['hete'] + 'stockwell/' ; make_dir(save_dir)

      for sta in n_sta:
        save_ref = save_dir + '{}_{}_stockwell.png'.format(key,str(sta))
        save_a10_s30 = save_dir + '{}_{}_{}_stockwell.png'.format(key,'a10_s30',str(sta))
        save_a50_s30 = save_dir + '{}_{}_{}_stockwell.png'.format(key,'a50_s30',str(sta))
        ref_obj[key].plot_stockwel(sta,key=key_title[key],label='reference trace',savefile=save_ref)
        hete_obj[key]['a10_s30'][6].plot_stockwel(sta,key=key_title[key],label=LABEL['a10_s30'],savefile=save_a10_s30)
        hete_obj[key]['a50_s30'][6].plot_stockwel(sta,key=key_title[key],label=LABEL['a50_s30'],savefile=save_a50_s30)


      #hete_st = self.apply_method(hete_obj[key],'compute_st',n_out=0,CL=CL)

      # commute statistics on stockwell
      #hete_st_mean = self.compute_stats(hete_st,option='mean')

      # Pickle stockwell transforms

      #file_names = { ref_filename : ref_st, hete_filename : hete_st_mean}

      #for filename, data in file_names.items() :
      #  with open(filename, 'wb') as f:
      #    pickle.dump(data,f,protocol=pickle.HIGHEST_PROTOCOL)



    # else :
    #   with open(ref_filename,'rb') as f :
    #     ref_st = pickle.load(f)
    #   with open(hete_filename,'rb') as f :
    #     hete_st = pickle.load(f)




  def map_simulation(self,key='elast_sh'):
    hete_obj = self.load(case='HETE')

    tag = np.arange(10)

    hete_pgv = { cor_l : [obj.compute_pv(op='pga') for obj in hete_obj[key][cor_l]]\
               for cor_l in hete_obj[key]}

    hete_stats = self.compute_stats(hete_pgv,option='max')
    toto = np.array(hete_pgv['a10'])

    t_max = np.argmax(toto,axis=0)

    color_list = ['b','r','k','g','c']
    cmap = mpl.colors.ListedColormap(color_list)
    bounds = np.arange(5)
    norm = mpl.colors.BoundaryNorm(bounds, cmap.N)

    for i in range(len(t_max)):
      plt.scatter(i,hete_stats['a10'][i],c=color_list[t_max[i]],label='{:s}'.format(t_max[i]))
    plt.show()


  def plot_cc_matrix(self,key='elast_sh'):
    import seaborn as sns
    #set_rcParams()
    ref_simul = self.load(case='HOMO')
    hete_simul = self.load(case='HETE')

    ref_obj = ref_simul[key]
    hete_obj = hete_simul[key]
    hete_obj = self.apply_method(hete_obj,'read_seismo',params={'filter_s':True},output=False)
    ref_obj.read_seismo(filter_s=True)

    # Get velocity of one heterogeneous simulation
    hete_veloc = hete_obj['a10_s30'][0].velocity[:,240]
    ref_veloc  = ref_obj.velocity[:,240]

    ################################################################
    # Select wave packages
    t = 400
    fig , ax = plt.subplots(figsize=(12,5))
    ax.plot(hete_obj['a10_s30'][0].velocity[:,140],'r')
    ax.plot(ref_obj.velocity[:,140],'k')

    hete_obj['a10_s30'][0].plot_Vs(vs_br=1000)
    hete_obj['a10_s30'][0].plot_wiggle(ssta=421)

    db.set_trace()
    rc = sp.correlate(hete_veloc,ref_veloc)
    rc2 = sp.correlate(hete_veloc[:t],ref_veloc[:t])

    fig1,ax1 = plt.subplots(2,2,figsize=(12,5))
    ax1[0,0].plot(hete_veloc,'r',label='hete')
    ax1[0,0].plot(ref_veloc,'k',label='ref')
    ax1[1,0].plot(hete_veloc[:t],'r')
    ax1[1,0].plot(ref_veloc[:t],'k')
    ax1[0,1].plot(rc)
    ax1[1,1].plot(rc2)
    ax1[0,0].legend()

    db.set_trace()
    ################################################################
    ref_obj.cc_matrix()
    cc_hete, _ , dt_hete = hete_obj['s30_a10'][0].cc_matrix()
    cc_hete_ref , _ , dt_hete_ref = cc_matrix_static(ref_obj.velocity,hete_veloc,ref_obj.dt)


  def compute_dtw(self,nsurface=421,key='visla_sh',save=True):
    """
      Compute the dynamic time warping (DTW) between different simulations
    """

    # Computing the DTW is time consuming, so compute once and pickle the objects
    save_dir = self.pickle_dir + 'DTW/'
    filename_hete = save_dir + key + '_hete_dtw.pk'

    if save :
      make_dir(save_dir)
      hete_obj = self.apply_method(self.hete_obj[key],'decimate_sig',params={'q':4,'filter_s':True},output=False)
      self.ref_obj[key].decimate_sig(q=4,filter_s=True)
      #ref_hete_dtw = self.apply_dtw(self.ref_obj[key].decimated_veloc,hete_obj,nsurface=nsurface,filename=filename_hete)
      ref_hete_dtw = self.apply_dtw(self.ref_obj[key].velocity,hete_obj,nsurface=nsurface,filename=filename_hete)

    return ref_hete_dtw


  def plot_dtw(self,nsta=421,key='visla_sh',load=True):

    save_dir = self.pickle_dir + 'DTW/'
    if load:
      filename_hete = save_dir + key + '_hete_dtw.pk'

      with open(filename_hete,'rb') as f:
        dtw_distances = pickle.load(f)
    else:
      dtw_distances = self.compute_dtw(nsurface=nsta,key=key)

    db.set_trace()

    dtw_distances = self.compute_stats(dtw_distances,'mean')

    #self.make_dtw_figures(dtw_distances,xcoord, key=key, fig_num=3, colors=colors, save=True, save_dir=savefile)
    return dtw_distances

  def plot_velocity_maps(self,s=3,key='visla_sh',brock_Vs=1000,clim=None,key_title=None,nreal=7,save=True,init=True):
    set_rcParams()

    x,y = read_profile(init=init)
    for cor_l in self.hete_obj[key].keys():
        print(cor_l)
        fig, ax = plt.subplots(figsize=(12,3.5))
        ax = self.hete_obj[key][cor_l][nreal].plot_Vs(vs_br=1000,s=s,cmap='jet',axis=ax,clim=clim, size='3%')
        #ax.fill_between(x,np.ones(x.shape)*-34,y[7,:],facecolor='#b26400')
        #ax.set_xlim(min(x),max(x))
        #ax.set_ylim(-34,max(y[0]+1))
        ax.set_xlim(490,1590)
        ax.set_title(key_title[cor_l],fontsize=17)
        ax.spines['right'].set_visible(False)
        ax.spines['top'].set_visible(False)
        if save:
          fig.savefig(self.save_dir + cor_l + '_bis.png', dpi=300, bbox_inches='tight', pad_inches=0.02)
    plt.show()


  ###################################################################################################################
  #           STATIC METHODS
  ###################################################################################################################

  #-- Load objects ---
  @staticmethod
  def load(case="HOMO"):
    sim_obj  = create_obj(case=case,load=True)
    return sim_obj

  @staticmethod
  def select_key(dic,key):
    output = { c : dic[c] for c in key }
    return output

  @staticmethod
  def compute_score(obj,option='slope',remove_10perc=False):
    output = {}

    for cor_l in obj:

      n_simul = len(obj[cor_l])
      ns = len(obj[cor_l][0])

      if option == 'predict':
        output[cor_l] = []
      else:
        output[cor_l] = np.zeros((n_simul,ns))

      for n in range(n_simul):
        for i in range(len(obj[cor_l][n])):

          x = obj[cor_l][n][i][0].reshape((-1,1))
          y = obj[cor_l][n][i][1]

          if remove_10perc:
            ind10 = int(round(len(y) * 0.1,0))
            x = x[ind10:-ind10,:]
            y = y[ind10:-ind10]

          model = LinearRegression().fit(x,y)

          if option == 'slope':
            output[cor_l][n,i] = model.coef_
          elif option == 'score':
            output[cor_l][n,i] = model.score(x,y)
          elif option == 'predict':
            output[cor_l].append(model.predict(x))

    return output

  @staticmethod
  def apply_dtw(ref,obj,nsurface,filename=None):
    output = {}

    for cor_l in obj.keys():
      print(cor_l)
      output[cor_l] = [ [ dtw.distance(ref[:,i], op.velocity[:,i]) for i in range(nsurface)] for
                         op in obj[cor_l] ]
      db.set_trace()

    # pickle output
    if filename :
      print('saving ... ')
      with open(filename,'wb') as f:
        pickle.dump(output,f,protocol=pickle.HIGHEST_PROTOCOL)
    return output

  @staticmethod
  def get_dtw_dim(obj,dim):
    assert (dim in ['dist','cost','path'])
    if dim == 'dist':
      indice = 0
    elif dim == 'cost':
      indice = 1
    elif dim == 'path':
      indice = 3

    output = {}
    for cor_l in obj:
      output[cor_l] = [ [i[indice] for i in op] for op in obj[cor_l]]
    return output

  @staticmethod
  def plot_stats(obj_list,n_sta):
    set_rcParams()
    from matplotlib.lines import Line2D

    N = len(obj_list)
    x_axis = np.arange(N)
    y_axis = np.array([ obj[n_sta] for obj in obj_list ])

    # mean
    mean = y_axis.mean()
    argmin = y_axis.argmin()
    argmax = y_axis.argmax()
    std    = (y_axis.std()/mean)*100

    fig , ax = plt.subplots(figsize=(8,5))
    ax.plot(y_axis,'*')
    ax.plot(x_axis[argmin],y_axis[argmin],'r*',label='Min (simu {}) : {:.4f}'.format(argmin + 1, y_axis[argmin]))
    ax.plot(x_axis[argmax],y_axis[argmax],'g*',label='Max (simu {}) : {:.4f}'.format(argmax + 1, y_axis[argmax]))
    ax.hlines(mean,np.min(x_axis),N,'b',label='Mean value : {:.4f}'.format(mean))

    handles , labels = ax.get_legend_handles_labels()
    handles.append(Line2D([0],[0],color='k'))
    labels.append('Std value : {:.2f}%'.format(std))
    ax.legend(handles=handles,labels=labels)

    ax.set_xlabel('Simulation number')
    ax.set_xlabel('')
    ax.set_xticks(x_axis)
    ax.set_xticklabels([str(i+1) for i in x_axis])
    ax.set_title('Statistics among {} different realisations of a simulation'.format(N))
    plt.show(block=False)

    # Variation of standard deviation among all stations
    obj_array = np.array(obj_list)
    array_mean = obj_array.mean(axis=0)
    array_std  = (obj_array.std(axis=0) / array_mean) * 100



    fig2, ax2 = plt.subplots(figsize=(12,6))
    ax2.plot(array_std,'.',markersize=4)
    ax2.set_xlabel('Recording stations')
    ax2.set_ylabel('Variation coefficient [%]')
    ax2.set_title('Variation coefficient along recording stations')
    plt.show(block=True)




  @staticmethod
  def apply_method(obj_dict,method,params={},output=True,**kwargs):
    """
      Apply class methods from strings

      Input :
        -- obj_dict [object]  :: class object
        -- method   [str]     :: obj's method
        -- params   [dict]    :: method's parameters
        -- output   [bool]    :: Returns the object dictionary
        -- n_out    [int]     :: integer to select the number of output arguments of a fxn
    """

    dict_keys = list(obj_dict.keys())

    if ('CL' in kwargs.keys()) and (bool(kwargs['CL'])):
      select = True
    else :
      select = False
    
    if isinstance(obj_dict[dict_keys[0]],dict):
      result = {}
      for key in obj_dict:
        result[key] = {}

        if select:
          for cor_l in kwargs['CL']:
            result[key][cor_l] = [ getattr(obj,method)(**params) for obj in obj_dict[cor_l] ]
        else :
          for cor_l in obj_dict[key]:
            result[key][cor_l] = [ getattr(obj,method)(**params) for obj in obj_dict[key][cor_l] ]

      return result

    elif isinstance(obj_dict[dict_keys[0]],list):

      if output == True:
        result = {}

        if 'n_out' in kwargs.keys():
          if select :
            for cor_l in kwargs['CL']:
              result[cor_l] = [ getattr(obj,method)(**params)[kwargs['n_out']] for obj in obj_dict[cor_l] ]
          else:
            for cor_l in obj_dict:
              result[cor_l] = [ getattr(obj,method)(**params)[kwargs['n_out']] for obj in obj_dict[cor_l] ]
        else:
          if select :
            for cor_l in kwargs['CL']:
              result[cor_l] = [ getattr(obj,method)(**params) for obj in obj_dict[cor_l] ]
          else:
            for cor_l in obj_dict:
              result[cor_l] = [ getattr(obj,method)(**params) for obj in obj_dict[cor_l] ]
        return result
      else:
        if select:
          for cor_l in kwargs['CL']:
            for obj in obj_dict[cor_l]:
              getattr(obj,method)(**params)
        else:
          for cor_l in obj_dict:
            for obj in obj_dict[cor_l]:
              getattr(obj,method)(**params)
        return obj_dict

    else:

      msg = 'Wrong object dictionary type given !!'
      raise Exception(msg)


  @staticmethod
  def compute_ratio(reference,data):
    import warnings
    warnings.filterwarnings('ignore',category=RuntimeWarning)

    obj_keys = list(data.keys())

    if isinstance(data[obj_keys[0]],dict):
      diff = {}
      for key in data:
        diff[key] = { cor_l : np.divide( data[key][cor_l] , reference) for cor_l in \
                          data[key] }
    elif isinstance(data[obj_keys[0]],np.ndarray):
      diff = { cor_l : np.divide( data[cor_l] , reference) for cor_l in data }
    else:
      msg = 'Wrong data type given as parameter'
      raise Exception(msg)

    return diff

  @staticmethod
  def compute_stats(data,option=None):

    obj_keys = list(data.keys())

    params = {'mean':'mean', 'median':'median', 'min':'minimum', 'max':'maximum',
              'perc':'perc', 'std':'std', 'gmean':'gmean', 'gmean_cov':'gmean_cov'}

    if option:

      stats = {}
      if isinstance(data[obj_keys[0]],dict):
        for key in data:
          stats[key] = {}
          if option == 'perc':
            for cor_l in data[key]:
              perc10 = np.percentile(np.array(data[key][cor_l]),16,axis=0)
              perc80 = np.percentile(np.array(data[key][cor_l]),84,axis=0)
              median = np.median(np.array(data[key][cor_l]),axis=0)
              stats[key][cor_l] = (perc80 - perc10) / ( 2 * median)
          elif option == 'median':
            for cor_l in data[key]:
              stats[key][cor_l] = np.median(np.array(data[key][cor_l]),axis=0)
          else:
            for cor_l in data[key]:
              stats[key][cor_l] = getattr(np.array(data[key][cor_l]),option)(axis=0)

      elif isinstance(data[obj_keys[0]],list):
        if option == 'perc':
          for cor_l in data:
            perc10 = np.percentile(np.array(data[cor_l]),16,axis=0)
            perc80 = np.percentile(np.array(data[cor_l]),84,axis=0)
            median = np.median(np.array(data[cor_l]),axis=0)
            stats[cor_l] = (perc80 - perc10) / (2 * median)
        elif option == 'median':
            for cor_l in data:
              stats[cor_l] = np.median(np.array(data[cor_l]),axis=0)
        else:
          for cor_l in data:
            stats[cor_l] = getattr(np.array(data[cor_l]),option)(axis=0)
      else:
        msg = 'Wrong data type given as parameter'
        raise Exception(msg)

    else:
      stats = {}

      if isinstance(data[obj_keys[0]],dict):
        for key in data:
          stats[key] = {}
          for cor_l in data[key]:
            stats[key][cor_l] = {}
            stats[key][cor_l].fromkeys(params)
            array = np.array(data[key][cor_l])
            for para,value  in params.items():
              if para == 'perc':
                perc16 = np.percentile(array,16,axis=0)
                perc84 = np.percentile(array,84,axis=0)
                median = np.median(array,axis=0)
                stats[key][cor_l][value] = (perc84 - perc16) / ( 2 * median)
                stats[key][cor_l]['84perc'] = perc84
                stats[key][cor_l]['16perc'] = perc16
              elif para == 'median':
                stats[key][cor_l][value] = np.median(array,axis=0)
              elif para == 'gmean':
                stats[key][cor_l][value] = gmean(array,axis=0)
              elif para == 'gmean_cov':
                std = np.std( np.log(array), axis=0)
                stats[key][cor_l][value] = np.sqrt( np.exp(std**2) - 1 ) * 1e2
              else:
                stats[key][cor_l][value] = getattr(array,para)(axis=0)

      elif isinstance(data[obj_keys[0]],list):
        for cor_l in data:
          stats[cor_l] = {}
          stats[cor_l].fromkeys(params)
          array = np.array(data[cor_l])

          for para,value  in params.items():
            if para == 'perc':
              perc16 = np.percentile(array,16,axis=0)
              perc84 = np.percentile(array,84,axis=0)
              median = np.median(array,axis=0)
              stats[cor_l][value] = (perc84 - perc16) / ( 2 * median)
              stats[cor_l]['84perc'] = perc84
              stats[cor_l]['16perc'] = perc16
            elif para == 'median':
              stats[cor_l][value] = np.median(array,axis=0)
            elif para == 'gmean':
              stats[cor_l][value] = gmean(array,axis=0)
            elif para == 'gmean_cov':
              std = np.std( np.log(array), axis=0)
              stats[cor_l][value] = np.sqrt( np.exp(std**2) - 1 )
            else:
              stats[cor_l][value] = getattr(array,para)(axis=0)

    return stats

  @staticmethod
  def velocity_maps(data,save_dir=None,brock_Vs=0, clim=[100,380]):
    set_rcParams()

    for cor_l in data.keys():

      fig , ax = plt.subplots(5,2,figsize=(12,9),sharex='col',sharey='row')
      axis_f = ax.flatten()
      n = 0
      for obj in data[cor_l]:

        # Plot
        axis = obj.plot_Vs(vs_br=brock_Vs,clim=clim,axis=axis_f[n])
        #if (n != 8) or (n != 9):
        #  axis.set_xticklabels([])
        #if n % 2:
        #  axis.set_yticklabels([])
        axis.set_ylabel('')
        axis.set_xlabel('')
        axis.set_title('Simulation {:d}'.format(n+1))
        n += 1

      fig.text(0.53,0.045, 'Horizontal profile [m]', ha='center', fontweight='bold',fontsize=15)
      fig.text(0.075,0.5, 'Depth [m]', va='center',rotation='vertical', fontweight='bold',fontsize=15)
      fig.suptitle(LABEL[cor_l], y=0.99, fontsize=20)

      if save_dir:
        save_file = save_dir + cor_l + '.png'
        fig.savefig(save_file, bbox_inches='tight', pad_inches=0.01)
      plt.show(block=True)

    pass

  @staticmethod
  def plot_psa_freq(data,fig,ax,xcoord):
    im = ax.imshow(data, cmap='jet', aspect='auto', vmin=0, vmax=15, \
                      origin='lower', extent=[min(xcoord),max(xcoord),0,8])
    divider = make_axes_locatable(ax)
    cax     = divider.append_axes('right', size='3%', pad=0.1)
    c  = fig.colorbar(im, cax=cax, fraction=0.046, shrink=0.6)
    im.set_clim(0,15)
    c.set_label('PSA [m$s^{-2}$]',fontsize=8)
    c.ax.minorticks_off()

    return ax

  @staticmethod
  def subplot_2ax(figsize=(8,6)):
    fig = plt.figure(figsize=figsize)
    ax1 = fig.add_axes([0.2,0.24,0.6,0.66])
    ax2 = fig.add_axes([0.2,0.075,0.6,0.15])
    return fig, ax1, ax2


def plot_ssr(data,title,xmin,xmax,clim,cmap='jet'):
    fig , ax = plt.subplots(figsize=(8,6))
    im = ax.imshow(data.T,origin='lower',extent=[xmin,xmax,0,50], interpolation='bilinear',
          aspect='auto',vmin=clim[0],vmax=clim[1],cmap=cmap)
    cb = fig.colorbar(im)
    cb.set_label('Amplification', rotation=90)
    cb.ax.minorticks_off()
    ax.tick_params(axis='x', which='minor', bottom=False)
    ax.set_ylim(0.2,10)
    ax.set_xlabel('Horizontal profile [m]')
    ax.set_ylabel('Frequency [Hz]')
    ax.set_title(title)
    return fig



