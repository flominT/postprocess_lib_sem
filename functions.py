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
sys.path.append('/Users/flomin/Desktop/cubit/Nice/scripts')
from test_bis import plot_nice, read_profile
import numpy as np

CASES          = [ "HETE", # Heterogeneous simulations
                   "HOMO", # Reference simulations
                  ]

def create_obj(case="HOMO", freqs=[0.1,10], load=False, save=True, verbose=True, kwargs={}):

    """
       Function to create, pickle and load sem2dpack objects depending on the case
       parameter.

       Parameters :: [type:default] for input variables
       -------------

         -- case [str:HOMO] : simulation type

         -- freqs [list:[0.1,10]] : frequency intervals for bandpass filtering the seismograms
                                    The values are used to initialize the state of the sem2dpack.freqs attribute.
                                    No filtering is applied till the sem2dpack.read_seismo method is called with
                                    filter_s = True. freqs values can be updated afterwards.

         -- load [bool:False]  : If load is True, Directly load the saved pickled objects (time saving)
                                else if load is False, create object.

         -- save [bool:True]   : Save the created sem2dpack objects.
    """

    assert case in CASES, "Wrong case parameter given : valid cases are {:s}".format( \
           ','.join([c for c in CASES]) )

    tic  = time.time()

    if not load :

        dict_out = {}

        if case == "HOMO":
            #------- Reference directories ---------------------------------------
            for key in kwargs["REF_SIMUL_TYPE"]:
                directory   = kwargs["REF_DIR"] + key + '/'

                if not os.path.isdir(directory):
                    msg = '{} does not exist !! \n Skipping ....'.format(directory)
                    print(msg)
                    pass
                else:
                    #------- Create objects --------------------------------------------
                    compo = get_compo(key)
                    dict_out[key] = sem2dpack(directory,freqs=freqs,component=compo)

        else :

            for key in kwargs["HETE_SIMUL_TYPE"]:

                dict_out[key] = {}
                compo = get_compo(key)
                for cl in kwargs["COR_LEN"]:

                    if verbose:
                      print('Creating objects for {}'.format(cl))

                    tmp_dir = kwargs["HETE_DIR"] + cl + '/' + key

                    if not os.path.isdir(tmp_dir):
                        msg = '{} does not exist !! \n Skipping ....'.format(tmp_dir)
                        print(msg)
                        pass
                    else:
                        dict_out[key][cl] = []

                        # Check the number of simulations and redefine N_SIMUL correspondinly

                        sim_num = len(glob.glob(tmp_dir + '/n[1-9]*'))
                        if sim_num != kwargs["N_SIMUL"]:
                            print(" {} contains {} simulations \n".format(tmp_dir,sim_num))
                            print("Redefining global N_SIMUL to {}".format(sim_num))

                        for i in range(sim_num):

                            directory = tmp_dir + '/n' + str(i+1) + '/'

                            dict_out[key][cl].append(sem2dpack(directory,freqs=freqs,component=compo))


        #-------------- Save -----------------------------------------------------

        save_file_dir   =  kwargs["PICKLE_DIR"] + '/' + case
        make_dir(save_file_dir)

        if case == "HOMO":
            sim_type = kwargs["REF_SIMUL_TYPE"]
        else:
            sim_type = kwargs["HETE_SIMUL_TYPE"]

        for key in sim_type :
            save_file_name = save_file_dir + '/' + key + '_object.pk'
            with open(save_file_name, 'wb') as f:
                pickle.dump(dict_out[key],f,protocol=pickle.HIGHEST_PROTOCOL)
    else:

        try :
            #-- Load pickles objects --------------------------------------------

            dict_out = {}

            if case == "HOMO":
                sim_type = kwargs["REF_SIMUL_TYPE"]
            else:
                sim_type = kwargs["HETE_SIMUL_TYPE"]

            #-- Load pickles objects --------------------------------------------
            for key in sim_type :
              save_file_dir  = kwargs["PICKLE_DIR"] + case
              save_file_name = save_file_dir + '/' + key + '_object.pk'

              with open(save_file_name, 'rb') as f:
                  loaded_obj = pickle.load(f)
              if case == "HETE":
                dict_out[key] = {cor_l : loaded_obj[cor_l] for cor_l in kwargs["COR_LEN"]}
              else:
                dict_out[key] = loaded_obj
        except:
            msg = 'Could not load objects , Run function with load = False before !!'
            raise Exception(msg)


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

  def time_traces(self,key=None):
    try :
        assert isinstance(key,(list,tuple))
    except:
        if key:
            key = [key,]
        else:
            key = list(self.ref_obj.keys())

    ref_trace  = {k : self.ref_obj[k].read_seismo() for k in key}
    hete_trace = self.apply_method(self.hete_obj,'read_seismo',params={})

    if key :
        hete_trace = {k : hete_trace[k] for k in key}

    return ref_trace, hete_trace

  def plot_ref_wiggle(self,nsta,stride,key=None,naxis=False,save=False,fig_title=''):
    set_rcParams()

    try:
        assert isinstance(key,(list,tuple))
    except:
        if key:
            key = [key,]
        else:
            key = self.hete_obj.keys()

    if naxis:
        # Reference wiggle
        for k in key:
            ref_fig, ax1, ax2 = self.subplot_2ax()
            divider = make_axes_locatable(ax1)
            cax     = divider.append_axes('right',size='3%',pad=0.2)
            cax.remove()

            ax1 = self.ref_obj[k].plot_wiggle(ssta=nsta,stride=stride,axis=ax1)
            plot_nice(op='norm',axis=ax2)
            ax1.set_ylabel('Time [s]')
            ref_fig.suptitle(fig_title[k], y=0.98)
            if save:
                savefile = self.save_dir + k + '_wiggle.png'
                ref_fig.savefig(savefile, bbox_inches='tight', pad_inches=0.01)
    else:
        for k in key:
            self.ref_obj[k].plot_wiggle(ssta=nsta,stride=stride)

    return

  def plot_hete_wiggle(self,nsta,stride,key=None,naxis=False,save=False,
                        fig_title='', axis_title=''):
    set_rcParams()
    x,y = read_profile()

    try:
        assert isinstance(key,(list,tuple))
    except:
        if key:
            key = [key,]
        else:
            key = self.hete_obj.keys()

    for k in key:
        for cor_l in self.hete_obj[k].keys():
            if naxis:
                hete_fig , ax1, ax2 = self.subplot_2ax()
                divider = make_axes_locatable(ax1)
                cax     = divider.append_axes('right',size='3%',pad=0.2)
                cax.remove()

                ax1 = self.hete_obj[k][cor_l][7].plot_wiggle(ssta=nsta,stride=stride,axis=ax1)
                ax2 = self.hete_obj[k][cor_l][7].plot_Vs(vs_br=1000,cmap='jet',axis=ax2,clim=[90,390])
                ax2.fill_between(x,np.ones(x.shape)*-34,y[7,:],facecolor='#b26400')
                ax2.set_xlim(min(x),max(x))
                ax1.set_ylabel('Time [s]')
                hete_fig.suptitle(fig_title[k], y=0.998)
                ax1.set_title(axis_title[cor_l])

                if save:
                    savefile = self.save_dir + k + '_' + cor_l + '_wiggle.png'
                    hete_fig.savefig(savefile, bbox_inches='tight', pad_inches=0.02)
            else:
                self.hete_obj[k][cor_l][7].plot_wiggle(ssta=nsta,stride=stride)

    plt.show()


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

    # Reference TF
    for k in key:
      tf_param.update({'brockName':self.pickle_dir + k + '_br_ssr'})
      ref_tf[k]  = self.ref_obj[k].compute_tf(**tf_param,saveBr=True)

    # Heterogeneous TF
    for k in key:
      hete_tf[k] = {}
      tf_param.update({'brockName':self.pickle_dir + k + '_br_ssr.npy'})
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




  def compare_peak_values(self, key=None, pv_type='pgv'):
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
      ref_pgv[k] = self.ref_obj[k].compute_pv(op=pv_type,freqs=freqs,n_surf=421,component='x')

    data = self.select_key(self.hete_obj,key)
    hete_pgv = self.apply_method(data,'compute_pv',params={'op':pv_type,'n_surf':421,'freqs':freqs,'component':'x'})

    #---- Compute statistics on the heterogeneous simulations --
    hete_stats = self.compute_stats(hete_pgv)

    return ref_pgv, hete_pgv, hete_stats


  def rms_velocity(self,key='visla_sh'):

    return


  def compute_psa(self, key='elast_sh', T=[2, 6], save=None, plot_op = 'default', pickle_load=True):
    """
      Computes the pseudo-spectral acceleration (PSA) as a function of frequency.

      Input :
       -- key :: Simulation type
       -- T   :: Frequencies at which to compute the PSA
    """

    make_dir(self.pickle_dir + 'PSA/')
    pickle_name = self.pickle_dir +  'PSA/peak_response.pk'

    if not pickle_load:
      T  = [1,2,3,4,5,6,7,8]

      #-- ref psa
      ref_pr = {}
      for key in ref_obj:
        ref_pr[key] = self.ref_obj[key].psa_sac(T,n_surf=421)

      hete_pr = self.apply_method(self.hete_obj,'psa_sac',params={'T':T,'n_surf':421})

      pickle_pr = {'ref_pr':ref_pr, 'hete_pr':hete_pr}

      # Save peak responses

      with open(pickle_name,'wb') as f:
        pickle.dump(pickle_pr, f, protocol=pickle.HIGHEST_PROTOCOL)

    else :
      with open(pickle_name,'rb') as f:
        pickle_pr = pickle.load(f)

      ref_pr  = pickle_pr['ref_pr']
      hete_pr = pickle_pr['hete_pr'][key]

    hete_pr = self.select_key(hete_pr,self.hete_obj[key].keys())
    
    # Select periods
    ref_pr = { k : ref_pr[k][T,:] for k in ref_pr.keys()}
    hete_pr = { cl : [hete_pr[cl][i][T,:] for i in range(len(hete_pr[cl]))] for cl in hete_pr.keys()}

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

    ref_simul = self.load(case='HOMO')
    hete_simul = self.load(case='HETE')

    ref_obj  = ref_simul[key]
    hete_obj = hete_simul[key]

    # Computing the DTW is time consuming, so compute once and pickle the objects
    save_dir = pickle_dir + 'DTW/'
    filename_hete = save_dir + key + '_hete_dtw.pk'

    if save :
      make_dir(save_dir)
      hete_obj = self.apply_method(hete_obj,'decimate_sig',params={'q':4,'filter_s':True},output=False)
      ref_obj.decimate_sig(q=4,filter_s=True)
      ref_hete_dtw = self.apply_dtw(ref_obj.decimated_veloc,hete_obj,nsurface=nsurface,filename=filename_hete,CL=CL)


  def plot_dtw(self,key='visla_sh'):

    save_dir = self.pickle_dir + 'DTW/'
    filename_hete = save_dir + key + '_hete_dtw.pk'

    with open(filename_hete,'rb') as f:
      ref_hete_dtw = pickle.load(f)

    # Get dtw distances
    ref_hete_path = self.get_dtw_dim(ref_hete_dtw,'path')
    dtw_distances = self.get_dtw_dim(ref_hete_dtw,'dist')

    dtw_distances = self.compute_stats(dtw_distances,'mean')

    #self.make_dtw_figures(dtw_distances,xcoord, key=key, fig_num=3, colors=colors, save=True, save_dir=savefile)
    return dtw_distances

  def plot_velocity_maps(self,key='visla_sh',brock_Vs=1000,clim=None,key_title=None,save=True):
    set_rcParams()
    x,y = read_profile()
    for cor_l in self.hete_obj[key].keys():
        fig, ax = plt.subplots(figsize=(12,3))
        ax = self.hete_obj[key][cor_l][7].plot_Vs(vs_br=1000,cmap='jet',axis=ax,clim=clim)
        ax.fill_between(x,np.ones(x.shape)*-34,y[7,:],facecolor='#b26400')
        ax.set_xlim(min(x),max(x))
        ax.set_ylim(-34,max(y[0]+1))
        ax.set_title(key_title[cor_l])
        if save:
          fig.savefig(self.save_dir + cor_l + '.png', bbox_inches='tight', pad_inches=0.02)
        plt.show(block=True)
    


  def plot_dtw_pair_dist(self,key='visla_psv',save=True):
    ref_obj = self.load(case='HOMO')[key]

    save_file = pickle_dir + 'DTW/' + key + '_ref_pair_wise_dtw.pk'

    if save :
      dist_path = ref_obj.pairwise_dtw()

      with open(save_file,'wb') as f:
        pickle.dump(dist_path,f,protocol=pickle.HIGHEST_PROTOCOL)
    else :

      with open(save_file, 'rb') as f:
                  dist_path = pickle.load(f)

    xcoord = ref_obj.rcoord[:,0]
    set_rcParams()
    fig , ax = plt.subplots()
    ax.imshow(dist_path['dist'],aspect='auto',extent=[xcoord[115],xcoord[315],dist_path['dist'].shape[0]*5,5])
    plt.show(block=True)
    db.set_trace()

    return

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
  def apply_dtw(ref,obj,nsurface,filename=None,parallel=False,**kwargs):
    output = {}

    if ('CL' in kwargs.keys()) and (bool(kwargs['CL'])):
      for cor_l in kwargs['CL']:
        print(cor_l)
        output[cor_l] = [ [ dtw(ref[:,i], op.decimated_veloc[:,i], dist=euclidean) for i in range(nsurface)] for
                         op in obj[cor_l] ]
    else:
      for cor_l in obj:
        print(cor_l)
        output[cor_l] = [ [ dtw(ref[:,i], op.decimated_veloc[:,i], dist=euclidean) for i in range(nsurface)] for
                           op in obj[cor_l] ]

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
  def make_psa_figures(ref_data, hete_data, xcoord, T, save_dir=None, colors='rbgc',
                        plot_op='stats', key='elast_sh',**kwargs):
    '''
      Make pseudo-spectral acceleration figures.
    '''

    set_rcParams()
    plt.rcParams['axes.linewidth'] = 0.8

    if plot_op == 'stats':
      fig = plt.figure(figsize=(14,6))
      grid = fig.add_gridspec(2,2)

      i = 0
      for stat in hete_data.keys():
        gax = grid[i].subgridspec(1,2)
        for j in range(2):
          ax = fig.add_subplot(gax[j])

          ax.plot(xcoord,ref_data[j,:],'k',label='reference medium')
          cc = 0
          for cor_l in hete_data[stat].keys():
            ax.plot(xcoord,hete_data[stat][cor_l][j,:],c=colors[cor_l],label=LABEL[cor_l])
            cc += 1

          ax.set_title('Central frequency : {} Hz'.format(T[j]))

        i += 1

    elif plot_op == 'maps' :

      for cor_l in hete_data:

        plt.rcParams['axes.labelsize'] = 10
        plt.rcParams['ytick.major.size'] = 2

        fig = plt.figure(figsize = (8,12))
        gs = mpl.gridspec.GridSpec(15 , 2 , wspace=0.4 , hspace=0.4, figure=fig, left=0.08, right=0.92,
                                bottom=0.05, top=0.95)
        sim = 0
        n = 0
        for it in range(5):
          i = n
          j = n+2
          for k in range(2):
            simu = hete_data[cor_l][sim]
            ax1 = fig.add_subplot(gs[i:j,k], xticklabels=[])
            ax2 = fig.add_subplot(gs[j,k], sharex=ax1)
            ax1.set_title('simulation {}'.format(sim+1),fontsize=12)

            im = ax1.imshow(simu, cmap='jet', aspect='auto', vmin=0, vmax=15, \
                      origin='lower', extent=[min(xcoord),max(xcoord),0,8])
            divider = make_axes_locatable(ax1)
            cax     = divider.append_axes('right', size='3%', pad=0.1)
            c  = fig.colorbar(im, cax=cax, fraction=0.046, shrink=0.6)
            im.set_clim(0,15)
            c.set_label('PSA [m$s^{-2}$]',fontsize=8)
            c.ax.minorticks_off()
            ax1.xaxis.set_visible(False)
            ax2 = getattr(kwargs['hete_obj'][cor_l][sim],'plot_Vs')(axis=ax2,clim=[90,390])


            if it == 4 :
              ax2.xaxis.set_visible(True)
            else:
              ax2.xaxis.set_visible(False)

            ax1.minorticks_off()
            ax1.set_yticks(np.arange(.5,8,1))
            ax1.set_yticklabels(np.arange(1,9,1))
            ax2.set_xlim([495,1575])
            ax1.set_ylabel('Frequency [Hz]')

            sim += 1

          n += 3
        fig.suptitle(key_title[key] + ', ' + LABEL[cor_l])

        if save_dir:
          savefile = save_dir + key + '_' + cor_l + '_psa_maps.png'
          fig.savefig(savefile)

    elif plot_op == 'ref':

      fig = plt.figure(figsize=(10,8))
      ax1 = fig.add_axes([0.2,0.28,0.6,0.65])
      ax2 = fig.add_axes([0.2,0.075,0.6,0.2])

      for sim in ref_data:
        im = ax1.imshow(ref_data[sim], cmap='jet', aspect='auto', vmin=0, vmax=15, \
                      origin='lower', extent=[min(xcoord),max(xcoord),0,8])
        divider = make_axes_locatable(ax1)
        cax     = divider.append_axes('right', size='3%', pad=0.2)
        c  = fig.colorbar(im, cax=cax, fraction=0.046, shrink=0.6)
        im.set_clim(0,15)
        c.set_label('PSA [m$s^{-2}$]',fontsize=12)
        c.ax.minorticks_off()

        ax1.xaxis.set_visible(False)

        ax1.minorticks_off()
        ax1.set_yticks(np.arange(.5,8,1))
        ax1.set_yticklabels(np.arange(1,9,1))
        ax2.set_xlim([495,1575])
        ax1.set_ylabel('Frequency [Hz]')
        ax1.set_title('Pseudo-spectral acceleration ')

        split = key.split('_')

        if split[1] == 'tab':
          plot_nice(op='tabular',axis=ax2)
        elif split[1] == 'homo':
          plot_nice(op='homo',axis=ax2)
        else:
          plot_nice(op='norm',axis=ax2)

        fig.suptitle(key_title[key], y=0.99)
        if save_dir:
          savefile = save_dir + sim + '_ref_psa.png'
          fig.savefig(savefile)

    elif plot_op == 'freq':
      stats_param = ['mean','min','max','median']
      cl  = ['a10_s5','a10_s30','a50_s5','a50_s30']
      fig, ax = plt.subplots(2,4,figsize=(12,6), sharex = 'col')
      fig.subplots_adjust(bottom = 0.15,left=0.06,right=0.99,hspace=0.2,top=0.9)

      axis = ax.flatten()
      j = 0
      for stat in stats_param:
        for i in range(2):
          if i == 0:
            for cor_l in cl :
              axis[j].plot(xcoord,hete_data[cor_l][stat][1,:],c=colors[cor_l])
            axis[j].plot(xcoord,ref_data[key][1,:],'k')
            axis[j].set_title(stat.capitalize() + ' at 2 Hz',fontsize=14)
          else :
            for cor_l in cl :
              axis[j+1].plot(xcoord,hete_data[cor_l][stat][5,:],c=colors[cor_l])
            axis[j+1].plot(xcoord,ref_data[key][5,:],'k')
            axis[j+1].set_title(stat.capitalize() + ' at 6 Hz',fontsize=14)
        j += 2

      fig.text(0.02,0.5, 'PSA',
                va='center',rotation='vertical', fontsize=14)
      fig.text(0.5,0.08, 'Horizontal profile [m]', ha='center', fontsize=14)


      fig.suptitle(key_title[key], y=0.99)
      #plt.tight_layout(pad=2.8,h_pad=0,w_pad=0.5)

      lines = [Line2D([0],[0], color=i) for i in colors ]
      lines.insert(0,Line2D([0],[0], color='k'))
      labels = [LABEL[cor_l] for cor_l in cl] ; labels.insert(0,'Reference medium')
      fig.legend(lines, labels, loc= (0.02,0.01), ncol=5)

      if save_dir:
        savefile = save_dir + key  + '_psa_stats.png'
        fig.savefig(savefile)

    else :
      stats_param = ['mean','min','max','median','perc']
      cl  = ['a10_s5','a10_s30','a50_s5','a50_s30']

      for t in T:
        fig, ax = plt.subplots(3, 2, figsize=(12,7), sharex='col')
        fig.subplots_adjust(wspace=0.1,top=0.87)
        axis = ax.flatten()

        i = 0
        for param in stats_param:
          for cor_l in cl:
            axis[i].plot(xcoord,hete_data[cor_l][param][t-1,:],c=colors[cor_l])

          if param != 'perc':
            axis[i].plot(xcoord,ref_data[key][t-1,:],'k')

          if i == 3:
            axis[i].xaxis.set_tick_params(labelbottom=True)
          if param == 'perc':
            axis[i].set_title('Normalized percentile range',fontsize=18)
          else:
            axis[i].set_title(param.capitalize(),fontsize=18)
          axis[i].set_xlim(500,1570)

          i += 1


        fig.text(0.06,0.5, 'PSA [m$s^{-2}$]',
                  va='center',rotation='vertical', fontsize=16)
        fig.text(0.5,0.025, 'Horizontal profile [m]', ha='center', fontsize=16)


        fig.suptitle(key_title[key] + '\n' + 'Pseudo-spectral response statistics at ' + str(t) + ' Hz', y=0.99)

        axis[-1].remove()
        lines  = []
        labels = []

        for cor_l in colors:
          lines.append(Line2D([0],[0], color=colors[cor_l]))
          labels.append(LABEL[cor_l])
        lines.insert(0,Line2D([0],[0], color='k'))
        labels.insert(0,'Reference medium')
        fig.legend(lines, labels, loc= (0.6,0.13))


        if save_dir:
          print('Saving .... ')
          savefile = save_dir + key  + '_psa_default_' + str(t) +'Hz.png'
          fig.savefig(savefile, bbox_inches='tight', pad_inches=0.01)

      db.set_trace()


  @staticmethod
  def plot_config_1dtf(n_sta, ref, h_obj,
                        h_max, h_min, h_mean,
                        max_ratio, min_ratio, mean_ratio,
                        lab, colors, CL, **kwargs ):

    # Time and frequency vectors
    tvec = ref.tvec
    freq = ref.fft_freq

    #-- Initialize plot parameters --
    tick_param  = set_plot_param(option='tick',fontsize=10)
    label_param = set_plot_param(option='label')

    fig , ax = plt.subplots(3,1,figsize=(9,7))

    #-- Plot reference ---
    ax[0].plot(tvec, ref.velocity[:,n_sta], c='k',label='Homogeneous medium',linewidth=1)
    ax[1].plot(freq, ref.raw_ssr[n_sta,:], c='k',linewidth=1.5)

    #-- Plot heterogeneous
    cycol = cycle(colors)
    i = 0

    for cor_l in CL :
      color = next(cycol)
      ax[0].plot(tvec,h_obj[cor_l][4].velocity[:,n_sta],c=color,label=lab[i], linewidth=1)

      ax[1].plot(freq,h_max[cor_l][n_sta,:],c=color,linestyle='-.',linewidth=1, \
                 label='max TF')
      ax[1].plot(freq,h_min[cor_l][n_sta,:],c=color,linestyle='--',linewidth=1, \
                 label='min TF')
      ax[1].plot(freq,h_mean[cor_l][n_sta,:],c=color,linestyle='-',linewidth=1, \
                 label='mean TF')

      ax[2].plot(freq,max_ratio[cor_l][n_sta,:],c=color,linestyle='-.',linewidth=1)
      ax[2].plot(freq,min_ratio[cor_l][n_sta,:],c=color,linestyle='--',linewidth=1)
      ax[2].plot(freq,mean_ratio[cor_l][n_sta,:],c=color,linestyle='-',linewidth=1)
      i += 1

    ax[0].set_ylabel('Velocity [ms$^{-1}$]',**label_param)
    ax[1].set_ylabel('Amplification',**label_param)
    ax[2].set_ylabel('Ampl. ratio (Ht/Hm)',**label_param)

    ax[0].set_xlabel('Time [s]',**label_param)
    ax[1].set_xlabel('Frequency [Hz]',**label_param)
    ax[2].set_xlabel('Frequency [Hz]',**label_param)
    ax[2].set_ylim(0.2,2)

    ax[0].set_title('Receiver at $x$ = {:.1f} m '.format(int(ref.rcoord[n_sta,0])),**label_param)

    if 'key' in kwargs:
      tit = kwargs['key'].split('_')
      if tit[0] == 'elast':
        tit[0] = 'Elastic'
      elif tit[0] == 'visla':
        tit[0] =  'Viscoelastic'

    #-- Legend --
    from matplotlib.lines import Line2D
    custom_lines = [Line2D([0], [0], color=colors[0], lw=1),
                Line2D([0], [0], color=colors[0], linestyle='-.', lw=1),
                Line2D([0], [0], color=colors[0], linestyle='--', lw=1)]

    L = ax[0].legend(loc=1)
    L2 = ax[1].legend(custom_lines, ['mean', 'max', 'min'])
    plt.setp(L.texts, family='serif')
    plt.setp(L2.texts, family='serif')

    for axis in ax.flatten():
      axis.ticklabel_format(axis='y',style='sci',scilimits=(1,1))
      axis.tick_params(**tick_param)
    ax[0].set_xlim(0,10)
    ax[1].set_xlim(0.5,10)
    ax[2].set_xlim(0.5,10)
    ax[1].set_ylim(0,10)
    ax[2].axhline(y=1.2, color='k', linestyle='--',label='20 % amplification')
    ax[2].legend()
    ax[1].grid()
    ax[2].grid()
    fig.suptitle('{} {} wave propagation'.format(tit[0],tit[1].upper()),fontname='serif',fontweight='bold')

    plt.tight_layout(pad=2.8,h_pad=0,w_pad=0.5)
    if 'save_dir' in kwargs:
      if kwargs['save_dir']:
        print('== saving ...')
        fig.savefig(kwargs['save_dir'])
    plt.show(block=True)

  @staticmethod
  def make_tf_figures(n_sta, r_obj, h_obj, r_tf, h_stats,
                      colors='rbgc', key='visla_sh', **kwargs ):

    set_rcParams()
    # Time and frequency vectors
    tvec = r_obj.tvec
    freq = r_obj.fft_freq

    fig , ax = plt.subplots(2,1,figsize=(9,7))
    fig.subplots_adjust(bottom = 0.17)


    #-- Plot heterogeneous
    cycol = cycle(colors)

    for cor_l in h_stats.keys() :
      color = next(cycol)
      ax[0].plot(tvec,h_obj[cor_l][6].velocity[:,n_sta],c=color, linewidth=1)

      ax[1].plot(freq,h_stats[cor_l][n_sta,:],c=color,linestyle='-',linewidth=1, \
                 label='')

      #ax[2].plot(freq,h_ratio[cor_l][n_sta,:],c=color,linestyle='-',linewidth=1)

    #-- Plot reference ---
    ax[0].plot(tvec, r_obj.velocity[:,n_sta], c='k',linewidth=1)
    ax[1].plot(freq, r_tf[n_sta,:], c='k',linewidth=1)

    ax[0].set_ylabel('Velocity [ms$^{-1}$]')
    ax[1].set_ylabel('Amplification')
    #ax[2].set_ylabel(r'TF ratio ($\frac{median}{reference}$)')

    ax[0].set_xlabel('Time [s]')
    ax[1].set_xlabel('Frequency [Hz]')
    #ax[2].set_xlabel('Frequency [Hz]')
    #ax[2].set_ylim(0.5,1.6)

    ax[0].set_title('Receiver at $x$ = {:.1f} m '.format(int(r_obj.rcoord[n_sta,0])))

    ax[0].set_xlim(0,10)
    ax[1].set_xlim(0.5,10)
    #ax[2].set_xlim(0.5,10)
    ax[1].set_ylim(0.2,5)

    fig.suptitle(key_title[key], y=0.99)
    #plt.tight_layout(pad=2.8,h_pad=0,w_pad=0.5)

    lines = [Line2D([0],[0], color=i) for i in colors ]
    lines.insert(0,Line2D([0],[0], color='k'))
    labels = [LABEL[cor_l] for cor_l in h_stats.keys()] ; labels.insert(0,'Reference medium')
    fig.legend(lines, labels, loc= (0.2,0.01), ncol=3)

    if 'save_dir' in kwargs:
      if kwargs['save_dir']:
        print('== saving ...')
        savefile = kwargs['save_dir'] + key + '_TF_' + str(n_sta) + '_mean.png'
        fig.savefig(savefile)
    plt.show(block=True)


  @staticmethod
  def make_pv_figures(data1, data2, xcoord, CL=[], colors=None, label=None,
                      save_dir=None, plot_op = 'all', key='elast_sh', **kwargs):

    set_rcParams()

    if plot_op == 'all':
      for cor_l in data2.keys():
        fig = plt.figure(figsize = (8,12))
        gs = mpl.gridspec.GridSpec(10 , 2 , wspace=0.4 , hspace=0.4, figure=fig, left=0.08, right=0.92,
                                  bottom=0.05, top=0.95)
        sim = 0
        n = 0
        for it in range(5):
          i = n
          j = n + 1

          for k in range(2):

            simu = data2[cor_l][sim]

            ax1 = fig.add_subplot(gs[i,k], xticklabels=[])
            ax2 = fig.add_subplot(gs[j,k], sharex=ax1)

            divider = make_axes_locatable(ax1)
            cax = divider.append_axes('right', size='3%', pad=0.1)

            ax1.set_title('simulation {}'.format(sim+1),fontsize=14)

            ax1.plot(xcoord,getattr(simu,kwargs['pv_type']),'.',c='r',markersize=2)

            if (cor_l == 'nice_homo') and (key == 'visla_sh') :
              ref_key = 'visla_homo_sh'
              ax1.plot(xcoord,getattr(data1[ref_key],kwargs['pv_type']),'.',c='k',markersize=2)

            elif (cor_l == 'nice_homo') and (key == 'visla_psv') :
              ref_key = 'visla_homo_psv'
              ax1.plot(xcoord,getattr(data1[ref_key],kwargs['pv_type']),'.',c='k',markersize=2)

            elif (cor_l == 'nice_tabular') and (key == 'visla_psv') :
              ref_key = 'visla_tab_psv'
              ax1.plot(xcoord,getattr(data1[ref_key],kwargs['pv_type']),'.',c='k',markersize=2)

            elif (cor_l == 'nice_tabular') and (key == 'visla_sh') :
              ref_key = 'visla_tab_sh'
              ax1.plot(xcoord,getattr(data1[ref_key],kwargs['pv_type']),'.',c='k',markersize=2)
            else:
              ax1.plot(xcoord,getattr(data1[key],kwargs['pv_type']),'.',c='k',markersize=2)

            simu.plot_Vs(axis=ax2)

            cax.remove()
            ax1.set_xlim(500,1570)
            ax1.xaxis.set_visible(False)
            if kwargs['pv_type'] == 'pgv':
              ax1.set_ylabel('PGV [m$s^{-1}$]', fontsize=10)
            elif kwargs['pv_type'] == 'pga':
              ax1.set_ylabel('PGA [m$s^{-2}$]', fontsize=10)

            if it == 4 :
              ax2.xaxis.set_visible(True)
            else:
              ax2.xaxis.set_visible(False)

            sim += 1

          n += 2

        fig.suptitle(key_title[key] + ', ' + LABEL[cor_l])
        lines = [Line2D([0],[0], color='k'), Line2D([0],[0], color='r')]
        labels = ['Homogeneous medium', 'Heterogeneous media']
        fig.legend(lines, labels, loc= (0.2,0), ncol=2)

        if save_dir:
          savefile = save_dir + key + '_' + cor_l + '_{}.png'.format(kwargs['pv_type'])
          fig.savefig(savefile)


    elif plot_op == 'stats':

      stat_param = kwargs['stat_param']

      fig, ax = plt.subplots(3, 2, figsize=(12,7), sharex='col')
      fig.subplots_adjust(wspace=0.15)
      axis = ax.flatten()

      i = 0
      for param in stat_param:
        c = 0
        for cor_l in CL:
          axis[i].plot(xcoord,data2[cor_l][param],c=colors[cor_l])
          c += 1
        if param != 'perc':
          axis[i].plot(xcoord,data1[key],'k')

        if i == 3:
          axis[i].xaxis.set_tick_params(labelbottom=True)
        if param == 'perc':
          axis[i].set_title('Normalized percentile range',fontsize=18)
        else:
          axis[i].set_title(param.capitalize(),fontsize=18)
        axis[i].set_xlim(500,1570)

        i += 1

      # Axis labels
      if  kwargs['pv_type'] == 'pgv':
        fig.text(0.06,0.5, kwargs['pv_type'].upper() + ' [m$s^{-1}$]',
                va='center',rotation='vertical', fontsize=16)
      elif kwargs['pv_type'] == 'pga':
        fig.text(0.05,0.5, kwargs['pv_type'].upper() + ' [m$s^{-2}$]',
                va='center',rotation='vertical', fontsize=16)
      fig.text(0.5,0.03, 'Horizontal profile [m]', ha='center', fontsize=16)
      fig.suptitle(key_title[key], y=0.995)

      # Legend
      axis[-1].remove()
      lines = []
      labels = []
      for cor_l in colors.keys():
        lines.append(Line2D([0],[0], color=colors[cor_l]))
        labels.append(LABEL[cor_l])
      lines.insert(0,Line2D([0],[0], color='k'))
      labels.insert(0,'Reference medium')
      fig.legend(lines, labels, loc= (0.6,0.13))

      if save_dir:
        savefile = save_dir + key + '_stats_' + '_{}.png'.format(kwargs['pv_type'])
        fig.savefig(savefile, bbox_inches='tight', pad_inches=0.01)
      plt.show(block=True)


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

    params = ['mean','median','min','max','perc']

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
            for para in params:
              if para == 'perc':
                perc10 = np.percentile(array,16,axis=0)
                perc80 = np.percentile(array,84,axis=0)
                median = np.median(array,axis=0)
                stats[key][cor_l][para] = (perc80 - perc10) / ( 2 * median)
              elif para == 'median':
                stats[key][cor_l][para] = np.median(array,axis=0)
              else:
                stats[key][cor_l][para] = getattr(array,para)(axis=0)

      elif isinstance(data[obj_keys[0]],list):
        for cor_l in data:
          stats[cor_l] = {}
          stats[cor_l].fromkeys(params)
          array = np.array(data[cor_l])

          for para in params:
            if para == 'perc':
              perc10 = np.percentile(array,16,axis=0)
              perc80 = np.percentile(array,84,axis=0)
              median = np.median(array,axis=0)
              stats[cor_l][para] = (perc80 - perc10) / ( 2 * median)
            elif para == 'median':
              stats[cor_l][para] = np.median(array,axis=0)
            else:
              stats[cor_l][para] = getattr(array,para)(axis=0)

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

  # -----
  @staticmethod
  def make_dtw_figures(data,xcoord,fig_num=1,save=False, colors='', key='visla_psv',option=None,save_dir=None):
    set_rcParams()

    if fig_num == 1:
      for cor_l in data.keys():
        n = 0
        fig , ax = plt.subplots(5,2,figsize=(8,8),sharex='col')
        for i in range(5):
          for j in range(2):
            ax[i,j].plot(xcoord,data[cor_l][n,:],'*')
            ax[i,j].set_title('Simulation {:d}'.format(n+1))
            maxv = np.abs(data[cor_l][n,:]).max()
            nten = len(str(int(1/maxv)))
            incr = 1/(10**nten)*2
            ax[i,j].axhline(y=0,color='k',linestyle='--')
            ax[i,j].set_ylim(-maxv-incr,maxv+incr)
            n += 1
        fig.suptitle(LABEL[cor_l])
        fig.text(0.5,0.03, 'Horizontal profile [m]', ha='center', fontsize=13)
        fig.text(0.04,0.5, 'Slope - 1 {:s}'.format(option), va='center',rotation='vertical', fontsize=13)

        if save:
          fig.savefig(save_dir + '{0:s}_{1:s}_{2:s}.png'.format(key,cor_l,option))

      plt.show(block=True)
    elif fig_num == 2:
      fig, ax = plt.subplots(2,2,figsize=(12,6),sharex='col')
      cl = 0
      for axis in ax.flatten():
        axis.plot(xcoord,data[CL[cl]][0],'k',label='median score')
        axis.plot(xcoord,data[CL[cl]][1],label='median slope')
        axis.set_title(LABEL[CL[cl]])
        cl += 1
        axis.legend()
      fig.text(0.5,0.03, 'Horizontal profile [m]', ha='center', fontsize=14)
      fig.text(0.04,0.5, r'$\vert (score|slope) - 1 \vert$', va='center',rotation='vertical', fontsize=14)
      fig.suptitle(key_title[key])

      if save:
        fig.savefig(save_dir + '{:s}_score_slope.png'.format(key))
      plt.show(block=True)

    elif fig_num == 3:
      fig, ax = plt.subplots(1,1,figsize=(12,7))
      i = 0
      for cor_l in data.keys():
        ax.plot(xcoord,data[cor_l],c=colors[cor_l],label=LABEL[cor_l])
        i += 1
      ax.set_xlabel('Horizontal profile [m]',fontsize=16)
      ax.set_ylabel('Distance',fontsize=16)
      ax.set_title('DTW distance between reference and heterogeneous simulation',fontsize=18)
      ax.legend()
      fig.suptitle(key_title[key],y=0.995)

      if save:
        print('Saving ..')
        print(save_dir)
        fig.savefig(save_dir + '{:s}_dtw_distances.png'.format(key),bbox_inches='tight',pad_inches=0.01)
    elif fig_num == 5:
      fig, ax = plt.subplots(1,1,figsize=(12,7))

    plt.show(block=True)

  @staticmethod
  def make_grid_figure(obj,method1,method2):

    fig = plt.figure(figsize = (8,12))
    gs = mpl.gridspec.GridSpec(15 , 2 , wspace=0.4 , hspace=0.4, figure=fig, left=0.08, right=0.92,
                              bottom=0.05, top=0.95)
    sim = 0
    n = 0
    for it in range(5):
      i = n
      j = n+2
      for k in range(2):
        simu = obj[sim]
        ax1 = fig.add_subplot(gs[i:j,k], xticklabels=[])
        ax2 = fig.add_subplot(gs[j,k], sharex=ax1)
        ax1.set_title('simulation {}'.format(sim+1),fontsize=10)

        ax1 = getattr(simu,method1)(axis=ax1)
        ax2 = getattr(simu,method2)(axis=ax2,clim=[100,380])


        if it == 4 :
          ax2.xaxis.set_visible(True)
        else:
          ax2.xaxis.set_visible(False)

        sim += 1

      n += 3
    return fig

  @staticmethod
  def make_grid_figure1d(data1, data2, data3, xcoord, label=None,
                      save_dir=None, key='visla_sh', plot_param={}):

    set_rcParams()
    for cor_l in data2.keys():
      fig = plt.figure(figsize = (8,12))
      gs = mpl.gridspec.GridSpec(10 , 2 , wspace=0.4 , hspace=0.4, figure=fig, left=0.08, right=0.92,
                                bottom=0.05, top=0.95)
      sim = 0
      n = 0
      for it in range(5):
        i = n
        j = n + 1

        for k in range(2):

          simu = data2[cor_l][sim]

          ax1 = fig.add_subplot(gs[i,k], xticklabels=[])
          ax2 = fig.add_subplot(gs[j,k], sharex=ax1)

          divider = make_axes_locatable(ax1)
          cax = divider.append_axes('right', size='3%', pad=0.1)

          ax1.set_title('simulation {}'.format(sim+1),fontsize=14)

          ax1.plot(xcoord,simu,'.',c='r',markersize=2)

          if (cor_l == 'nice_homo') and (key == 'visla_sh') :
            ref_key = 'visla_homo_sh'
            ax1.plot(xcoord,data1[ref_key],'.',c='k',markersize=2)

          elif (cor_l == 'nice_homo') and (key == 'visla_psv') :
            ref_key = 'visla_homo_psv'
            ax1.plot(xcoord,data1[ref_key],'.',c='k',markersize=2)

          elif (cor_l == 'nice_tabular') and (key == 'visla_psv') :
            ref_key = 'visla_tab_psv'
            ax1.plot(xcoord,data1[ref_key],'.',c='k',markersize=2)

          elif (cor_l == 'nice_tabular') and (key == 'visla_sh') :
            ref_key = 'visla_tab_sh'
            ax1.plot(xcoord,data1[ref_key],'.',c='k',markersize=2)
          else:
            ax1.plot(xcoord,data1[key],'.',c='k',markersize=2)

          data3[cor_l][sim].plot_Vs(axis=ax2,clim=[100,380])

          cax.remove()
          ax1.set_xlim(500,1570)
          ax1.xaxis.set_visible(False)

          ax1.set_ylabel(plot_param['xlabel'], fontsize=10)
          ax1.ticklabel_format(axis='y',style='sci',scilimits=(0,0))


          if it == 4 :
            ax2.xaxis.set_visible(True)
          else:
            ax2.xaxis.set_visible(False)

          sim += 1

        n += 2

      fig.suptitle(key_title[key] + ', ' + LABEL[cor_l], y=0.985)
      lines = [Line2D([0],[0], color='k'), Line2D([0],[0], color='r')]
      labels = ['Homogeneous medium', 'Heterogeneous media']
      fig.legend(lines, labels, loc= (0.2,0), ncol=2)

      if save_dir:
        savefile = save_dir + key + '_' + cor_l + '_{}.png'.format(plot_param['plot_type'])
        fig.savefig(savefile)

    return

  @staticmethod
  def make_stats_fig1d(data1,data2,xcoord,CL,colors=None,save_dir=None,plot_param={},key='visla_sh'):

    stats_keys = ['mean','min','max','median','perc']
    set_rcParams()
    fig, ax = plt.subplots(3, 2, figsize=(12,7), sharex='col')
    fig.subplots_adjust(wspace=0.1)
    axis = ax.flatten()

    i = 0
    for param in stats_keys:
      c = 0
      for cor_l in CL:
        axis[i].plot(xcoord,data2[cor_l][param],c=colors[c])
        c += 1
      if param != 'perc':
        axis[i].plot(xcoord,data1[key],'k')

      if i == 3:
        axis[i].xaxis.set_tick_params(labelbottom=True)
      if param == 'perc':
        axis[i].set_title('Normalized percentile range',fontsize=14)
      else:
        axis[i].set_title(param.capitalize(),fontsize=14)
      axis[i].set_xlim(500,1570)
      axis[i].ticklabel_format(axis='y',style='sci',scilimits=(0,0))

      i += 1

    # Axis labels
    fig.text(0.08,0.5, plot_param['xlabel'],
              va='center',rotation='vertical', fontsize=14)
    fig.text(0.5,0.03, 'Horizontal profile [m]', ha='center', fontsize=14)
    fig.suptitle(key_title[key],y=0.985)

    # Legend
    axis[-1].remove()
    lines = [Line2D([0],[0], color=i) for i in colors ]
    lines.insert(0,Line2D([0],[0], color='k'))
    labels = [LABEL[cor_l] for cor_l in CL] ; labels.insert(0,'Reference medium')
    fig.legend(lines, labels, loc= (0.6,0.15))

    if save_dir:
      savefile = save_dir + key + '_stats_' + '_{}.png'.format(plot_param['plot_type'])
      fig.savefig(savefile)

    return

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
  def subplot_2ax(figsize=(9,7)):
    fig = plt.figure(figsize=figsize)
    ax1 = fig.add_axes([0.2,0.28,0.6,0.65])
    ax2 = fig.add_axes([0.2,0.075,0.6,0.2])
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



