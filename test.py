#!/usr/bin/env python3

from sem2d import sem2dpack
import ipdb as db
import matplotlib.pyplot as plt

directory = '/Users/flomin/Desktop/thesis/test/sem/'
#directory2 = '/Users/flomin/Desktop/thesis/test/sem2/'
savedir = '/Users/flomin/Desktop/thesis/report/year_1/presenation/phosphore_octobre/hete.flv'
#savedir2 = '/Users/flomin/Desktop/thesis/report/year_1/presenation/phosphore_octobre/homo.flv'
ob = sem2dpack(directory)
#ob2 = sem2dpack(directory2)
#ob.animate(save=True,savefile=savedir)
#ob2.animate(save=True,savefile=savedir2)
ob.plot_snapshot('vx_001_sem2d.dat')
db.set_trace()



