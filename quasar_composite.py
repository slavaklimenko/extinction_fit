#!/usr/bin/env python


from bisect import bisect_left
import matplotlib.pyplot as plt
from matplotlib import patches
from matplotlib.ticker import AutoMinorLocator, MultipleLocator, FormatStrFormatter
import numpy as np
from pathlib import Path
from scipy import interpolate
import sys


sys.path.append('/home/slava/science/codes/python/spectro/')
sys.path.append('/home/slava/science/codes/python/dust_extinction_model/')
sys.path.append('/home/slava/anaconda3')
sys.path.append('/home/slava/science/codes/python/spectro/sviewer/')
from astropy import constants as const
import os
import pickle
from scipy.interpolate import interp1d
from matplotlib import rcParams
from astropy.io import ascii, fits
rcParams['font.family'] = 'serif'
import emcee
from chainconsumer import ChainConsumer

from astropy.io import fits
from scipy import signal
import scipy.signal
import time, glob
from numpy.polynomial.polynomial import polyval
from scipy.signal import savgol_filter
import csv
from scipy.signal import savgol_filter
from spectrum_model import * #spectrum, rebin_arr,rebin_weight_mean


path_to_composite_spectra = '/home/slava/science/codes/python/dust_extinction_model/data/composite/'



qso_composite_vanderberk = path_to_composite_spectra + 'VandenBerk.txt'
uv_qso_composite = np.loadtxt(qso_composite_vanderberk)
col1 = uv_qso_composite[:, 0]
col2 = uv_qso_composite[:, 1]
col3 = uv_qso_composite[:, 2]
s_uv_qso = spectrum(x=col1,y=col2*col1**2*1e-7,err = col3*col1**2*1e-7)
s_uv_qso.interp = interp1d(s_uv_qso.x, s_uv_qso.y,fill_value='extrapolate')


qso_composite_glikman = path_to_composite_spectra + 'Glikman.dat'
nir_qso_composite = np.loadtxt(qso_composite_glikman)
col1 = nir_qso_composite[:, 0]
col2 = nir_qso_composite[:, 1]
col3 = nir_qso_composite[:, 2]
n = -1
if n>1:
    col1 = rebin_arr(col1, n)
    col2,col3 = rebin_weight_mean(col2,col3,n)
#correction for absorptions:
if 1:
    mask_abs = col1>0
    abs_lines_list = [6311,6374.6406,6439,6543,7367.7434,7559,8301,13742,14673,21685,21866,22019,22117,22323,22100,22411,
                      23802,23927,23825,24007,24524,24604,25177,27286,27740,27636,33976]
    for l in abs_lines_list:
        mask_abs[np.abs(col1 - l) < 15] = False
    f_interp = interp1d(col1[mask_abs],col2[mask_abs],)
    col2_1= np.array(col2)
    col2_1[~mask_abs] = f_interp(col1[~mask_abs])
    if 0:
        plt.subplots()
        plt.errorbar(x=col1,y=col2,yerr=col3)
        plt.errorbar(x=col1,y=col2_1,yerr=col3)
        plt.show()
#correction for high dispersion in the red part
if 1:
    mask = col1>21000
    xx = np.array(col1[mask])
    yy = np.array(col2_1[mask])
    err = np.array(col3[mask])
    n = 20
    xx = rebin_arr(xx,n)
    yy,err = rebin_arr(yy,n), rebin_arr(err,n)
    x = np.append(col1[~mask],xx)
    y = np.append(col2_1[~mask],yy)
    err = np.append(col3[~mask],err)
    s_glikman_qso = spectrum(x=x, y=y * x ** 2 * 1e-7, err=err * x ** 2 * 1e-7)
else:
    s_glikman_qso = spectrum(x=col1,y=col2_1*col1**2*1e-7,err=col3*col1**2*1e-7)
if 0:
    plt.subplots()
    plt.plot(col1,col2_1)
    plt.errorbar(x=x,y=y,yerr=err,fmt='s')
    plt.show()

qso_composite_selsing_path = path_to_composite_spectra + 'Selsing.dat'
f = np.loadtxt(qso_composite_selsing_path)
col1 = f[:, 0]
col2 = f[:, 1]
col3 = f[:, 2]
mask_absorption = col2 == 0
s_selsing_qso = spectrum(x=col1[~mask_absorption], y=col2[~mask_absorption] * col1[~mask_absorption] ** 2 * 1e-7,err=col3[~mask_absorption] * col1[~mask_absorption] ** 2 * 1e-7)
s_selsing_qso.ynu = np.array(col2[~mask_absorption])


qso_composite_jiang = path_to_composite_spectra + 'Jiang2011.dat'
jiang_qso_composite = np.loadtxt(qso_composite_jiang)
col1 = jiang_qso_composite[:, 0]
col2 = jiang_qso_composite[:, 1]
col3 = jiang_qso_composite[:, 2]
if 0:
    plt.subplots()
    plt.errorbar(x=col1, y=col2, yerr=col3)
    plt.show()
s_jiang_qso = spectrum(x=np.array(col1),y=np.array(col2*col1**2*1e-7),err=np.array(col3*col1**2*1e-7))
s_jiang_qso.interp = interp1d(s_jiang_qso.x, s_jiang_qso.y,fill_value='extrapolate')


#renormaliztion
def renorm_qso(x,y,err,lc=8000,dl=200): #3250
    mask = np.abs(x-lc)<dl
    ynorm = np.nanmean(y[mask])
    return y/ynorm,err/ynorm

#fnorm = np.mean(s_uv_qso.y[np.abs(s_uv_qso.x - 3250)<200]) / np.mean(
#    s_selsing_qso.y[np.abs(s_selsing_qso.x - 3250)<200])
s_uv_qso.y,s_uv_qso.err = renorm_qso(s_uv_qso.x,s_uv_qso.y,s_uv_qso.err)
s_uv_qso.interp = interp1d(s_uv_qso.x, s_uv_qso.y, fill_value='extrapolate')

s_jiang_qso.y,s_jiang_qso.err = renorm_qso(s_jiang_qso.x,s_jiang_qso.y,s_jiang_qso.err)
s_jiang_qso.interp = interp1d(s_jiang_qso.x, s_jiang_qso.y, fill_value='extrapolate')

s_selsing_qso.y,s_selsing_qso.err = renorm_qso(s_selsing_qso.x,s_selsing_qso.y,s_selsing_qso.err)
s_selsing_qso.interp = interp1d(s_selsing_qso.x, s_selsing_qso.y, fill_value='extrapolate')

s_glikman_qso.y,s_glikman_qso.err = renorm_qso(s_glikman_qso.x,s_glikman_qso.y,s_glikman_qso.err)
s_glikman_qso.interp = interp1d(s_glikman_qso.x, s_glikman_qso.y, fill_value='extrapolate')


#MIR models
mir_hatzi_qso_composite = np.loadtxt(path_to_composite_spectra + 'J_AJ_129_1198/table2.dat')
#noramlized at 0.3 micron
col1 = mir_hatzi_qso_composite[:, 0]
col2 = mir_hatzi_qso_composite[:, 1]
snr = 50
s_hatzi_qso = spectrum(x=np.array(col1)*1e4,y=np.array(col2*col1**2),err=np.array(col2*col1**2)/snr)
s_hatzi_qso.interp = interp1d(s_hatzi_qso.x, s_hatzi_qso.y,fill_value='extrapolate')
s_hatzi_qso.y,s_hatzi_qso.err = renorm_qso(s_hatzi_qso.x,s_hatzi_qso.y,s_hatzi_qso.err)

mir_hernan_qso_composite = np.loadtxt(path_to_composite_spectra +'Hernan-Caballero-16/table1.dat',skiprows=17)
col1 = mir_hernan_qso_composite[:, 0]
col2 = mir_hernan_qso_composite[:, 1]
snr = 30
s_hernan_qso = spectrum(x=np.array(col1)*1e4,y=np.array(col2),err=np.array(col2)/snr)
s_hernan_qso.interp = interp1d(s_hernan_qso.x, s_hernan_qso.y,fill_value='extrapolate')
s_hernan_qso.y,s_hernan_qso.err = renorm_qso(s_hernan_qso.x,s_hernan_qso.y,s_hernan_qso.err)




#fnorm = np.mean(s_selsing_qso.y[np.abs(s_selsing_qso.x - 11000)<200]) / np.mean(s_nir_qso.y[np.abs(s_nir_qso.x - 11000)<200])
#s_nir_qso.y *= fnorm
#s_nir_qso.interp = interp1d(s_nir_qso.x, s_nir_qso.y, fill_value='extrapolate')



mask_uv = (s_uv_qso.x<=1100)
mask_opt = (s_selsing_qso.x > 1100) * (s_selsing_qso.x <= 11000)
mask_nir = (s_glikman_qso.x > 11000)

x = np.append(s_uv_qso.x[mask_uv],s_selsing_qso.x[mask_opt])
x = np.append(x,s_glikman_qso.x[mask_nir])
y = np.append(s_uv_qso.y[mask_uv],s_selsing_qso.y[mask_opt])
y = np.append(y,s_glikman_qso.y[mask_nir])

s_combined_composite = spectrum(x,y)


#s_combined_composite.x /= 1e4 #convert to microns
class qso_composite():
    def __init__(self, units='angstrom',smooth_ir=False,mir_interp=False,mode='VSG',flux_units='Jy',debug=False,wave_mode = 'normal'):
        # F in Jy (F_nu)
        if mode == 'VSG':
            mask_uv = (s_uv_qso.x <= 1100)
            mask_opt = (s_selsing_qso.x > 1100) * (s_selsing_qso.x <= 10400)
            mask_nir = (s_glikman_qso.x > 10400)

            x,y,err = [],[],[]
            x = np.append(x,s_uv_qso.x[mask_uv])
            y = np.append(y,s_uv_qso.y[mask_uv])
            err = np.append(err,s_uv_qso.err[mask_uv])

            f_norm_opt = np.nanmean(s_uv_qso.y[np.abs(s_uv_qso.x-1100)<100])/np.nanmean(s_selsing_qso.y[np.abs(s_selsing_qso.x-1100)<100])
            x = np.append(x,s_selsing_qso.x[mask_opt])
            y = np.append(y,s_selsing_qso.y[mask_opt]*f_norm_opt)
            err = np.append(err,s_selsing_qso.err[mask_opt]*f_norm_opt)

            f_norm_nir =   np.nanmean(y[np.abs(x - 10300) < 100])/ np.nanmean(s_glikman_qso.y[np.abs(s_glikman_qso.x - 10300) < 100])
            x = np.append(x, s_glikman_qso.x[mask_nir])
            y = np.append(y, s_glikman_qso.y[mask_nir] * f_norm_nir)
            err = np.append(err, s_glikman_qso.err[mask_nir] * f_norm_nir)

            if wave_mode == 'normal':
                self.x = x
                self.y = y
                #errorbars underestimated by a factor of 3-5: by default set it to 2
                self.err = 2*err
            elif wave_mode == 'extended':
                self.x = np.append(x,np.linspace(x[-1],30*1e4,50))
                self.y = np.append(y,np.zeros(50))
                self.err = np.append(y,np.zeros(50))

            self.y,self.err = renorm_qso(self.x, self.y,self.err)

            if mir_interp:
                self.y[self.x > 2e4] = 3.08672 * (self.x[self.x > 2e4] / 19470.561) ** (1.17085)

        if mode == 'VG':
            mask_uv = (s_uv_qso.x <= 3000)
            mask_nir = (s_glikman_qso.x > 3000)

            x, y,err = [], [],[]
            x = np.append(x, s_uv_qso.x[mask_uv])
            y = np.append(y, s_uv_qso.y[mask_uv])
            err = np.append(err, s_uv_qso.err[mask_uv])

            f_norm_nir = np.nanmean(y[np.abs(x - 3000) < 100]) / np.nanmean(s_glikman_qso.y[np.abs(s_glikman_qso.x - 3000) < 100])
            x = np.append(x, s_glikman_qso.x[mask_nir])
            y = np.append(y, s_glikman_qso.y[mask_nir] * f_norm_nir)
            err = np.append(err, s_glikman_qso.err[mask_nir] * f_norm_nir)


            self.x = x
            self.y = y
            self.err = err
            self.y,self.err = renorm_qso(self.x, self.y,self.err)

            if mir_interp:
                self.y[self.x > 2e4] = 3.08672 * (self.x[self.x > 2e4] / 19470.561) ** (1.17085)

        #Vandenberg-Selsing-Glickman-Hernan
        if mode == 'VSGH':
            mask_uv = (s_uv_qso.x <= 1100)
            mask_opt = (s_selsing_qso.x > 1100) * (s_selsing_qso.x <= 10400)
            mask_nir = (s_glikman_qso.x > 10400)*(s_glikman_qso.x < 34000)
            mask_mir = (s_hernan_qso.x > 34000)

            x, y,err = [], [], []
            x = np.append(x, s_uv_qso.x[mask_uv])
            y = np.append(y, s_uv_qso.y[mask_uv])
            err = np.append(err, s_uv_qso.err[mask_uv])

            f_norm_opt = np.nanmean(s_uv_qso.y[np.abs(s_uv_qso.x - 1100) < 100]) / np.nanmean(
                s_selsing_qso.y[np.abs(s_selsing_qso.x - 1100) < 100])
            x = np.append(x, s_selsing_qso.x[mask_opt])
            y = np.append(y, s_selsing_qso.y[mask_opt] * f_norm_opt)
            err = np.append(err, s_selsing_qso.err[mask_opt] * f_norm_opt)

            f_norm_nir = np.nanmean(y[np.abs(x - 10300) < 100]) / np.nanmean(
                s_glikman_qso.y[np.abs(s_glikman_qso.x - 10300) < 100])
            x = np.append(x, s_glikman_qso.x[mask_nir])
            y = np.append(y, s_glikman_qso.y[mask_nir] * f_norm_nir)
            err = np.append(err, s_glikman_qso.err[mask_nir] * f_norm_nir)

            f_norm_mir = np.nanmean(y[np.abs(x - 34000) < 100]) / np.nanmean(s_hernan_qso.y[np.abs(s_hernan_qso.x - 34000) < 100])
            x = np.append(x, s_hernan_qso.x[mask_mir])
            y = np.append(y, s_hernan_qso.y[mask_mir] * f_norm_mir)
            err = np.append(err,  s_hernan_qso.err[mask_mir] * f_norm_mir)

            if wave_mode == 'normal':
                self.x = x
                self.y = y
                self.err = err
            elif wave_mode == 'extended':
                self.x = np.append(x, np.linspace(x[-1], 30 * 1e4, 50))
                self.y = np.append(y, np.zeros(50))
                self.err =  np.append(err, np.zeros(50))
            self.y,self.err = renorm_qso(self.x, self.y,self.err)

            #if mir_interp:
            #    self.y[self.x > 2e4] = 3.08672 * (self.x[self.x > 2e4] / 19470.561) ** (1.17085)

        if mode == 'VSGHa':
            mask_uv = (s_uv_qso.x <= 1100)
            mask_opt = (s_selsing_qso.x > 1100) * (s_selsing_qso.x <= 10400)
            mask_nir = (s_glikman_qso.x > 10400)*(s_glikman_qso.x < 34000)
            mask_mir = (s_hatzi_qso.x > 32000)

            x, y,err = [], [], []
            x = np.append(x, s_uv_qso.x[mask_uv])
            y = np.append(y, s_uv_qso.y[mask_uv])
            err = np.append(err, s_uv_qso.err[mask_uv])

            f_norm_opt = np.nanmean(s_uv_qso.y[np.abs(s_uv_qso.x - 1100) < 100]) / np.nanmean(
                s_selsing_qso.y[np.abs(s_selsing_qso.x - 1100) < 100])
            x = np.append(x, s_selsing_qso.x[mask_opt])
            y = np.append(y, s_selsing_qso.y[mask_opt] * f_norm_opt)
            err = np.append(err, s_selsing_qso.err[mask_opt] * f_norm_opt)

            f_norm_nir = np.nanmean(y[np.abs(x - 10300) < 100]) / np.nanmean(
                s_glikman_qso.y[np.abs(s_glikman_qso.x - 10300) < 100])
            x = np.append(x, s_glikman_qso.x[mask_nir])
            y = np.append(y, s_glikman_qso.y[mask_nir] * f_norm_nir)
            err = np.append(err, s_glikman_qso.err[mask_nir] * f_norm_nir)

            f_norm_mir = np.nanmean(y[np.abs(x - 32000) < 500]) / np.nanmean(s_hatzi_qso.y[np.abs(s_hatzi_qso.x - 32000) < 500])
            x = np.append(x, s_hatzi_qso.x[mask_mir])
            y = np.append(y, s_hatzi_qso.y[mask_mir] * f_norm_mir)
            err = np.append(err,  s_hatzi_qso.err[mask_mir] * f_norm_mir)

            if wave_mode == 'normal':
                self.x = x
                self.y = y
                self.err = err
            elif wave_mode == 'extended':
                self.x = np.append(x, np.linspace(x[-1], 30 * 1e4, 50))
                self.y = np.append(y, np.zeros(50))
                self.err =  np.append(err, np.zeros(50))
            self.y,self.err = renorm_qso(self.x, self.y,self.err)

            #if mir_interp:
            #    self.y[self.x > 2e4] = 3.08672 * (self.x[self.x > 2e4] / 19470.561) ** (1.17085)

        elif mode == 'JG':
            mask_opt = (s_jiang_qso.x <= 8000)
            mask_nir = (s_glikman_qso.x > 8000)

            x, y,err = [], [],[]
            x = np.append(x, s_jiang_qso.x[mask_opt])
            y = np.append(y, s_jiang_qso.y[mask_opt])
            err = np.append(err, s_jiang_qso.err[mask_opt])

            f_norm_nir = np.nanmean(y[np.abs(x - 7900) < 100]) / np.nanmean(
                s_glikman_qso.y[np.abs(s_glikman_qso.x - 7900) < 100])
            x = np.append(x, s_glikman_qso.x[mask_nir])
            y = np.append(y, s_glikman_qso.y[mask_nir] * f_norm_nir)
            err = np.append(err, s_glikman_qso.err[mask_nir] * f_norm_nir)

            self.x = x
            self.y = y
            self.err = err

            self.y,self.err = renorm_qso(self.x, self.y,self.err)

            if mir_interp:
                self.y[self.x > 2e4] = 4.852695841834076 * (self.x[self.x > 2e4] / 20051.60) ** (1.170858849426)


        if flux_units == 'Jy':
            print()
        elif flux_units == 'F_lambda':
            self.y,self.err = renorm_qso(self.x, self.y/self.x**2,self.err//self.x**2)


        if smooth_ir:
            self.y[self.x <  11000] = savgol_filter(self.y[self.x <  11000], 50, 3)

        self.units = units
        if units == 'micron':
            self.x /= 1e4
        self.interp()

        if debug:
            self.cont_plot()

    def interp(self):
        self.interp = interp1d(self.x,self.y)

    def cont_plot(self):
        fig, ax = plt.subplots()
        ax.plot(self.x,self.y,'o',label='composite',zorder=10)
        ax.plot(s_uv_qso.x ,s_uv_qso.y,label='UV')
        ax.plot( s_glikman_qso.x, s_glikman_qso.y,label='Glikman')
        plt.legend()
        plt.show()

if __name__ == '__main__':
    fig, ax = plt.subplots()
    ax.plot(s_selsing_qso.x, s_selsing_qso.y, label='Selsing')
    ax.plot(s_glikman_qso.x, s_glikman_qso.y, label='Glikman')
    ax.plot(s_uv_qso.x, s_uv_qso.y, label='Vanderberk',ls='--')
    ax.plot(s_jiang_qso.x, s_jiang_qso.y, label='Jiang')
    ax.plot(s_hatzi_qso.x,s_hatzi_qso.y,label='Hatzi')
    ax.plot(s_hernan_qso.x, s_hernan_qso.y, label='Hernan')

    q_comp = qso_composite(mir_interp=False,mode='VSGHa',flux_units = 'Jy', debug=False)
    ax.errorbar(x=q_comp.x, y=q_comp.y, yerr=q_comp.err,fmt='s', label='Composite', zorder=-100,color='blue',alpha=0.2)
    ax.step(q_comp.x, q_comp.y,where='mid',color='blue')
    ax.set_xscale('log')
    plt.legend()
    plt.show()

    if 1:
        d = np.zeros((np.size(q_comp.x),3))
        d[:,0] = q_comp.x
        d[:, 1] = q_comp.y
        d[:, 2] = q_comp.err
        np.savetxt('composite_spectrum.dat',d)
        plt.show()





    if 0:
        from astropy import modeling


        fitter = modeling.fitting.LevMarLSQFitter()
        #fitter = modeling.fitting.PowerLaw1D()
        #astropy.modeling.powerlaws.PowerLaw1D
        model = modeling.powerlaws.PowerLaw1D(x_0=20000)
        mask = (q_comp.x>16500)*(np.abs(q_comp.x-18700)>300)*(np.abs(q_comp.x-23000)>2500)
        y =  q_comp.y[mask]
        x=q_comp.x[mask]
        fitted_model = fitter(model, x, y)
        amp = fitted_model.amplitude.value
        x_0 = fitted_model.x_0.value
        alpha = fitted_model.alpha.value
        print(amp,x_0,alpha)

        plt.subplots()
        plt.plot(q_comp.x,q_comp.y,label='data')
        plt.plot(q_comp.x[mask],fitted_model(q_comp.x[mask]),label='fit')
        plt.legend()
        plt.show()


        q_comp = qso_composite(mir_interp=True,flux_units='F_lambda')
        plt.subplots()
        plt.plot(q_comp.x,q_comp.y)

        d = np.zeros((np.size(q_comp.x),2))
        d[:,0] = q_comp.x
        d[:, 1] = q_comp.y
        np.savetxt('composite_qso.dat',d)
        plt.show()

