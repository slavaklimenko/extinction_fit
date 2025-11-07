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


class photometry():
    def __init__(self, l=None, nu=None, f=None,name='',err=None,lmin=None,lmax=None,mag=None,magerr=0.1,
                 transmission_filename=None):
        self.name = name
        if l is not None:
            self.l=l
        if nu is not None:
            self.nu=nu
            self.calc_lambda()
        if err is not None:
            self.err=err
        if lmin is not None:
            self.lmin=lmin
            self.lmax=lmax
        else:
            self.lmin = self.l
            self.lmax = self.l
        if mag is not None:
            self.mag = mag
            self.mag_err= magerr
            self.calc_flux()
        if f is not None:
            self.fnu=f
            self.flam = f* 3e18 / (self.l*1e4) ** 2 * 1e-23
            self.flam_err = err* 3e18 / (self.l*1e4) ** 2 * 1e-23
            #self.f_nu = self.f /((self.l * 1e4) ** 2 / 3e18 * 1e-17 / 1e-23)
        self.read_transmission_profile(transmission_filename)

    def calc_lambda(self):
        if hasattr(self,'nu'):
            self.l = 3e10/self.nu*1e4 #in micron
    def read_transmission_profile(self,transmission_filename=None):
        if transmission_filename != None:
            d = np.loadtxt(transmission_filename)
            self.transmission = interp1d(d[:,0],d[:,1]/np.nanmax(d[:,1]),fill_value=0, bounds_error=False)
            self.l
    def weight_function(self,xgrid = [1,2],flux=None,debug=False):
        x = np.array(xgrid)*1e4
        y = self.transmission(x)
        x_range = np.trapz(y,x)
        w = np.sum(y)
        if flux is not None:
            fit_value = np.trapz(y*flux,x)/x_range
            if debug:
                plt.subplots()
                plt.plot(x,flux,label='flux')
                plt.plot(x,y,label=self.name)
                plt.axhline(fit_value,color='purple',ls='--',lw=4,label='integrated flux')
                plt.legend()
                plt.show()
            #print(np.size(x))
            return w,fit_value
        else:
            return w

    def calc_flux(self):
        if self.name == '2MASSJ':
            self.zero_flux = 1594  # Jy
        elif self.name == '2MASSH':
            self.zero_flux = 1024  # Jy
        elif self.name == '2MASSK':
            self.zero_flux = 666.7  # Jy
        elif self.name == 'WISE1':
            self.zero_flux = 309.54  # Jy
        elif self.name == 'WISE2':
            self.zero_flux = 171.78  # Jy
        elif self.name == 'WISE3':
            self.zero_flux = 31.674  # Jy
        elif self.name == 'WISE4':
            self.zero_flux = 8.363  # Jy
        else:
            self.zero_flux = 0

        self.l = (self.lmax+self.lmin)/2 #in micron
        self.fnu = self.zero_flux * 10 ** (-0.4 * self.mag)  # in Vega magnitudes
        self.flam = (self.zero_flux * 10 ** (-0.4 * self.mag))* 3e18 / (self.l*1e4) ** 2 * 1e-23  # in 1e-17 erg/s/cm2/A
        if 'SDSS' in self.name:
            offset = 0
            if self.name == 'SDSSg':
                offset = -0.04
            if self.name == 'SDSSz':
                offset = 0.02
            self.flam = 10 ** (-0.4 * (self.mag + 48.6 + offset)) / 1e-23 * 3e18 / (self.l*1e4) ** 2 * 1e-23 # in 1e-17 erg/s/cm2/A
        self.flam_err = (self.zero_flux * 10 ** (-0.4 * (self.mag-self.mag_err))* 3e18 / (self.l*1e4) ** 2 * 1e-23 -
                         self.flam) # in 1e-17 erg/s/cm2/A

        print(self.name, self.mag,self.fnu, self.flam,self.flam_err)

#define classes and functions
#
class spectrum():
    def __init__(self, x=None, y=None, err=None,name=None):
        if any([v is not None for v in [x, y, err]]):
            self.set_data(x=x, y=y, err=err,name=name)

    def set_data(self, x=None, y=None, err=None,name=None):
        if x is not None:
            self.x = np.asarray(x)
        if y is not None:
            self.y = np.asarray(y)
        if y is not None:
            self.y = np.asarray(y)
        if err is not None:
            self.err = np.asarray(err)
        if name is not None:
            self.name = name

    def normalize(self,x0=12,delta_x = 0.1):
        if x0>self.x[0] and x0<self.x[-1]:
            mask = (self.x >= x0-delta_x)*(self.x <= x0+delta_x)
            norm_f = np.mean(self.y[mask])
            self.y /=norm_f
            self.err /=norm_f

    def append(self,s,mode = 'mean disp',debug=False):
        if not hasattr(self, 'x'):
            if hasattr(s,'x'):
                self.x = s.x.copy()
                self.y = s.y.copy()
                self.err = s.err.copy()
        else:
            if np.sum(s.x)>0:
                mask_intersection = s.x <= self.x[-1]
                mask_extension = ~mask_intersection
                if debug:
                    import matplotlib.pyplot as plt
                    fig, ax = plt.subplots(1, 2)
                    ax[0].plot(s.x,s.y/s.err,label='s')
                    ax[0].plot(self.x,self.y/self.err,label='self')
                    f = np.linspace(0.1,10,100)
                    ax[1].plot(f,(f+5)*5/(25 + f**2))
                    ax[0].legend()
                    plt.show()


                if np.sum(mask_intersection) > 0:
                    s_interp = interp1d(s.x, s.y, bounds_error=False, fill_value=np.NaN)
                    s_interp_err = interp1d(s.x, s.err, bounds_error=False, fill_value=np.NaN)
                    mask_selfx_intersection = self.x>=s.x[0]
                    comb = [self.y[mask_selfx_intersection],s_interp(self.x[mask_selfx_intersection])]
                    e_comb = [self.err[mask_selfx_intersection],s_interp_err(self.x[mask_selfx_intersection])]
                    if mode == 'mean weighted':
                        w = np.power(e_comb,-2)
                        f2 = np.nansum(comb * w, axis=0) / np.nansum(w, axis=0)
                        self.y[mask_selfx_intersection] = f2
                        self.err[mask_selfx_intersection] = np.power(np.nansum(w, axis=0), -0.5)
                    elif mode == 'mean':
                        self.y[mask_selfx_intersection] = np.nanmean(comb)
                        self.err[mask_selfx_intersection] = np.power(np.nansum(np.power(e_comb, -2), axis=0), -0.5)
                    elif mode == 'mean disp':
                        self.y[mask_selfx_intersection] = np.nansum(comb,axis=0)/2
                        f = comb-self.y[mask_selfx_intersection]
                        f1 = np.power(f,2)
                        self.err[mask_selfx_intersection] = np.power(np.nansum(f1,axis=0)/2, 0.5)

                self.x = np.append(self.x,s.x[mask_extension])
                self.y = np.append(self.y,s.y[mask_extension])
                self.err = np.append(self.err,s.err[mask_extension])


    def copy(self):
        return spectrum(self.x,self.y,self.err)

class em_line():
    def __init__(self,name='',l=1,w=0):
        self.l = l
        self.w = w
        self.name = name



def rebin_arr(a, factor):
    '''
        How to rebin 1d array:
        :params: a - array
                 factor - binning factor
        :return: rebinned array by a factor = input factor
        '''
    n = a.shape[0] // factor
    return a[:n*factor].reshape(a.shape[0] // factor, factor).sum(1)/factor

def rebin_weight_mean(y, err,factor):
    '''
        How to rebin 1d spectrum using weighted mean method:
        :params: y,err - arrays flux and flux uncertinties
                 factor - binning factor
        :return: 2 rebinned array for flux and flux uncertainty by a factor = input factor
    '''

    w = np.array(np.power(err,-2))
    a = np.array(y)
    n = a.shape[0] // factor
    a*=w
    a = a[:n*factor].reshape(n, factor)
    w = w[:n * factor].reshape(n, factor)
    a = a.sum(1)
    w = w.sum(1)
    return a/w,np.power(w,-0.5)


if __name__ == '__main__':

    print('')