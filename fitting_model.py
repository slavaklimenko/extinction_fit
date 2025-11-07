# define the fitting model




from bisect import bisect_left
import matplotlib.pyplot as plt
import copy


plt.close('all')
from matplotlib import patches
from matplotlib.ticker import AutoMinorLocator, MultipleLocator, FormatStrFormatter
import numpy as np
from pathlib import Path
from scipy import interpolate
import sys

from statsmodels.sandbox.tsa.try_arma_more import rescale

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
import csv
from scipy.signal import savgol_filter

from spectrum_model import *
from gordon_extinction import *
from spec_summary import QSO_list
from spec_summary import QSO_list
import copy


class parameters():
    def __init__(self, name=None,val=None, errp=None, errm=None, var = False,vrange=[] ,disp=None,prior=0):
        if name is not None:
            self.name = name
        if val is not None:
            self.val = val
        if errp is not None:
            self.errp = errp
        if errm is not None:
            self.errm = errm
        if disp is not None:
            self.disp =disp
        if prior is not None:
            self.prior = prior
        self.var = var
        self.vrange = vrange
    def set_prior(self,cen,p,m):
        self.prior = 1
        self.cen = cen
        self.p =p
        self.m = m


'''
f0 - model flux scaling coefficient
c1,c2,c3,x0,gamma - FM parameters
b,alpha - the normalization and slope of MIR extinction
fsdss,fnir,fmiri - scailing coefficients
fukirtHK - scailing coefficients for UKIRT HK, which are ususally observed separately from YJ bands
'''
pars = {}
pars['f0'] = parameters('f0', 1, var=1, disp=0.1, vrange=[0.5, 2])
pars['fvar'] = parameters('fvar', 0, var=1, disp=0.1, vrange=[-0.3, 0.3])
pars['Av'] = parameters('Av', 1, var=0, disp=0.2, vrange=[0, 10])
pars['Rv'] = parameters('Rv', 3.1, var=0, disp=0.2, vrange=[2.5, 5.5])
pars['c1'] = parameters('c1', 0.5, var=1, disp=0.5, vrange=[-5, 10])
pars['c2'] = parameters('c2', 0.2, var=1, disp=0.2, vrange=[0, 10])
pars['c3'] = parameters('c3', 1.0, var=1, disp=0.5, vrange=[0, 10])
pars['c4'] = parameters('c4', 0.0, var=0, disp=0.05, vrange=[0, 0.3])
pars['x0'] = parameters('x0', 4.60, var=1, disp=0.1, vrange=[4.4, 4.8])
pars['E0'] = parameters('E0', -0.35, var=1, disp=0.1, vrange=[-2, 2])
pars['E3'] = parameters('E3', -0.05, var=0, disp=0.1, vrange=[-1, 1])
pars['E4'] = parameters('E4', 0.006, var=0, disp=0.1, vrange=[-0.5, 0.5])
pars['gamma'] = parameters('gamma', 1.0, var=0, disp=0.2, vrange=[0.6, 1.5])
pars['b'] = parameters('b', 0.4, var=1, disp=0.1, vrange=[0.01, np.inf])
pars['alpha'] = parameters('alpha', 1.5, var=1, disp=0.3, vrange=[0.2, 5])
pars['fsdss'] = parameters('fsdss', 1, var=1, disp=0.1, vrange=[0.5, 1.5])
pars['fnir'] = parameters('fnir', 1, var=1, disp=0.1, vrange=[0.5, 1.5])
pars['fukirtHK'] = parameters('fukirtHK', 1, var=0, disp=0.1, vrange=[0.5, 1.5])
pars['fmiri'] = parameters('fmiri', 1, var=0, disp=0.1, vrange=[0.5, 1.5])
pars['fnuv'] = parameters('fnuv', 1, var=0, disp=0.1, vrange=[0.5, 1.5])

if 0:
    pars['fnir'].var = 0
    pars['E0'].var = 0
    pars['E0'].val = 0
    pars['E3'].var = 0
    pars['E3'].val = 0
    pars['E4'].var = 0
    pars['E4'].val = 0

theta = [el.val for el in pars.values() if el.var]


def init_pars(pars):
    pars['f0'] = parameters('f0', 1, var=1, disp=0.1, vrange=[0.5, 2])
    pars['fvar'] = parameters('fvar', 0, var=1, disp=0.1, vrange=[-0.3, 0.3])
    pars['Av'] = parameters('Av', 1, var=0, disp=0.2, vrange=[0, 10])
    pars['Rv'] = parameters('Rv', 3.1, var=0, disp=0.2, vrange=[2.5, 5.5])
    pars['c1'] = parameters('c1', 0.5, var=1, disp=0.5, vrange=[-5, 10])
    pars['c2'] = parameters('c2', 0.2, var=1, disp=0.2, vrange=[0, 10])
    pars['c3'] = parameters('c3', 1.0, var=1, disp=0.5, vrange=[0, 10])
    pars['c4'] = parameters('c4', 0.2, var=1, disp=0.05, vrange=[0, 0.3])
    pars['x0'] = parameters('x0', 4.60, var=1, disp=0.1, vrange=[4.4, 4.8])
    pars['E0'] = parameters('E0', -0.35, var=1, disp=0.1, vrange=[-2, 2])
    pars['E3'] = parameters('E3', -0.05, var=0, disp=0.1, vrange=[-1, 1])
    pars['E4'] = parameters('E4', 0.006, var=0, disp=0.1, vrange=[-0.5, 0.5])
    pars['gamma'] = parameters('gamma', 1.0, var=0, disp=0.2, vrange=[0.6, 1.5])
    pars['b'] = parameters('b', 0.4, var=1, disp=0.1, vrange=[0.01, np.inf])
    pars['alpha'] = parameters('alpha', 1.5, var=1, disp=0.3, vrange=[0.2, 5])
    pars['fsdss'] = parameters('fsdss', 1, var=1, disp=0.1, vrange=[0.5, 1.5])
    pars['fnir'] = parameters('fnir', 1, var=1, disp=0.1, vrange=[0.5, 1.5])
    pars['fukirtHK'] = parameters('fukirtHK', 1, var=0, disp=0.1, vrange=[0.5, 1.5])
    pars['fmiri'] = parameters('fmiri', 1, var=0, disp=0.1, vrange=[0.5, 1.5])
    pars['fnuv'] = parameters('fnuv', 1, var=0, disp=0.1, vrange=[0.5, 1.5])
    return pars

def init_MW_template(pars):
    pars['f0'] = parameters('f0', 1, var=1, disp=0.1, vrange=[0.5, 2])
    pars['fvar'] = parameters('fvar', 0, var=1, disp=0.1, vrange=[-0.3, 0.3])
    pars['Av'] = parameters('Av', 1, var=1, disp=0.2, vrange=[0, 10])
    pars['Rv'] = parameters('Rv', 3.1, var=1, disp=0.2, vrange=[2.5, 5.5])
    pars['c1'] = parameters('c1', 0.812, var=0, disp=0.5, vrange=[-5, 10])
    pars['c2'] = parameters('c2', 0.277, var=0, disp=0.2, vrange=[0, 10])
    pars['c3'] = parameters('c3', 1.0629, var=0, disp=0.5, vrange=[0, 10])
    pars['c4'] = parameters('c4', 0.113, var=0, disp=0.05, vrange=[0, 0.3])
    pars['x0'] = parameters('x0', 4.60, var=0, disp=0.1, vrange=[4.4, 4.8])
    pars['E0'] = parameters('E0', -0.358, var=0, disp=0.1, vrange=[-2, 2])
    pars['E3'] = parameters('E3', -0.054, var=0, disp=0.1, vrange=[-1, 1])
    pars['E4'] = parameters('E4', 0.0067, var=0, disp=0.1, vrange=[-0.5, 0.5])
    pars['gamma'] = parameters('gamma', 0.99, var=0, disp=0.2, vrange=[0.6, 1.5])
    pars['b'] = parameters('b', 0.385, var=0, disp=0.1, vrange=[0.01, np.inf])
    pars['alpha'] = parameters('alpha', 1.68, var=0, disp=0.3, vrange=[0.2, 5])
    pars['fsdss'] = parameters('fsdss', 1, var=0, disp=0.1, vrange=[0.5, 1.5])
    pars['fnir'] = parameters('fnir', 1, var=0, disp=0.1, vrange=[0.5, 1.5])
    pars['fukirtHK'] = parameters('fukirtHK', 0, var=0, disp=0.1, vrange=[0.5, 1.5])
    pars['fmiri'] = parameters('fmiri', 1, var=0, disp=0.1, vrange=[0.5, 1.5])
    pars['fnuv'] = parameters('fnuv', 1, var=0, disp=0.1, vrange=[0.5, 1.5])
    return pars


# create the quasar continuum
from quasar_composite import qso_composite

# function for qso composite construction
def calc_composite_cont(s=qso_composite(units='micron',mode='VSG',flux_units='Jy'), debug=False,  f_units='F_lam'):
    '''
    mode - set the continuum model:
    JG  = Jiang+Glikman
    VSG = Vanderberk+Selsing+Glikman
    VG =   Vanderberk+Glikman
    '''
    xx = np.array(s.x)
    fnorm = 1
    if f_units == 'F_lam':
        fnorm = (xx * 1e4) ** 2 / 3e18 * 1e-17 / 1e-23  # in 1e-17 erg/s/cm2/A
    if debug:
        plt.subplots()
        plt.plot(s.x, s.y, label='cont')
        plt.plot(s.x, s.err, label='err')
        plt.legend()
        plt.show()
    return spectrum(xx, np.array(s.y) / fnorm, err=np.array(s.err) / fnorm)

# create the composite
s_composite = calc_composite_cont()



    # a function for calculation the extinction curve
def extinction_fit(l=[1], c1=1, c2=1, c3=0, c4=0, x0=4.6, gamma=1,  E0=1, E3=0,E4=0, b=1, alpha=1,debug=False,mode='MW'):
    l = np.array(l) # wavelength array
    # wavelength ref points to join uv, optical, ir branches of extinction curve
    x1 = 0.3 #micron
    x2 = 1.0 #micron
    # arrays for uv, optical, ir parts
    a_uv, a_opt, a_mir = np.zeros_like(l), np.zeros_like(l), np.zeros_like(l)
    # uv part
    a_uv[l <= x1] = FM_ext(xx=l[l <= x1], c1=c1, c2=c2, c3=c3, c4=c4, x0=x0, g=gamma)
    # ir part
    a_mir[l >= x2] = b * np.power(l[l >= x2], -alpha)
    # optical part - E0 + E1/l + E2/l**2
    fm1 = FM_ext(xx=[x1], c1=c1, c2=c2, c3=c3, c4=c4, x0=x0, g=gamma)
    mir2 =  b * np.power(x2, -alpha)
    #define E2 and E1 throug fm1, mir2 and E0, E3,E4
    if mode=='MW':
        opt_ext_part1 = fm1 - E0 -E3/x1**3 - E4/x1**4
        opt_ext_part2 = mir2 - E0 -E3/x2**3 - E4/x2**4
        E2 = (opt_ext_part1*x1 - opt_ext_part2*x2)/(x2-x1)*x1*x2
        E1 = opt_ext_part2*x2 - E2/x2
    elif mode == '1-order':
        E1 = x1*x2*(fm1-mir2)/(x2-x1)
        E0 = fm1 - E1/x1
        E2= 0
        E3 = 0
        E4 = 0
    #elif mode=='MW':
    #    E2 = 0.087 #(x1 ** 2 * x2 * (fm1 - E0) - x1 * x2 ** 2 * (mir2 - E0)) / (x2 - x1)
    #    E1 = 0.7122 #x1 * (fm1 - E0) - E2 / x1
    elif mode == '2-order':
        E2 = (x1**2 * x2 * (fm1 - E0) -  x1*x2**2*(mir2-E0))/(x2 - x1)
        E1 = x1*(fm1 - E0) -  E2 / x1
        E3 = 0
        E4 = 0
    # define optical extinction
    a_opt[(l > x1) * (l <= x2)] = (E0 + E1 * np.power(l[(l > x1) * (l < x2)], -1)
                + E2 * np.power(l[(l > x1) * (l < x2)], -2)+ E3 * np.power(l[(l > x1) * (l < x2)], -3)
                                   + E4 * np.power(l[(l > x1) * (l < x2)], -4)
                                   )
    #
    f = (a_uv + a_mir + a_opt)

    #
    if debug == True:
        plt.subplots()
        plt.plot(l, f,color='black',ls='--',lw=4)
        plt.plot(l, a_uv)
        plt.plot(l, a_mir)
        plt.plot(l, a_opt)
        #plt.xscale('log')
        #plt.yscale('log')
        plt.plot(l,extinction_gordon23(l=l),color='blue',ls='--')
        plt.show()
    return f

# a function acoounting for a variability power law component
def calc_quasar_variability(s_composite=s_composite, variability=0):
    # calcualte total flux in "UV-opt" range (0.13,0.6 micron):
    alpha_vary = -2
    lmin, lmax = 0.13, 0.6

    mask_vary = (s_composite.x > lmin) * (s_composite.x < lmax)

    Fc_tot = np.trapz(s_composite.y[mask_vary], s_composite.x[mask_vary])
    Fadd_tot = np.trapz(np.power(s_composite.x, alpha_vary)[mask_vary], s_composite.x[mask_vary])

    y_vary = variability * Fc_tot / Fadd_tot * np.power(s_composite.x, alpha_vary)
    y_vary[s_composite.x < 0.09] = 0
    return y_vary

def derive_ext_parameters(chain,pars):
    # calculate parameters distribution in Av-Abump parameter space
    chain_lenght = chain.shape[0]
    chain_tmp = np.zeros((chain_lenght, 11))

    for i in range(chain_lenght):
        theta = chain[i, :]
        for el, val in zip([p for p in pars.values() if p.var], theta):
            el.val = val

        x = np.power(10,np.linspace(-1,1,100))
        y = pars['Av'].val*extinction_fit(l=x, c1=pars['c1'].val,
                                          c2=pars['c2'].val, c3=pars['c3'].val, c4=pars['c4'].val,
                                          x0=pars['x0'].val, gamma=pars['gamma'].val,
                                          E0=pars['E0'].val, E3=pars['E3'].val, E4=pars['E4'].val,
                                          b=pars['b'].val, alpha=pars['alpha'].val)
        y_savgoal = np.array(y)
        #mask = np.abs(x-1)<0.3
        #_savgoal[mask] = savgol_filter(y, 20, 3)[mask]
        #plt.subplots()
        #plt.plot(x,y)
        #plt.plot(x, y_savgoal,label='savgoal')
        #lt.legend()
        #plt.show()

        f_interp = interp1d(x,y_savgoal)

        if 0:
            phot_x = np.linspace(0.5, 0.6, 50)
            phot_y = np.trapz(extinction_fit(l=phot_x, c1=pars['c1'].val,
                                             c2=pars['c2'].val, c3=pars['c3'].val, c4=pars['c4'].val,
                                             x0=pars['x0'].val, gamma=pars['gamma'].val,
                                             E0=pars['E0'].val, E3=pars['E3'].val, E4=pars['E4'].val,
                                             b=pars['b'].val, alpha=pars['alpha'].val), phot_x)
            ext_model_av = phot_y / 0.1

            phot_x = np.linspace(0.4, 0.5, 50)
            phot_y = np.trapz(extinction_fit(l=phot_x, c1=pars['c1'].val,
                                             c2=pars['c2'].val, c3=pars['c3'].val, c4=pars['c4'].val,
                                             x0=pars['x0'].val, gamma=pars['gamma'].val,
                                             E0=pars['E0'].val, E3=pars['E3'].val, E4=pars['E4'].val,
                                             b=pars['b'].val, alpha=pars['alpha'].val), phot_x)
            ext_model_ab = phot_y / 0.1
        else:
            ext_model_av = f_interp(0.55)
            ext_model_ab = f_interp(0.45)
        chain_tmp[i, 0] = ext_model_av
        chain_tmp[i, 1] = pars['c3'].val * np.pi / 2 / pars['gamma'].val
        chain_tmp[i, 2] = pars['c3'].val / pars['gamma'].val ** 2

        chain_tmp[i, 3] = ext_model_av / (ext_model_ab - ext_model_av)
        chain_tmp[i, 4] = f_interp(0.3) / ext_model_av
        # B,I,J,H,K
        chain_tmp[i, 5] = ext_model_ab / ext_model_av
        chain_tmp[i, 6] = f_interp(0.8) / ext_model_av
        chain_tmp[i, 7] = f_interp(1.25) / ext_model_av
        chain_tmp[i, 8] = f_interp(1.65) / ext_model_av
        chain_tmp[i, 9] = f_interp(2.2) / ext_model_av
        chain_tmp[i, 10] = f_interp(0.365) / ext_model_av

    par_names2 = ['Av', 'Surf2175', 'Size2175', 'RV', '0.3/V', 'B/V', 'I/V', 'J/V', 'H/V', 'K/V','U/V']
    c2 = ChainConsumer()
    c2.add_chain(chain_tmp, parameters=par_names2)
    c2.plotter.plot( figsize="column")
    res2 = c2.analysis.get_summary(parameters=par_names2)
    print('res2', res2)
    return res2,par_names2


if __name__ == '__main__':
    # create the composite
    s_composite = calc_composite_cont()
    pars = init_MW_template(pars)
    x = np.linspace(0.1,5,1000)
    ext_model = extinction_fit(l=x, c1=pars['c1'].val,
                               c2=pars['c2'].val, c3=pars['c3'].val*0, c4=pars['c4'].val,
                               x0=pars['x0'].val, gamma=pars['gamma'].val,
                               E0=pars['E0'].val, E3=pars['E3'].val, E4=pars['E4'].val,
                               b=pars['b'].val, alpha=pars['alpha'].val, debug=True)
    plt.show()

