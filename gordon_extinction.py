#!/usr/bin/env python


from bisect import bisect_left
import matplotlib.pyplot as plt
from matplotlib import patches
from matplotlib.pyplot import subplots
from matplotlib.ticker import AutoMinorLocator, MultipleLocator, FormatStrFormatter
import numpy as np
from pathlib import Path
from scipy import interpolate
import sys
sys.path.append('/home/slava/science/codes/python/spectro/')
sys.path.append('/home/slava/anaconda3')
sys.path.append('/home/slava/science/codes/python/spectro/sviewer/')
sys.path.append('/home/slava/science/codes/python/dust_extinction_model/')
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
from spectrum_model import * #spectrum, rebin_arr,rebin_weight_mean

def drude(x=1,x0=4.6,gamma=1.5):
    f = x**2/((x**2-x0**2)**2 + gamma**2*x**2)
    return f
def drude_modified(l=20,l0=10,g0=1,a=1):
    g = 2*g0/(1+np.exp(a*(l-l0)))
    f = (g/l0)**2/((l/l0-l0/l)**2 + g**2/l0**2)
    return f
def f_farUV(x=0):
    # fit from Gordon et al. 2009
    mask = x > 5.9
    y = np.zeros_like(x)
    if np.sum(mask) > 0:
        y[mask] = 0.5392 * (x[mask] - 5.9) ** 2 + 0.05644 * (x[mask] - 5.9) ** 3
    return y

def FM_ext(xx,c1=-0.5,c2=0.7,c3=3.5,x0=4.6,g=1.5,c4=0):
    x = 1/np.array(xx)
    d = drude(x=x,x0=x0,gamma=g)
    return c1 + c2 * x + c3 * d + c4*f_farUV(x)

def MIR_ext(l=1,B=1,alpha=1,S1=0,l0=10,g0=1,a=1,S2=0,l02=20,g02=13,a2=-0.3):
    d = drude_modified(l=l,l0=l0,g0=g0,a=a)
    d2 = drude_modified(l=l,l0=l02,g0=g02,a=a2)
    return B*np.power(l,alpha)+S1*d + S2*d2


#uniform curve as a fucntion on Rv from Gordon+2023
a_coeff = {}
b_coeff = {}
#UV part
a_coeff['C1AV'] = 0.81297
a_coeff['C2AV'] = 0.2775
a_coeff['C3AV'] = 1.06295
a_coeff['C4AV'] = 0.11303
a_coeff['x0'] = 4.60
a_coeff['gamma'] = 0.99
#
b_coeff['C1AV'] = -2.97868
b_coeff['C2AV'] = 1.89808
b_coeff['C3AV'] = 3.10334
b_coeff['C4AV'] = 0.65484
b_coeff['x0'] = 4.60
b_coeff['gamma'] = 0.99
#optical part
def optical_extinction_curve(l=1,E=[0,1,2,3,4],F=[0,1,2],xi=[2.288,2.054,1.587],gi=[0.243,0.179,0.243]):
    xx= 1/l # l in microns
    E = np.array(E)
    F = np.array(F)
    xi = np.array(xi)
    gi = np.array(gi)
    func = np.zeros_like(xx)
    for j in range(5):
        func+=E[j]*np.power(xx,j)
    for j in range(3):
        func+=F[j]*drude(x=xx,x0=xi[j],gamma=gi[j])*gi[j]**2
    return func
#
a_coeff['E0'] = -0.35848
a_coeff['E1'] =  0.7122
a_coeff['E2'] =  0.08746
a_coeff['E3'] = -0.05403
a_coeff['E4'] =  0.00674
#
a_coeff['F1'] =  0.03893
a_coeff['F2'] =  0.02965
a_coeff['F3'] =  0.01747
#
b_coeff['E0'] = 0.12354
b_coeff['E1'] = -2.68335
b_coeff['E2'] = 2.01901
b_coeff['E3'] = -0.39299
b_coeff['E4'] = 0.03355
b_coeff['F1'] = 0.18453
b_coeff['F2'] = 0.19728
b_coeff['F3'] = 0.1713
#
#NIR
def z_function(l,lb,delta):
    return (l - (lb - 0.5*delta)) / delta

def W_function(l,lb,delta):
    f = np.zeros_like(l)
    z = z_function(l,lb,delta)
    mask_z = (z>=0)*(z<=1)
    f[mask_z] += 3*z[mask_z]**2 - 2*z[mask_z]**3
    return f
#
a_coeff['g1'] = 0.38526
a_coeff['alpha1'] = 1.68467
a_coeff['alpha2'] = 0.78791
a_coeff['lb'] = 4.30578
a_coeff['delta'] = 4.78338
a_coeff['S1']  = 0.06652
a_coeff['l01'] = 9.8434
a_coeff['g01'] = 2.21205
a_coeff['a1']  = -0.24703
a_coeff['S2']  = 0.0267
a_coeff['l02'] = 19.258294
a_coeff['g02'] = 17
a_coeff['a2']  = -0.27
#
b_coeff['g1'] = -1.01251
b_coeff['alpha1'] = -1.06099

def IR_extinction(l,g1=a_coeff['g1'],alpha1=a_coeff['alpha1'],alpha2=a_coeff['alpha2'],lb=a_coeff['lb'],delta=a_coeff['delta'],S1=a_coeff['S1'],l01=a_coeff['l01'],
                  g01=a_coeff['g01'],a1=a_coeff['a1'],S2=a_coeff['S2'],l02=a_coeff['l02'],g02=a_coeff['g02'],a2=a_coeff['a2']):
    f = np.zeros_like(l)
    f += g1 * np.power(l,-alpha1) *(1-W_function(l,lb,delta))
    f += g1  * np.power(l,-alpha2) * W_function(l, lb, delta) * np.power(lb+delta/2, -alpha1)/np.power(lb+delta/2, -alpha2)
    f += S1 * drude_modified(l, l0=l01, g0=g01, a=a1)
    f += S2 * drude_modified(l, l0=l02, g0=g02, a=a2)
    return f

def extinction_gordon23(l=1,a_coef = a_coeff,b_coef=b_coeff,Rv=3.1):
    l = np.array(l)
    W1 = W_function(l,0.315,0.03)
    W2 = W_function(l,1.0,0.2)
    # apply fit form Gordon+23
    a, b = np.zeros_like(l), np.zeros_like(l)
    mask_uv = l<0.3
    ll = np.array(l[mask_uv]) #in inverse micron
    a[mask_uv] +=  a_coef['C1AV'] +  a_coef['C2AV']/ll +a_coef['C3AV']*drude(x=1/ll,x0=a_coef['x0'],gamma=a_coef['gamma']) + a_coef['C4AV']*f_farUV(x=1/ll)
    b[mask_uv] +=  b_coef['C1AV'] +  b_coef['C2AV']/ll +b_coef['C3AV']*drude(x=1/ll,x0=b_coef['x0'],gamma=b_coef['gamma']) + b_coef['C4AV']*f_farUV(x=1/ll)

    mask_uv_opt = (l >= 0.3)*(l<0.33)
    ll = np.array(l[mask_uv_opt]) #in inverse micron
    a_opt = optical_extinction_curve(l=ll,E=[a_coef['E0'],a_coef['E1'],a_coef['E2'],a_coef['E3'],a_coef['E4']],
                                     F=[a_coef['F1'],a_coef['F2'],a_coef['F3']])
    b_opt = optical_extinction_curve(l=ll,E=[b_coef['E0'],b_coef['E1'],b_coef['E2'],b_coef['E3'],b_coef['E4']],
                                     F=[b_coef['F1'],b_coef['F2'],b_coef['F3']])
    a_uv = (a_coef['C1AV'] +  a_coef['C2AV']/ll +a_coef['C3AV']*drude(x=1/ll,x0=a_coef['x0'],gamma=a_coef['gamma']) +
                                           a_coef['C4AV']*f_farUV(x=1/ll))
    b_uv = (b_coef['C1AV'] +  b_coef['C2AV']/ll +b_coef['C3AV']*drude(x=1/ll,x0=b_coef['x0'],gamma=b_coef['gamma']) +
                                           b_coef['C4AV']*f_farUV(x=1/ll))
    a[mask_uv_opt] += (1-W1[mask_uv_opt])*np.array(a_uv)
    a[mask_uv_opt] +=     W1[mask_uv_opt]*np.array(a_opt)
    b[mask_uv_opt] += (1-W1[mask_uv_opt])*np.array(b_uv)
    b[mask_uv_opt] +=     W1[mask_uv_opt]*np.array(b_opt)

    mask_opt = ( l >= 0.33 ) * ( l < 0.9 )
    ll = np.array(l[mask_opt])       # in inverse micron
    a_opt = optical_extinction_curve(l=ll,
                                     E=[a_coef['E0'], a_coef['E1'], a_coef['E2'], a_coef['E3'], a_coef['E4']],
                                     F=[a_coef['F1'], a_coef['F2'], a_coef['F3']])
    b_opt = optical_extinction_curve(l=ll,
                                     E=[b_coef['E0'], b_coef['E1'], b_coef['E2'], b_coef['E3'], b_coef['E4']],
                                     F=[b_coef['F1'], b_coef['F2'], b_coef['F3']])
    a[mask_opt] += a_opt
    b[mask_opt] += b_opt

    mask_opt_ir = (l >= 0.9) * (l < 1.1)
    ll = np.array(l[mask_opt_ir])  # in inverse micron

    a_opt = optical_extinction_curve(l=ll,E=[a_coef['E0'],a_coef['E1'],a_coef['E2'],a_coef['E3'],a_coef['E4']],
                                     F=[a_coef['F1'],a_coef['F2'],a_coef['F3']])
    b_opt = optical_extinction_curve(l=ll,E=[b_coef['E0'],b_coef['E1'],b_coef['E2'],b_coef['E3'],b_coef['E4']],
                                     F=[b_coef['F1'],b_coef['F2'],b_coef['F3']])

    a_ir =  IR_extinction(ll,a_coef['g1'],a_coef['alpha1'],a_coef['alpha2'],
                         a_coef['lb'],a_coef['delta'],a_coef['S1'],
                         a_coef['l01'],a_coef['g01'],a_coef['a1'],
                         a_coef['S2'],a_coef['l02'],a_coef['g02'],a_coef['a2'])
    b_ir = b_coef['g1']*np.power(ll,b_coef['alpha1'])

    a[mask_opt_ir] += (1-W2[mask_opt_ir])*a_opt
    a[mask_opt_ir] += W2[mask_opt_ir] * a_ir
    b[mask_opt_ir] += (1-W2[mask_opt_ir])*b_opt
    b[mask_opt_ir] += W2[mask_opt_ir] * b_ir

    mask_ir = (l >= 1.1) * (l <= 32)
    #plt.subplots()
    #plt.plot(l,a)
    #plt.show()
    ll = np.array(l[mask_ir])  # in inverse micron
    a_ir = IR_extinction(ll, a_coef['g1'], a_coef['alpha1'], a_coef['alpha2'],
                         a_coef['lb'], a_coef['delta'], a_coef['S1'],
                         a_coef['l01'], a_coef['g01'], a_coef['a1'],
                         a_coef['S2'], a_coef['l02'], a_coef['g02'], a_coef['a2'])
    b_ir = b_coef['g1'] * np.power(ll, b_coef['alpha1'])
    a[mask_ir] += a_ir
    b[mask_ir] += b_ir


    ext_func = a + b*(1/Rv - 1/3.1)
    return ext_func




if __name__ == '__main__':
    #data for J1007 from from Zhou2010
    c1 = -0.5
    c2=0.67
    c3=3.34
    x0=4.656
    gamma = 1.449

    def Ext_curve_Zhou(xx,g=gamma,x0=x0,c1=c1,c2=c2,c3=c3):
        x = 1/xx
        return c1 + c2*x + c3*drude(x,gamma=g,x0=x0)


    plot_fig = 0
    if plot_fig:
        fig2, ax2 = plt.subplots()
        path = '/home/slava/science/research/kulkarni/JWST-DLAs/ID2155/cassis/Gordon21/spitzer_mir_ext_data_26Mar21/Ext_Curves/'
        stars = {}
        lst = ['hd029647_hd195986_ext_POWLAW2DRUDE.fits', 'hd192660_hd204172_ext_POWLAW2DRUDE.fits',
               'hd147701_hd195986_ext_POWLAW2DRUDE.fits', 'hd283809_hd064802_ext_POWLAW2DRUDE.fits']
        for f in lst:
            hdu = fits.open(path + f)
            data1 = hdu[1].data  # ext in bands
            data2 = hdu[2].data  # uv ext data
            data3 = hdu[3].data  # IR ext data
            head = hdu[0].header
            col1 = data1['WAVELENGTH']
            col2 = data1['EXT']
            col3 = data1['UNC']
            s_BAND = spectrum(np.array(col1), np.array(col2), err=col3)
            col1 = data2['WAVELENGTH']
            col2 = data2['EXT']
            s_V = spectrum(np.array(col1), np.array(col2))
            col1 = data3['WAVELENGTH']
            col2 = data3['EXT']
            s_IR = spectrum(col1, col2)
            Av = float(head['Av'])
            name = f.split('_')[0]
            stars[name] = s_IR
            stars[name].Av = Av
            stars[name].name = name
            stars[name].optical = s_V
            stars[name].band = s_BAND
        if 1:
            list = [1.203, 0.103, 0.981, 4.616, 1.23, 0.217]
            stars['hd147701'].ext_coef = list
            list = [1.135, 0.265, 1.008, 4.587, 0.996, 0.122]
            stars['hd192660'].ext_coef = list

        # s = stars['hd147701']
        s = stars['hd192660']
        mask = s.y / s.Av != 0
        ax2.plot(s.x[mask], s.y[mask] / s.Av + 1)  # IR

        mask = s.optical.y / s.Av != 0
        ax2.plot(s.optical.x[mask], s.optical.y[mask] / s.Av + 1)  # OPTIC
        # ax2.plot(s.optical.x, s.optical.y)       # OPTIC
        ax2.errorbar(x=s.band.x, y=s.band.y / s.Av + 1, yerr=s.band.err / s.Av, fmt='o')  # photometry

        x = 10 ** np.linspace(-1, -0.2, 500)
        ax2.plot(x, FM_ext(xx=x, c1=s.ext_coef[0], c2=s.ext_coef[1], c3=s.ext_coef[2], x0=s.ext_coef[3],
                                   g=s.ext_coef[4], c4=s.ext_coef[5]))
        # ax2.plot(x, ext_curve_func(xx=x, c1=0.00, c2=0.503, c3=0.981, x0=4.616, g=1.23, c4=0.217))
        # ax2.plot(x, IR_ext_curve_func(l=x, B=0.51, alpha=-1))
        x = 10 ** np.linspace(-0.2, 1.6, 500)
        ax2.plot(x, MIR_ext(l=x, B=0.51, alpha=-2.65, S1=0.047, l0=9.64, g0=1.68, a=-1.25), ls='--')
        ax2.plot(x, MIR_ext(l=x, B=0.51, alpha=-2.65, S1=0.047, l0=9.64, g0=1.68, a=-1.25, S2=0.0359, g02=13,
                                      l02=20, a2=-0.30))
        # NIR part
        # x = np.linspace(1, 5, 500)
        # ax2.plot(x,0.386*x**(-1.71),color='tab:green')

        if 1:
            x = 10 ** np.linspace(np.log10(0.0913), np.log10(32), 500)
            f = extinction_gordon23(l=x, Rv=3.1)
            ax2.plot(x, f, color='green')
            f = extinction_gordon23(l=x, Rv=2.5)
            ax2.plot(x, f, color='red')
            f = extinction_gordon23(l=x, Rv=5.5)
            ax2.plot(x, f, color='purple')
            f = extinction_gordon23(l=x, Rv=4.)
            ax2.plot(x, f, color='tab:blue')

        if 1:
            for l in [0.0912, 0.3, 0.33, 0.9, 1.1, 32]:
                ax2.axvline(l, ls='--')
            ax2.axhline(0, ls=':', color='black')

        ax2.set_xscale('log')
        # ax2.set_yscale('log')
        ax2.set_title(s.name)
        ax2.set_xlabel('Observed wavelength, $\\mu$m')
        ax2.set_ylabel('$A_{\\lambda}/A(V)$')
        plt.show()

    #plot
    if 1:
        plt.subplots()
        x = np.linspace(0.1,1,100)
        plt.plot(x,Ext_curve_Zhou(x,c1=0.09,c2=0.47,c3=1.46,g=1.11,x0=4.66))
        plt.plot(x, Ext_curve_Zhou(x, c1=0.16, c2=-0.02, c3=1.65, g=1.85, x0=4.55)+1-Ext_curve_Zhou(0.55, c1=0.16, c2=-0.02, c3=1.65, g=1.85, x0=4.55))
        plt.plot(x, Ext_curve_Zhou(x, c1=-1.07, c2=0.262, c3=0.386, g=0.93, x0=4.46)+1-Ext_curve_Zhou(0.55, c1=-1.07, c2=0.262, c3=0.386, g=0.93, x0=4.46))
        plt.plot(x, Ext_curve_Zhou(x, c1=-0.55, c2=0.626, c3=3.34, g=1.44, x0=4.465)+1-Ext_curve_Zhou(0.55,  c1=-0.55, c2=0.626, c3=3.34, g=1.44, x0=4.465))
        plt.axvline(0.55,ls='--')
        plt.plot(x,extinction_gordon23(x),ls='--')
        plt.plot(x,extinction_gordon23(x,Rv=2.5),ls='--')
        plt.plot(x,extinction_gordon23(x,Rv=5),ls='--')
        plt.plot(x,extinction_gordon23(x,Rv=1.8),ls='--')
        plt.plot(x,extinction_gordon23(x,Rv=10),ls='--')
        plt.show()


    if 1:
        plt.subplots()
        x = np.linspace(0.1,1,100)
        plt.plot(x,Ext_curve_Zhou(x,c1=0.09,c2=0.47,c3=1.46,g=1.11,x0=4.66))
        plt.plot(x, Ext_curve_Zhou(x, c1=0.16, c2=-0.02, c3=1.65, g=1.85, x0=4.55)+1-Ext_curve_Zhou(0.55, c1=0.16, c2=-0.02, c3=1.65, g=1.85, x0=4.55))
        plt.plot(x, Ext_curve_Zhou(x, c1=-1.07, c2=0.262, c3=0.386, g=0.93, x0=4.46)+1-Ext_curve_Zhou(0.55, c1=-1.07, c2=0.262, c3=0.386, g=0.93, x0=4.46))
        plt.plot(x, Ext_curve_Zhou(x, c1=-0.55, c2=0.626, c3=3.34, g=1.44, x0=4.465)+1-Ext_curve_Zhou(0.55,  c1=-0.55, c2=0.626, c3=3.34, g=1.44, x0=4.465))
        plt.axvline(0.55,ls='--')
        plt.plot(x,extinction_gordon23(x),ls='--')
        plt.plot(x,extinction_gordon23(x,Rv=2.5),ls='--')
        plt.plot(x,extinction_gordon23(x,Rv=5),ls='--')
        plt.plot(x,extinction_gordon23(x,Rv=1.8),ls='--')
        plt.plot(x,extinction_gordon23(x,Rv=10),ls='--')
        plt.show()
