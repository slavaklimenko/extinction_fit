#!/usr/bin/env python


from bisect import bisect_left
import matplotlib.pyplot as plt
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
from read_dataset import *
from spec_summary import QSO_list
import copy

###################################################################################################
#SET path to folder containing spectra and dust templates
if 1:
    folder = '/home/slava/science/codes/python/dust_extinction_model/'
    #path_to_sdss_spectrum = folder+'spectrum/spec-J0745-3673-55178-0352.fits'
    path_to_composite_spectra = '/home/slava/science/article/absorption_spectra/Composite_spectra/'

from spec_summary import load_list
from fitting_model import extinction_fit,calc_composite_cont, calc_quasar_variability



 #list_name ='qso_list_init.csv'
list_name = 'qso_list_total.csv'
s = load_list(list_name = list_name, sdss_folder='./sdss/')
print()

from fitting_model import pars, init_pars
par_names = [el.name for el in pars.values() if el.var]

Q = s['J0927+1543']

Q,fitting_data= set_data(Q=read_data(Q),debug=False)
print('Quasar', Q.name)

#import fitting parameters
from fitting_model import pars,init_pars
pars = init_pars(pars)
print('Init pars', pars)

#correct fitting parameters
if 'UKIRT-H' in Q.photo_data.keys():
    pars['fukirtHK'].var = 1
if 'NUV' in Q.photo_data.keys():
    pars['fnuv'].var = 1

theta = [el.val for el in pars.values() if el.var]
print('theta',theta)
theta_best_fit =  [ 1.05530745e+00,  2.99442412e-01, -3.44687687e-01,  3.10791868e-01,
  2.54287949e-01,  1.77645894e-03,  4.64127095e+00,  1.54231843e-01,
  2.01108439e-02,  8.58477342e-01,  9.58731839e-01,  1.48023335e+00]

s_composite = calc_composite_cont()

# a function for comparison data with the redenned composite
def fit_quasar_spectrum(theta=theta, pars=pars, Q=Q, scale_phot_to_spec_res=10,
                        mask_sdds=False,
                        mask_ir=False, debug=False, photometry=True, return_data=False):
    for el, val in zip([p for p in pars.values() if p.var], theta):
        el.val = val
    # print("Parameter values:", [(p.name, p.val) for p in pars.values()])

    # create the composite spectrum in quasar restframe
    s_composite = calc_composite_cont()

    # create variable component of the composite
    y_vary = calc_quasar_variability(s_composite=s_composite, variability=pars['fvar'].val)
    s_composite.y += y_vary

    # create a mask for composite array covering spectral region near w_norm wavelength
    mask_norm = np.abs(s_composite.x * (1 + Q.z_qso) - Q.w_norm) < Q.w_norm_disp

    # normalize composite at flux at the w_norm wavelength
    s_composite.norm_value = np.array(np.nanmean(s_composite.y[mask_norm]))
    s_composite.y /= s_composite.norm_value
    s_composite.err /= s_composite.norm_value

    ext_model = extinction_fit(l=s_composite.x * (1 + Q.z_qso) / (1 + Q.z_abs), c1=pars['c1'].val,
                               c2=pars['c2'].val, c3=pars['c3'].val, c4=pars['c4'].val,
                               x0=pars['x0'].val, gamma=pars['gamma'].val,
                               E0=pars['E0'].val, E3=pars['E3'].val, E4=pars['E4'].val,
                               b=pars['b'].val, alpha=pars['alpha'].val)

    # create a model for the scaled redenned composite
    fit_model = pars['f0'].val * np.array(s_composite.y) * np.exp(-ext_model / 1.086)
    fit_err = pars['f0'].val * np.array(s_composite.err) * np.exp(-ext_model / 1.086)
    # interpolation the model in absorption restframe
    fit_model_interp = interp1d(s_composite.x * (1 + Q.z_qso) / (1 + Q.z_abs), fit_model,
                                fill_value='extrapolate')
    fit_err_interp = interp1d(s_composite.x * (1 + Q.z_qso) / (1 + Q.z_abs), fit_err, fill_value='extrapolate')
    fit_composite_interp = interp1d(s_composite.x * (1 + Q.z_qso) / (1 + Q.z_abs), pars['f0'].val
                                    * np.array(s_composite.y), fill_value='extrapolate')

    # apply scaling for sdss data
    scale_spec = np.ones_like(fitting_data.x)
    scale_spec[fitting_data.x < Q.sp['sdss'].x[-1] / (1 + Q.z_abs)] = pars['fsdss'].val
    scale_spec[fitting_data.x > Q.sp['sdss'].x[-1] / (1 + Q.z_abs)] = pars['fmiri'].val
    # add errors on composite to data_err
    y_err = np.sqrt(fitting_data.err ** 2 + fit_err_interp(fitting_data.x) ** 2) * scale_spec
    # calculate chi2 for spectral data

    # mask uv or ir data_
    weight_data = np.ones_like(fitting_data.x)
    if mask_ir:
        weight_data[fitting_data.x > 1] = 0
    chiq = np.sum(weight_data * np.power((fitting_data.y * scale_spec - fit_model_interp(fitting_data.x)) / (y_err), 2))
    chiq = np.array(chiq)

    # calculate additional chi2 for photometric data
    chiq_phot = 0
    weight_tot = 0
    if photometry:
        # set rescale_photometry as True to correct photometry offset 2mass_vs_sdss and wise_vs_miri
        # consistently with fit_model. The procedure calculated f_corr_NIR,f_corr_MIR.
        # if False, f_corr_NIR,f_corr_MIR is calculated once for composite spectrum
        rescale_photometry = True
        if rescale_photometry:
            f_corr_NIR = pars['fnir'].val  # scaling for 2mass/ukirt(Y,J)
            f_corr_UKIRTHK = pars['fukirtHK'].val  # scaling for 2mass/ukirt(Y,J)
            f_corr_NUV = pars['fnuv'].val  # scaling for wise
            f_corr_wise = 1
        else:
            f_corr_NIR, f_corr_NUV = 1, 1

        for f in Q.photo_data.values():
            weight = 0
            spec_resolution = fitting_data.res / scale_phot_to_spec_res

            if '2MASS' in f.name and 1:
                if f.name not in ['']:  # set bands to ignore
                    phot_flux = np.array(f.f / fitting_data.flux_norm) * f_corr_NIR
                    phot_err = np.array(f.err / fitting_data.flux_norm) * f_corr_NIR
                    bin_size = (f.lmax + f.lmin) / 2 / spec_resolution
                    delta_x, nbin = 50 * bin_size, int((f.lmax - f.lmin + 2 * 50 * bin_size) / bin_size)
                    phot_x = np.linspace(f.lmin - delta_x, f.lmax + delta_x, nbin)
                    weight, model_flux = f.weight_function(xgrid=phot_x,
                                                           flux=fit_model_interp(phot_x / (1 + Q.z_abs)),
                                                           debug=False)
                    model_err = np.nanmean(fit_err_interp(phot_x / (1 + Q.z_abs))) / np.sqrt(weight)
                    if (0.6563 * (1 + Q.z_qso) > f.lmin) and (0.6563 * (1 + Q.z_qso) < f.lmax):
                        weight = 0
            elif 'UKIRT' in f.name and 1:
                if f.name in ['UKIRT-Y', 'UKIRT-J']:  # set bands to ignore
                    phot_flux = np.array(f.f / fitting_data.flux_norm) * f_corr_NIR
                    phot_err = np.array(f.err / fitting_data.flux_norm) * f_corr_NIR
                    bin_size = (f.lmax + f.lmin) / 2 / spec_resolution
                    delta_x, nbin = 50 * bin_size, int((f.lmax - f.lmin + 2 * 50 * bin_size) / bin_size)
                    phot_x = np.linspace(f.lmin - delta_x, f.lmax + delta_x, nbin)
                    weight, model_flux = f.weight_function(xgrid=phot_x,
                                                           flux=fit_model_interp(phot_x / (1 + Q.z_abs)),
                                                           debug=False)
                    model_err = np.nanmean(fit_err_interp(phot_x / (1 + Q.z_abs))) / np.sqrt(weight)
                elif f.name in ['UKIRT-H', 'UKIRT-K']:  # set bands to ignore
                    phot_flux = np.array(f.f / fitting_data.flux_norm) * f_corr_UKIRTHK
                    phot_err = np.array(f.err / fitting_data.flux_norm) * f_corr_UKIRTHK
                    bin_size = (f.lmax + f.lmin) / 2 / spec_resolution
                    delta_x, nbin = 50 * bin_size, int((f.lmax - f.lmin + 2 * 50 * bin_size) / bin_size)
                    phot_x = np.linspace(f.lmin - delta_x, f.lmax + delta_x, nbin)
                    weight, model_flux = f.weight_function(xgrid=phot_x,
                                                           flux=fit_model_interp(phot_x / (1 + Q.z_abs)),
                                                           debug=False)
                    model_err = np.nanmean(fit_err_interp(phot_x / (1 + Q.z_abs))) / np.sqrt(weight)

            elif 'WISE' in f.name:
                if f.name not in ['']:  # set bands to ignore
                    phot_flux = np.array(f.f / fitting_data.flux_norm) * f_corr_wise
                    phot_err = np.array(f.err / fitting_data.flux_norm) * f_corr_wise
                    bin_size = (f.lmax + f.lmin) / 2 / spec_resolution
                    delta_x, nbin = 50 * bin_size, int((f.lmax - f.lmin + 2 * 50 * bin_size) / bin_size)
                    phot_x = np.linspace(f.lmin - delta_x, f.lmax + delta_x, nbin)
                    weight, model_flux = f.weight_function(xgrid=phot_x,
                                                           flux=fit_model_interp(phot_x / (1 + Q.z_abs)),
                                                           debug=False)
                    model_err = np.nanmean(fit_err_interp(phot_x / (1 + Q.z_abs))) / np.sqrt(weight)
            elif 'NUV' in f.name:
                phot_flux = np.array(f.f / fitting_data.flux_norm) * f_corr_NUV
                phot_err = np.array(f.err / fitting_data.flux_norm) * f_corr_NUV
                bin_size = (f.lmax + f.lmin) / 2 / spec_resolution
                delta_x, nbin = 50 * bin_size, int((f.lmax - f.lmin + 2 * 50 * bin_size) / bin_size)
                phot_x = np.linspace(f.lmin - delta_x, f.lmax + delta_x, nbin)
                weight, model_flux = f.weight_function(xgrid=phot_x,
                                                       flux=fit_model_interp(phot_x / (1 + Q.z_abs)),
                                                       debug=False)
                model_err = np.nanmean(fit_err_interp(phot_x / (1 + Q.z_abs))) / np.sqrt(weight)

            if weight > 0:
                weight_tot += weight
                alpha = phot_flux - model_flux
                if 'NUV' in f.name and 1:
                    if model_flux <= phot_flux - phot_err:
                        alpha = np.exp(alpha)
                    else:
                        alpha = 0
                chiq_phot += weight * np.power((alpha) / np.sqrt(phot_err ** 2 + 0 * model_err ** 2), 2)
                if debug:
                    chi_band = np.power(alpha / phot_err, 2)
                    print(f.name, 'chi_band', chi_band, 'obs_band', phot_flux, 'f_model', model_flux,
                          'phot_err',
                          phot_err, 'weight', weight, ' chiq_phot', chiq_phot)
        # add  photometric chi2 to spectral data chi2
        chiq += chiq_phot
    # print stats:
    if debug:
        print('N_spec_points:', np.size(fitting_data.x))
        print('N_phot_points:', weight_tot)
        print('chi(spec):', chiq - chiq_phot, 'chi(phot):', chiq_phot)
        print('chiq_reduced:', chiq / (np.sum(weight_data) + weight_tot - len(pars)))
    # plot model and data
    if debug:
        print("Parameter values:", [(p.name, p.val, p.var) for p in pars.values()])
        print('Parameter values list:', [p.val for p in pars.values()])
        fit_composite = interp1d(s_composite.x * (1 + Q.z_qso) / (1 + Q.z_abs), pars['f0'].val * s_composite.y,
                                 fill_value='extrapolate')
        ext_profile_no_bump = (
            extinction_fit(l=s_composite.x * (1 + Q.z_qso) / (1 + Q.z_abs), c1=pars['c1'].val,
                           c2=pars['c2'].val, c3=0, c4=pars['c4'].val,
                           x0=pars['x0'].val, gamma=pars['gamma'].val,
                           E0=pars['E0'].val, E3=pars['E3'].val, E4=pars['E4'].val,
                           b=pars['b'].val, alpha=pars['alpha'].val))
        Av = extinction_fit(l=[0.55], c1=pars['c1'].val,
                            c2=pars['c2'].val, c3=pars['c3'].val, c4=pars['c4'].val,
                            x0=pars['x0'].val, gamma=pars['gamma'].val,
                            E0=pars['E0'].val, E3=pars['E3'].val, E4=pars['E4'].val,
                            b=pars['b'].val, alpha=pars['alpha'].val)
        A2 = extinction_fit(l=[0.2], c1=pars['c1'].val,
                            c2=pars['c2'].val, c3=pars['c3'].val, c4=pars['c4'].val,
                            x0=pars['x0'].val, gamma=pars['gamma'].val,
                            E0=pars['E0'].val, E3=pars['E3'].val, E4=pars['E4'].val,
                            b=pars['b'].val, alpha=pars['alpha'].val)
        print('AV:', Av)

        fig, ax = plt.subplots(2, 1, sharex=True)
        fig.subplots_adjust(hspace=0.1)

        ax[0].plot(s_composite.x * (1 + Q.z_qso) / (1 + Q.z_abs), ext_model / Av, label='ext_model', color='red')
        ax[0].plot(fitting_data.x, -1.086 * np.log(fitting_data.y * scale_spec / fit_composite(fitting_data.x)) / Av,
                   label='Data',
                   color='black',
                   zorder=-100, lw=1)
        ax[0].plot(s_composite.x * (1 + Q.z_qso) / (1 + Q.z_abs), ext_profile_no_bump / Av,
                   label='ext_no bump',
                   color='red', ls='--')
        l = np.linspace(0.1, 10, 5000)
        ax[0].plot(l, extinction_gordon23(l=l, Rv=2.7), color='green', lw=2, label='MW ISM, $R_V$=3.1',
                   zorder=100, ls='--', alpha=0.6)

        ax[1].plot(s_composite.x * (1 + Q.z_qso) / (1 + Q.z_abs), pars['f0'].val * s_composite.y, label='Cont')
        ax[1].plot(s_composite.x * (1 + Q.z_qso) / (1 + Q.z_abs), fit_model, label='Fit', color='red')
        ax[1].plot(s_composite.x * (1 + Q.z_qso) / (1 + Q.z_abs), pars['f0'].val * np.array(s_composite.y) *
                   np.exp(-ext_profile_no_bump / 1.086), label='Fit_no_bump', color='red', ls='--')
        ax[1].plot(fitting_data.x, fitting_data.y * scale_spec, label='Data', color='black', lw=1, zorder=-100)
        if 1:
            (x, y) = (s_composite.x * (1 + Q.z_qso) / (1 + Q.z_abs),
                      pars['f0'].val * s_composite.y *
                      np.exp(-Av * extinction_gordon23(l=s_composite.x * (1 + Q.z_qso) / (1 + Q.z_abs),
                                                       Rv=2.7) / 1.086))
            ax[1].plot(x[::100], y[::100],
                       color='green', zorder=100, lw=2, ls='--', alpha=0.6, label='Fit with the MW')

        plot_photometry = photometry
        if plot_photometry:
            for el in Q.photo_data.values():
                if 'SDSS' not in el.name:
                    if 'SDSS' in el.name:
                        color = 'orange'
                        f_corr = pars['fsdss'].val
                    elif '2MASS' in el.name:
                        color = 'green'
                        f_corr = f_corr_NIR
                    elif el.name in ['UKIRT-Y', 'UKIRT-J']:
                        color = 'green'
                        f_corr = f_corr_NIR
                    elif el.name in ['UKIRT-H', 'UKIRT-K']:
                        color = 'green'
                        f_corr = f_corr_UKIRTHK
                    elif 'WISE' in el.name:
                        color = 'purple'
                        f_corr = f_corr_wise
                    elif 'NUV' in el.name:
                        color = 'red'
                        f_corr = f_corr_NUV
                    else:
                        color = 'blue'
                        f_corr = 1
                    ax[1].errorbar(x=el.l / (1 + Q.z_abs), y=el.f * f_corr / fitting_data.flux_norm,
                                   xerr=[[(el.l - el.lmin) / (1 + Q.z_abs)],
                                         [(el.lmax - el.l) / (1 + Q.z_abs)]],
                                   yerr=el.err * f_corr / fitting_data.flux_norm, fmt='s',
                                   markerfacecolor=color, markeredgecolor='black', ecolor=color)
                    # calculate best fit 'model flux' in the band
                    bin_size = (el.lmax + el.lmin) / 2 / spec_resolution
                    delta_x, nbin = 50 * bin_size, int((el.lmax - el.lmin + 2 * 50 * bin_size) / bin_size)
                    phot_x = np.linspace(el.lmin - delta_x, el.lmax + delta_x, nbin)
                    weight, model_flux = el.weight_function(xgrid=phot_x,
                                                            flux=fit_composite_interp(phot_x / (1 + Q.z_abs)),
                                                            debug=False)
                    ax[0].errorbar(x=el.l / (1 + Q.z_abs),
                                   y=-1.086 * np.log(el.f * f_corr / fitting_data.flux_norm / model_flux) / Av,
                                   fmt='s', markerfacecolor=color,
                                   xerr=[[(el.l - el.lmin) / (1 + Q.z_abs)],
                                         [(el.lmax - el.l) / (1 + Q.z_abs)]],
                                   yerr=-1.086 * np.log(el.f / (el.f + el.err)) / Av,
                                   markeredgecolor='black', ecolor=color)

        ax[1].set_title('chi2=' + str(chiq))
        ax[1].legend()
        ax[1].set_xscale('log')
        # ax[1].set_yscale('log')
        ax[0].set_ylabel('Ext curve')
        ax[1].set_ylabel('Flux')
        ax[0].axvline(Q.w_norm / (1 + Q.z_abs), ls='--')
        ax[1].axvline(Q.w_norm / (1 + Q.z_abs), ls='--')

        ax[0].set_ylim(-0.5, 1.2 * A2 / Av)
        f_v = np.nanmean(fit_model[np.abs(s_composite.x * (1 + Q.z_qso) / (1 + Q.z_abs) - 0.55) < 0.1])
        ax[1].set_ylim(-0.5, 5 * f_v)
        fig.savefig('images/' + Q.name + '_ext_fit.pdf', bbox_inches='tight')

    ln = -chiq

    # return ln (by default) or model details
    if return_data:
        print('chi2=', chiq, ' chi2/red', chiq / np.size(fitting_data.y))
        return (s_composite.x, s_composite.y, ext_model, fit_model)
    else:
        return ln


#make a figure with 3 panels
if 1:
    for el, val in zip([p for p in pars.values()], theta_best_fit):
        el.val = val
        el.var = 1
    # calc Av and A2175 stregth for the best fit solution
    #Av = c1 + c2/ 0.55
    #A2175 = c3 * np.pi / 2 / gamma
    #
    print('plot debug')
    print('theta_best_fit',theta_best_fit)
    (s_composite.x, s_composite.y, ext_model, fit_model) = fit_quasar_spectrum(theta=theta_best_fit,pars=pars, debug=True,
                                                        photometry=False, return_data = True)
    fit_model_interp = interp1d(s_composite.x * (1 + Q.z_qso) / (1 + Q.z_abs), fit_model, fill_value='extrapolate')
    fit_composite_interp = interp1d(s_composite.x * (1 + Q.z_qso) / (1 + Q.z_abs), pars['f0'].val*s_composite.y, fill_value='extrapolate')
    #
    ext_model_no_bump =  (ext_model - pars['c3'].val*drude(x=1/(s_composite.x * (1 + Q.z_qso) / (1 + Q.z_abs)),
                                                           x0=pars['x0'].val,gamma=pars['gamma'].val))
    fit_model_no_bump = pars['f0'].val*s_composite.y  * np.exp(-ext_model_no_bump / 1.086)
    #
    fontsize = 12
    fig, ax = plt.subplots(2, 1, figsize=(6, 6))
    fig.subplots_adjust(hspace=0.1)
    pos = ax[0].get_position()
    print(pos)
    #
    # plot composite
    ax[0].plot(s_composite.x * (1 + Q.z_qso) / (1 + Q.z_abs), pars['f0'].val*s_composite.y*fitting_data.flux_norm, label='QSO Composite',
               color='blue', zorder=-40, lw=1)
    # plot best fit
    ax[0].plot(s_composite.x * (1 + Q.z_qso) / (1 + Q.z_abs), fit_model*data_norm, label='Best Fit',
               color='red', zorder=10, lw=1)

    ax[1].plot(s_composite.x * (1 + Q.z_qso) / (1 + Q.z_abs), ext_model/AV_best_fit, zorder=100, color='red', lw=2,label='Best Fit')
    #plot fit to extinction curve with no bump
    # fill 2175 bump region
    if 1:
        ax[0].fill_between(x=s_composite.x * (1 + Q.z_qso) / (1 + Q.z_abs), y1=fit_model_no_bump*data_norm ,
                           y2=fit_model*data_norm , color='red', zorder=-10, alpha=0.1)
        ax[1].fill_between(x=s_composite.x * (1 + Q.z_qso) / (1 + Q.z_abs), y1=ext_model/AV_best_fit,
                           y2=ext_model_no_bump/AV_best_fit, color='red', zorder=-100,
                           alpha=0.2)


    #plot data
    scale_spec = np.ones_like(data_x)
    scale_spec[data_x<Q.sp['sdss'].x[-1] / (1 + Q.z_abs)] *= pars['fsdss'].val
    scale_spec[data_x>1] *=pars['fmiri'].val
    if 'spitzer' in Q.sp.keys():


        mask = data_tot_x <1
        ax[1].plot(data_tot_x[mask], -1.086 * np.log(data_tot_y[mask] * scale_spec_tot[mask] / fit_composite_interp(data_tot_x[mask]))/AV_best_fit,
                   color='gray', zorder=-100, alpha=0.5)
        ax[1].plot(data_tot_x[~mask], -1.086 * np.log(data_tot_y[~mask] * scale_spec_tot[~mask] / fit_composite_interp(data_tot_x[~mask]))/AV_best_fit,
                   color='gray', zorder=-100, alpha=0.5)
        ax[0].errorbar(x=data_tot_x[mask], y=data_tot_y[mask] * data_norm * scale_spec_tot[mask], color='grey', ls='-', lw=0.5,
                       zorder=-100)
        ax[0].errorbar(x=data_tot_x[~mask], y=data_tot_y[~mask] * data_norm * scale_spec_tot[~mask], color='grey', ls='-', lw=0.5,
                       zorder=-100)

        mask = data_x < 1
        ax[0].errorbar(x=data_x[mask], y=data_y[mask] * data_norm * scale_spec[mask],
                       yerr=data_err[mask] * data_norm, color='black',
                       zorder=-10, ls='-', lw=0.5)
        ax[0].errorbar(x=data_x[~mask], y=data_y[~mask] * data_norm * scale_spec[~mask],
                       yerr=data_err[~mask] * data_norm, color='black',
                       zorder=-10, ls='-', lw=0.5)

        ax[1].plot(data_x[mask], -1.086 * np.log(data_y[mask] * scale_spec[mask] / fit_composite_interp(data_x[mask]))/AV_best_fit,
                   color='black', zorder=-10, alpha=1, lw=0.5)
        ax[1].plot(data_x[~mask], -1.086 * np.log(data_y[~mask] * scale_spec[~mask] / fit_composite_interp(data_x[~mask]))/AV_best_fit,
                   color='black', zorder=-10, alpha=1, lw=0.5)

    else:
        mask = data_tot_x < 1
        ax[0].errorbar(x=data_tot_x[mask], y=data_tot_y[mask]* data_norm*scale_spec_tot[mask], color='grey', ls='-', lw=0.5, zorder=-100)

        ax[1].plot(data_tot_x[mask], -1.086 * np.log(data_tot_y[mask] * scale_spec_tot[mask] / fit_composite_interp(data_tot_x[mask]))/AV_best_fit,
                   color='gray', zorder=-100, alpha=0.5)


        mask = data_x < 1
        ax[0].errorbar(x=data_x[mask], y=data_y[mask] * data_norm * scale_spec[mask],
                       color='black', #yerr=data_err[mask] * data_norm
                       zorder=-10, ls='-', lw=0.5)
        ax[1].plot(data_x[mask], -1.086 * np.log(data_y[mask] * scale_spec[mask] / fit_composite_interp(data_x[mask]))/AV_best_fit,
                   color='black', zorder=-10, alpha=1, lw=0.5)

        mask = data_tot_x > 1
        ax[0].errorbar(x=data_tot_x[mask], y=data_tot_y[mask] * data_norm * scale_spec_tot[mask], color='grey', ls='-',
                       lw=0.5, zorder=-100)
        ax[1].plot(data_tot_x[mask],
                   -1.086 * np.log(data_tot_y[mask] * scale_spec_tot[mask] / fit_composite_interp(data_tot_x[mask]))/AV_best_fit,
                   color='gray', zorder=-100, alpha=0.5)

        mask = data_x > 1
        ax[0].errorbar(x=data_x[mask], y=data_y[mask] * data_norm * scale_spec[mask],
                       #yerr=data_err[mask] * data_norm,
                       color='black',
                       zorder=-10, ls='-', lw=0.5)
        ax[1].plot(data_x[mask], -1.086 * np.log(data_y[mask] * scale_spec[mask] / fit_composite_interp(data_x[mask]))/AV_best_fit,
                   color='black', zorder=-10, alpha=1, lw=0.5)


    #plot photometry
    if 1:
        rescale_photometry = True
        scale_photometry_resolution = 5
        spec_resolution = data_res / scale_photometry_resolution
        if rescale_photometry:
            f_corr_NIR = pars['fnir'].val  # scaling for 2mass/ukirt(Y,J)
            f_corr_UKIRTHK = pars['fukirtHK'].val  # scaling for 2mass/ukirt(Y,J)
            f_corr_MIR = 1
        else:
            f_corr_NIR, f_corr_MIR = 1, 1

        for el in Q.photo_data.values():
            if 'SDSS' in el.name:
                color = 'orange'
                f_corr = pars['fsdss'].val
            elif '2MASS' in el.name:
                color = 'green'
                f_corr = f_corr_NIR
            elif el.name in ['UKIRT-Y', 'UKIRT-J']:
                color = 'green'
                f_corr = f_corr_NIR
            elif el.name in ['UKIRT-H', 'UKIRT-K']:
                color = 'green'
                f_corr = f_corr_UKIRTHK
            elif 'WISE' in el.name:
                color = 'purple'
                f_corr = f_corr_MIR
            elif 'NUV' in el.name:
                color = 'tab:blue'
                f_corr = pars['fnuv'].val

            else:
                color = 'blue'
                f_corr = 1
            if el.name not in ['MIRI1A','SDSSu','SDSSg','SDSSr','SDSSi','SDSSz']:
                msize = 8
                uplims,lolims=False,False
                if 'NUV' in el.name:
                    uplims, lolims = True,True
                ax[0].errorbar(x=el.l / (1 + Q.z_abs), y=(el.f-el.err*lolims)*f_corr ,
                               xerr=[[(el.l - el.lmin) / (1 + Q.z_abs)], [(el.lmax - el.l) / (1 + Q.z_abs)]],
                               yerr=el.err*f_corr , fmt='s',markersize=msize,
                               markerfacecolor=color, markeredgecolor='black', ecolor=color,zorder=100,lolims=lolims)

                # calculate best fit 'model flux' in the band
                bin_size = (el.lmax + el.lmin) / 2 / spec_resolution
                delta_x, nbin = 50 * bin_size, int((el.lmax - el.lmin + 2 * 50 * bin_size) / bin_size)
                phot_x = np.linspace(el.lmin - delta_x, el.lmax + delta_x, nbin)
                weight, model_flux = el.weight_function(xgrid=phot_x, flux=fit_composite_interp(phot_x / (1 + Q.z_abs)),
                                                       debug=False)
                #
                ax[1].errorbar(x=el.l / (1 + Q.z_abs),
                               y=-1.086 * np.log((el.f-el.err*uplims)*f_corr / data_norm /model_flux)/AV_best_fit,
                               fmt='s', markerfacecolor=color,markersize=msize,
                               xerr=[[(el.l - el.lmin) / (1 + Q.z_abs)], [(el.lmax - el.l) / (1 + Q.z_abs)]],
                               yerr=-1.086 * np.log(el.f / (el.f + el.err))/AV_best_fit,
                               markeredgecolor='black', ecolor=color,zorder=100,uplims=uplims)


    #add MW fit
    if 1:
        mask_v = np.abs(s_composite.x * (1 + Q.z_qso) / (1 + Q.z_abs)-0.55)<0.05
        print('AV:', np.median(ext_model[mask_v]), 'A2175:', pars['c3'].val / pars['gamma'].val ** 2)
        l = np.linspace(0.1, 10, 5000)
        ax[1].plot(l, extinction_gordon23(l=l, Rv=2.7), color='green', lw=2, label='MW ISM, $R_V$=3.1',
                   zorder=100,ls='--',alpha=0.6)
        #[::100]
        (x,y) = (s_composite.x * (1 + Q.z_qso) / (1 + Q.z_abs),pars['f0'].val * s_composite.y * data_norm *
                   np.exp(-AV_best_fit*extinction_gordon23(l=s_composite.x * (1 + Q.z_qso) / (1 + Q.z_abs),  Rv=2.7) / 1.086))
        ax[0].plot(x[::100], y[::100],
                   color='green', zorder=100, lw=2, ls='--',alpha=0.6,label='Fit with the MW')
        #
        if 1:
            phot_x = np.linspace(0.5, 0.6, 100)
            phot_y = np.trapz(extinction_gordon23(l=phot_x, Rv=2.7), phot_x)
            ext_model_av = phot_y / 0.1

            phot_x = np.linspace(0.4, 0.5, 100)
            phot_y = np.trapz(extinction_gordon23(l=phot_x, Rv=2.7), phot_x)
            ext_model_ab = phot_y / 0.1
        print('Gordon Rv:',ext_model_av/(ext_model_ab-ext_model_av))


    #ax[0].set_yscale('log')
    ax[0].set_xscale('log')
    ax[1].set_xscale('log')
    #ax[1].set_yscale('log')
    #



    if Q.name == 'J1007':
        ax[0].set_ylim(-1, 13)
        ax[1].set_ylim(-0.2, 4.2)


        yl, yu = ax[1].get_ylim()
        ax[1].text(0.7, yl + (yu - yl) * 0.8, 'J1007+2853', fontsize=fontsize, rotation=0, color='black', alpha=1)
        ax[1].text(0.7, yl + (yu - yl) * 0.7, '$A_V=1.93$', fontsize=fontsize, rotation=0,
                   color='black', alpha=1)
        ax[1].text(0.7, yl + (yu - yl) * 0.6, '$A_{2175}=1.26$, $R_V=5.6$', fontsize=fontsize, rotation=0,
                   color='black', alpha=1)
        ax[0].yaxis.set_minor_locator(AutoMinorLocator(2))
        ax[0].yaxis.set_major_locator(MultipleLocator(2))
        ax[1].yaxis.set_minor_locator(AutoMinorLocator(2))
        ax[1].yaxis.set_major_locator(MultipleLocator(1))


    ax[0].set_ylabel('$F_{\\lambda}$, $\\rm 10^{-17} erg/s/cm^2$', fontsize=fontsize)
    ax[1].set_ylabel('$A(\\lambda)/A_V$', fontsize=fontsize)

    #set legends and text
    #ax[0].legend(loc='lower left', fontsize=fontsize - 1, framealpha=1)
    yl, yup = ax[0].get_ylim()
    l = Q.name + ', $z_{\\rm abs}=' + str(round(Q.z_abs, 2)) + '$'
    #ax[0].text(0.6, yl + 0.45 * (yup - yl),  l , fontsize=fontsize)
    yl, yu = ax[1].get_ylim()
    ax[1].text(0.52,yl+(yu-yl)*0.9, 'V',fontsize=fontsize,rotation = 0,color='black',alpha=1)
    ax[1].text(2.1,yl+(yu-yl)*0.9, 'K',fontsize=fontsize,rotation = 0,color='black',alpha=1)


    #ax[0].text(4.2, yl + 0.05 * (yup - yl), '$\\lambda_{\\rm Norm}$', fontsize=fontsize, rotation=90, color='tab:blue', alpha=0.5)
    #ax[2].text(0.3, yl + 0.7 * (yup - yl),'$z_{\\rm abs}=' + str(round(Q.z_abs, 2)) + '$' + ', $z_{\\rm qso}=' + str(
    #               round(Q.z_qso, 2)) + '$', fontsize=fontsize)

    ax[1].set_xlabel('Absorber Rest Wavelength, $\mu$m', fontsize=fontsize)



    for axs in ax[:]:


        #axs.axvline(0.5, c='orange', ls='--',alpha=0.5)
        #axs.axvline(0.6, c='orange', ls='--',alpha=0.5)

        #axs.axvline(2.2, c='gray', ls='--', alpha=0.5)
        axs.fill_betweenx(y=[-1,400],x1=0.5,x2=0.6,color='orange',alpha=0.2)
        axs.fill_betweenx(y=[-1,400],x1=2.1,x2=2.3,color='orange',alpha=0.2)
        #axs.fill_betweenx(y=[-1, 100], x1=0.94/(1+Q.z_abs), x2=2.45/(1+Q.z_abs), color='green', alpha=0.35)
        axs.axhline(0, ls=':', c='black')
        axs.tick_params(which='both', width=1, direction='in',
                        labelsize=fontsize,
                        right='True',
                        top='True')
        axs.tick_params(which='major', length=5)
        axs.tick_params(which='minor', length=3)
        if 1:
            axs.set_xlim(0.08, 5.1) #int(3.5 * (1 + Q.z_qso) / (1 + Q.z_abs)) + 1)
        else:
            axs.set_xlim(0.15, 5.1) #int(3.5 * (1 + Q.z_qso) / (1 + Q.z_abs)) + 1)
        axs.set_xscale('log')
        #for el in Q.sp.values():
        #    axs.fill_betweenx(x1=el.x[0] / (1 + Q.z_abs), x2=el.x[-1] / (1 + Q.z_abs), y=[-1, 1e2], color='grey',
        #                  alpha=0.2)

    for axs in ax[:]:
        axs.set_xticks([0.2, 0.5, 1, 2, 3, 5])
        axs.get_xaxis().set_major_formatter(
            plt.ScalarFormatter())  # Optional: shows numbers instead of scientific notation
        axs.tick_params(which='major', length=5)
        axs.tick_params(which='minor', length=3)



    ax[0].legend(loc='upper right', fontsize=fontsize)




    #ax[2].set_yscale('log')
    plt.show()

    save_fig = False
    if save_fig:
        fig.savefig('./images/fig_ext_curve_j1017.pdf', bbox_inches='tight')
    #plt.show()