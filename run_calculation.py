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







if __name__ == '__main__':

    #main settings
    run_mcmc = False
    if run_mcmc:
        mcmc_part1_name = 'mcmc_part1.pkl'
        mcmc_part2_name = 'mcmc_part2.pkl'

    #list_name ='qso_list_init.csv'
    list_name = 'qso_list_total.csv'
    s = load_list(list_name = list_name, sdss_folder='./sdss/')
    print()

    from fitting_model import pars, init_pars
    par_names = [el.name for el in pars.values() if el.var]
    with open('./results/fit_results.txt', 'w') as file:
        line = ''
        for p in par_names:
                line += ', ' + p + ',' + p+'_err'
        file.write('target' + line + '\n')
    file.close()


    for Q in s.values():
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



            ext_model = pars['Av'].val*extinction_fit(l=s_composite.x * (1 + Q.z_qso) / (1 + Q.z_abs), c1=pars['c1'].val,
                                       c2=pars['c2'].val, c3=pars['c3'].val, c4=pars['c4'].val,
                                       x0=pars['x0'].val, gamma=pars['gamma'].val,
                                       E0=pars['E0'].val, E3=pars['E3'].val, E4=pars['E4'].val,
                                       b=pars['b'].val, alpha=pars['alpha'].val,mode = '2-order')

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
                            if (0.6563*(1+Q.z_qso)>f.lmin) and (0.6563*(1+Q.z_qso)<f.lmax):
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
                            phot_flux = np.array(f.f /fitting_data.flux_norm) * f_corr_UKIRTHK
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
                            phot_flux = np.array(f.f / fitting_data.flux_norm)*f_corr_wise
                            phot_err = np.array(f.err / fitting_data.flux_norm)*f_corr_wise
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

                ext_model_no_bump = (
                        ext_model - pars['Av'].val*pars['c3'].val * drude(x=1 / (s_composite.x * (1 + Q.z_qso) / (1 + Q.z_abs)),
                                                               x0=pars['x0'].val, gamma=pars['gamma'].val))
                fit_model_no_bump = pars['f0'].val * np.array(s_composite.y) * np.exp(-ext_model_no_bump / 1.086)
                A_V = pars['Av'].val*extinction_fit(l=[0.55], c1=pars['c1'].val,
                                       c2=pars['c2'].val, c3=pars['c3'].val, c4=pars['c4'].val,
                                       x0=pars['x0'].val, gamma=pars['gamma'].val,
                                       E0=pars['E0'].val, E3=pars['E3'].val, E4=pars['E4'].val,
                                       b=pars['b'].val, alpha=pars['alpha'].val,mode = '2-order')

                fig, ax = plt.subplots(2, 1, figsize=(6, 6))
                fig.subplots_adjust(hspace=0.1)
                fontsize= 12


                ax[0].plot(s_composite.x * (1 + Q.z_qso) / (1 + Q.z_abs), ext_model/A_V, label='ext_model', color='red')
                ax[0].plot(fitting_data.x, -1.086 * np.log(fitting_data.y * scale_spec / fit_composite(fitting_data.x))/A_V,
                           color='black',
                           zorder=-100, lw=1)
                l = np.linspace(0.1, 10, 5000)
                ax[0].plot(l, extinction_gordon23(l=l, Rv=3.1), color='green', lw=2, label='MW ISM, $R_V$=3.1',
                           zorder=100, ls='--', alpha=0.6)

                ax[1].plot(fitting_data.x, fitting_data.y * scale_spec, color='black', lw=1, zorder=-100)
                mask_data = s_composite.x * (1 + Q.z_qso) / (1 + Q.z_abs) > fitting_data.x[0]-0.1
                ax[1].plot(s_composite.x[mask_data] * (1 + Q.z_qso) / (1 + Q.z_abs), pars['f0'].val * s_composite.y[mask_data], label='QSO Composite',color='tab:blue',lw=1)
                ax[1].plot(s_composite.x[mask_data] * (1 + Q.z_qso) / (1 + Q.z_abs), fit_model[mask_data], label='Best Fit', color='red')

                #show bump
                ax[1].fill_between(x=s_composite.x[mask_data] * (1 + Q.z_qso) / (1 + Q.z_abs), y1=fit_model_no_bump[mask_data],
                                   y2=fit_model[mask_data], color='purple', zorder=-10, alpha=0.3)
                ax[0].fill_between(x=s_composite.x * (1 + Q.z_qso) / (1 + Q.z_abs), y1=ext_model /A_V,
                                   y2=ext_model_no_bump / A_V, color='purple', zorder=-100,
                                   alpha=0.2)


                plot_photometry = photometry
                if 1:
                    phot_flux_max = 1
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
                                f_corr = f_corr_NIR
                            elif 'WISE' in el.name:
                                color = 'purple'
                                f_corr = 1
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
                            spec_resolution = fitting_data.res / scale_phot_to_spec_res
                            bin_size = (el.lmax + el.lmin) / 2 / spec_resolution
                            delta_x, nbin = 50 * bin_size, int((el.lmax - el.lmin + 2 * 50 * bin_size) / bin_size)
                            phot_x = np.linspace(el.lmin - delta_x, el.lmax + delta_x, nbin)
                            weight, model_flux = el.weight_function(xgrid=phot_x,
                                                                    flux=fit_composite_interp(phot_x / (1 + Q.z_abs)),
                                                                    debug=False)
                            ax[0].errorbar(x=el.l / (1 + Q.z_abs),
                                           y=-1.086 * np.log(el.f * f_corr / fitting_data.flux_norm / model_flux)/A_V,
                                           fmt='s', markerfacecolor=color,
                                           xerr=[[(el.l - el.lmin) / (1 + Q.z_abs)],
                                                 [(el.lmax - el.l) / (1 + Q.z_abs)]],
                                           yerr=-1.086 * np.log(el.f / (el.f + el.err))/A_V,
                                           markeredgecolor='black', ecolor=color)
                            if phot_flux_max < -1.086 * np.log(el.f * f_corr / fitting_data.flux_norm / model_flux)/A_V:
                                phot_flux_max = -1.086 * np.log(el.f * f_corr / fitting_data.flux_norm / model_flux)/A_V

                ax[1].legend(loc='lower left', fontsize=fontsize)
                ax[0].set_ylabel('$A(\\lambda)/A_V$', fontsize=fontsize)
                ax[1].set_ylabel('Flux/F(W2)', fontsize=fontsize)
                ax[1].set_xlabel('Absorber Rest Wavelength, $\mu$m', fontsize=fontsize)

                for axs in ax[:]:


                    axs.fill_betweenx(y=[-1, 400], x1=0.5, x2=0.6, color='orange', alpha=0.2)
                    axs.fill_betweenx(y=[-1, 400], x1=2.1, x2=2.3, color='orange', alpha=0.2)

                    axs.axhline(0, ls=':', c='black')
                    axs.tick_params(which='both', width=1, direction='in',
                                    labelsize=fontsize,
                                    right='True',
                                    top='True')
                    axs.tick_params(which='major', length=5)
                    axs.tick_params(which='minor', length=3)
                    axs.set_xlim(0.08, 5.1)  # int(3.5 * (1 + Q.z_qso) / (1 + Q.z_abs)) + 1)
                    axs.set_xscale('log')

                for axs in ax[:]:
                    axs.set_xticks([0.2, 0.5, 1, 2, 3, 5])
                    axs.get_xaxis().set_major_formatter(
                        plt.ScalarFormatter())  # Optional: shows numbers instead of scientific notation
                    axs.tick_params(which='major', length=5)
                    axs.tick_params(which='minor', length=3)



                f_3 = np.nanmean(fit_model[np.abs(s_composite.x * (1 + Q.z_qso) / (1 + Q.z_abs) - 0.2) < 0.1])
                a_3 = (np.nanmean(ext_model[np.abs(s_composite.x * (1 + Q.z_qso) / (1 + Q.z_abs) - 0.2) < 0.1]))
                ax_0_max = np.max([phot_flux_max,  2*a_3/A_V])
                ax[0].set_ylim(-0.5*a_3/A_V, ax_0_max)
                #ax[1].set_ylim(-0.5*f_3, 2*f_3)

                ax[0].yaxis.set_minor_locator(AutoMinorLocator(2))
                step = int(ax_0_max/5)
                if step==0:
                    step = 1
                ax[0].yaxis.set_major_locator(MultipleLocator(step))
                step = int(2*f_3/5)
                if step==0:
                    step = 1
                #ax[1].yaxis.set_minor_locator(AutoMinorLocator(2))
                #ax[1].yaxis.set_major_locator(MultipleLocator(step))
                ax[1].set_yscale('log')
                ax[1].set_ylim(0.5, 4 * f_3)

                fig.savefig('images/'+Q.name+'_ext_fit_all.pdf', bbox_inches='tight')
                #plt.show()

            ln = -chiq

            # return ln (by default) or model details
            if return_data:
                print('chi2=', chiq, ' chi2/red', chiq / np.size(fitting_data.y))
                return (s_composite.x, s_composite.y, ext_model, fit_model)
            else:
                return ln


        # test procedure
        #fit_quasar_spectrum(debug=True,theta=theta, pars=pars, Q=Q)
        #plt.show()

        # set likelihood
        if 1:
            x, y = fitting_data.x, fitting_data.y

            # define likelihood function for data
            def log_likelihood_ir(theta, x, y):
                ln = fit_quasar_spectrum(theta=theta,Q=Q, mask_sdds=True, mask_ir=False, debug=False)
                return ln

            def log_likelihood_sdss(theta, x, y):
                ln = fit_quasar_spectrum(theta=theta,Q=Q, mask_sdds=False, mask_ir=True)
                return ln


            # define prior function for paramters
            def log_prior(theta, pars=pars):
                for el, val in zip([p for p in pars.values() if p.var], theta):
                    el.val = val
                if 1:
                    # measure extimction at x1,x2, and V-band
                    x1, x2 = 0.3, 1

                    f_x1 = extinction_fit(l=[x1], c1=pars['c1'].val,
                                          c2=pars['c2'].val, c3=0, c4=pars['c4'].val,
                                          x0=pars['x0'].val, gamma=pars['gamma'].val,
                                          E0=pars['E0'].val, E3=pars['E3'].val, E4=pars['E4'].val,
                                          b=pars['b'].val, alpha=pars['alpha'].val,mode = '2-order')
                    f_x2 = extinction_fit(l=[x2 - 0.01], c1=pars['c1'].val,
                                          c2=pars['c2'].val, c3=0, c4=pars['c4'].val,
                                          x0=pars['x0'].val, gamma=pars['gamma'].val,
                                          E0=pars['E0'].val, E3=pars['E3'].val, E4=pars['E4'].val,
                                          b=pars['b'].val, alpha=pars['alpha'].val,mode = '2-order')
                    Av = extinction_fit(l=[0.55], c1=pars['c1'].val,
                                        c2=pars['c2'].val, c3=pars['c3'].val, c4=pars['c4'].val,
                                        x0=pars['x0'].val, gamma=pars['gamma'].val,
                                        E0=pars['E0'].val, E3=pars['E3'].val, E4=pars['E4'].val,
                                        b=pars['b'].val, alpha=pars['alpha'].val,mode = '2-order')

                    A_5 = extinction_fit(l=[5.0], c1=pars['c1'].val,
                                         c2=pars['c2'].val, c3=pars['c3'].val, c4=pars['c4'].val,
                                         x0=pars['x0'].val, gamma=pars['gamma'].val,
                                         E0=pars['E0'].val, E3=pars['E3'].val, E4=pars['E4'].val,
                                         b=pars['b'].val, alpha=pars['alpha'].val,mode = '2-order')

                    A_1 = pars['b'].val
                # measure max and min optical extinction
                if 1:
                    l = np.linspace(0.33, 0.97, 20)
                    f_max = np.nanmax(extinction_fit(l=l, c1=pars['c1'].val,
                                                     c2=pars['c2'].val, c3=pars['c3'].val, c4=pars['c4'].val,
                                                     x0=pars['x0'].val, gamma=pars['gamma'].val,
                                                     E0=pars['E0'].val, E3=pars['E3'].val, E4=pars['E4'].val,
                                                     b=pars['b'].val, alpha=pars['alpha'].val,mode = '2-order'))
                    f_min = np.nanmin(extinction_fit(l=l, c1=pars['c1'].val,
                                                     c2=pars['c2'].val, c3=pars['c3'].val, c4=pars['c4'].val,
                                                     x0=pars['x0'].val, gamma=pars['gamma'].val,
                                                     E0=pars['E0'].val, E3=pars['E3'].val, E4=pars['E4'].val,
                                                     b=pars['b'].val, alpha=pars['alpha'].val,mode = '2-order'))

                chi = 0
                if Av > 0:
                    for p in pars.values():
                        p_scale = 1
                        if p.name in ['c4']:
                            p_scale = Av
                        if p.val < p.vrange[0] * p_scale or p.val > p.vrange[1] * p_scale:
                            chi = -np.inf

                if f_x1 < 0 or Av < 0 or f_x1 < f_x2 or f_x1 <= f_max or f_x2 >= f_min:
                    chi = -np.inf

                if A_1/Av<0.1:
                    chi = -np.inf

                if A_5 / Av > 0.1:
                    chi = -np.inf


                return chi

            # define combined likelihood
            def log_probability_ir(theta, x, y):
                lp = log_prior(theta)
                if not np.isfinite(lp):
                    return -np.inf
                return lp + log_likelihood_ir(theta, x, y)


            def log_probability_sdss(theta, x, y):
                lp = log_prior(theta)
                if not np.isfinite(lp):
                    return -np.inf
                return lp + log_likelihood_sdss(theta, x, y)

        #run mcmc

        # part I
        # define mcmc fitting parameters
        ndim = np.sum([el.var for el in pars.values()])  # number of parameters
        nwalkers = 500
        nsteps = 2000
        par_names = [el.name for el in pars.values() if el.var]
        print('par_names',par_names)

        # set initial values and run mcmc and save calculation to mcmc.pkl
        if run_mcmc and 1:
            init = np.array([el.val for el in pars.values() if el.var])
            init_range = np.array([el.disp for el in pars.values() if el.var])

            pos = []
            for i in range(nwalkers):
                prob = -np.inf
                while np.isinf(prob):
                    rndm = np.random.randn(ndim)
                    wal_pos = init + init_range * rndm
                    prob = log_probability_ir(theta=wal_pos, x=x, y=y)
                pos.append(wal_pos)

            from multiprocessing import Pool

            with Pool() as pool:
                sampler = emcee.EnsembleSampler(nwalkers, ndim, log_probability_ir, pool=pool, args=(x, y))
                start = time.time()
                sampler.run_mcmc(pos, nsteps, progress=True)
                end = time.time()
                multi_time = end - start
                print("Multiprocessing took {0:.1f} seconds".format(multi_time))

            samples = sampler.chain[:, :, :]
            lnprob = sampler.get_log_prob()
            if 1:
                with open('./results/mcmc_'+Q.name+'.pkl', 'wb') as f:
                    pickle.dump((samples, lnprob), f)
        #calc stats mcmc

        with open('./results/mcmc_'+Q.name+'.pkl', 'rb') as f:
            (samples, lnprob) = pickle.load(f)
        nwalkers, nsteps = samples.shape[0], samples.shape[1]
        # make a plot for chain statistic - mean, dispersion, and random walker path
        if 0:
            means = np.zeros((ndim, nsteps))
            vars = np.zeros((ndim, nsteps))
            single = np.zeros((ndim, nsteps))
            chi2 = np.zeros((nsteps))
            chi2mean = np.zeros((nsteps))
            chi2max = np.zeros((nsteps))
            for i in range(nsteps):
                for j in range(ndim):
                    means[j, i] = np.mean(samples[:, i, j])
                    vars[j, i] = np.std(samples[:, i, j])
                    single[j, i] = samples[5, i, j]
            for i in range(nsteps):
                print('i=', i)
                chi2[i] = np.nanmin(-lnprob[i, :])
                chi2mean[i] = np.nanmean(-lnprob[i, :])
                chi2max[i] = np.nanmax(-lnprob[i, :])

            # chain stats
            print('chain stats')
            fig, ax = plt.subplots(nrows=1, ncols=ndim + 1, figsize=(3 * (ndim + 1), 3))
            if ndim > 1:
                i = 0
                for col in ax:
                    print(i)
                    if i < ndim:
                        col.errorbar(np.arange(nsteps), means[i, :], yerr=vars[i, :],
                                     fmt='-', color='black',
                                     markeredgecolor='black', markeredgewidth=2, capsize=2,
                                     ecolor='royalblue', alpha=0.7)
                        col.plot(np.arange(nsteps), single[i, :], color='red')
                        col.set_title(par_names[i])
                    else:
                        mask = chi2 > 0
                        col.plot(np.arange(nsteps)[mask], chi2[mask], color='red')
                        col.plot(np.arange(nsteps)[mask], chi2mean[mask], color='black')
                        col.plot(np.arange(nsteps)[mask], chi2max[mask], color='blue')

                        col.set_yscale('log')
                    col.axvline(nsteps * 0.8, ls='--', color='red')
                    i += 1

            else:
                i = 0
                ax[0].errorbar(np.arange(nsteps), means[i, :], yerr=vars[i, :],
                               fmt='-', color='black',
                               markeredgecolor='black', markeredgewidth=2, capsize=2,
                               ecolor='royalblue', alpha=0.7)
                ax[0].plot(np.arange(nsteps), single[i, :], color='red')
            plt.show()

        # cut the chain at 0.8 of total lenght
        burnin = int(nsteps * 0.9)
        chain = samples[:, burnin:, :].reshape((-1, ndim))
        chain_ln = -lnprob[burnin:, :].reshape(-1)
        # analyse the chain using chainconsumer

        # calculate and plot contours for parameters
        from chainconsumer import ChainConsumer

        c = ChainConsumer()
        if 1:
            c.add_chain(chain, parameters=par_names)
            c.plotter.plot( figsize="column")
            plt.close()
            res = c.analysis.get_summary(parameters=par_names)
            for p in par_names:
                if res[p][2] == None:
                    print(res[p])
                else:
                    print(p, ',', round(res[p][1], 3), ',', round(res[p][2] - res[p][1], 3), ',',
                          round(res[p][1] - res[p][0], 3))

            for p in pars.values():
                if p.var:
                    p.val = res[p.name][1]
                    # p.set_prior(res[p.name][1],res[p.name][2]-res[p.name][1],res[p.name][1]-res[p.name][0])
        if 1:
            ind_best = np.nanargmin(chain_ln)
            theta_best = chain[ind_best, :]

            theta_MPE = np.array([res[p][1] for p in par_names])
            print('theta_MPE', theta_MPE)
            print('theta_best',theta_best)
            ln = log_probability_ir(theta=theta_best, x=x, y=y)

        if 0:
            theta_for_plot = [res[p][1] for p in par_names]
        else:
            theta_for_plot = theta_best

        # plot the best fit solution
        print('plot the best fit solution')
        fit_quasar_spectrum(theta=theta_for_plot, debug=True)

        if 0:
            with open('./results/fit_results.txt','a') as file:
                line = ''
                for p in par_names:

                    if res[p][2] == None or res[p][0] == None:
                        err = 'None'
                    else:
                        err = (res[p][2] - res[p][0])/2
                    line += ', ' + str(res[p][1])  + ', ' + str(err)
                file.write(Q.name + line + '\n')
        #plt.close()
        #plt.show()

        #estimate Extinction parameters
        if 0:
            from fitting_model import derive_ext_parameters
            res2,par_names2 = derive_ext_parameters(chain=chain, pars=pars)

            with open('./results/extinction_results.txt', 'a') as file:
                line = ''
                for p in par_names2:

                    if res2[p][2] == None or res2[p][0] == None:
                        err = 'None'
                    else:
                        err = (res2[p][2] - res2[p][0]) / 2
                    line += ', ' + str(res2[p][1]) + ', ' + str(err)
                file.write(Q.name + line + '\n')
            file.close()
        plt.close()
        #plt.show()

plt.show()
