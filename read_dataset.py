
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
from matplotlib import rcParams
from astropy.io import ascii, fits
rcParams['font.family'] = 'serif'
from astropy.io import fits

#from dust_extinction_model.spectrum_model import *
from gordon_extinction import *
from spec_summary import QSO_list

emission_lines = {}
emission_lines['Lya'] = em_line('Lya', 1215.67, 40)
emission_lines['SiIV'] = em_line('SiIV', 1398.3, 10)
emission_lines['CIV'] = em_line('CIV', 1546.15, 60)
emission_lines['CIII]'] = em_line('CIII]', 1905.9, 40)
emission_lines['MgII'] = em_line('MgII', 2800, 100)
emission_lines['[NeV]'] = em_line('[NeV]', 3426, 20)
emission_lines['[OII]'] = em_line('[OII]', 3729, 20)
emission_lines['[NeIII]'] = em_line('[NeIII]', 3869, 15)
emission_lines['Hg'] = em_line('Hg', 4341, 60)
emission_lines['FeII'] = em_line('FeII', 4564, 61.9)
emission_lines['Hb'] = em_line('Hb', 4862, 80.4)
emission_lines['[OIII]5008'] = em_line('[OIII]5008', 5008.2, 40)
emission_lines['[OIII]4960'] = em_line('[OIII]4960', 4960.3, 40)
emission_lines['[OIII]4364'] = em_line('[OIII]4364', 4364, 40)
emission_lines['Ha'] = em_line('Ha', 6564.9, 47.4)
emission_lines['Pa'] = em_line('Pa', 1.875*1e4, 0.05*1e4)


def read_data(Q=QSO_list['J1007+2852'],f_units='F_lam',add_sdss=True,add_IR=False, add_hst=False):
    '''
    add_sdss - read SDSS spectroscopy
    add_IR - read IR spectroscopy
    ir_instr = 'miri','spitzer','none'
    '''

    # set normalization point for continuum
    Q.w_norm = Q.photo_data['WISE2'].l
    Q.w_norm_disp = 0.1
    Q.f_units = 'F_lam'

    #convert flux units to Jy or Flam
    if Q.f_units == 'F_lam':
        for el in Q.photo_data.values():
            fnorm =  1e-17 # notmalize to 1e-17 erg/s/cm2/A
            el.f = el.flam/fnorm
            el.err = el.flam_err/fnorm

    # correct photometry fluxes for galactic extinction
    if Q.AvMW > 0:
        for el in Q.photo_data.values():
            el.f *= np.exp(+Q.AvMW * extinction_gordon23(l=np.array(el.l), Rv=3.1) / 1.086)


    # read spectral data
    Q.sp = {}
    # read sdss spectrum
    if add_sdss:
        rebin =False
        qname = Q.path_sdss_spec
        if qname.endswith('fits'):
            hdulist = fits.open(qname)
            data = hdulist[1].data
            l = 10 ** data.field('loglam')  # in Angstrom
            fl = data.field('flux')  # in 1e-17 erg/cm2/s/Ang
            sig = (data.field('ivar')) ** (-0.5)
        elif qname.endswith('.dat'):
            data = np.loadtxt(qname)
            l = data[:,0]
            fl = data[:,1]
            sig =  data[:,2]
        fnorm = 1
        if f_units == 'Jy':
            fnorm = l ** 2 / 3e18 * 1e-17 / 1e-23  # in Jy
        col1 = l / 1e4 # convert to microns
        col2 = fl * fnorm
        col3 = sig * fnorm

        #rebin the data
        if rebin:
            n = 4
            x = rebin_arr(col1, n)
            y, err = rebin_weight_mean(col2, col3, n)
        else:
            x, y, err = col1, col2, col3

        # save spectral data as the "spectrum" object
        mask = y > 0
        Q.sp['sdss'] = spectrum(x=x[mask], y=y[mask], err=err[mask], name=qname)

        # correction for galactic extinction
        if Q.AvMW > 0:
            Q.sp['sdss'].y *= np.exp(+Q.AvMW * extinction_gordon23(l=np.array(Q.sp['sdss'].x), Rv=3.1) / 1.086)
            Q.sp['sdss'].err *= np.exp(+Q.AvMW * extinction_gordon23(l=np.array(Q.sp['sdss'].x), Rv=3.1) / 1.086)

        # correct spectral data for the offset with sdss photometry in sdss i filter
        #weight, model_flux = Q.photo_data['SDSSi'].weight_function(xgrid=Q.sp['sdss'].x, flux=Q.sp['sdss'].y,debug=False)
        #Q.sp['sdss'].y*=Q.photo_data['SDSSi'].f /model_flux
        #Q.sp['sdss'].err*=Q.photo_data['SDSSi'].f / model_flux

    return Q

# define data for fitting (data_x,data_y)
def set_data(Q=read_data(),debug=True):
    #add spectral data
    if 1:
        # mask absorption and emission features for sdss
        if 'sdss' in Q.sp.keys():
            # create a mask for sdss spectrum
            Q.sp['sdss'].mask_fit = Q.sp['sdss'].x > 0
            #mask emissions
            for l in emission_lines.values():
                Q.sp['sdss'].mask_fit[np.abs(Q.sp['sdss'].x * 1e4 / (1 + Q.z_qso) - l.l) < l.w] = False
            #mask region with poor SNR >1micron
            Q.sp['sdss'].mask_fit[Q.sp['sdss'].x > 1] = False
            #mask absorptions
            #for l in Q.mask_bad_pixels_sdss_spec:
            #    Q.sp['sdss'].mask_fit[np.abs(Q.sp['sdss'].x - l / 1e4) < 9 / 1e4] = False
            if 1:
                #
                f = savgol_filter(Q.sp['sdss'].y, 50, 1)
                #plt.plot(Q.sp['sdss'].x, f, c='blue', lw=1)
                #
                mask_sdss = ((Q.sp['sdss'].y - f) / Q.sp['sdss'].err) ** 2 < 2 ** 2
                Q.sp['sdss'].mask_fit[~mask_sdss] = False
            Q.sp['sdss'].res = np.mean(Q.sp['sdss'].x)/np.median(np.diff(Q.sp['sdss'].x))
            print('sdss res',Q.sp['sdss'].res)
            print('sdds npix:',np.sum(Q.sp['sdss'].mask_fit))
        # mask aborption and emission features for IR
        if 'miri' in Q.sp.keys():
            Q.sp['miri'].mask_fit = Q.sp['miri'].x > 0
            # mask silicate feature
            Q.sp['miri'].mask_fit = (Q.sp['miri'].x <= 8 * (1 + Q.z_abs))*(Q.sp['miri'].x<=3.5*(1+Q.z_qso))
            # mask emission lines
            for l in emission_lines.values():
                Q.sp['miri'].mask_fit[np.abs(Q.sp['miri'].x * 1e4 / (1 + Q.z_qso) - l.l) < l.w] = False
            # excluded absorption lines
            for l in Q.mask_bad_pixels_jwst_spec:
                Q.sp['miri'].mask_fit[np.abs(Q.sp['miri'].x - l ) < 0.01] = False
            Q.sp['miri'].res = np.mean(Q.sp['miri'].x) / np.median(np.diff(Q.sp['miri'].x))
            print('miri res', Q.sp['miri'].res)
        if 'spitzer' in Q.sp.keys():
            Q.sp['spitzer'].mask_fit = Q.sp['spitzer'].x > 0
            # mask silicate feature
            #Q.sp['spitzer'].mask_fit *= (np.abs(Q.sp['spitzer'].x - 10 * (1 + Q.z_abs))>2*(1+Q.z_abs)) * (Q.sp['spitzer'].x <= w_norm+5)
            Q.sp['spitzer'].mask_fit[Q.sp['spitzer'].x >  9 * (1 + Q.z_abs)] = 0
            # mask emission lines
            for l in emission_lines.values():
                Q.sp['spitzer'].mask_fit[np.abs(Q.sp['spitzer'].x * 1e4 / (1 + Q.z_qso) - l.l) < l.w] = False
            # excluded absorption lines
            for l in Q.mask_bad_pixels_spitzer_spec:
                Q.sp['spitzer'].mask_fit[np.abs(Q.sp['spitzer'].x - l) < 0.1] = False
            Q.sp['spitzer'].res = np.mean(Q.sp['spitzer'].x) / np.median(np.diff(Q.sp['spitzer'].x))
            print('spitzer res', Q.sp['spitzer'].res)
        if 'hst' in Q.sp.keys():
            Q.sp['hst'].mask_fit = Q.sp['hst'].x > 0
            #mask emission lines
            for l in emission_lines.values():
                Q.sp['hst'].mask_fit[np.abs(Q.sp['hst'].x * 1e4 / (1 + Q.z_qso) - l.l) < l.w] = False
            #mask region with poor SNR >1micron
            Q.sp['hst'].mask_fit[Q.sp['hst'].x<0.195] = False
            Q.sp['hst'].mask_fit[Q.sp['hst'].x > 0.47] = False
            #mask absorption lines
            for l in Q.mask_bad_pixels_hst_spec:
                Q.sp['hst'].mask_fit[np.abs(Q.sp['hst'].x - l / 1e4) < 9 / 1e4] = False
            Q.sp['hst'].res = np.mean(Q.sp['hst'].x)/np.median(np.diff(Q.sp['hst'].x))
            print('hst res',Q.sp['hst'].res)
            print('hst npix:',np.sum(Q.sp['hst'].mask_fit))

        # derive data: dat_x,data_y and data_err - lambda, flux, flux uncertainty
        data_x,data_y,data_err = [],[],[]
        data_res = 1000
        for q in Q.sp.values():
            if q.res >data_res:
                data_x = np.append(data_x,q.x[q.mask_fit])
                data_y = np.append(data_y,q.y[q.mask_fit])
                data_err = np.append(data_err,q.err[q.mask_fit])

        #rebin data to a common resolution
        def rebin_spectral_data(data_x=data_x,data_y=data_y,data_err=data_err,data_res=data_res,debug=False):
            nbin = int(np.log(np.nanmax(data_x)/np.nanmin(data_x))/np.log(1+1/data_res))
            x_new = np.array([np.nanmin(data_x)*(1+1/data_res)**k for k in range(nbin)])
            y_new = np.zeros_like(x_new)
            err_new = np.zeros_like(x_new)
            for k in range(nbin):
                if k==0:
                    mask = (data_x <= x_new[0]+(x_new[1]-x_new[0])/2)
                elif k<nbin-1:
                    mask = ((data_x>=x_new[k-1]+(x_new[k]-x_new[k-1])/2)*
                            (data_x<=x_new[k]+(x_new[k+1]-x_new[k])/2))
                else:
                    mask = (data_x >= x_new[k-1]+(x_new[k]-x_new[k-1])/2)
                if np.sum(mask)>0:
                    y_new[k] = np.mean(data_y[mask])
                    err_new[k] = np.sqrt(np.sum(data_err[mask]**2))/np.sum(mask)
            mask = y_new>0
            # show rebinning procedure
            if debug:
                xx_new = x_new[mask]
                yy_new = y_new[mask]
                err_new2 = err_new[mask]

                plt.subplots()
                plt.plot(data_x,data_y,label='data')
                plt.errorbar(x=xx_new,y=yy_new,yerr=err_new2,label='rebinned')
                plt.legend()
                plt.show()
            else:
                del(data_x,data_y,data_err)
                data_x=x_new[mask]
                data_y=y_new[mask]
                data_err =err_new[mask]
            return data_x,data_y,data_err

        data_x, data_y, data_err = rebin_spectral_data()
        #for low resolutions data - add without rebinning
        for q in Q.sp.values():
            if q.res < data_res:
                data_x = np.append(data_x, q.x[q.mask_fit])
                data_y = np.append(data_y, q.y[q.mask_fit])
                data_err = np.append(data_err, q.err[q.mask_fit])

        # normalize to flux at wavelength = w_norm
        data_norm = Q.photo_data['WISE2'].f
        data_y /= data_norm
        data_err /= data_norm

        # shift data_x to the absorption restframe
        data_x /= (1 + Q.z_abs)
        print('N points:',np.size(data_x),[np.sum(q.mask_fit) for q in Q.sp.values()])
        fitting_data = spectrum(x=data_x,y=data_y,err=data_err)
        fitting_data.res = data_res
        fitting_data.flux_norm = data_norm
    # check data for absorption/emission details
    if debug:
        plt.subplots()
        plt.title(Q.name)
        # plot unmasked spectra
        for q in Q.sp.values():
            plt.plot(q.x, q.y / data_norm, color='red', zorder=-10)
            plt.errorbar(x=q.x, y=q.y / data_norm, yerr=q.err / data_norm, ecolor='red', fmt='none', zorder=-10)
        plt.plot(Q.sp['sdss'].x,Q.sp['sdss'].y / data_norm,c='blue')
        # plot masked sdss spectra
        plt.plot(data_x * (1 + Q.z_abs), data_y, 'o')
        # plot ned fluxes
        for el in Q.photo_data.values():
            if 'SDSS' in el.name:
                plt.errorbar(x=el.l, y=el.f / data_norm,xerr=[[el.l-el.lmin],[el.lmax-el.l]],yerr=el.err/data_norm,fmt= 's', markerfacecolor='yellow',markeredgecolor='black')
            elif '2MASS' in el.name or 'UKIRT' in el.name:
                plt.errorbar(x=el.l, y=el.f / data_norm, xerr=[[el.l - el.lmin], [el.lmax - el.l]],yerr=el.err/data_norm, fmt='s',
                             markerfacecolor='green', markeredgecolor='black')
            elif 'WISE' in el.name:
                plt.errorbar(x=el.l, y=el.f / data_norm, xerr=[[el.l - el.lmin], [el.lmax - el.l]],yerr=el.err/data_norm, fmt='s',
                             markerfacecolor='red', markeredgecolor='black')
            else:
                plt.errorbar(x=el.l, y=el.f / data_norm,xerr=[[el.l-el.lmin],[el.lmax-el.l]],yerr=el.err/data_norm,fmt= 'o', markerfacecolor='magenta',markeredgecolor='black')
        #plot normalization wavelength
        plt.axvline(Q.w_norm, ls='--')
        plt.xlabel('Wavelength (Observed)')
        plt.ylabel('Flux')
        # set scale of y-axis
        #plt.yscale('log')
        plt.show()

    return Q,fitting_data


if __name__ == '__main__':

    Q = read_data()
    Q,s = set_data(Q=Q)