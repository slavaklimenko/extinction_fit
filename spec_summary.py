import numpy as np
from spectrum_model import *

path_to_transmission_files = './data/filters/'

def mag_flux_conv(AB, ref=48.6, offset=0):
    return 10 ** (-(AB + ref + offset) / 2.5) / 1e-23
def mag_flux_err(m,merr,offset=0):
    return -mag_flux_conv(m + merr+offset) + mag_flux_conv(m+offset)

class sys():
    def __init__(self, z_abs=-1,z_qso=-1, pars_arch = [], AvMW = -1,ref = None):
        self.z_abs = z_abs
        self.z_qso = z_qso
        self.pars_arch = pars_arch
        self.AvMW = AvMW
        self.ref = ref
        self.photo_data = {}
    def get_data(self):
        return (self.z_abs, self.z_qso, self.pars_arch, self.AvMW)
    def add_sdss_photometry(self,band='u',mag=19,mag_err=0.02):
        if band == 'u':
            self.photo_data['SDSSu'] = photometry(l=3680 / 1e4, lmin=3055 / 1e4, lmax=4030 / 1e4, f=mag_flux_conv(mag, offset=-0.04),
                                       err=mag_flux_err(mag-0.04,merr=mag_err), name='SDSSu',
                                       transmission_filename=path_to_transmission_files+'SLOAN_SDSS.u.dat')
        elif band == 'g':
            self.photo_data['SDSSg'] = photometry(l=5180 / 1e4, lmin=3797 / 1e4, lmax=5553 / 1e4, f=mag_flux_conv(mag),
                       err=mag_flux_err(mag,merr=mag_err), name='SDSSg',
                                       transmission_filename=path_to_transmission_files+'SLOAN_SDSS.g.dat')
        elif band == 'r':
            self.photo_data['SDSSr'] = photometry(l=6530 / 1e4, lmin=5418 / 1e4, lmax=6994 / 1e4, f=mag_flux_conv(mag),
                                                      err=mag_flux_err(mag,merr=mag_err), name='SDSSr',
                                       transmission_filename=path_to_transmission_files+'SLOAN_SDSS.r.dat')
        elif band == 'i':
            self.photo_data['SDSSi'] = photometry(l=7105 / 1e4, lmin=6692 / 1e4, lmax=8400 / 1e4, f=mag_flux_conv(mag),
                                               err=mag_flux_err(mag,merr=mag_err), name='SDSSi',
                                       transmission_filename=path_to_transmission_files+'SLOAN_SDSS.i.dat')
        elif band == 'z':
            self.photo_data['SDSSz'] = photometry(l=8680 / 1e4, lmin=7964 / 1e4, lmax=10873 / 1e4, f=mag_flux_conv(mag),
                                               err=mag_flux_err(mag,merr=mag_err), name='SDSSz',
                                       transmission_filename=path_to_transmission_files+'SLOAN_SDSS.z.dat')

    def add_2mass_photometry(self,band= 'J', f=None, err=None, mag=None):
        if band == 'J':
            self.photo_data['2MASSJ'] = photometry(nu=2.4e14, lmin=10806.47 / 1e4, lmax=14067.97 / 1e4,
                                            f=f, err=err, mag=mag, name='2MASSJ',
                                           transmission_filename=path_to_transmission_files+'2MASS_2MASS.J.dat')
        elif band == 'H':
            self.photo_data['2MASSH'] = photometry(nu=1.82e14, lmin=14787.38 / 1e4, lmax=18231.02 / 1e4,
                                                                  f=f, err=err, mag=mag, name='2MASSH',
                                       transmission_filename=path_to_transmission_files+'2MASS_2MASS.H.dat')
        elif band == 'K':
            self.photo_data['2MASSK'] = photometry(nu=1.38e14, lmin=19543.69 / 1e4, lmax=23552.40 / 1e4,
                                                                   f=f, err=err, mag=mag, name='2MASSK',
                                       transmission_filename=path_to_transmission_files+'2MASS_2MASS.Ks.dat')
    def add_UKIRT_photometry(self,band= 'Y', mag=None,err=None):
        # from https://about.ifa.hawaii.edu/ukirt/calibration-and-standards/astronomical-utilities/zero-mag-fluxes-and-conversions/
        if band == 'Y':
            self.photo_data['UKIRT-Y'] = photometry(l=1.03, lmin=9790 / 1e4, lmax=10810 / 1e4,
                                f=mag_flux_conv(AB=mag,offset=0.634), err=mag_flux_err(mag,err,offset=0.634),
                                name='UKIRT-Y', transmission_filename=path_to_transmission_files+'UKIRT_UFTI.Y.dat')
        if band == 'J':
            self.photo_data['UKIRT-J'] = photometry(l=1.25, lmin=11690/1e4,lmax=13280/1e4,
                                                    f=mag_flux_conv(AB=mag, offset=0.938),
                                                    err=mag_flux_err(mag, err, offset=0.938),
                                                    name='UKIRT-J',
                                                    transmission_filename=path_to_transmission_files + 'UKIRT_UFTI.J.dat')
        if band == 'H':
            self.photo_data['UKIRT-H'] = photometry(l=1.64, lmin=14920/1e4,lmax=17840/1e4,
                                                    f=mag_flux_conv(AB=mag, offset=1.379),
                                                    err=mag_flux_err(mag, err, offset=1.379),
                                                    name='UKIRT-H',
                                                    transmission_filename=path_to_transmission_files + 'UKIRT_UFTI.H.dat')
        if band == 'K':
            self.photo_data['UKIRT-K'] = photometry(l=2.21,lmin=20290/1e4,lmax=23800/1e4,
                                                    f=mag_flux_conv(AB=mag, offset=1.9),
                                                    err=mag_flux_err(mag, err, offset=1.9),
                                                    name='UKIRT-K',
                                                    transmission_filename=path_to_transmission_files + 'UKIRT_UFTI.K.dat')
    def add_wise_photometry(self, band='W1', f=None, err=None,mag=None):
        if band == 'W1':
            self.photo_data['WISE1'] = photometry(nu=8.94e13, f=f, err=err, mag=mag, name='WISE1', lmin=27540.97 / 1e4,
                        lmax=38723.88 / 1e4, transmission_filename=path_to_transmission_files+'WISE_WISE.W1.dat')
        elif band == 'W2':
            self.photo_data['WISE2'] = photometry(nu=6.51e13, f=f, err=err,mag=mag, name='WISE2', lmin=39633.26 / 1e4,
                        lmax=53413.60 / 1e4, transmission_filename=path_to_transmission_files+'WISE_WISE.W2.dat')
        elif band == 'W3':
            self.photo_data['WISE3'] = photometry(nu=2.59e13, f=f, err=err, mag=mag,name='WISE3', lmin=74430.44/1e4,
                            lmax=172613.43/1e4, transmission_filename=path_to_transmission_files+'WISE_WISE.W3.dat')
        elif band == 'W4':
            self.photo_data['WISE4'] = photometry(nu=1.36e13, f=f, err=err,mag=mag, name='WISE4', lmin=195200.83/1e4,
                        lmax=279107.24/1e4, transmission_filename=path_to_transmission_files+'WISE_WISE.W4.dat')
    def add_miri_photometry(self, band='1A', f=None, err=None):
        if band == '1A':
            self.photo_data['MIRI1A'] = photometry(l=5.32, f=f, err=err,name='MIRI1A',lmin=4.9,lmax =5.7,
                                               transmission_filename=path_to_transmission_files+'JWST_MIRI.1A.dat')

    def add_GALEX_photometry(self, band='NUV', f=None, err=None,mag=None):
        if band == 'NUV':
            if mag=='None':
                self.photo_data['NUV'] = photometry(l=0.234, f=f, err=err, name='NUV', lmin=0.1693, lmax=0.3007,
                                                   transmission_filename=path_to_transmission_files + 'GALEX_GALEX.NUV.dat')
            else:
                mag = float(mag)
                f_lam = 10 ** (-(mag-20.08) / 2.5) * 2.06 * 1e-16  # erg/cm2/s/A
                fnorm = (0.234 * 1e4) ** 2 / 3e18 * 1e-17 / 1e-23  # in 1e-17 erg/s/cm2/A
                f_nu = f_lam/1e-17*fnorm
                if err is None:
                    err = (10 ** (-(mag-0.5-20.08) / 2.5) * 2.06 * 1e-16-f_lam)/1e-17*fnorm  #err = 0.5mag
                else:
                    err = (10 ** (-(mag - err - 20.08) / 2.5) * 2.06 * 1e-16 - f_lam) / 1e-17 * fnorm
                print('flam,f_nu,,fnu_err', f_lam, f_nu, err)
                self.photo_data['NUV'] = photometry(l=0.234, f=f_nu, err=err, name='NUV', lmin=0.1693, lmax=0.3007,
                                                    transmission_filename=path_to_transmission_files + 'GALEX_GALEX.NUV.dat')
        if band == 'FUV':
            self.photo_data['FUV'] = photometry(l=0.1545, mag= None, f=f, err=err, name='FUV', lmin=0.134, lmax=0.1809,
                                                   transmission_filename=path_to_transmission_files + 'GALEX_GALEX.FUV.dat')


if 0:
    # add abs sys data
    folder = '/home/slava/science/codes/python/dust_extinction_model/'
    QSO_list = {}
    QSO_list['J1007'] = sys(z_abs=0.884, z_qso=1.047, pars_arch=[1, -0.55, 0.626, 3.34, 4.65, 1.449], AvMW=0.063,
                            ref='Zhou2010')
    QSO_list['J1007'].name = 'J1007+2853'
    # add photometry
    QSO_list['J1007'].add_GALEX_photometry(band='NUV', mag=22.75)
    QSO_list['J1007'].add_2mass_photometry(band='J', f=3.74e-4, err=4.8e-5)
    QSO_list['J1007'].add_2mass_photometry(band='H', f=5.2e-4, err=7.3e-5)
    QSO_list['J1007'].add_2mass_photometry(band='K',  f=7.93e-4, err=7e-5)
    QSO_list['J1007'].add_wise_photometry(band='W1', f=1.21e-3, err=7e-5)
    QSO_list['J1007'].add_wise_photometry(band='W2', f=2.33e-3, err=7e-5)
    QSO_list['J1007'].add_miri_photometry(band='1A', f=2.44e-3, err=1e-4)
    #spectral data
    QSO_list['J1007'].path_sdss_spec = folder + 'spectrum/spec-10453-58136-0558.fits'

############

if 0:
    QSO_list['J1459'] = sys(z_abs=1.39, z_qso=3.01, pars_arch=[1, 0.09, 0.47, 1.46, 4.66, 1.11], AvMW=0.132,
                            ref='unknowm')
    QSO_list['J1459'].name = 'J1459+0024'
    # add photometry
    QSO_list['J1459'].add_sdss_photometry('u',20.67,0.07)
    QSO_list['J1459'].add_sdss_photometry('g', 19.09, 0.03)
    QSO_list['J1459'].add_sdss_photometry('r', 18.46, 0.02)
    QSO_list['J1459'].add_sdss_photometry('i', 17.93, 0.03)
    QSO_list['J1459'].add_sdss_photometry('z', 17.63, 0.04)
    QSO_list['J1459'].add_2mass_photometry(band='J', f=6.68E-4, err=4e-5)
    QSO_list['J1459'].add_2mass_photometry(band='H', f=5.25E-4, err=6e-5)
    QSO_list['J1459'].add_2mass_photometry(band='K', f=7.09E-4, err=6e-5)
    QSO_list['J1459'].add_wise_photometry(band= 'W1', f=8.50E-4,err=7.97e-6)
    QSO_list['J1459'].add_wise_photometry(band= 'W2', f=1.03E-3,err=1.6e-4)
    #QSO_list['J1459'].add_wise_photometry(band= 'W3', f=4.98E-3,err=1.6e-4)
    #spectral data
    QSO_list['J1459'].path_sdss_spec = '/home/slava/science/research/Proposals/ESO/2026/XS_archive/XS_J1459/rebinned.dat'
    #QSO_list['J1459'].path_IR_spec = folder + ('spectrum/IR/spitzer-J0852.txt')
    # set masked regions for sdss spectra
    QSO_list['J1459'].mask_bad_pixels_sdss_spec = [4580,4740,4860,4904,5024,5090,5104,5564,6690,6714,7590,7600,7636,7660]
    QSO_list['J1459'].mask_bad_pixels_jwst_spec =  []
    QSO_list['J1459'].mask_bad_pixels_spitzer_spec = []
    QSO_list['J1459'].w_norm = 18

def load_list(list_name = 'qso_list_init.csv',sdss_folder=''):
    QSO = {}
    from astropy.table import Table, join, hstack
    lst = Table.read(list_name)
    for i in range(len(lst)):
        s_name = lst['target'][i]
        s = sys(z_abs = lst['zabs'][i],  z_qso = lst['zqso'][i], AvMW = lst['AV_MW'][i], ref = 'unknown')
        s.name = s_name
        # photometry
        for el in ['J','H','K']:
            if lst[el][i] != 'None':
                s.add_2mass_photometry(band=el, mag=  lst[el][i])
        for el in ['W1', 'W2']:
            if lst[el][i] != 'None':
                mag = lst[el][i]
                s.add_wise_photometry(band=el, mag=mag)
        if lst['NUV'][i] != 'None' and 1:
            print('NUV',lst['NUV'][i])
            p = lst['NUV'][i]
            s.add_GALEX_photometry(band='NUV', mag=lst['NUV'][i],err=0.2)

        # spectral data
        s.path_sdss_spec = sdss_folder + lst['sdss_name'][i]+'.fits'
        s.mask_bad_pixels_sdss_spec = []
        QSO[s_name] = s
    return QSO

list_name = 'qso_list_init.csv'
QSO_list = load_list(list_name = list_name, sdss_folder='./sdss/')

if __name__ == '__main__':

    if 1:
        list_name = 'qso_list_init.csv'
        s = load_list(list_name = list_name, sdss_folder='./sdss/')
        print(s)