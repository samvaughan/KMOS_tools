from cube_tools import Cube
import numpy as np 
import voronoi_2d_binning as V
from stellarpops.tools import extractTools as ET
from astropy.io import fits
from scipy import ndimage

import glob

from ppxf import ppxf
import ppxf_util as util
#import sys

#from stellarpops.tools import fspTools as FT
#import lmfit_SPV as LMSPV
import scipy.constants as const
#from spectools import *
from stellarpops.tools import CD12tools as CT
#from python_utils import sky_shift
#import matplotlib.pyplot as plt

#import argparse

#from KMOS_tools import cube_tools as C

from tqdm import tqdm

#import kinematics_helper_functions as KHF
import numpy.ma as ma

#import plotting as P
import lmfit_SPV as LMSPV

class CubeKinematics(Cube):


    #A subclass of Cube to deal with the kinematics

    def __init__(self, cube, bins_spectra_path='/home/vaughan/Science/KCLASH/Kinematics/Kin_Results_fits_files/Bins_and_spectra', 
        fits_file_out_path='/home/vaughan/Science/KCLASH/Kinematics/Kin_Results_fits_files/Kinematic_and_Flux_measurements', 
        text_file_outname='/home/vaughan/Science/KCLASH/Kinematics/Kin_Results_txt_files',
        **kwargs):

        super(self.__class__, self).__init__(cube, **kwargs)
        self.bins_spectra_path = bins_spectra_path
        self.fits_file_out_path = fits_file_out_path
        self.text_file_out_path = text_file_outname

        #The extra attributes we'll create and fill
        #Question- should I just call either functions to make these or functions to load these here?

        #After the voronoi Binning
        self.x_coords_1d=None
        self.y_coords_1d=None
        self.x_coords_2d=None
        self.y_coords_2d=None

        self.bins_1d=None
        self.bins_2d=None

        self.nPixels=None
        self.spectra=None
        self.noise_spectra=None
        self.bin_mask=None


        #For PPXF
        self.rest_lamdas=None
        self.velscale=None
        self.vel=None
        self.vel_err=None
        self.sigmas=None
        self.sigmas_err=None
        self.weights=None
        self.chisqs=None
        self.gas_templates=None
        self.line_names=None


        #State variables
        self.has_voronoi_bins=False
        self.has_extracted_spectra=True
        self.has_extracted_noise_spectra=True
        self.emission_lines_been_fit=False



    def voronoi_bin_cube(self, SN_TARGET, save=True):

        """
        Take a KCLASH cube, and run Michele's voronoi binning code to bin it to a minimum signal to noise. Then extract the spectra and noise spectra
        from those bins
        Outputs are saved as $CUBENAME_bins_spectra.fits in folder $savepath.
        
        The output file has 4 fits extenstions:
            0. an N_spaxel x 3 array of columns x, y and binNumber. Each spaxel is assigned either a bin or -999 if it's unbinned
            1. an N_lamda array of the lamda values for each pixel in the spectral direction. It's just cube.lamdas
            2. an N_lamda x N_bins array of the spectrum corresponding to each bin
            3. an N_lamda x N_bins array of the noise spectrum corresponding to each bin


        Inputs:
            cube- a Cube object (from the Cube class in cube_tools)
            SN_TARGET- int. Target S/N of each bin 
            savepath- string. Folder to save output fits file to. 

        Returns:
            None
        """

        #Interpolate to 0.1 arcsecond sampling
        self.interpolate_point_1_arcsec_sampling()
        

        d_lam=self.lamdas[1]-self.lamdas[0]

        #Mask just around the H-alpha wavelength, to get the signal value
        signal_mask=self.get_spec_mask_around_wave(0.65628*(1+self.z), 0.001)

        #Not a typo- x and y axes are reversed
        self.y_coords_2d, self.x_coords_2d=np.indices((self.ny, self.nx))
        self.x_coords_1d=self.x_coords_2d.ravel()
        self.y_coords_1d=self.y_coords_2d.ravel()


        galmedian=np.nanmedian(self.data[signal_mask, :, :], axis=0)
        signal=np.abs(galmedian)*np.nansum(self.data[signal_mask, :, :]/galmedian, axis=0)*d_lam  
        noise=np.nansum(self.noise[signal_mask, :, :], axis=0)*d_lam    

        #Mask invalid things
        nan_mask=~((np.isfinite(noise))&(noise>0))

        #Do the binning
        binNum, xNode, yNode, xBar, yBar, sn, nPixels, scale=V.voronoi_2d_binning(self.x_coords_2d[~nan_mask].ravel(), self.y_coords_2d[~nan_mask].ravel(), signal[~nan_mask].ravel(), noise[~nan_mask].ravel(), SN_TARGET, plot=False, quiet=True)

        #Make sure everything left unbinned has a value of -999
        #The normal output of voronoi binning just omits these pixels, meaning we don't know the length of binNum beforehand. 
        all_bins=np.full_like(self.x_coords_1d, -999)
        all_bins[~nan_mask.ravel()]=binNum.copy() 
        self.bins_1d=all_bins
        self.bins_2d=self.bins_1d.reshape(self.ny, self.nx)

        self.bin_mask=np.where(self.bins_1d>=0.0)

        self.spectra, self.noise_spectra=ET.simple_extraction(self.y_coords_2d, self.x_coords_2d, self.bins_2d, self.data, self.noise**2, type='median')
        
        outname='{}/{}_bins_spectra.fits'.format(self.bins_spectra_path, self.object_name)

        if save:
            self.save_spectra_to_fits(self.x_coords_1d, self.y_coords_1d, self.bins_1d, self.spectra, self.noise_spectra, self.nPixels, outname)

        
        self.nPixels=nPixels


        self.has_extracted_spectra=True
        self.has_extracted_noise_spectra=True
        self.has_voronoi_bins=True

        
        return self.x_coords_1d, self.y_coords_1d, self.bins_1d, self.nPixels


    @staticmethod
    def save_spectra_to_fits(x, y, twoD_bins, lamdas, spectra, noise_spectra, outname, nPixels, overwrite=True):
 

        #Write the fits file
        hdu1 = fits.PrimaryHDU(np.column_stack((x.ravel(), y.ravel(), twoD_bins.ravel())))
        hdu2 = fits.ImageHDU(lamdas)
        hdu3 = fits.ImageHDU(spectra)
        hdu4 = fits.ImageHDU(noise_spectra)
        hdu5 = fits.ImageHDU(nPixels)
        new_hdul = fits.HDUList([hdu1, hdu2, hdu3, hdu4, hdu5])

        new_hdul.writeto('{}'.format(outname), overwrite=overwrite)

        return new_hdul

    def load_voronoi_bin_attributes(self, filename):
        """Load values of voronoi bins from a fits file and assign them to class attributes"""

        hdul=fits.open(filename)

        self.x_coords_1d, self.y_coords_1d, self.bins_1d=hdul[0].data
        #FIXME- assign lamdas here?
        self.lamdas=hdul[1].data
        self.spectra=hdul[2].data
        self.noise_spectra=hdul[3].data
        self.nPixels=hdul[4].data

        #assign the other attributes we need

        self.bin_mask=np.where(self.bins_1d>=0.0)
        self.bins_2d=self.bins_1d.reshape(self.ny, self.nx)
        self.x_coords_2d = self.x_coords_1d.reshape(self.ny, self.nx)
        self.y_coords_2d = self.y_coords_1d.reshape(self.ny, self.nx)

        self.has_extracted_spectra=True
        self.has_extracted_noise_spectra=True
        self.has_voronoi_bins=True

    #Emission Line Fitting
    @staticmethod
    def load_gas_templates(lamRange_galaxy, velscale, FWHM_gal):
        """
        Load the emission line templates we use for the ppxf fitting.
        Do this by loading a CvD stellar template, then log-rebinning to the appropriate velscale. Then we use this
        logLamda array to give us the correct length gas templates. 

        Inputs:
            lamRange_galaxy: List.  A two component list with the start and stop wavelength values of the galaxy spectrum. In Angstroms!
            velscale: Float. In a log-rebinned spectrum, the difference in wavelength between two adjacent pixels, converted to in km/s. 
            FWHM_gal. Float. The FWHM of the galaxy spectrum. Check a skyline
        Outputs:
            gas_templates: An N_templates x N_lamda array of template spectra
            line_names: An N_templates list of names for each line we have loaded
            line_wave: An N_templates list of the central wavelength of each emission line 
            lamRange_template: List. A two component list of the start and stop wavelengths of the templates. In Angstroms
        """
        cvd = glob.glob('/Data/stellarpops/CvD1.2/t*.ssp')
        cvd.sort()

        #CvD Templates are at resolution 2000, so work out lamda/R for the middle wavelength in your array
        FWHM_tem = np.median(lamRange_galaxy)/2000
        
        


        #Use Simon's CvDTools function to read in the CvD models and get them into proper units
        cvd_data=CT.loadCD12spec(cvd[0])
        #They're returned in Ryan's spectrum class. spec.lam is wavelengths, spec.flam is flux in lamda units
        cvd_lams=cvd_data.lam

        #Pad the templates so they're longer than the data
        pad=100
        lamRange_template=(lamRange_galaxy[0]-pad, lamRange_galaxy[-1]+pad)
        template_mask=np.where((cvd_lams>lamRange_template[0])&(cvd_lams<lamRange_template[1]))[0]
        cdelt=cvd_lams[10]-cvd_lams[9]    


        FWHM_dif = np.sqrt((FWHM_gal**2 - FWHM_tem**2).clip(0))
        sigma = FWHM_dif/2.355/cdelt # Sigma difference in pixels
        
        #Log Rebin one spectrum to get the length of the templates array right
        ssp=cvd_data.flam[0][template_mask]
        ssp = ndimage.gaussian_filter1d(ssp,sigma)
        sspNew, logLam_template, velscale = util.log_rebin(lamRange_template, ssp, velscale=velscale)

        gas_templates, line_names, line_wave = util.emission_lines(logLam_template, lamRange_galaxy, FWHM_gal)

        return gas_templates, line_names, line_wave, lamRange_template


    @staticmethod
    def get_bins_and_spectra(bins_spectra_path, object_name):

        spectra_noise_hdu=fits.open('{}/{}_bins_spectra.fits'.format(bins_spectra_path, object_name))
        bin_information=spectra_noise_hdu[0].data
        lamdas=spectra_noise_hdu[1].data
        spectra=spectra_noise_hdu[2].data
        noise_spectra=spectra_noise_hdu[3].data

        return bin_information, lamdas, spectra, noise_spectra


    def make_MEF_of_quantities(self, things, labels):

        
        hdu_extensions=[]
        for thing, label in zip(things, labels):
            hdu=fits.ImageHDU(self.display_binned_quantity(self.xcoords_1d[self.bin_mask], self.ycoords_1d[self.bin_mask], thing[self.bins_1d[self.bin_mask]]))
            hdu.header['QUANTITY']=label
            hdu_extensions.append(hdu)

        return hdu_extensions


    def fit_emission_lines(self, save=True, plot=False):

        """
        Fit emission lines to the spectra of a K-CLASH observation
        """
        # bin_information, lamdas, spectra, noise_spectra=self.get_bins_and_spectra(self.bins_spectra_path, self.object_name)

        if not self.has_extracted_spectra:
            raise AttributeError("You can't run fit_emission_lines without a set of spectra extracted from (e.g voronoi) bins")
        elif not self.has_extracted_noise_spectra:
            raise AttributeError("You can't run fit_emission_lines without a set of noise spectra extracted from (e.g voronoi) bins")

        nbins=self.spectra.shape[0]
        #Convert to angstroms and de-redshift
        self.rest_lamdas=self.lamdas*(10**4/(1+self.z))

        lamRange_galaxy=[self.rest_lamdas[0], self.rest_lamdas[-1]]
        


        #Get the velscale
        _, _, self.velscale=util.log_rebin(lamRange_galaxy, self.spectra[0, :])
        #And the FWHM of the galaxy
        FWHM_gal = 2.0/(1+self.z)

        #Load the gas templates
        self.gas_templates, self.line_names, line_wave, lamRange_template=self.load_gas_templates(lamRange_galaxy, self.velscale, FWHM_gal)


        

        #Empty arrays for results
        self.vel=np.empty(nbins)
        self.vel_err=np.empty(nbins)
        self.sigmas=np.empty(nbins)
        self.sigmas_err=np.empty(nbins)
        self.weights=np.empty((nbins, self.gas_templates.shape[-1]))
        self.chisqs=np.empty(nbins)



        
        #Fit each spectrum with pPXF
        for i, (spectrum, noise_spectrum) in enumerate(tqdm(zip(self.spectra, self.noise_spectra), leave=False)):


            #logrebin the galaxy spectrum and noise spectrum
            log_galaxy, logLam_galaxy, self.velscale = util.log_rebin(lamRange_galaxy, spectrum)
            log_noise, logLam_galaxy, _=util.log_rebin(lamRange_galaxy, noise_spectrum)


            ##Mask pixels
            #Make a mask the correct length...
            mask=np.ones_like(log_galaxy, dtype='bool')

            run_ppxf=True
            if not np.any(log_noise>0.0):
                run_ppxf=False

            #... and mask all noise elements which are 0...
            zero_noise=log_noise<=0.0
            log_noise[zero_noise]=np.nanmedian(log_noise)
            mask[zero_noise]=0.0

            #... or NANs
            nan_noise=~np.isfinite(log_noise)
            log_noise[nan_noise]=np.nanmedian(log_noise)
            mask[nan_noise]=0.0


            #Work out the wavelength difference (in km/s) between the start of the templates and the start of the galaxy. 
            dv=(const.c/1000.0)*np.log(lamRange_template[0]/lamRange_galaxy[0])

            #PPXF starting guess
            start=[0,3*self.velscale[0]]

            #Call ppxf
            if run_ppxf:
                pp = ppxf.ppxf(self.gas_templates, log_galaxy, log_noise, self.velscale, start, plot=False, moments=2, degree=4, vsyst=dv, clean=True, quiet=True)
                chi2=pp.chi2
            else:
                #If we have a bad spectrum, set the chi_squared to be huge and catch it in the bad results below 
                chi2=10000000

            #Only save the results if the ChiSquared is good


            if chi2<5:
                self.vel[i]=pp.sol[0]
                self.vel_err[i]=pp.error[0]*np.sqrt(pp.chi2)
                self.sigmas[i]=pp.sol[1]
                self.sigmas_err[i]=pp.error[1]*np.sqrt(pp.chi2)

                self.chisqs[i]=pp.chi2

                self.weights[i, :]=pp.weights

                # if plot:
                #     line, =ax.plot(self.lamdas, spectrum)
                #     ax.plot(self.lamdas, pp.bestfit, c=line.get_color(), linewidth=2.0)

            else:
                print("Bin {} returns a bad result".format(i))

                self.vel[i]=np.nan
                self.vel_err[i]=np.nan
                self.sigmas[i]=np.nan
                self.sigmas_err[i]=np.nan

                self.chisqs[i]=pp.chi2

                self.weights[i, :]=[np.nan]*len(pp.weights)


                # if plot:
                #     line, =ax.plot(self.lamdas, spectrum, linestyle='dotted')
                #     ax.plot(self.lamdas, pp.bestfit, c=line.get_color(), linewidth=2.0,  linestyle='dotted')



        if save:
            self.save_ppxf_results_to_text_file(self.text_file_outname)
            self.save_ppxf_results_to_MEF(self.fits_file_out_path)

        self.emission_lines_been_fit=True




    def save_ppxf_results_to_text_file(self, out_file_path):
    #Clear the filename
    
        results_filename='{}/{}_results.txt'.format(out_file_path, self.object_name)
        #Saving things to our text file
        with open(results_filename, "w") as f:             
            np.savetxt(f, np.column_stack((self.vel, self.vel_err, self.sigmas, self.sigmas_err, self.weights, self.chisqs)))#, delimiter='\t', newline='\t')



    def save_ppxf_results_to_MEF(self, out_file_path):
        #The list which we'll fill with fits extensions
        hdu_extensions=[]

        #Saving everything to a Multi extension fits file
        kinematic_quantities=['Velocity', 'VelocityError', 'Sigma', 'SigmaError', 'Chisq']

        #Empty primary HDU
        #Just to have a header containing all the info
        hdu_primary=fits.PrimaryHDU()
        for i, label in enumerate(['VoronoiBins'] + kinematic_quantities + self.line_names.tolist()):
            hdu_primary.header['EXT{}'.format(i+1)]=label
        hdu_extensions.append(hdu_primary)

        #Extenstion with the voronoi bins
        hdu_bins=fits.ImageHDU(self.display_binned_quantity(self.x_coords_1d[self.bin_mask], self.y_coords_1d[self.bin_mask], self.bins[self.bin_mask]))
        hdu_bins.header['QUANTITY']='VoronoiBins'
        hdu_extensions.append(hdu_bins)

        #Kinematic Quantities
        hdu_extensions.extend(self.make_MEF_of_quantities([self.vel, self.vel_err, self.sigmas, self.sigmas_err, self.chisqs], kinematic_quantities))

        #Weights of templates
        hdu_extensions.extend(self.make_MEF_of_quantities(self.weights.T, self.line_names))

        final_fits_file = fits.HDUList(hdu_extensions)
        final_fits_file.writeto('{}/{}_kin_flux.fits'.format(out_file_path, self.object_name), overwrite=True)



        





    def display_results(self):

        #FIXME

        fig, ax=P.display_kinematics(self, self.fits_file_out_path, self.nPixels)




    def mask_2D_map(self, attribute, mask):

        map_2d=self.getattr(attribute)   
        masked=map_2d[mask]=np.nan     
        self.setattr(attribute, masked)



    #Fit the kinematic map

    def fit_map(self):


        start_r0=self.table['r50_int_z']/self.table['pixscale']

        #hdu=fits.open('{}/{}_kin_flux.fits'.format(fits_file_out_path, self.object_name))
        #spectra_noise_hdu=fits.open('{}/{}_bins_spectra.fits'.format(bins_spectra_path, self.object_name))

        # bin_information=spectra_noise_hdu[0].data
        # x, y, bins=bin_information.T
        # #Load the arrays and create the x and y arrays
        # TwoD_bins=hdu[1].data
        # vel_data=hdu[2].data
        # vel_errs=hdu[3].data

        bad_bins=np.where(self.nPixels>15.0)
        # #Get indices which correspond to the bad bins
        mask=np.isin(self.twoD_bins, bad_bins)

        self.mask_2D_map('vel', mask)
        self.mask_2D_map('vel_errs', mask)
  
        #Get to a velocity around 0
        self.vel-=np.nanmedian(self.vel)

        self.vel_errs[self.vel_errs>50.0]=np.nan

        data=ma.masked_invalid(self.vel)
        noise=ma.array(self.vel_errs, mask=data.mask)
        max_y, max_x=data.shape



        fit_params=LMSPV.Parameters()
        #Theta controls how elliptical the contours are
        #it's arccos(short axis / long axis)
        fit_params.add('theta', value=17.0, min=1, max=np.arccos(1/5.)*180.0/np.pi, vary=True)
        fit_params.add('xc', value=max_x/2, min=10, max=20, vary=True)
        fit_params.add('yc', value=max_y/2, min=10, max=20, vary=True)
        fit_params.add('r0', value=start_r0, min=1.0, max=100.0, vary=True)
        fit_params.add('log_s0', value=8, min=3.0, max=8.0, vary=True)
        fit_params.add('v0', value=np.nanmedian(data), min=-300.0, max=300.0)
        fit_params.add('PA', value=341.3, min=0.0, max=360.0, vary=True)


        # a=1.0
        # s=3.0

        #Select the parameters we're varying, ignore the fixed ones
        #variables=[thing for thing in fit_params if fit_params[thing].vary]
        #ndim=len(variables)
        #Vice versa, plus add in the fixed value
        #fixed=[ "{}={},".format(thing, fit_params[thing].value) for thing in fit_params if not fit_params[thing].vary]


        nwalkers=50
        nsteps=1000
        

        minimiser = LMSPV.Minimizer(objective_function, fit_params, fcn_args=(data, noise, self.xcoords_1d, self.ycoords_1d, self.bins_1d))
        quick_result = minimiser.minimize(method='differential_evolution')




        start_vals=np.array([quick_result.params[thing].value for thing in quick_result.params if quick_result.params[thing].vary])
        p0=np.array([start_vals+ 1e-2*np.random.randn(len(start_vals)) for i in range(nwalkers)])


        minimiser = LMSPV.Minimizer(lnprob, fit_params, fcn_args=(data, noise, self.xcoords_1d, self.ycoords_1d, self.bins_1d))
        print('Running emcee')
        fit_result = minimiser.emcee(steps=nsteps, nwalkers=nwalkers, params=fit_params, pos=p0, emcee_sample_kwargs={'progress':True})


        # fit_params.add('ln_a', value=1.0, min=-2, max=5, vary=True)
        # fit_params.add('ln_s', value=1.0, min=-2.0, max=5.0, vary=True)

        # start_vals=np.append(start_vals, [1.0, 1.0])
        # p0=np.array([start_vals+ 1e-2*np.random.randn(len(start_vals)) for i in range(nwalkers)])
        
        # # start_vals=np.array([fit_params[thing].value for thing in fit_params if fit_params[thing].vary])
        # # p0=np.array([start_vals+ np.random.randn(len(start_vals)) for i in range(nwalkers)])


        # minimiser = LMSPV.Minimizer(KHF.lnlike_covariance, fit_params, fcn_args=(data, noise, x, y, bins))
        # # #minimiser = LMSPV.Minimizer(KHF.lnprob, fit_params, fcn_args=(data, noise, x, y, bins))
        # print('Running emcee')
        # covar_result = minimiser.emcee(steps=nsteps, nwalkers=nwalkers, params=fit_params, pos=p0, emcee_sample_kwargs={'progress':True})


        #if best_result is not None:
        #result = minimiser.emcee(steps=2000, nwalkers=50)
        #LMSPV.report_fit(result)
        #chisq = result.chisqr/(result.ndata-result.nvarys)


        (fig, ax), stds=P.plot_model(self, fit_result.params, data, noise, fit_result.flatchain.values, self.x_coords_2d, self.y_coords_2d, self.bins_2d)

        best_model=KHF.velfield(fit_result.params, data)
        max_v=np.max(best_model-np.nanmedian(best_model))
        min_v=np.min(best_model-np.nanmedian(best_model))
        max_v_err=np.max(best_model-np.nanmedian(best_model)+stds)
        min_v_err=np.min(best_model-np.nanmedian(best_model)-stds)
        LMSPV.report_fit(fit_result)


        #fig.savefig('/home/vaughan/Science/KCLASH/Kinematics/KinMapFits/{}_fit.pdf'.format(GalName), bbox_inches='tight')

        # #Ignore errors here
        # with open(results_filename, 'a') as f:
        #     f.write('{}\t{}\t{}\t{}\n'.format(GalName, self.table['M*'][0], max_v, chisq))
        return (fig, ax), fit_result, (max_v, max_v_err, min_v, min_v_err), data, noise


#Likelihood function here: saves pickling the parameters dictionary
def _lnprob(T, theta, var_names, bounds, data, errors, x, y, bins, a, s):

    #Log prob function. T is an array of values

    
    assert len(T)==len(var_names), 'Error! The number of variables and walker position shapes are different'

    #Prior information comes from the parameter bounds now
    if np.any(T > bounds[:, 1]) or np.any(T < bounds[:, 0]):
        return -np.inf


    #make theta from the emcee walker positions
    for name, val in zip(var_names, T):
        theta[name].value = val


    ll=KHF.test_lnlike(theta, data, errors, x, y, bins, a, s)
    return ll



def velfield(params, data):

    """
    Make a 2d array containing a velocity field
    """

    #This is the 'angular eccentricity'
    #Shapes the flattening of the elliptical coordinates
    #cos(theta) is just b/a for the ellipse
    #sin(theta) is sqrt(1-b**2/a**2), or the eccentricity e
    #Should limit a=5b for reasonable galaxies
    theta=params['theta'].value
    
    xc=params['xc'].value
    yc=params['yc'].value
    r0=params['r0'].value
    s0=params['log_s0'].value
    v0=params['v0'].value
    PA=params['PA'].value

    #Get the integer shift and the float shift
    xc_int=int(xc)
    yc_int=int(yc)
    xc_float=xc-xc_int
    yc_float=yc-yc_int

    theta_rad=theta*np.pi/180.

    #xcen=min(max(0,xc),sz[1]-1)
    #ycen=min(max(0,yc),sz[0]-1)

    Y, X=np.indices(data.shape)
    R = np.sqrt((X-xc_int)**2 + ((Y-yc_int)/np.cos(theta_rad))**2)

    #Get the simple axisymettric velfield, then scale by (X-Xc)/R)
    velfield = v_circ_exp_quick(R, params)*(X-xc_int)/(R*np.sin(theta_rad))
    
    velfield[yc_int, xc_int] = 0   

    velfield_rotated_and_offset = ndi.rotate(velfield,-PA, reshape=False,mode='nearest') + v0

    
    velfield_final = ndi.shift(velfield_rotated_and_offset,[yc_float, xc_float], mode='nearest')

    return velfield_final


def aggregate_bins(model, x, y, bins):

    vals=np.empty(len(np.unique(bins[bins>=0])))
    for i in np.unique(bins[bins>=0]):
        mask=np.where(bins==i)
        vals[i]=np.mean(model[mask])

    return vals

def v_circ_exp_quick(R,params):

    """
    Make a rotation curve, following:
    # exponential disk model velocity curve (Freeman 1970; Equation 10)

    # v^2 = R^2*!PI*G*nu0*a*(I0K0-I1K1)
    """
    

    # param = [r0,s0,v0,roff,theta]

    # r0 = 1/a = disk radii

    # R = radial distance 

    # roff = offset of velocity curve

    # from 0 -> might want to set to fixed at 0?)

    # s0 = nu0 = surface density constant (nu(R) = nu0*exp(-aR))

    # v0 is the overall velocity offset

    

    # G

    G = 6.67408e-11 #m*kg^-1*(m/s)^2
    G = G*1.989e30  #m*Msol^-1*(m/s)^2
    G = G/3.0857e19 #kpc*Msol^-1(m/s)^2
    G = G/1000./1000.

    

    # parameters

    R0=params['r0'].value
    log_s0=params['log_s0'].value
    s0  = 10**log_s0



    

    

    # evaluate bessel functions (evaluated at 0.5aR; see Freeman70)

    half_a_R=(0.5*(R)/R0)

    #temp[temp>709.]=709.

    #Bessel Functions
    I0K0 = iv(0,half_a_R)*kv(0,half_a_R)
    I1K1 = iv(1,half_a_R)*kv(1,half_a_R)

    #bsl  = I0K0 - I1K1

    

    # velocity curve

    V_squared  =  R*((np.pi*G*s0)*(I0K0 - I1K1)/R0)

    V=np.sqrt(V_squared)   

    return V




def get_slit_profile(params, data, model, noise):

    
    xc=params['xc'].value
    yc=params['xc'].value
    max_y, max_x=data.shape
    

    #Slit along PA: y=mc+c, where m=tan(y/x), c=yc-xc*tan(theta)
    x_slit=np.arange(max_x).astype(int)
    y_slit=np.ones_like(x_slit)*max_y/2+1
    #y_slit=x_slit*np.tan(PA*np.pi/180.0)+yc-(xc*np.tan(PA*np.pi/180.0))

    #mask so we don't go outside of the data
    #subtract 0.5 to ensure that the last valye of y doesn't get rounded up and throw an error
    #mask=(y_slit>0)&(y_slit<max_y-0.5)

    #The x and y values along the slit
    #Not sure why this is necessary! Come back to!
    #x_slit_indices=x_slit[mask]#-1-int(np.abs(xc-max_x/2))
    #y_slit_indices=np.around(y_slit[mask]).astype(int)


    #take a stripe along the PA axis of 5 pixels (0.5") and median/mean combine
    #with inverse variance weighting
    s=data.shape
    v_profile=model[s[0]/2, :]

    ivars=1./(noise[s[0]/2-2:s[0]/2+3, :]**2)
    vels=data[s[0]/2-2:s[0]/2+3, :]
    v_obs=np.nansum(ivars*vels, axis=0)/np.nansum(ivars, axis=0)
    #v_obs=np.nanmean(, axis=0)
    #v_err=np.sqrt(np.nansum(noise[s[0]/2-2:s[0]/2+3, :]**2, axis=0))
    v_err=np.sqrt(1./np.nansum(ivars, axis=0))



    return v_profile, v_obs, v_err, [x_slit, y_slit]



def objective_function(params, data, errors, x, y, bins):

    model=velfield(params, data)

    #Bin the model in the same way as the data

    binned_model=display_binned_quantity(x, y, model.ravel()[bins])

    residuals=((binned_model[~data.mask]-data[~data.mask])/errors[~data.mask]).flatten()

    return ma.getdata(residuals)

def lnprob(params, data, errors, x, y, bins):

    # a=np.exp(params['lnA'].value)
    # tau=np.exp(params['lnTau'].value)
    # gp = george.GP(a * kernels.Matern32Kernel(tau, ndim=2))

    # T=np.array([np.ravel(x[~data.mask]), np.ravel(y[~data.mask])]).T
    # E=errors[~data.mask].ravel()    
    # gp.compute(T, E)

    model=velfield(params, data)

    #Bin the model in the same way as the data

    binned_model=display_binned_quantity(x, y, model.ravel()[bins])


    residuals=(((binned_model[~data.mask]-data[~data.mask])/errors[~data.mask]).flatten())**2

    likelihood=-0.5*np.sum(residuals) #- 0.5*np.sum(np.log(errors[~data.mask]))

   
    return likelihood



def lnlike_covariance(params, data, errors, x, y, bins):
    

    a=np.exp(params['ln_a'])
    s=np.exp(params['ln_s'])

    model=velfield(params, data)

    #Bin the model in the same way as the data
    bin_mask=np.where(bins>=0)
    binned_model=display_binned_quantity(x[bin_mask], y[bin_mask], model[bin_mask])

    residuals=(binned_model[~data.mask]-data[~data.mask])

    r2=(x[~data.mask, None]-x[None, ~data.mask])**2+(y[~data.mask, None]-y[None, ~data.mask])**2

    C=np.diag(errors[~data.mask]**2) + a*np.exp(-0.5*r2/s*s)

    factor, flag=cho_factor(C)

    logdet=2*np.sum(np.log(np.diag(factor)))

    lnlike=-0.5*(np.dot(residuals, cho_solve((factor, flag), residuals))+ logdet + len(x)*np.log(2*np.pi))


    return lnlike





def display_binned_quantity(x, y, quantity):

    """
    Display pixels at coordinates (x, y) coloured with "counts".
    This routine is fast but not fully general as it assumes the spaxels
    are on a regular grid. This needs not be the case for Voronoi binning.

    Edited from Michele's function '_display_pixels'

    """
    x=x.ravel()
    y=y.ravel()
    pixelSize=1.0

    xmin, xmax = np.min(x), np.max(x)
    ymin, ymax = np.min(y), np.max(y)
    nx = int(round((xmax - xmin)/pixelSize) + 1)
    ny = int(round((ymax - ymin)/pixelSize) + 1)
    img = np.full((nx, ny), np.nan)  # use nan for missing data
    j = np.round((x - xmin)/pixelSize).astype(int)
    k = np.round((y - ymin)/pixelSize).astype(int)
    img[j, k] = quantity

    return img


