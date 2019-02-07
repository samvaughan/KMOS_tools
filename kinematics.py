from .cube_tools import Cube, twoD_Gaussian_with_slope
import numpy as np 
import voronoi_2d_binning as V
from stellarpops.tools import extractTools as ET
from astropy.io import fits
from scipy import ndimage
import scipy.signal as S

import glob

import ppxf.ppxf as PPXF
import ppxf.ppxf_util as util


import scipy.constants as const

from stellarpops.tools import CD12tools as CT


from tqdm import tqdm
from . import kin_functions as KF

import numpy.ma as ma
import lmfit as LM  

from . import plotting as P
import lmfit_SPV as LMSPV
import pandas as pd

import os
from . import settings

class CubeKinematics(Cube):


    #A subclass of Cube to deal with the kinematics

    def __init__(self, cube, bins_spectra_path=os.path.expanduser('~/Science/KCLASH/Kinematics/Kin_Results_fits_files/Bins_and_spectra'), 
        fits_file_out_path=os.path.expanduser('~/Science/KCLASH/Kinematics/Kin_Results_fits_files/Kinematic_and_Flux_measurements'), 
        text_file_out_path=os.path.expanduser('~/Science/KCLASH/Kinematics/Kin_Results_txt_files'),
        **kwargs):

        #Initialise the parent class
        super(self.__class__, self).__init__(cube, **kwargs)
        self.bins_spectra_path = bins_spectra_path
        self.fits_file_out_path = fits_file_out_path
        self.text_file_out_path = text_file_out_path

        #The extra attributes we'll create and fill
        #Question- should I just call functions to make these (or functions to load these) here?

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
        self.vel_2d=None
        self.vel_err_2d=None
        self.sigmas_2d=None
        self.sigmas_err_2d=None
        self.halpha_2d=None
        self.n2_2d=None
        self.chisqs=None
        self.gas_templates=None
        self.line_names=None


        #State variables
        self.has_voronoi_bins=False
        self.has_extracted_spectra=True
        self.has_extracted_noise_spectra=True
        self.emission_lines_been_fit=False



    def voronoi_bin_cube(self, SN_TARGET, extract_type='sum', save=True):

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
        

        d_lam=self.lamdas[1]-self.lamdas[0]

        #Mask just around the H-alpha wavelength, to get the signal value
        self.signal_mask=self.get_spec_mask_around_wave(self.Ha_lam, 0.001)

        #Not a typo- x and y axes are reversed
        self.y_coords_2d, self.x_coords_2d=np.indices((self.ny, self.nx))
        self.x_coords_1d=self.x_coords_2d.ravel()
        self.y_coords_1d=self.y_coords_2d.ravel()


        galmedian=np.nanmedian(self.data[self.signal_mask, :, :], axis=0)
        signal=np.abs(galmedian)*np.nansum(self.data[self.signal_mask, :, :]/galmedian, axis=0)  
        #noise=np.nansum(self.noise[signal_mask, :, :], axis=0)
        noise=np.sqrt(np.nansum(self.noise[self.signal_mask, :, :]**2, axis=0))

        #Mask invalid things
        nan_mask=~((np.isfinite(noise))&(noise>0))

        #Do the binning
        binNum, xNode, yNode, xBar, yBar, sn, nPixels, scale=V.voronoi_2d_binning(self.x_coords_2d[~nan_mask].ravel(), self.y_coords_2d[~nan_mask].ravel(), signal[~nan_mask].ravel(), noise[~nan_mask].ravel(), SN_TARGET, plot=False, quiet=True)

        #Make sure everything left unbinned has a value of -1
        #The normal output of voronoi binning just omits these pixels, meaning we don't know the length of binNum beforehand. 
        all_bins=np.full_like(self.x_coords_1d, -1)
        all_bins[~nan_mask.ravel()]=binNum.copy() 
        self.bins_1d=all_bins
        self.bins_2d=self.bins_1d.reshape(self.ny, self.nx)

        self.bin_mask=np.where(self.bins_1d>=0.0)



        self.spectra, self.noise_spectra=ET.simple_extraction(self.x_coords_2d, self.y_coords_2d, self.bins_2d, self.data, self.noise**2, type=extract_type)
        
        outname='{}/{}_bins_spectra.fits'.format(self.bins_spectra_path, self.object_name)

        if save:
            self.save_spectra_to_fits(self.x_coords_1d, self.y_coords_1d, self.bins_1d, self.lamdas, self.spectra, self.noise_spectra, self.nPixels, outname)

        
        self.nPixels=nPixels


        self.has_extracted_spectra=True
        self.has_extracted_noise_spectra=True
        self.has_voronoi_bins=True


        return self.x_coords_1d, self.y_coords_1d, self.bins_1d, self.nPixels


    @staticmethod
    def save_spectra_to_fits(x, y, bins, lamdas, spectra, noise_spectra, nPixels, outname, overwrite=True):
        
        """Take spectra extracted from a set of Voronoi bins and save them to a multi extension fits_file

        Inputs:
        x, y: 1 dimensional arrrays of x and y coordinates, corresponding to the pixels in the cube
        bins: 1 dimensional array of the bin assigned to each pixel
        lamdas: 1 dimensional array of wavelengths
        spectra: an (N_bins, N_lamdas) array of floats. Spectra from each voronoi bin
        noise_spectra: an (N_bins, N_lamdas) array of floats. Noise spectra from each voronoi bin
        outname: string. The name of the output file
        nPixels: 1 dimensional array which is len(np.unique(bins)) long. The number of pixels in each bin
        overwrite: Bool, optional. Do we want to overwrite the MEF file if it already exsists?

        Returns:
        new_hdul: The HDU list object we've just saved
        """

        if not x.shape==y.shape==bins.shape:
            raise ValueError('Input arrays for x, y and bins must be the same size')

        #Write the fits file
        hdu1 = fits.PrimaryHDU(np.column_stack((x, y, bins)))
        hdu2 = fits.ImageHDU(lamdas)
        hdu3 = fits.ImageHDU(spectra)
        hdu4 = fits.ImageHDU(noise_spectra)
        hdu5 = fits.ImageHDU(nPixels)
        new_hdul = fits.HDUList([hdu1, hdu2, hdu3, hdu4, hdu5])

        new_hdul.writeto('{}'.format(outname), overwrite=overwrite)

        return new_hdul

    def load_voronoi_bin_attributes(self, filename):
        """Load values of voronoi bins from a fits file and assign them to class attributes

        Inputs:
        filename: string. Name of the file to load from

        Returns:
        None
        """

        hdul=fits.open(filename)

        self.x_coords_1d, self.y_coords_1d, self.bins_1d=hdul[0].data.T
        #FIXME- assign lamdas here?
        self.lamdas=self.header['CRVAL3']+(np.arange(self.header['NAXIS3'])-self.header['CRPIX3'])*self.header['CDELT3']
        #self.lamdas=hdul[1].data
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



    def inspect_spectra(self):
        """
        Plot the spectra we've extracted from the voronoi bins
        """
        import matplotlib.pyplot as plt 
        fig, ax=plt.subplots(figsize=(17, 4))

        ax.plot(np.exp(self.ppxf_logLam)/10**4, self.ppxf_log_galaxy.T)
        ax.plot(np.exp(self.ppxf_logLam)/10**4, self.ppxf_best_fit.T, c='r', alpha=0.8)

        ax.axvline(self.Ha_lam, c='k', linestyle='dashed')

        return ax








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
        cvd = glob.glob(os.path.expanduser('~/z/Data/stellarpops/CvD1.2/t*.ssp'))
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

        """
        Given a fits file containing the bins and spectra, return the bin information, lamdas, spectra and noise_spectra
        """

        spectra_noise_hdu=fits.open('{}/{}_bins_spectra.fits'.format(bins_spectra_path, object_name))
        bin_information=spectra_noise_hdu[0].data
        lamdas=spectra_noise_hdu[1].data
        spectra=spectra_noise_hdu[2].data
        noise_spectra=spectra_noise_hdu[3].data

        return bin_information, lamdas, spectra, noise_spectra


    def make_MEF_of_quantities(self, things, labels):

        """
        Add a series of arrays to an HDU extension, ready to be saved
        """
        hdu_extensions=[]

        for thing, label in zip(things, labels):
            hdu=fits.ImageHDU(thing)
            hdu.header['QUANTITY']=label
            hdu_extensions.append(hdu)

        return hdu_extensions


    def fit_emission_lines(self, save=True, median_filter=1, **kwargs):

        """
        Fit emission lines to the spectra of a K-CLASH observation cube
        """


        if not self.has_extracted_spectra:
            raise AttributeError("You can't run fit_emission_lines without a set of spectra extracted from (e.g voronoi) bins")
        elif not self.has_extracted_noise_spectra:
            raise AttributeError("You can't run fit_emission_lines without a set of noise spectra extracted from (e.g voronoi) bins")

        nbins=self.spectra.shape[0]

        #Convert to angstroms and de-redshift
        self.rest_lamdas=self.lamdas*(10**4/(1+self.z))

        lamRange_galaxy=[self.rest_lamdas[0], self.rest_lamdas[-1]]

        #Get the velscale of the data
        _, _, self.velscale=util.log_rebin(lamRange_galaxy, self.spectra[0, :])

        #And the FWHM of the galaxy
        FWHM_gal = 2.0/(1+self.z)

        #Load the gas templates
        self.gas_templates, self.line_names, line_wave, lamRange_template=self.load_gas_templates(lamRange_galaxy, self.velscale, FWHM_gal)

        #Empty arrays for results
        self.vel_1d=np.empty(nbins)
        self.vel_err_1d=np.empty(nbins)
        self.sigmas_1d=np.empty(nbins)
        self.sigmas_err_1d=np.empty(nbins)
        self.weights_1d=np.empty((nbins, self.gas_templates.shape[-1]))
        self.chisqs_1d=np.empty(nbins)

        self.ppxf_best_fit=np.empty_like(self.spectra)
        self.ppxf_log_galaxy=np.empty_like(self.spectra)
        self.ppxf_logLam=None
        #Fit each spectrum with pPXF

        self.medfilt_spectra=S.medfilt(self.spectra, (1, median_filter))
        for i, (spectrum, noise_spectrum) in enumerate(tqdm(list(zip(self.medfilt_spectra, self.noise_spectra)), leave=False)):


            #logrebin the galaxy spectrum and noise spectrum
            log_galaxy, logLam_galaxy, self.velscale = util.log_rebin(lamRange_galaxy, spectrum)
            log_noise, logLam_galaxy, _=util.log_rebin(lamRange_galaxy, noise_spectrum)

            run_ppxf=True
            ##Mask pixels
            #If our noise is all 0s, skip this spectrum            
            if not np.any(log_noise>0.0):
                run_ppxf=False



            if run_ppxf:
                #Make a mask the correct length...
                mask=np.ones_like(log_galaxy, dtype='bool')

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

                #Call ppxf and set default arguments
                kwargs.setdefault('clean', True)
                kwargs.setdefault('quiet', True)
                kwargs.setdefault('plot', False)
                kwargs.setdefault('moments', 2)
                kwargs.setdefault('degree', 4)


                #Mask bits where the galaxy is 0 and clip the edges
                mask=np.ones_like(log_galaxy, dtype=bool)
                mask[log_galaxy==0]=False
                mask[:140]=False
                mask[1980:]=False

                pp = PPXF.ppxf(self.gas_templates, log_galaxy, log_noise, self.velscale, start, vsyst=dv, mask=mask, **kwargs)

                chi2=pp.chi2
            else:
                #If we have a bad spectrum, set the chi_squared to be huge and catch it in the bad results below 
                chi2=10000000

            #Only save the results if the ChiSquared is good
            if chi2<5:
                self.vel_1d[i]=pp.sol[0]
                self.vel_err_1d[i]=pp.error[0]*np.sqrt(pp.chi2)
                self.sigmas_1d[i]=pp.sol[1]
                self.sigmas_err_1d[i]=pp.error[1]*np.sqrt(pp.chi2)
                self.chisqs_1d[i]=pp.chi2
                self.weights_1d[i, :]=pp.weights

                self.ppxf_best_fit[i, :]=pp.bestfit
                self.ppxf_log_galaxy[i, :]=log_galaxy
                self.ppxf_logLam=logLam_galaxy
                # if plot:
                #     line, =ax.plot(lamdas, spectrum)
                #     ax.plot(lamdas, pp.bestfit, c=line.get_color(), linewidth=2.0)

            else:
                print(("Bin {} returns a bad result".format(i)))

                self.vel_1d[i]=np.nan
                self.vel_err_1d[i]=np.nan
                self.sigmas_1d[i]=np.nan
                self.sigmas_err_1d[i]=np.nan

                self.chisqs_1d[i]=pp.chi2

                self.weights_1d[i, :]=[np.nan]*len(pp.weights)

                self.ppxf_best_fit[i, :]=np.zeros_like(log_galaxy)
                self.ppxf_log_galaxy[i, :]=log_galaxy
                self.ppxf_logLam=logLam_galaxy
                # if plot:
                #     line, =ax.plot(self.lamdas, spectrum, linestyle='dotted')
                #     ax.plot(self.lamdas, pp.bestfit, c=line.get_color(), linewidth=2.0,  linestyle='dotted')


        for thing, name in zip([self.vel_1d, self.vel_err_1d, self.sigmas_1d, self.sigmas_err_1d, self.weights_1d[:, 0], self.weights_1d[:, -1]], ['vel_2d', 'vel_err_2d', 'sigmas_2d', 'sigmas_err_2d', 'halpha_2d', 'n2_2d']):
            self.expand_to_2d_map(thing, name)
        #Save the results to a text file and a multi extension fits file
        if save:
            self.save_ppxf_results_to_text_file(self.text_file_out_path)
            self.save_ppxf_results_to_MEF(self.fits_file_out_path)




        self.emission_lines_been_fit=True




    def expand_to_2d_map(self, thing, name):
        """Take a list of quantities corresponding to each bin and 'expand' those to make a 2D image
        """
        thing=np.append(thing, np.nan)
        value=KF.display_binned_quantity(self.y_coords_1d, self.x_coords_1d, thing[self.bins_1d])
        setattr(self, name, value)
        return value



    def save_ppxf_results_to_text_file(self, out_file_path):
        """Save the results from ppxf to a text file"""
    
        results_filename='{}/{}_results.txt'.format(out_file_path, self.object_name)
        #Saving things to our text file
        #with open(results_filename, "w") as f:             
        np.savetxt(results_filename, np.column_stack((self.vel_1d, self.vel_err_1d, self.sigmas_1d, self.sigmas_err_1d, self.weights_1d, self.chisqs_1d)))#, delimiter='\t', newline='\t')



    def save_ppxf_results_to_MEF(self, out_file_path):

        """Save the results from ppxf to a multi extension fits file"""

        #The list which we'll fill with fits extensions
        hdu_extensions=[]

        #Kinematic quatities we're saving
        kinematic_quantities=['Velocity', 'VelocityError', 'Sigma', 'SigmaError', 'Chisq']

        #Empty primary HDU
        #Just to have a header containing all the info
        hdu_primary=fits.PrimaryHDU()
        for i, label in enumerate(['VoronoiBins'] + kinematic_quantities + ['Halpha', 'N2']):
            hdu_primary.header['EXT{}'.format(i+1)]=label
        hdu_extensions.append(hdu_primary)

        #Extenstion with the voronoi bins
        hdu_bins=fits.ImageHDU(KF.display_binned_quantity(self.x_coords_1d[self.bin_mask], self.y_coords_1d[self.bin_mask], self.bins_1d[self.bin_mask]))
        hdu_bins.header['QUANTITY']='VoronoiBins'
        hdu_extensions.append(hdu_bins)

        #Kinematic Quantities
        hdu_extensions.extend(self.make_MEF_of_quantities([self.vel_2d, self.vel_err_2d, self.sigmas_2d, self.sigmas_err_2d, self.chisqs], kinematic_quantities))

        #Weights of templates
        em_line_extensions=self.make_MEF_of_quantities([self.halpha_2d, self.n2_2d], ['Halpha', 'N2'])
        hdu_extensions.extend(em_line_extensions)


        #save
        final_fits_file = fits.HDUList(hdu_extensions)
        final_fits_file.writeto('{}/{}_kin_flux.fits'.format(out_file_path, self.object_name), overwrite=True)



    def load_emission_line_attributes(self, filename):
        """Load values of fits to emission line from a fits file and assign them to class attributes

        Inputs:
        filename: string. Name of the file to load from

        Returns:
        None
        """

        self.rest_lamdas=self.lamdas*(10**4)/(1+self.z)
        #we don't save the velscale but we also probably don't need it...
        self.velscale=np.nan


        hdu_list=fits.open('{}'.format(filename))


        self.vel_2d=hdu_list[2].data
        self.vel_err_2d=hdu_list[3].data
        self.sigmas_2d=hdu_list[4].data
        self.sigmas_err_2d=hdu_list[5].data
        self.halpha_2d=hdu_list[7].data
        self.n2_2d=hdu_list[8].data
        self.chisqs=None
        self.gas_templates=None
        self.line_names=None





    def display_results(self):

        #FIXME
        fig, ax=P.display_kinematics(self, self.vel, self.sigma, self.H_alpha, self.N2, self.bins_1d, self.nPixels)




    def mask_2D_map(self, attribute, mask):

        map_2d=getattr(self, attribute)   
        map_2d[mask]=np.nan     
        setattr(self, attribute, map_2d)



    def plot_chains(self, plot_starting_vals=False):

        fig, axs=plt.subplots(nrows=2, ncols=3)
        labels=self.fit_result.flatchain.columns
        ch=self.fit_result.chain
        for i, (ax, label) in enumerate(zip(axs.flatten(), labels)):
            ax.plot(ch[:, :, i].T, c='k', alpha=0.2)
            ax.set_title(label)
            if plot_starting_vals:
                for p in self.p0[:, i]:
                    ax.axhline(p, c='r', linestyle='dashed')

    def trim_outlier_walkers(self, chain):

        
        means=np.mean(np.mean(chain, axis=1), axis=0)
        stds=np.std(chain.reshape(-1, chain.shape[-1]), axis=0)

        mean_per_walker=np.mean(chain, axis=1)

        distances_from_mean=np.sqrt(np.sum((mean_per_walker-means)**2/stds**2, axis=1))

        #get rid out outliers
        good_walkers=np.where(distances_from_mean<5)

        return self.fit_result.chain[good_walkers[0], :, :]
    #Fit the kinematic map

    def fit_map(self, save=True):



        print('\n\tAbout to fit kinematic map of {}- save = {}\n'.format(self.object_name, save))


        start_r0=np.log10(self.table['r50_disk']/self.pix_scale)
        start_theta=self.table['inc_disk']
        #hdu=fits.open('{}/{}_kin_flux.fits'.format(fits_file_out_path, self.object_name))
        #spectra_noise_hdu=fits.open('{}/{}_bins_spectra.fits'.format(bins_spectra_path, self.object_name))

        # bin_information=spectra_noise_hdu[0].data
        # x, y, bins=bin_information.T
        # #Load the arrays and create the x and y arrays
        # TwoD_bins=hdu[1].data
        # vel_data=hdu[2].data
        # vel_errs=hdu[3].data

        # bad_bins=np.where(self.nPixels>30.0)
        # # #Get indices which correspond to the bad bins
        # mask=np.isin(self.bins_2d, bad_bins)

        # self.mask_2D_map('vel_2d', mask)
        # self.mask_2D_map('vel_err_2d', mask)
  

        #Get rid of velocities which large errros
        #self.vel_err_2d[np.abs(self.vel_err_2d)>50.0]=np.nan

        self.kinfit_data=ma.masked_invalid(self.vel_2d)
        self.kinfit_noise=ma.masked_invalid(self.vel_err_2d)

        #Mask anything which is bad in either the data or noise mask
        data_mask=self.kinfit_data.mask.copy()
        noise_mask=self.kinfit_noise.mask.copy()

        light=self.get_bestfit_lightprofile(oversample=1, seeing=None, X=self.x_coords_2d, Y=self.y_coords_2d)

        d=self.data[self.signal_mask, :, :]
        n=self.noise[self.signal_mask, :, :]
        sn=np.sum(d, axis=0)/np.sqrt(np.sum(n**2, axis=0))

        #light_mask=sn


        combined_mask=data_mask|noise_mask#|light_mask

        self.kinfit_data.mask=combined_mask
        self.kinfit_noise.mask=combined_mask

        #Get to a velocity around 0
        start_V0=ma.median(self.kinfit_data)
        self.kinfit_data-=start_V0



        #Oversample the arrays        
        print("oversampling by a factor of {}".format(settings.oversample))
        seeing_pixels=settings.seeing/self.pix_scale
        #Oversampled X and Y arrays
        Y, X=np.meshgrid(np.linspace(0.0, self.kinfit_data.shape[0], settings.oversample*self.kinfit_data.shape[0]), np.linspace(0.0, self.kinfit_data.shape[1], settings.oversample*self.kinfit_data.shape[1]))
        #Mask around H alpha and collapse the cube again
        Ha_mask=self.get_spec_mask_around_wave(self.Ha_lam, 0.01)
        im=self.collapse(wavelength_mask=Ha_mask)
        self.light_image=self.get_bestfit_lightprofile(oversample=1, seeing=settings.seeing, X=self.x_coords_2d, Y=self.y_coords_2d)
        #self.light_image=None

        #get the centre of the continuum image
        result=self.get_continuum_centre(plot=False)
        xc, yc=result.params['X'].value, result.params['Y'].value

        fit_params=LMSPV.Parameters()
        #Theta controls how elliptical the contours are
        #it's arccos(short axis / long axis)
        fit_params.add('theta', value=start_theta, min=1, max=np.arccos(1/5.)*180.0/np.pi, vary=False)
        fit_params.add('xc', value=xc, min=xc-settings.max_centre_shift, max=xc+settings.max_centre_shift, vary=True)
        fit_params.add('yc', value=yc, min=yc-settings.max_centre_shift, max=yc+settings.max_centre_shift, vary=True)
        fit_params.add('log_r0', value=start_r0, min=0.0, max=2.0, vary=True)
        fit_params.add('log_s0', value=10, min=5.0, max=20.0, vary=True)
        fit_params.add('v0', value=0.0, min=-300.0, max=300.0)
        fit_params.add('PA', value=0.0, min=-180.0, max=180.0, vary=True)

        #Select the parameters we're varying, ignore the fixed ones
        #variables=[thing for thing in fit_params if fit_params[thing].vary]
        #ndim=len(variables)
        #Vice versa, plus add in the fixed value
        #fixed=[ "{}={},".format(thing, fit_params[thing].value) for thing in fit_params if not fit_params[thing].vary]

        nwalkers=100
        nsteps=5000
        nburn=1000

        start_vals=np.array([fit_params[thing].value for thing in fit_params if fit_params[thing].vary])
        p0=np.array([start_vals+ 1e-1*np.random.randn(len(start_vals)) for i in range(nwalkers)])
        self.p0=p0




        minimiser = LMSPV.Minimizer(KF.lnprob, fit_params, 
            fcn_args=(self.kinfit_data, self.kinfit_noise, self.x_coords_1d, self.y_coords_1d, self.bins_1d, self.light_image, seeing_pixels, settings.oversample)
            #fcn_kwargs={'light_image':self.light_image, 'seeing_pixels':seeing_pixels, 'oversample':self.oversample}
            )
        print('Running emcee')
        sampler_kwargs={'progress':True}
       
        fit_result = minimiser.emcee(steps=nsteps, nwalkers=nwalkers, burn=nburn, params=fit_params, pos=p0, emcee_sample_kwargs=sampler_kwargs)

        best_model=KF.velfield(fit_result.params, self.kinfit_data, settings.oversample)
        max_v=np.max(best_model-np.nanmedian(best_model))
        min_v=np.min(best_model-np.nanmedian(best_model))
        max_v_err=0.0
        min_v_err=0.0
        #max_v_err=np.max(best_model-np.nanmedian(best_model)+stds)
        #min_v_err=np.min(best_model-np.nanmedian(best_model)-stds)
        LMSPV.report_fit(fit_result)

        self.fit_result=fit_result

        self.chain=self.trim_outlier_walkers(self.fit_result.chain)
        self.flatchain=pd.DataFrame(self.chain.reshape(-1, self.chain.shape[-1]), columns=self.fit_result.flatchain.columns)


        self.bestfit_model=KF.make_binned_model(self.fit_result.params, self.kinfit_data, self.x_coords_1d, self.y_coords_1d, self.bins_1d, self.light_image, seeing_pixels, settings.oversample)
        self.bestfit_model[self.kinfit_data.mask]=np.nan


        if save:
            print("Saving {} results".format(self.object_name))
            self.save_kinematic_fit_to_MEF(self.fits_file_out_path)


        return fit_result, (max_v, max_v_err, min_v, min_v_err)



    def save_kinematic_fit_to_MEF(self, out_file_path):

        """Save the results from the kinematic fitting to a multi extension fits file"""

        #The list which we'll fill with fits extensions
        hdu_extensions=[]

        #Kinematic quatities we're saving
        kinematic_quantities=['Velocity', 'VelocityError', 'Model']

        #Empty primary HDU
        #Just to have a header containing all the info
        hdu_primary=fits.PrimaryHDU()
        for i, label in enumerate(kinematic_quantities):
            hdu_primary.header['EXT{}'.format(i+1)]=label
        hdu_extensions.append(hdu_primary)

        #Extenstion with the voronoi bins
        for data, name in zip([self.kinfit_data, self.kinfit_noise, self.bestfit_model], kinematic_quantities):
            try:
                hdu=fits.ImageHDU(data.filled(fill_value=np.nan))
            except:
                hdu=fits.ImageHDU(data)
            hdu.header['QUANTITY']=name
            hdu_extensions.append(hdu)

        #save
        final_fits_file = fits.HDUList(hdu_extensions)
        final_fits_file.writeto('{}/{}_kin_fit.fits'.format(out_file_path, self.object_name), overwrite=True)


    def plot_kinematic_fit(self, params=None):

        if params is None:
            params=self.fit_result.params
        r_e=self.table['r50_disk']

        seeing_pixels=settings.seeing/self.pix_scale

        #stds=KF.get_errors_on_fit(params, self.kinfit_data, self.kinfit_noise, self.fit_result.flatchain.values, self.x_coords_1d, self.y_coords_1d, self.bins_1d)
        gaussian_fit_to_light, fit_to_light_result=self.get_bestfit_lightprofile(oversample=1, seeing=None, X=self.x_coords_2d, Y=self.y_coords_2d, return_full=True)
        #self.fit_errors=stds
        (fig, ax), (d, v_obs)=P.plot_model(params, self.kinfit_data, self.kinfit_noise, self.bestfit_model, self.x_coords_1d, self.y_coords_1d, self.bins_1d, r_e, self.light_image, seeing_pixels, self.collapsed.copy(), gaussian_fit_to_light, self.object_name, fit_to_light_result)#, stds)


        return (fig, ax), (d, v_obs)


    def plot_kinematic_maps(self, mask=None):

        fig, ax=P.display_kinematics(self, self.vel_2d, self.sigmas_2d, self.halpha_2d, self.n2_2d, self.bins_2d, self.nPixels, mask=mask)


        return fig, ax

    def get_bestfit_lightprofile(self, X, Y,oversample, seeing=None, return_full=False, **kwargs):

        """Fit a 2D gaussian with a slope and offset to a collapsed image of the cube. Then _remove_ the slope and offset
        to be left with the shape of the light profile (normalised)
        """
        
        Ha_mask=self.get_spec_mask_around_wave(self.Ha_lam, 0.01)
        im=self.collapse(wavelength_mask=Ha_mask)

        image, errors, minimiser, result=self.fit_gaussian(im=im, **kwargs)

        result.params['X_GRAD'].set(0.0)
        result.params['Y_GRAD'].set(0.0)
        result.params['OFFSET'].set(0.0)

        if seeing is not None:
            #print "'Removing' seeing of {} arcseconds in quadrature".format(seeing)
            seeing_pixels=seeing/self.pix_scale/np.sqrt(8*np.log(2))
            #print "This is {} pixels".format(seeing_pixels)

            xsig=result.params['XWIDTH'].value
            ysig=result.params['YWIDTH'].value

            result.params['XWIDTH'].set(np.sqrt((xsig**2-seeing_pixels**2).clip(0.2)))
            result.params['YWIDTH'].set(np.sqrt((ysig**2-seeing_pixels**2).clip(0.2)))



        new_x=np.linspace(X.min(), X.max(), X.shape[0]*oversample)
        new_y=np.linspace(Y.min(), Y.max(), Y.shape[1]*oversample)

        x, y=np.meshgrid(new_x, new_y)

        best_gaussian=twoD_Gaussian_with_slope(result.params, x, y)
        #norm=np.trapz(np.trapz(best_gaussian, X[:, 0], axis=1), Y[0, :])
        #norm=np.trapz(np.trapz(best_gaussian, Y[0, :], axis=1), X[:, 0])

        

        if return_full:
            ret=best_gaussian/np.max(best_gaussian).T, result
        else:
            ret=best_gaussian/np.max(best_gaussian).T

        return ret

    # def get_seeing_disk(self, seeing, X, Y, oversample=1):
    #     #Get the seeing Gaussian and the image of the galaxy light
            
    #     #Seeing is the FWHM of the Gaussian
    #     #So convert this into sigma pixels
    #     seeing_sigma=seeing/self.pix_scale/np.sqrt(8*np.log(2))
    #     #Seeing Gaussian

    #     seeing_xc=self.kinfit_data.shape[1]/2.0
    #     seeing_yc=self.kinfit_data.shape[0]/2.0
    #     seeing_params=LMSPV.Parameters()
    #     seeing_params.add('Amp', value=1.0)
    #     seeing_params.add('X', value=seeing_xc)
    #     seeing_params.add('Y', value=seeing_yc)
    #     seeing_params.add('XWIDTH', value=seeing_sigma)
    #     seeing_params.add('YWIDTH', value=seeing_sigma)
    #     seeing_params.add('ROTATION', value=0.0)
    #     seeing_params.add('OFFSET', value=0.0)
    #     seeing_params.add('X_GRAD', value=0.0) 
    #     seeing_params.add('Y_GRAD', value=0.0) 


        
        
    #     fullsize_seeing_gaussian=twoD_Gaussian_with_slope(seeing_params, Y, X)
    #     #Trim off things 5 sigma away from the centre to make the convolution faster
    #     ylow=int(seeing_yc*oversample-3*oversample*seeing_sigma)
    #     yhigh=int(seeing_yc*oversample+3*oversample*seeing_sigma)
    #     xlow=int(seeing_xc*oversample-3*oversample*seeing_sigma)
    #     xhigh=int(seeing_xc*oversample+3*oversample*seeing_sigma)
    #     seeing_gaussian=fullsize_seeing_gaussian[ylow:yhigh, xlow:xhigh]


    #     norm=np.trapz(np.trapz(seeing_gaussian, X[xlow:xhigh, 0], axis=1), Y[0, ylow:yhigh])
        

    #     return seeing_gaussian/norm




