from astropy.io import fits
import numpy as np 
import matplotlib.pyplot as plt 
import scipy.optimize as opt
from astropy.wcs import WCS 
import warnings
from astropy.coordinates import SkyCoord
import astropy.units as u
import astropy.stats as S

from ppxf.cap_mpfit import mpfit

import lmfit as LM   
import scipy.constants as const
from stellarpops.tools import fspTools as FT

import plotting as P

def twoD_Gaussian_list((x, y), amplitude, xo, yo, sigma_x, sigma_y, theta, offset):
    xo = float(xo)
    yo = float(yo)    
    a = (np.cos(theta)**2)/(2*sigma_x**2) + (np.sin(theta)**2)/(2*sigma_y**2)
    b = -(np.sin(2*theta))/(4*sigma_x**2) + (np.sin(2*theta))/(4*sigma_y**2)
    c = (np.sin(theta)**2)/(2*sigma_x**2) + (np.cos(theta)**2)/(2*sigma_y**2)
    g = offset + amplitude*np.exp( - (a*((x-xo)**2) + 2*b*(x-xo)*(y-yo) 
                            + c*((y-yo)**2)))
    return g.ravel()

def twoD_Gaussian_with_slope(params, X, Y):

    xo = params['X']
    yo = params['Y']
    theta = params['ROTATION']
    sigma_x = params['XWIDTH']
    sigma_y = params['YWIDTH']
    offset = params['OFFSET']
    amplitude = params['Amp']
    slope_X = params['X_GRAD']
    slope_Y = params['Y_GRAD']

    a = (np.cos(theta)**2)/(2*sigma_x**2) + (np.sin(theta)**2)/(2*sigma_y**2)
    b = -(np.sin(2*theta))/(4*sigma_x**2) + (np.sin(2*theta))/(4*sigma_y**2)
    c = (np.sin(theta)**2)/(2*sigma_x**2) + (np.cos(theta)**2)/(2*sigma_y**2)


    g = offset + slope_X*X +slope_Y*Y + amplitude*np.exp( - (a*((X-xo)**2) + 2*b*(X-xo)*(Y-yo) 
                            + c*((Y-yo)**2)))

    return g



def twoD_Gaussian(params, X, Y):

    xo = params['X']
    yo = params['Y']
    theta = params['ROTATION']
    sigma_x = params['XWIDTH']
    sigma_y = params['YWIDTH']
    offset = params['OFFSET']
    amplitude = params['Amp']

    a = (np.cos(theta)**2)/(2*sigma_x**2) + (np.sin(theta)**2)/(2*sigma_y**2)
    b = -(np.sin(2*theta))/(4*sigma_x**2) + (np.sin(2*theta))/(4*sigma_y**2)
    c = (np.sin(theta)**2)/(2*sigma_x**2) + (np.cos(theta)**2)/(2*sigma_y**2)
    g = offset + amplitude*np.exp( - (a*((X-xo)**2) + 2*b*(X-xo)*(Y-yo) 
                            + c*((Y-yo)**2)))

    return g



def moments(data):
    """Returns (height, x, y, width_x, width_y)
    the gaussian parameters of a 2D distribution by calculating its
    moments """
    total = np.nansum(np.abs(data))
    Y, X = np.indices(data.shape)
    x = np.nansum((X*np.abs(data)))/total
    y = np.nansum((Y*np.abs(data)))/total
    # col = data[:, int(y)]
    # width_x = np.sqrt(np.nansum(np.abs((np.arange(col.size)-y)**2*col))/np.nansum(col))
    # row = data[int(x), :]
    # width_y = np.sqrt(np.nansum(np.abs((np.arange(row.size)-x)**2*row))/np.nansum(row))
    height = np.nanmax(data)

    theta=np.pi/2.0
    offset=np.nanmedian(data)
    width_x=2.0
    width_y=2.0

    slope_X=0.0
    slope_Y=0.0

    return height, x, y, width_x, width_y, theta, offset



def get_av_seeing(opt, pixel_scale=0.2):
    """
    Get average seeing in x and y directions from fitting a gaussian to a point source. 
    Take sqrt(sigma_x**2+sigma_y**2) of the best fitting Gaussian and multiply by the pixel scale.
    """

    return pixel_scale*np.sqrt(opt[3]**2+opt[4]**2)





class Cube(object):


    def __init__(self, cube_filename, extension=1, object_name=None, noise_cube=None):

        """
        A Cube class for working with (primarily) KMOS cubes.
        Arguments:
            cube_filename: string. Filename of a 3D data cube.
            extension. Int. Which extension is the cube data in? Default is 1, since all KMOS cubes have their data in the 1 extension (not the first, 0th extension).
            object_name. String. Name of the object in the cube, used for plotting. Default is None, in which case we try and get it from the header.
            noise_cube. 3D numpy array. Array containing the noise cube. If None, assume we're dealing with a KMOS cube and use the cube in the (extension+1)th hdu extension
        Notes:

            Assumes the noise cube is a separate extension of 
        """
        self.filename=cube_filename
        self.hdu=fits.open(cube_filename)
        self.data=self.hdu[extension].data
        self.pri_header=self.hdu[0].header
        self.header=self.hdu[extension].header

        #WCS information
        self.wcs=WCS(self.header)
        self.wcs2d=self.wcs.celestial

        #Noise Cube
        if noise_cube is None:
            self.noise=self.hdu[extension+1].data
            
        else:
            self.noise=noise_cube


        

        #Try getting the units of the cube. If that fails, just use counts
        try:
            self.flux_unit=self.header['HIERARCH ESO QC CUBE_UNIT']
        except:
            self.flux_unit='Counts'

        if object_name is None:
            self.object_name=self.pri_header['HIERARCH ESO PRO REFLEX SUFFIX']



           
        data=self.get_KMOS_fits_data(self.object_name)
        if data is not None:
            self.Ha_flag=bool(data['Detected Emission?'])
            self.continuum_flag=bool(data['Detected Continuum?'])
            self.Ha_lam=float(data['Lamda_Ha'])
            self.other_lines_flag=bool(data['Other lines?'])
            self.cluster=data['Cluster_2']
            self.quadrant=data['Quadrant']
            self.skysub_method=data['SkySub']
            self.cubecomments=data['Comment']

            self.table=data
            if self.Ha_flag:
                self.z=(self.Ha_lam/0.65628)-1.0
            else:
                warnings.warn("No emission found to get an accurate z. Setting z=0.0")
                self.z=0.0
        else:
            warnings.warn("No data found for {}".format(self.object_name))


        # #Get the IFU arm, if we have it:
        # try:
        #     self.arm=get_KMOS_arm_number(self.object_name)
        # except KeyError:
        #     warnings.warn("No Arm data found for {}".format(self.object_name))


        #Get the number of dimensions. Should be 3
        self.ndims=self.header['NAXIS']
        assert self.ndims == 3, 'This Class is for dealing with cubes! Check we have 3 dimensions (or maybe the header is wrong)'


        #Get the wavelength axis
        if self.header['CTYPE1'] == 'WAVE':

            self.lam_axis=2
            self.nlam=self.header['NAXIS1']
            self.nx=self.header['NAXIS2']
            self.ny=self.header['NAXIS3']

            self.lamdas=self.header['CRVAL1']+(np.arange(self.header['NAXIS1'])-self.header['CRPIX1'])*self.header['CDELT1']

            self.lam_unit=self.header['CUNIT1']
            #Rotate the cube so the wavelength axis is first.
            self.lam_axis=0
            print 'Rolling the cube so the wavelength axis is first'
            self.data=np.rollaxis(self.data, 2, 0)


        elif self.header['CTYPE2'] == 'WAVE':
            
            self.nlam=self.header['NAXIS2']
            self.nx=self.header['NAXIS1']
            self.ny=self.header['NAXIS3']

            self.lamdas=self.header['CRVAL2']+(np.arange(self.header['NAXIS2'])-self.header['CRPIX2'])*self.header['CDELT2']
            self.lam_unit=self.header['CUNIT2']

            #Rotate the cube so the wavelength axis is first.
            self.lam_axis=0
            print 'Rolling the cube so the wavelength axis is first'
            self.data=np.rollaxis(self.data, 1, 0)

        elif self.header['CTYPE3'] == 'WAVE':
            self.lam_axis=0
            self.nlam=self.header['NAXIS3']
            self.nx=self.header['NAXIS1']
            self.ny=self.header['NAXIS2']

            self.lamdas=self.header['CRVAL3']+(np.arange(self.header['NAXIS3'])-self.header['CRPIX3'])*self.header['CDELT3']
            self.lam_unit=self.header['CUNIT3']

        else:
            raise NameError("Can't find Wavelength axis")

        #make a quick spectrum by median-combining the whole cube
        self.quick_spec=np.nanmedian(np.nanmedian(self.data, axis=2), axis=1)

        #Set the pixel scale
        self.pix_scale=self.pri_header['HIERARCH ESO PRO REC1 PARAM8 VALUE']

        #The cube hasn't been collapsed yet
        self.collapsed=None

        self.has_been_collapsed=False




    @staticmethod
    def get_KMOS_csv_data(object_name):

        data=np.genfromtxt('/Data/KCLASH/KCLASH_cube_data.csv', delimiter=',', dtype=str, skip_header=1, autostrip=True)
        
        
        galaxies=list(data[:, 0])


        
        obj_index=galaxies.index(object_name)
        return data[obj_index, :]



    @staticmethod 
    def get_KMOS_fits_data(object_name):

        hdu=fits.open('/Data/KCLASH/KCLASH_all_parameters.fits')
        data=hdu[1].data
        
        
        mask=data['Galaxy Name']==object_name
        n_results=len(np.where(mask==True)[0])
        if n_results==1:
            return data[mask]
        else:
            warnings.warn('Found {} entries for {}'.format(n_results, object_name))
            return None

        




    @staticmethod
    def get_KMOS_arm_number(object_name):

        arm_dict={
                    'MACS1931_bluefield_50044' : 1 ,
                    'MACS1931_bluefield_41168' : 10,
                    'MACS1931_blueclust_42661' : 11,
                    'MACS1931_blueclust_36441' : 12,
                    'ARMSTAR401' : 13,
                    'MACS1931_redclust_43187' : 14,
                    'MACS1931_bluefield_44837' : 15,
                    'MACS1931_bluefield_45168' : 16,
                    'MACS1931_redclust_40753' : 17,
                    'ARMSTAR423' : 18,
                    'MACS1931_blueclust_40782' : 19,
                    'MACS1931_redclust_58042' : 2 ,
                    'MACS1931_bluefield_43122' : 20,
                    'MACS1931_blueclust_46452' : 21,
                    'MACS_1931_ARM22_SCI'  : 22,
                    'MACS_1931_ARM23_SCI' : 23,
                    'MACS1931_redclust_55075' : 24,
                    'MACS1931_bluefield_53332' : 3 ,
                    'MACS1931_blueclust_52577' : 4 ,
                    'MACS1931_BCG_59407' : 5 ,
                    'MACS1931_blueclust_48927' : 6 ,
                    'MACS1931_bluefield_51071' : 7 ,
                    'ARMSTAR382' : 8 ,
                    'MACS1931_bluefield_46041' : 9

                    }

        return arm_dict[object_name]

    def collapse(self, wavelength_mask=None, collapse_func=np.nansum, plot=False,  plot_args={}, savename=None, save_args={}):
        """
        Collapse a cube along the wavelength axis, according to the function collapse_func.

        Arguments:
            collapse_func: function. Function to use to do the collapsing. Normally np.nanmedian or np.nansum. 
            plot: Bool. If true, plot the resulting collapsed cube.
            plot_args: Dict. Extra arguments to pass to plot_collapsed_cube
            savename: String or None. If string, pass to save_collapsed_cube with that filename. If None, don't save
            save_args: Dict. Extra args to pass to save_collapsed_cube.

        """
        if wavelength_mask is not None:
            self.collapsed=collapse_func(self.data[wavelength_mask, :, :], axis=self.lam_axis)
        else:
            self.collapsed=collapse_func(self.data, axis=self.lam_axis)

        if plot:
            fig, ax=self.plot_collaped_cube(savename=savename, **plot_args)
            return fig, ax
        if savename is not None:
            raise ValueError('Code not yet written!!')

        self.has_been_collapsed=True

    def plot_collaped_cube(self, fig=None, ax=None, savename=None, **kwargs):

        if ax is None or ax is None:
            fig, ax=plt.subplots(figsize=(10, 10))

        #vmin = kwargs.pop('vmin', 0.0)
        vmax = kwargs.pop('vmax', 0.98*np.nanmax(self.collapsed))
        im=ax.imshow(self.collapsed, aspect='auto', origin='lower',vmax=vmax, **kwargs)
        fig.colorbar(im, ax=ax, label='{}'.format(self.flux_unit))
        ax.set_xlabel('x')
        ax.set_ylabel('y')
        ax.set_title('{}'.format(self.object_name))

        if savename is not None:
            fig.savefig(savename)

        return fig, ax
    
    def get_emission_lines(self, z):

        ###
        #Get all emission lines within the cube
        #Taken and modified from ppxf_utils.py (Thanks Michele!)
        # Balmer Series:      Hdelta   Hgamma    Hbeta   Halpha
        line_wave = np.array([4101.76, 4340.47, 4861.33, 6562.80])/(10.0**4)  # air wavelengths
        line_names = np.array(['Hdelta', 'Hgamma', 'Hbeta', 'Halpha'])

        #                 -----[OII]-----    -----[SII]-----
        lines = np.array([3726.03, 3728.82, 6716.47, 6730.85])/(10.0**4)   # air wavelengths
        names = np.array(['[OII]3726', '[OII]3729', '[SII]6716', '[SII]6731'])
        line_names = np.append(line_names, names)
        line_wave = np.append(line_wave, lines)

        # To keep the flux ratio of a doublet fixed, we place the two lines in a single template
        #                 -----[OIII]-----
        lines = np.array([4958.92, 5006.84])/(10.0**4)     # air wavelengths
        line_names = np.append(line_names, '[OIII]5007d') # single template for this doublet
        line_wave = np.append(line_wave, lines[1])

        # To keep the flux ratio of a doublet fixed, we place the two lines in a single template
        #                  -----[OI]-----
        lines = np.array([6300.30, 6363.67])/(10.0**4)     # air wavelengths
        line_names = np.append(line_names, ['[OI]6300d', '[OI]6300d']) # single template for this doublet
        line_wave = np.append(line_wave, lines)

        # To keep the flux ratio of a doublet fixed, we place the two lines in a single template
        #                 -----[NII]-----
        lines = np.array([6548.03, 6583.41])/(10.0**4)     # air wavelengths
        line_names = np.append(line_names, ['[NII]6583d', '[NII]6583d']) # single template for this doublet
        line_wave = np.append(line_wave, lines)

        # Only include lines falling within the cube wavelength range.
        line_wave*=(1+z)
        w = (line_wave > self.lamdas[0]) & (line_wave < self.lamdas[-1])
        line_names = line_names[w]
        line_wave = line_wave[w]

        return line_names, line_wave


    def get_continuum_mask(self, z, line_width=50.0/(10**4), mask_skylines=False):
        """
        Get a wavelength mask which avoids all emission lines and the worst skylines
        """

        line_names, line_wave=self.get_emission_lines(z)        

        mask=np.ones_like(self.lamdas, dtype=bool)
        for line in line_wave:
            m=~((self.lamdas > (line-line_width/2.0)) & (self.lamdas < (line+line_width/2.0)))

            mask=mask & m

        #Avoid the last pixels which are often bad
        mask[1950:]=False


        if mask_skylines:
            #Avoid skylines. Needs work
            #At a skyline, the gradient sharply changs from positive to negative
            #If we sigma_clip the gradient and keep the outliers, we find where the worse skylines are
            clipped=S.sigma_clip(np.gradient(self.quick_spec), sigma=5, iters=3)
            mask=mask & ~ clipped.mask

        return mask



    def plot_continuum_map(self, z, plot_args={}, savename=None, line_width=50.0/(10**4)):
        """
        Make a map of the Continuum Emission in a cube. This collapses the cube along the wavelength axis, but masks strong skylines and emission lines

        Arguments:
            self
            z: float. The redshift of the cube (for masking emission lines) 
            linewidth: float. *Total* width around the line centre to mask out (i.e we mask out centre-(width/2) to centre+(width/2) ). Note this is in the same units as the cube (so usually microns for KMOS cubes). 
            plot_args: Dictionary. Passes to plot_collapsed_cube

        """



        mask=self.get_continuum_mask(z, line_width)
        


        fig, ax=self.collapse(wavelength_mask=mask, plot=True, plot_args=plot_args, savename=savename)
        title=ax.get_title()
        ax.set_title('{}: Continuum Map'.format(title))
        return fig, ax


    def get_spec_mask_around_wave(self, wave, width):

        """
        Mask a spectrum around a certain OBSERVED wavelength. Length of mask is defined by width. Width is in microns
        """

        m=((self.lamdas > (wave-width/2.0)) & (self.lamdas < (wave+width/2.0)))

        return m




    def get_cube_mask_around_wave(self, wave, width):

        """
        Mask an entire cube around a certain wavelength
        """
        m=self.get_spec_mask_around_wave(wave, width)

        cube_mask=np.repeat(m, self.nx*self.ny).reshape(-1, self.nx, self.ny)

        return cube_mask



    def mask_cube_around_wave(self, wave, width):
        """
        Apply get_cube_mask_around_wave to a cube and noise cube
        """

        cube_mask=self.get_cube_mask_around_wave(wave, width)

        return self.data*cube_mask, self.noise*cube_mask





    def plot_line_map(self, z, line_name, line_width=50.0/(10**4), plot_args={}, savename=None, show_spec=False):

        line_names, line_wave=self.get_emission_lines(z)
        assert line_name in line_names, 'Pick a line from here: {}'.format(line_names)

        #Select the desired line from the names
        #If we have a doublet then we'll get two wavelengths back
        wavelengths=line_wave[line_names==line_name]

        mask=np.zeros_like(self.lamdas, dtype=bool)
        for line in wavelengths:
            m=self.get_spec_mask_around_wave(line, line_width)
            mask=mask | m
        
        #Plot the quick spectrum, showing the lines we're mapping
        if show_spec == True:
            fig=plt.figure(figsize=(12, 10))
            #Make a gridspec. Want the height ratios such that the main image is larger than the spectrum
            import matplotlib.gridspec as gridspec
            gs = gridspec.GridSpec(2, 1, height_ratios=[4, 1])
            #Add the axes. 
            ax_linemap = plt.subplot(gs[0, 0])
            ax_spec = plt.subplot(gs[1, 0])

            plot_args['fig']=fig
            plot_args['ax']=ax_linemap
            fig, ax_linemap=self.collapse(wavelength_mask=mask, plot=True, plot_args=plot_args, savename=None)

            #Plot the whole spectrum in grey, then just the linemap region in red
            ax_spec.plot(self.lamdas, self.quick_spec, c='k', alpha=0.5, linewidth=2)
            ax_spec.plot(self.lamdas[mask], self.quick_spec[mask], c='r', alpha=1.0, linewidth=2)
            ax_spec.set_title('Median spec')
            ax_spec.set_xlabel(r'$\lambda$ ($\mu$m)')

            if savename is not None:
                fig.savefig(savename)
            return fig, [ax_linemap, ax_spec]

        else:
            fig, ax=self.collapse(wavelength_mask=mask, plot=True, plot_args=plot_args, savename=savename)
            title=ax.get_title()
            ax.set_title('{}: Line map around {}'.format(title, line_name))

            return fig, ax

    def plot_all_lines(self, z, line_width=50.0/(10**4), plot_args={}, savename=None):

        """
        Plot all the lines within the wavelength range of the cube, and a continuum map. Just calls plot_line_map many times, with
        plot_continuum_line at the end. Assumes that we have less than 5 lines! Otherwise we'd need to think of a better way to set up the axes
        """

        line_names, line_wave=self.get_emission_lines(z)

        #Get rid of duplicate names (i.e of the line doublets) but preserve the order of the labels
        from collections import OrderedDict
        names=list(OrderedDict.fromkeys(line_names))


        assert len(names) <=7, 'Change the code to include more than 5 emission lines!'
        
        if len(names) <=5:
            fig, axs=plt.subplots(nrows=2, ncols=3, figsize=(24, 12))
            for name, ax in zip(names, axs.flatten()):
                fig, ax=self.plot_line_map(z, name, line_width=50.0/(10**4), plot_args={'fig':fig, 'ax':ax}, show_spec=False)
                ax.set_title('{}'.format(name))
                ax.axis('off')

        elif 5 < len(names) <=7:
            fig, axs=plt.subplots(nrows=2, ncols=4, figsize=(24, 12))
            for name, ax in zip(names, axs.flatten()):
                fig, ax=self.plot_line_map(z, name, line_width=50.0/(10**4), plot_args={'fig':fig, 'ax':ax}, show_spec=False)
                ax.set_title('{}'.format(name))
                ax.axis('off')




        #Fill the last cube with a continuum map
        fig, ax=self.plot_continuum_map(z, line_width=50.0/(10**4), plot_args={'fig':fig, 'ax':axs.flatten()[-1]})
        ax.set_title('Continuum Map')
        ax.axis('off')


        fig.suptitle('{}'.format(self.object_name), fontsize=24)
        if savename is not None:
            fig.savefig(savename)
        return fig, axs





    def unwrap(self, plot=False,  plot_args={}, savename=None, save_args={}):
        """
        Unwrap a cube to make a 2D representation, to easily find emission lines.
        Arguments:
            self
            plot: Bool. If true, plot the resulting unwrapped cube.
            plot_args: Dict. Extra arguments to pass to plot_unwrapped_cube
            savename: String or None. If string, pass to save_unwrapped_cube with that filename. If None, don't save
            save_args: Dict. Extra args to pass to save_unwrapped_cube.

        """
        self.unwrapped=self.data.reshape(self.nlam, -1).T

        if plot:
            self.plot_unwrapped_cube(**plot_args)
        if savename is not None:
            self.save_unwrapped_cube(savename, **save_args)


        
    def plot_unwrapped_cube(self, fig=None, ax=None, savename=None, n_median=5.0, show_plot=True, **kwargs):
        """
        Plot an unwrapped cube, with vmin and vmax being 0.0 and 5.0 times the median value of the cube. I find that this gives nice results for the resulting image. 
        Arguments:
            fig: a matplotlib figure. Default is none, in which case we plot on a new figure.
            ax: a matplotlib axes. Default is none, in which case we plot on a new axis
            save: string or None. If string, save the image to that filename. If None, don't save. 
            n_median: float. Vmax is n_median*median value of the array. 
            **kwargs: extra args to pass to imshow

        """
        if ax is None or ax is None:
            fig, ax=plt.subplots(figsize=(18, 4))

        zero_mask=self.unwrapped!=0.0

        median_value=np.abs(np.nanmedian(self.unwrapped[zero_mask]))
        im=ax.imshow(self.unwrapped, vmin=-1.0*median_value, vmax=n_median*median_value, extent=[self.lamdas[0], self.lamdas[-1], self.unwrapped.shape[0], 0], aspect='auto', **kwargs)
        fig.colorbar(im, ax=ax, label='{}'.format(self.flux_unit))
        ax.set_xlabel(r'$\lambda$ ({})'.format(self.lam_unit))
        ax.set_ylabel('Spectrum Number')
        ax.set_title('{}'.format(self.object_name))

        import matplotlib.ticker as ticker
        ax.xaxis.set_minor_locator(ticker.MultipleLocator(0.01))
        ax.tick_params(which='major', length=7)
        ax.tick_params(which='minor', length=4)

        #ax.set_xticks(np.linspace())

        if savename is not None:
            fig.savefig(savename)

        return fig, ax



    def save_unwrapped_cube(self, savename, **kwargs):

        """
        Save an unwrapped cube to a filename, with an updated header.

        Args:
            self
            savename: the filename to save the cube to.
            **kwargs: extra args to pass to hdu.writeto (e.g clobber)

        """

        save_hdr=self.pri_header.copy()
        save_hdr.set("CRVAL1",self.lamdas[0])
        save_hdr.set("CRPIX1",1)
        save_hdr.set("CDELT1",self.lamdas[1]-self.lamdas[0])
        save_hdr.set("CTYPE1","LINEAR")
        save_hdr.set("DC-FLAG",0)
        save_hdr.set("OBJECT", self.object_name)

        hdu=fits.PrimaryHDU(self.unwrapped)
        hdu.header=save_hdr
        hdu.writeto(savename, **kwargs)



    def fit_gaussian(self, fit_funct=twoD_Gaussian, method='leastsq', clip=False):

        """
        Fit a Gaussian to a 2D collapsed cube using lmfit
        """

        
        

        
        try:
            collapsed_cube=self.collapsed
        except AttributeError:
            self.collapse()
            collapsed_cube=self.collapsed


        xmin=2
        xmax=-2
        ymin=2
        ymax=-2

        image=collapsed_cube.copy()[xmin:xmax, ymin:ymax]
        errors=np.sqrt(np.nansum(self.noise[:, xmin:xmax, ymin:ymax]**2, axis=0))
        errors[errors<=0.0]=100 

        Y, X = np.indices(image.shape)

        #Tidy up the image so we have no infs or nans
        image[~np.isfinite(image)]=0.0

        if clip:
            clipped_img=S.sigma_clip(image, sigma_lower=2.0, sigma_upper=3.0, iters=3)
            image=clipped_img.filled(0.0)

        
        #Initial Guess. Use the maxval of the image for the Gaussian height. Maybe use mean instead?
        #Best guess is centre, with sigma of 2 in each direction. Theta is 0.0 and so is the overall y offset
        max_val=np.nanmax(self.collapsed)

        #Normalise things to avoid having some parameters at 1e-20 and others at 1
        median=np.abs(np.median(image))
        image/=median
        errors/=median


        initial_guess=moments(image)

        params=LM.Parameters()
        params.add('Amp', value=initial_guess[0], min=1e-3)
        params.add('X', value=initial_guess[1], min=0.5, max=image.shape[1]-0.5)
        params.add('Y', value=initial_guess[2], min=0.5, max=image.shape[0]-0.5)
        params.add('XWIDTH', value=initial_guess[3], min=0.1, max=6.0)
        params.add('YWIDTH', value=initial_guess[4], min=0.1, max=6.0)
        params.add('ROTATION', value=initial_guess[5], min=0.0, max=2*np.pi)
        params.add('OFFSET', value=initial_guess[6])
        params.add('X_GRAD', value=0.0) 
        params.add('Y_GRAD', value=0.0) 

        def lmfitfun(p, data, err, X, Y):
            return (data.ravel()-twoD_Gaussian_with_slope(p, X, Y).ravel())/err.ravel()




        minimiser = LM.Minimizer(lmfitfun, params, fcn_args=(image, errors, X, Y))
        result = minimiser.minimize(method=method)   


        

        # try:
        #     popt, pcov = opt.curve_fit(fit_funct, (X, Y), image.ravel(), sigma=errors.ravel(), p0=initial_guess)
        # except:
        #     popt=None


        #Deal with the amount of the cube we clipped off:
        result.params['X'].set(value=result.params['X']+xmin, min=0.0, max=image.shape[1])
        result.params['Y'].set(value=result.params['Y']+ymin, min=0.0, max=image.shape[0])

        #import pdb; pdb.set_trace()

        return image, errors, minimiser, result

    def get_continuum_centre(self, fit_funct=twoD_Gaussian, plot=True, savename=None, verbose=False, fig=None, ax=None, clip=False, return_full=False, fit_args={}, plot_collaped_cube_args={}, contour_args={}):

        """
        Fit a guassian to a collapsed cube and get the x, y coordinates of the centre. 
        """
        


        image, errors, minimiser, result=self.fit_gaussian(fit_funct, clip=clip)
        
      
        ret=[result]
        

        if plot:
            if fig is None or ax is None:
                fig, ax=plt.subplots(figsize=(10, 10))
            fig, ax=self.plot_collaped_cube(fig=fig, ax=ax, **plot_collaped_cube_args)

            Y, X = np.indices(self.collapsed.shape)
            best_gaussian=twoD_Gaussian(result.params, X, Y)
            
            ax.contour(X, Y, best_gaussian, linewidth=1.0, colors='r', **contour_args)

            ax.set_title(r"{}: $X={:.2f}$, $Y={:.2f}$".format(ax.get_title(), result.params['X'].value, result.params['Y'].value))

            if savename is not None:
                fig.savefig(savename)
            ret.append((fig, ax))


            if verbose:
                print "\nObject: {}".format(self.object_name)
                print "Best Fitting Gaussian:"
                print "\t(x, y)={:.2f}, {:.2f}".format(result.params['X'].value, result.params['Y'].value)
                LM.report_fit(result)



        if return_full:
            ret.append((image, errors, minimiser))

        if len(ret)>1:
            ret=tuple(ret)
        else:
            ret=ret[0]

        return ret

    def get_PSF(self, fit_funct=twoD_Gaussian, plot=True, savename=None, verbose=True, fig=None, ax=None, plot_collaped_cube_args={}, contour_args={}):

        """ 
        Get the PSF of a cube containing an arm star. Use cube.fit_gaussian to fit a 2D gaussian, then use that to get the average seeing (from get_av_seeing)
        and return the average seeing and the optimal gaussian parameters.

        Can plot if necessary and print results. 
        """


        popt, pcov=self.fit_gaussian(fit_funct)
        Y, X = np.indices(self.collapsed.shape)
        best_gaussian=twoD_Gaussian((X, Y), *popt)



        if verbose:
            print "\nObject: {}".format(self.object_name)
            print "Best Fitting Gaussian:"
            print "\t(x, y)={:.2f}, {:.2f}".format(popt[1], popt[2])
            print "\tsigma_x={:.3f}, sigma_y={:.3f}".format(popt[3], popt[4])
            print '\ttheta={:.3f}, amplitude={}'.format(popt[5], popt[0])
            print '\tReconstructed seeing: {:.3f}"'.format(get_av_seeing(popt))

        if plot:
            if fig is None or ax is None:
                fig, ax=plt.subplots(figsize=(10, 10))
            fig, ax=self.plot_collaped_cube(fig=fig, ax=ax, **plot_collaped_cube_args)

            levels=np.array([0.2, 0.4, 0.6, 0.8, 0.95])*np.max(best_gaussian)
            ax.contour(X, Y, best_gaussian.reshape(self.ny, self.nx), levels=levels, linewidth=2.0, colors='w', **contour_args)

            ax.set_title(r"{}: $\sigma_{{x}}={:.2f}$, $\sigma_{{y}}={:.2f}$".format(ax.get_title(), popt[3], popt[4]))

            if savename is not None:
                fig.savefig(savename)
            return get_av_seeing(popt), popt, (fig, ax)

        return get_av_seeing(popt), popt 

    def interpolate_point_1_arcsec_sampling(self):
        """
        Interpolate a cube to go from 0.2" spaxels to 0.1" spaxels, ensuring the flux in each wavelength slice is the same.

        This assumes the spatial sampling is already 0.2"! And we just want to halve that.

        This works IN PLACE! So it overwrites cube.data, cube.noise and cube.nx, cube.ny.
        """

        if self.pix_scale != 0.2:
            import scipy.interpolate as si

            x=np.arange(self.nx)
            y=np.arange(self.ny)

            x_new=np.arange(0, self.nx, 0.5)
            y_new=np.arange(0, self.ny, 0.5)

            

            interp=si.RegularGridInterpolator((self.lamdas, y, x), self.data, bounds_error=False, fill_value=np.nan)
            noise_interp=si.RegularGridInterpolator((self.lamdas, y, x), self.noise, bounds_error=False, fill_value=np.nan)

            new_points = np.meshgrid(self.lamdas, y_new, x_new, indexing='ij')
            flat = np.array([m.flatten() for m in new_points])

            out_array = interp(flat.T)
            new_cube = out_array.reshape(*new_points[0].shape)

            out_noise = noise_interp(flat.T)
            new_noise = out_noise.reshape(*new_points[0].shape)

            #Ensure the flux in each wavelength slice is the same

            old_flux=np.nansum(np.nansum(self.data, axis=1), axis=1)
            new_flux=np.nansum(np.nansum(new_cube, axis=2), axis=1)

            ratio=old_flux/new_flux

            ratio[~np.isfinite(ratio)]=1.0
            ratio[ratio<0.0]=1.0

            ratio_cube=np.repeat(ratio, x_new.shape[0]*y_new.shape[0]).reshape(-1, y_new.shape[0], x_new.shape[0])

            

            self.data=new_cube*ratio_cube
            self.noise=new_noise*ratio_cube
            self.nx=x_new.shape[0]
            self.ny=y_new.shape[0]

            self.pix_scale=0.1
            return new_cube*ratio_cube, new_noise*ratio_cube

        else:
            warnings.warn('Pixel scale is {}, so interpolation has already been done!'.format(self.pix_scale))









def get_north_east_arrow(wcs):
    """
    Returns the (unit vectors) dx and dy of an arrow pointing North, as defined from a WCS instance
    """
    #Check the wcs we pass is celestial
    if not wcs.is_celestial:
        raise TypeError("Must pass a celestial WCS instance")

    cd11=wcs.wcs.cd[0, 0]
    cd12=wcs.wcs.cd[0, 1]
    cd21=wcs.wcs.cd[1, 0]
    cd22=wcs.wcs.cd[1, 1]

    cdelt1=np.sqrt(cd11**2+cd21**2)
    cdelt2=np.sqrt(cd12**2+cd22**2)

    #from definitions at http://danmoser.github.io/notes/gai_fits-imgs.html
    dx_N=cd21/cdelt1
    dy_N=cd11/cdelt1
    
    dx_E=cd22/cdelt2
    dy_E=-1.0*cd12/cd22

    return np.array([dx_N, dy_N, dx_E, dy_E])



    
def get_cutout(cube, main_cutout):
    from astropy.nddata import Cutout2D

    
    cubecentre_RA, cubecentre_DEC=cube.wcs2d.all_pix2world(np.array([[8.5, 8.5]]), 1)[0]

    position=SkyCoord(cubecentre_RA*u.deg, cubecentre_DEC*u.deg) 

    try:
        cutout=Cutout2D(main_cutout.data, position, (10*u.arcsec, 10*u.arcsec), wcs=main_cutout.wcs)
        return cutout, position
      
    except:
        import pdb; pdb.set_trace()
        print 'No overlap for {}'.format(cube.object_name)
        return None, None

    

def get_rectangle_coordinates(cube, cutout):
    #We want to plot a rectangle of the size of the IFU on each cutout. 
    #To do this, get the bottom left coordinate of each cube, and translate that pixel coordinate into the cube world coordinate
    #Then, translate that world coordinate into the pixel coordinates of the galaxy cutout

    cubeleft_RA, cubebottom_DEC=cube.wcs2d.all_pix2world(np.array([[0, 17]]), 1)[0]
    cuberight_RA, cubetop_DEC=cube.wcs2d.all_pix2world(np.array([[17, 0]]), 1)[0]

    pix_x_left, pix_y_top=cutout.wcs.all_world2pix(np.array([[cubeleft_RA, cubebottom_DEC]]), 1)[0]
    pix_x_right, pix_y_bottom=cutout.wcs.all_world2pix(np.array([[cuberight_RA, cubetop_DEC]]), 1)[0]

    pix_width=pix_x_right-pix_x_left
    pix_height=pix_y_top-pix_y_bottom

    return pix_x_left, pix_y_bottom, pix_width, pix_height


def get_1_arcsec_line(cube, cutout):
    

    cubeleft_RA, cubebottom_DEC=cube.wcs2d.all_pix2world(np.array([[0, 17]]), 1)[0]
    cuberight_RA, cubetop_DEC=cube.wcs2d.all_pix2world(np.array([[17, 0]]), 1)[0]

    pix_x_left, pix_y_left=cutout.wcs.all_world2pix(np.array([[cubeleft_RA, cubebottom_DEC]]), 1)[0]
    
    end_of_line=cubeleft_RA*u.deg + 1.0*u.arcsec
    pix_x_right, pix_y_right=cutout.wcs.all_world2pix(np.array([[end_of_line.value, cubebottom_DEC]]), 1)[0]

    return (pix_x_left, pix_y_left), (pix_x_right, pix_y_right)












# def get_gas_emission_templates(lamRange1, velscale, FWHM_gal):

#     from stellarpops.tools import CD12tools as CT
#     import glob
#     import ppxf_util as util

#     #Set up the templates to fit
#     cvd = glob.glob('/Data/stellarpops/CvD1.2/t*.ssp')
#     cvd.sort()

#     #CvD Templates are at resolution 2000, so work out lamda/R for the middle wavelength in your array
#     FWHM_tem = np.median(lamRange1)/2000



#     #Use Simon's CvDTools function to read in the CvD models and get them into proper units
#     cvd_data=CT.loadCD12spec(cvd[0])
#     #They're returned in Ryan's spectrum class. spec.lam is wavelengths, spec.flam is flux in lamda units
#     lams=cvd_data.lam
#     pad=100
#     lamRange2=(lamRange1[0]-pad, lamRange1[1]+pad)
#     template_mask=np.where((lams>lamRange2[0])&(lams<lamRange2[1]))[0]
#     cdelt=lams[10]-lams[9]





#     FWHM_dif = np.sqrt((FWHM_gal**2 - FWHM_tem**2).clip(0))
#     sigma = FWHM_dif/2.355/cdelt # Sigma difference in pixels
#     #Log Rebin one spectrum to get the length of the templates array right
#     ssp=cvd_data.flam[0][template_mask]
#     #ssp = ndimage.gaussian_filter1d(ssp,sigma)
#     sspNew, logLam2, velscale = util.log_rebin(lamRange2, ssp, velscale=velscale)

#     logtemplate=sspNew.copy()



#     gas_templates, line_names, line_wave = util.emission_lines(logLam2, lamRange1, FWHM_gal)

#     gas_templates=np.stack((gas_templates[:, 0], gas_templates[:, -1])).T
#     #gas_templates=gas_templates[:, 0].reshape(-1, 1)

#     return gas_templates, lamRange2, logLam2


# def get_SN(pp, loggalaxy, lognoise, baseline_noise):



#     #Get the polynomial from pPXF
#     x = np.linspace(-1, 1, len(loggalaxy))
#     apoly = np.polynomial.legendre.legval(x, pp.polyweights)



#     emission_lines=pp.bestfit-apoly

#     emission_mask=emission_lines>1e-23



#     SN=np.nansum(emission_lines[emission_mask])/np.sqrt(np.nansum((lognoise[emission_mask]**2)))
#     #SN=np.sum(emission_lines[emission_mask])/(baseline_noise*len(lognoise[emission_mask]))



#     if not np.isfinite(SN):
#         SN=0.0

#     return SN

# def call_ppxf(parameters, quiet=True):
#     import time
#     from ppxf import ppxf

#     gas_templates, loggalaxy, lognoise, velscale, start, logLam1, logLam2, goodPixels=parameters

#     #print("Running pPXF on Cube Spectra {}".format(i))
#     #Logarithmically rebin the galaxy and noise spectrum using the ppxf utils function

#     # loggalaxy/=galmedian # Normalize spectrum to avoid numerical issues
#     # lognoise/=galmedian
#     # #gas_templates*=100
 
#     c = 299792.458

#     dv = (logLam2[0]-logLam1[0])*c # km/s

#     #z = np.exp(vel/c) - 1   # Relation between velocity and redshift in pPXF
#     #goodPixels = util.determine_goodpixels(logLam1, lamRange2, 0.0)


#     t= time.clock()


#     pp = ppxf.ppxf(gas_templates, loggalaxy, lognoise, velscale, start, mask=goodPixels, plot=False, moments=2, degree=1, vsyst=dv, clean=False, quiet=quiet)



#     if not quiet:

#         print("Formal errors:")
#         print("     dV    dsigma   dh3      dh4")
#         print("".join("%8.2g" % f for f in pp.error*np.sqrt(pp.chi2)))
#         print('Elapsed time in PPXF: %.2f s' % (time.clock() - t))
#         print(pp.sol,  pp.error*np.sqrt(pp.chi2))

#     return pp


# def fit_Halpha_emission_lines_ppxf(cube, spectrum, noise_spectrum, templates=None, clip=True, baseline_subtract=True, quiet=True):

#     """
#     templates: tuple, optional.
#         A tuple of the emission line templates, lamRange2,  logLam2 and velscale
#     """

#     import ppxf_util as util
#     from stellarpops.tools import CD12tools as CT
#     import astropy.stats as S



#     #Set up the various masks we apply to the spectra
#     #Mask clips off the edges around Halpha
#     mask=cube.get_spec_mask_around_wave(0.65628*(1+cube.z), 0.1)
#     #fit_mask clips just around the H alpha emission line
#     fit_mask=cube.get_spec_mask_around_wave(0.65628*(1+cube.z), 0.05)
#     #This mask takes the main part of the spectrum but ignores the emission lines
#     mask_for_noise_std=(mask) & (~fit_mask)


#     #Get the wavelength range
#     wave=cube.lamdas.copy()
#     wave*=((10**4)/(1+cube.z))
#     low_mask=wave[fit_mask][0]
#     high_mask=wave[fit_mask][-1]

#     lamRange1 =np.array([low_mask, high_mask])


#     if templates is None:
#         #Read in one galaxy spec to get the velscale:
#         example_spec=cube.data[:, 10, 10]
#         _, _, velscale = util.log_rebin(lamRange1, example_spec[fit_mask])

#         FWHM_gal=2.0
#         #Make the templates
#         gas_templates, lamRange2, logLam2=get_gas_emission_templates(lamRange1, velscale, FWHM_gal)
#     else:
#         gas_templates, lamRange2, logLam2, velscale=templates

#     #start values
#     start=[0.0, 3*velscale]

#     #Mask we can edit to remove bad pixels
#     goodPixels=np.ones_like(spectrum, dtype=bool)

#     if clip:
#         clipped=S.sigma_clip(spectrum, sigma_upper=5.0, sigma_lower=1.0, iters=3)

#         spectrum=clipped.filled(0.0)
#         goodPixels[clipped.mask]=False


#     if baseline_subtract:
#         import continuum                        
#         cont=continuum.fit_continuum(wave, spectrum, noise_spectrum, clip=[1, 10, 10], order=5) 
#         spectrum-=cont
            


#     #A spectrum which is cut around the emission lines, but also has the emission lines masked out
#     spec_for_std=spectrum[mask_for_noise_std]
#     #RMS of this emission line free spectrum- separate noise estimate
#     rms=np.std(spec_for_std)
#     #Median of this spectrum
#     baseline_noise=np.nanmedian(spec_for_std)



#     loggalaxy, logLam1, velscale = util.log_rebin(lamRange1, spectrum[fit_mask])
#     lognoise, logLam1, _=util.log_rebin(lamRange1, noise_spectrum[fit_mask])

#     parameters=[gas_templates, loggalaxy, lognoise, velscale, start, logLam1, logLam2, goodPixels[fit_mask]]

#     pp=call_ppxf(parameters, quiet=quiet)

#     SN=get_SN(pp, loggalaxy, lognoise, baseline_noise)

#     return pp, SN

# def fit_Halpha_emission_lines(cube, spectrum, noise_spectrum, templates=None, clip=False, baseline_subtract=False, quiet=True, return_specs=False):

#     """
#     templates: tuple, optional.
#         A tuple of the emission line templates, lamRange2,  logLam2 and velscale
#     """

#     import ppxf_util as util
#     from stellarpops.tools import CD12tools as CT
#     import astropy.stats as S



#     #Set up the various masks we apply to the spectra
#     #Mask clips off the edges around Halpha
#     mask=cube.get_spec_mask_around_wave(0.65628*(1+cube.z), 0.1)
#     fit_mask=cube.get_spec_mask_around_wave(0.65628*(1+cube.z), 0.05)
#     mask_for_noise_std=(mask) & (~fit_mask)


#     #Get the wavelength range
#     wave=cube.lamdas.copy()
#     wave*=((10**4)/(1+cube.z))
#     low_mask=wave[fit_mask][0]
#     high_mask=wave[fit_mask][-1]

#     lamRange1 =np.array([low_mask, high_mask])


#     if templates is None:
#         #Read in one galaxy spec to get the velscale:
#         example_spec=cube.data[:, 10, 10]
#         _, _, velscale = util.log_rebin(lamRange1, example_spec[fit_mask])

#         FWHM_gal=2.0
#         #Make the templates
#         gas_templates, lamRange2, logLam2=get_gas_emission_templates(lamRange1, velscale, FWHM_gal)
#     else:
#         gas_templates, lamRange2, logLam2, velscale=templates


#     temps=gas_templates/np.max(gas_templates)

#     #start values
#     start=[0.0, 3*velscale]

#     #Mask we can edit to remove bad pixels
#     goodPixels=np.ones_like(spectrum, dtype=bool)

#     if clip:
#         clipped=S.sigma_clip(spectrum, sigma_upper=5.0, sigma_lower=1.0, iters=3)

#         spectrum=clipped.filled(0.0)
#         goodPixels[clipped.mask]=False


#     if baseline_subtract:
#         import continuum                        
#         cont=continuum.fit_continuum(wave, spectrum, noise_spectrum, clip=[1, 10, 10], order=5) 
#         spectrum-=cont
    

#     #A spectrum which is cut around the emission lines, but also has the emission lines masked out
#     spec_for_std=spectrum[mask_for_noise_std]
#     #RMS of this emission line free spectrum- separate noise estimate
#     rms=np.std(spec_for_std)
#     #Median of this spectrum
#     baseline_noise=np.nanmedian(spec_for_std[spec_for_std!=0.0])



#     loggalaxy, logLam1, velscale = util.log_rebin(lamRange1, spectrum[fit_mask])
#     logWeights, logLam1, _=util.log_rebin(lamRange1, noise_spectrum[fit_mask])
#     lognoise=np.ones_like(loggalaxy)*baseline_noise

#     galmedian=np.abs(np.median(loggalaxy[loggalaxy!=0]))

#     data=loggalaxy/galmedian
#     noise=lognoise/galmedian



#     weights=1./(1+(logWeights/galmedian)**2)

#     c_light=const.c/1000.0
#     dv=c_light*np.log(lamRange2[0]/lamRange1[0])

#     try:
#         result, bestfit=fit_kinematics(data, noise, temps, velscale, dv, weights)
#     except:
#         import pdb; pdb.set_trace()

#     bestfit_Ha=get_just_Ha_model(result.params, temps, velscale, dv, data)
#     bestfit_NII=get_just_NII_model(result.params, temps, velscale, dv, data)

#     SN=get_SN(bestfit-result.params['offset'].value, data, noise, weights)


#     if return_specs:
#         return result, SN, galmedian, [data, bestfit, noise, bestfit_Ha, bestfit_NII]

#     return result, SN

# def make_model(params, templates, velscale, dv, data):


#     V=params['Vel'].value
#     SIG=params['sigma'].value
#     HA_SCALE=params['Ha_scale'].value
#     NII_SCALE=params['NII_scale'].value
#     OFFSET=params['offset'].value

#     model=HA_SCALE*FT.convolve_template_with_losvd(templates[:, 0], V, SIG, velscale=velscale, vsyst=dv)[:len(data)]+NII_SCALE*FT.convolve_template_with_losvd(templates[:, -1], V, SIG, velscale=velscale, vsyst=dv)[:len(data)]+OFFSET

#     return model

# def get_just_Ha_model(params, templates, velscale, dv, data):
#     V=params['Vel'].value
#     SIG=params['sigma'].value
#     HA_SCALE=params['Ha_scale'].value
#     NII_SCALE=params['NII_scale'].value
#     OFFSET=params['offset'].value

#     model=HA_SCALE*FT.convolve_template_with_losvd(templates[:, 0], V, SIG, velscale=velscale, vsyst=dv)[:len(data)]

#     return model

# def get_just_NII_model(params, templates, velscale, dv, data):
#     V=params['Vel'].value
#     SIG=params['sigma'].value
#     HA_SCALE=params['Ha_scale'].value
#     NII_SCALE=params['NII_scale'].value
#     OFFSET=params['offset'].value

#     model=NII_SCALE*FT.convolve_template_with_losvd(templates[:, -1], V, SIG, velscale=velscale, vsyst=dv)[:len(data)]

#     return model


# def objective_function(parameters, data, noise, weights, templates, velscale, dv):

#     model=make_model(parameters, templates, velscale, dv, data)

#     chisqs=weights*(data-model)/noise

#     return chisqs

# def fit_kinematics(data, noise, templates, velscale, dv, weights):






#     params=LM.Parameters()
#     params.add('Vel', value=0.0, vary=True, min=-500.0, max=500.0)
#     params.add('sigma', value=3*velscale, vary=True, min=0.5*velscale[0], max=500.0)
#     params.add('Ha_scale', value=10.0, vary=True, min=0.0, max=1000.0)
#     params.add('NII_scale', value=10.0, vary=True, min=0.0, max=1000.0)
#     params.add('offset', value=0.0, vary=True, min=-100, max=100)


#     #model=make_model(params, template, velscale, dv, data)

    
#     minner = LM.Minimizer(objective_function, params, fcn_args=(data, noise, weights, templates, velscale, dv))
#     result = minner.minimize()

#     bestfit=make_model(result.params, templates, velscale, dv, data)

#     return result, bestfit


# def get_SN(bestfit, data, noise, weights):



#     #Get the polynomial from pPXF
#     #x = np.linspace(-1, 1, len(loggalaxy))
#     #apoly = np.polynomial.legendre.legval(x, pp.polyweights)



#     #emission_lines=make_model(params, template, velscale, dv, loggalaxy)

#     emission_mask=bestfit>1e-3

#     SN=(np.nansum(bestfit[emission_mask]*weights[emission_mask])/np.sqrt(np.nansum((noise[emission_mask]**2))))
#     #SN=np.sum(emission_lines[emission_mask])/(baseline_noise*len(lognoise[emission_mask]))

#     if not np.isfinite(SN):
#         SN=0.0

#     return SN