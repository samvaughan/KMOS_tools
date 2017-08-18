from astropy.io import fits
import numpy as np 
import matplotlib.pyplot as plt 
import scipy.optimize as opt

def twoD_Gaussian((x, y), amplitude, xo, yo, sigma_x, sigma_y, theta, offset):
    xo = float(xo)
    yo = float(yo)    
    a = (np.cos(theta)**2)/(2*sigma_x**2) + (np.sin(theta)**2)/(2*sigma_y**2)
    b = -(np.sin(2*theta))/(4*sigma_x**2) + (np.sin(2*theta))/(4*sigma_y**2)
    c = (np.sin(theta)**2)/(2*sigma_x**2) + (np.cos(theta)**2)/(2*sigma_y**2)
    g = offset + amplitude*np.exp( - (a*((x-xo)**2) + 2*b*(x-xo)*(y-yo) 
                            + c*((y-yo)**2)))
    return g.ravel()

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


        elif self.header['CTYPE2'] == 'WAVE':
            self.lam_axis=1
            self.nlam=self.header['NAXIS2']
            self.nx=self.header['NAXIS1']
            self.ny=self.header['NAXIS3']

            self.lamdas=self.header['CRVAL2']+(np.arange(self.header['NAXIS2'])-self.header['CRPIX2'])*self.header['CDELT2']
            self.lam_unit=self.header['CUNIT2']

        elif self.header['CTYPE3'] == 'WAVE':
            self.lam_axis=0
            self.nlam=self.header['NAXIS3']
            self.nx=self.header['NAXIS1']
            self.ny=self.header['NAXIS2']

            self.lamdas=self.header['CRVAL3']+(np.arange(self.header['NAXIS3'])-self.header['CRPIX3'])*self.header['CDELT3']
            self.lam_unit=self.header['CUNIT3']

        else:
            raise NameError("Can't find Wavelength axis")


    def collapse(self, collapse_func=np.nanmedian, plot=False,  plot_args={}, savename=None, save_args={}):
        """
        Collapse a cube along the wavelength axis, according to the function collapse_func.

        Arguments:
            collapse_func: function. Function to use to do the collapsing. Normally np.nanmedian or np.nansum. 
            plot: Bool. If true, plot the resulting collapsed cube.
            plot_args: Dict. Extra arguments to pass to plot_collapsed_cube
            savename: String or None. If string, pass to save_collapsed_cube with that filename. If None, don't save
            save_args: Dict. Extra args to pass to save_collapsed_cube.

        """

        self.collapsed=collapse_func(self.data, axis=self.lam_axis)

        if plot:
            self.plot_collaped_cube(**plot_args)
        if savename is not None:
            raise ValueError('Code not yet written!!')


    def plot_collaped_cube(self, fig=None, ax=None, savename=None, **kwargs):

        if ax is None or ax is None:
            fig, ax=plt.subplots(figsize=(10, 10))

        
        im=ax.imshow(self.collapsed, aspect='auto', origin='lower', **kwargs)
        fig.colorbar(im, ax=ax, label='{}'.format(self.flux_unit))
        ax.set_xlabel('x')
        ax.set_ylabel('y')
        ax.set_title('{}'.format(self.object_name))

        if savename is not None:
            fig.savefig(savename)

        return fig, ax
        





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

        median_value=np.nanmedian(self.unwrapped)
        im=ax.imshow(self.unwrapped, vmin=0.0, vmax=n_median*median_value, extent=[self.lamdas[0], self.lamdas[-1], self.unwrapped.shape[0], 0], aspect='auto', **kwargs)
        fig.colorbar(im, ax=ax, label='{}'.format(self.flux_unit))
        ax.set_xlabel(r'$\lambda$ ({})'.format(self.lam_unit))
        ax.set_ylabel('Spectrum Number')
        ax.set_title('{}'.format(self.object_name))

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



    def get_PSF(self, fit_funct=twoD_Gaussian, plot=True, savename=None, verbose=True, fig=None, ax=None, plot_collaped_cube_args={}, contour_args={}):
        """
        Fit a Gaussian to a 2D collapsed cube
        """

        X, Y=np.meshgrid(np.arange(self.nx), np.arange(self.ny))
        

        
        try:
            image=self.collapsed
        except AttributeError:
            self.collapse()
            image=self.collapsed

        #Tidy up the image so we have no infs or nans
        image[~np.isfinite(image)]=0.0
        
        #Initial Guess. Use the maxval of the image for the Gaussian height. Maybe use mean instead?
        #Best guess is centre, with sigma of 2 in each direction. Theta is 0.0 and so is the overall y offset
        max_val=np.nanmax(self.collapsed)
        initial_guess=[max_val, 8, 8, 2, 2, 0.0, 0.0]


        popt, pcov = opt.curve_fit(fit_funct, (X, Y), image.ravel(), p0=initial_guess)
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
            ax.contour(X, Y, best_gaussian.reshape(self.nx, self.ny), levels=levels, linewidth=2.0, colors='w', **contour_args)

            ax.set_title(r"{}: $\sigma_{{x}}={:.2f}$, $\sigma_{{y}}={:.2f}$".format(ax.get_title(), popt[3], popt[4]))

            if savename is not None:
                fig.savefig(savename)
            return get_av_seeing(popt), popt, (fig, ax)

        return get_av_seeing(popt), popt 







