from cube_tools import Cube
import numpy as np 
import matplotlib.pyplot as plt 


class CubePlot(Cube):

    """
    A class for plotting things to do with the cube
    """
    def __init__(self, *args, **kwargs):
        super(self.__class__, self).__init__(*args, **kwargs)
        # self.time = datetime.now()


    @staticmethod
    def show_bins(x, y, bins, ax=None, cmap='prism'):
        """
        Display pixels at coordinates (x, y) coloured with "counts".
        This routine is fast but not fully general as it assumes the spaxels
        are on a regular grid. This needs not be the case for Voronoi binning.

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
        img[j, k] = bins

        if ax is None:
            fig, ax=plt.subplots()
        ax.imshow(np.rot90(img), interpolation='nearest', cmap=cmap,
                   extent=[xmin - pixelSize/2, xmax + pixelSize/2,
                           ymin - pixelSize/2, ymax + pixelSize/2])

        return ax


def display_kinematics(cube, fits_file_out_path, nPixels, plot=False):

    """
    Display the kinematics of a K-CLASH cube in a handy format
    """


    #Load the kinematics and flux measurements
    hdu_list=fits.open('{}/{}_kin_flux.fits'.format(fits_file_out_path, cube.object_name))
    

    #Load the plot settings file
    V_min, V_max, S_min, S_max, flux_max=np.genfromtxt('/home/vaughan/Science/KCLASH/Kinematics/plot_params/{}.txt'.format(cube.object_name), unpack=True)
 

    #Plotting extras- titles
    titles=[r'V$_{\mathrm{gas}}$', r'$\sigma_{\mathrm{gas}}$', r'H$\alpha$', '[NII]6583d']#'[SII]6716', '[SII]6731',

    #Arguments for the imshow call for each quantity
    extra_args=[{'vmin':V_min, 'vmax':V_max}, {'vmin':S_min, 'vmax':S_max}, {'vmin':0.0, 'vmax':flux_max}, {'vmin':0.0, 'vmax':flux_max}]#, {'vmin':0.0, 'vmax':flux_max}, {'vmin':0.0, 'vmax':flux_max}]

    #Labels of the colourbars
    labels=[r'kms$^{-1}$', r'kms$^{-1}$', r'erg sec$^{-1} $cm$^{-2}$ A$^{-1}$', r'erg sec$^{-1} $cm$^{-2}$ A$^{-1}$']#, r'erg sec$^{-1} $cm$^{-2}$ A$^{-1}$', r'erg sec$^{-1} $cm$^{-2}$ A$^{-1}$']



    #Plot the H-Alpha linemap and the postage stamp cutout
    #This sets up the gridspec    
    fig=plt.figure(figsize=(16, 8))
    gs = gridspec.GridSpec(4, 6)
    ax_image=fig.add_subplot(gs[0:2, 0:2])
    ax_continuum=fig.add_subplot(gs[2:, 0:2])
    ax2=fig.add_subplot(gs[0:2, 2:4])
    ax3=fig.add_subplot(gs[0:2, 4:6])
    ax5=fig.add_subplot(gs[2:, 2:4])
    ax6=fig.add_subplot(gs[2:, 4:6])

    #These are the only axes which we'll fill using the for loop
    #Axes 1 and 4 are 'special'- the postage stamp and the linemap
    axs=np.array([ax2, ax3, ax5, ax6])

    #Load the cutout
    cutout_img=mpimg.imread('/home/vaughan/Science/KCLASH/Cutouts/imgs/{}.png'.format(cube.object_name))
    ax_image.imshow(cutout_img)
    ax_image.axis('off')
    ax_image.set_title('{}'.format(cube.object_name))


    #Get the sauron colormap
    cm=plt.get_cmap(sauron)


    #Plot the continuum image
    cont_mask=cube.get_continuum_mask(cube.z)
    cube.collapse(wavelength_mask=cont_mask)

    #Plot the continuum map
    fig, ax_continuum=cube.plot_line_map(cube.z, 'Halpha', show_spec=False, plot_args={'fig':fig, 'ax':ax_continuum, 'cmap':cm, 'vmin':0.0, 'vmax':5e-18})
    ax_continuum.tick_params(axis='both', which='both', labelbottom='off', labeltop='off', labelleft='off', labelright='off')
    ax_continuum.set_xlabel('')
    ax_continuum.set_ylabel('')
    ax_continuum.set_title(r'H$\alpha$ Line Map')


    ax_image.set_aspect('equal', adjustable='box')
    ax_continuum.set_aspect('equal', adjustable='box')


    #Plot the images from the kinematic fits
    #Loop through the axes, in the order Velocity, Sigma, H-alpha weight, NII weight
    #Also loop through titles, the extra kwargs which we pass to imshow and the labels for the colorbars
    for i, (image, ax, title, kwargs, label) in enumerate(zip([hdu_list[2].data, hdu_list[4].data, hdu_list[7].data, hdu_list[-1].data], axs.flatten(), titles, extra_args, labels)):


        bins=hdu_list[1].data
        good_bins=np.where(nPixels>15.0)

        # #Get indices which correspond to the good bins
        mask=np.isin(bins, good_bins)
        image_copy=image.copy()
        image[mask]=np.nan
        image_copy[~mask]=np.nan

        if i==0:
            image-=np.nanmedian(image)

        img=ax.imshow(image.T, cmap=cm, **kwargs)
        x, y=np.indices((image.shape[0], image.shape[1]))

        ax.imshow(image_copy.T, cmap=cm, alpha=0.5, **kwargs)
        #Add the colorbar and label
        fig.colorbar(img, ax=ax, label=label)
        ax.set_title(title)
        ax.tick_params(axis='both', which='both', labelbottom='off', labeltop='off', labelleft='off', labelright='off')
        ax.set_xlabel('')
        ax.set_ylabel('')

        


    if plot:
        plt.show()
    else:
        plt.close('all')

    return fig, ax


def plot_model(cube, params, data, errors, chain_samples, x, y, bins):


    r_e=cube.table['r50_int_z']#/cube.table['pixscale']
    #Do this better!
    FWHM_seeing=0.5
    r22_disk=np.sqrt((1.3*r_e)**2 + (FWHM_seeing/2.35)**2)
    r3_disk=np.sqrt((1.8*r_e)**2 + (FWHM_seeing/2.35)**2)

    model=velfield(params, data)
    # m[data.mask]=np.nan
    # model=ma.masked_invalid(m)

    max_y, max_x=data.shape

    yc=int(params['yc'].value)
    xc=int(params['xc'].value)
    PA=params['PA'].value



    shifted_data=ndi.shift(data.filled(-9999), [max_x/2-xc, max_y/2-yc], mode='nearest', order=0)
    shifted_model=ndi.shift(model, [max_x/2-xc, max_y/2-yc], mode='nearest', order=3)
    shifted_errors=ndi.shift(errors.filled(-9999), [max_x/2-xc, max_y/2-yc], mode='nearest', order=0)

    final_data=ndi.rotate(shifted_data,PA, reshape=False,mode='nearest', order=0)
    final_model=ndi.rotate(shifted_model,PA, reshape=False,mode='nearest', order=3)
    final_errors=ndi.rotate(shifted_errors,PA, reshape=False,mode='nearest', order=0)
    #final_residuals=ndi.rotate(data.filled(-9999)-model.filled(-9999),PA, reshape=False,mode='nearest', order=0)



    nan_mask=final_data<-9000
    final_data[nan_mask]=np.nan
    final_model[nan_mask]=np.nan
    final_errors[nan_mask]=np.nan
    final_residuals=final_data-final_model

    x, y=np.indices((bins.shape[0], bins.shape[1]))
    bins=bins.astype(int)
    bin_mask=np.where(bins>=0)
    final_binned_model=display_binned_quantity(x[bin_mask], y[bin_mask], final_model.ravel()[bins[bin_mask]])
    import pdb; pdb.set_trace()
    
    v_profile, v_obs, v_err, [x_slit, y_slit]=get_slit_profile(params, final_data, final_model, final_errors)


    #########################################################################################################
    #Plotting

    min_vel=np.nanmin(model)
    max_vel=np.nanmax(model)

    fig, axs=plt.subplots(nrows=1, ncols=4, figsize=(18, 5))   
    cbaxes = fig.add_axes([0.1, 0.1, 0.01, 0.8])

    img=axs[0].imshow(final_data, origin='lower', cmap=sauron, vmin=min_vel, vmax=max_vel)
    cbar=fig.colorbar(img, cax=cbaxes)
    cbar.set_label(label=r'$V_{\mathrm{rot}}$ (kms$^{-1}$)', fontsize=15)
        
    cbaxes.yaxis.set_label_position('left')
    cbaxes.yaxis.set_ticks_position('left')

    axs[1].imshow(final_binned_model, origin='lower', cmap=sauron, vmin=min_vel, vmax=max_vel)
    axs[2].imshow(final_residuals, origin='lower', cmap=sauron, vmin=min_vel, vmax=max_vel)
    axs[2].tick_params(axis='both', which='both', labelbottom='off', labelleft='off')


    #Get x-axis in arcseconds
    axs[3].plot((x_slit-xc)*0.1, v_profile, c='r')
    axs[3].errorbar((x_slit-xc)*0.1, v_obs, yerr=v_err, c='k', marker='o')
    #axs[3].set_ylim([min_vel-20, max_vel+20])
  

    #Overlay error regions for the fit
    ndraws=1000
    variables=[p for p in params]
    all_profiles=np.empty((len(v_profile), ndraws))
    for i, pars in enumerate(chain_samples[np.random.randint(len(chain_samples), size=ndraws)]):
        
        sample_pars=LMSPV.Parameters()
        for p, name in zip(pars, variables):
            sample_pars.add(name, value=p)
        sample_pars['PA'].set(params['PA'].value)
        
        p_xc=sample_pars['xc']
        p_yc=sample_pars['yc']
        model=velfield(sample_pars, data)
        shifted_model=ndi.shift(model, [max_x/2-p_xc, max_y/2-p_yc], mode='nearest', order=3)
        final_model=ndi.rotate(shifted_model,PA, reshape=False,mode='nearest', order=3)
        final_model[nan_mask]=np.nan

        v_p, v_obs, v_err, [_, _]=get_slit_profile(sample_pars, final_data, final_model, final_errors)
        all_profiles[:, i]=v_p
    
    stds=2.0*np.nanstd((all_profiles-v_profile[:, None]), axis=1)
    #old_stds=2.0*np.nanstd(all_profiles, axis=1)
    #import pdb; pdb.set_trace()
    axs[3].fill_between((x_slit-xc)*0.1, v_profile-stds, v_profile+stds, facecolor='r', alpha=0.2)


    #Plot the slit
    for ax in axs[:2]:

        
        ax.axhline(max_y/2+1, c='k', linewidth=2.0)
        ax.axhline(max_y/2+1+2, c='k', linestyle='dotted')
        ax.axhline(max_y/2+1-2, c='k', linestyle='dotted')
        ax.tick_params(axis='both', which='both', labelbottom='off', labelleft='off')


    #2.2 and 3Re lines
    axs[3].axvline(r22_disk, linestyle='dotted', c='k')
    axs[3].axvline(r3_disk, linestyle='dashed', c='k')
    axs[3].axvline(-1.0*r22_disk, linestyle='dotted', c='k')
    axs[3].axvline(-1.0*r3_disk, linestyle='dashed', c='k')

    axs[3].annotate(r'2.2 R$_{\mathrm{d}}$', xy=(r22_disk, min_vel), xytext=(-2, 10), textcoords='offset points', horizontalalignment='right', verticalalignment='top')
    axs[3].annotate(r'3 R$_{\mathrm{d}}$', xy=(r3_disk, min_vel+10), xytext=(-2, 10), textcoords='offset points', horizontalalignment='right', verticalalignment='top')

    #Titles
    axs[0].set_title('Data', fontsize=25, loc='left')
    axs[1].set_title('Binned Model', fontsize=25, loc='left')
    axs[2].set_title('Residuals', fontsize=25, loc='left')

    axs[3].set_title('Rotation Curve', fontsize=25, loc='left')
    axs[3].set_xlabel(r'$r (^{\prime\prime})$', fontsize=15)
    axs[3].set_ylabel(r'$V_{\mathrm{rot}}$ (kms$^{-1}$)', fontsize=15)

    axs[3].yaxis.set_label_position('right')
    axs[3].yaxis.set_ticks_position('right')
    

    for ax in axs:
        x0,x1 = ax.get_xlim()
        y0,y1 = ax.get_ylim()
      
        ax.set_aspect((x1-x0)/(y1-y0)) 
        
    fig.subplots_adjust(hspace=0.2, wspace=0.16)


    return (fig, axs), stds