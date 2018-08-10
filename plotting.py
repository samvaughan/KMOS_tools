from .cube_tools import Cube
import numpy as np 
import matplotlib.pyplot as plt 
import matplotlib.gridspec as gridspec
import matplotlib.image as mpimg
from voronoi_binning.display_pixels.sauron_colormap import sauron
import os


from . import kin_functions as KF
from scipy import ndimage as ndi

import settings

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


def display_kinematics(cube, vel, sigma, H_alpha, N2, bins, nPixels, mask=None):

    """
    Display the kinematics of a K-CLASH cube in a handy format
    """


    #Load the kinematics and flux measurements
    #hdu_list=fits.open('{}/{}_kin_flux.fits'.format(fits_file_out_path, cube.object_name))
    

    #Load the plot settings file
    #V_min, V_max, S_min, S_max, flux_max=np.genfromtxt('/home/vaughan/Science/KCLASH/Kinematics/plot_params/{}.txt'.format(cube.object_name), unpack=True)
    V_min=1.1*np.nanpercentile(vel, 3)
    V_max=1.1*np.nanpercentile(vel, 97)

    S_min=1.1*np.nanpercentile(sigma, 3)
    S_max=1.1*np.nanpercentile(sigma, 97)

    flux_max=1.1*np.nanpercentile(H_alpha, 97)



    #Plotting extras- titles
    titles=[r'V$_{\mathrm{gas}}$', r'$\sigma_{\mathrm{gas}}$', r'H$\alpha$', '[NII]6583d']#'[SII]6716', '[SII]6731',

    #Arguments for the imshow call for each quantity
    extra_args=[{'vmin':-300.0, 'vmax':300.0}, {'vmin':0.0, 'vmax':200.0}, {'vmin':0.0, 'vmax':flux_max}, {'vmin':0.0, 'vmax':flux_max}]#, {'vmin':0.0, 'vmax':flux_max}, {'vmin':0.0, 'vmax':flux_max}]

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
    cutout_img=mpimg.imread(os.path.expanduser('~/z/Data/KCLASH/Cutouts/imgs/{}.png'.format(cube.object_name)))
    ax_image.imshow(cutout_img)
    ax_image.axis('off')
    ax_image.set_title('{}'.format(cube.object_name))


    #Get the sauron colormap
    cm=plt.get_cmap(sauron)


    #Plot the continuum image
    if not cube.has_been_collapsed:
        cont_mask=cube.get_continuum_mask(cube.z)
        cube.collapse(wavelength_mask=cont_mask)

    #Plot the continuum map
    fig, ax_continuum=cube.plot_line_map(cube.z, 'Halpha', show_spec=False, plot_args={'fig':fig, 'ax':ax_continuum, 'cmap':cm, 'vmin':0.0, 'vmax':0.1*np.nanpercentile(cube.data, 95)*cube.data.shape[0]})
    ax_continuum.tick_params(axis='both', which='both', labelbottom='off', labeltop='off', labelleft='off', labelright='off')
    ax_continuum.set_xlabel('')
    ax_continuum.set_ylabel('')
    ax_continuum.set_title(r'H$\alpha$ Line Map')


    ax_image.set_aspect('equal', adjustable='box')
    ax_continuum.set_aspect('equal', adjustable='box')


    #Plot the images from the kinematic fits
    #Loop through the axes, in the order Velocity, Sigma, H-alpha weight, NII weight
    #Also loop through titles, the extra kwargs which we pass to imshow and the labels for the colorbars
    for i, (image, ax, title, kwargs, label) in enumerate(zip([vel, sigma, H_alpha, N2], axs.flatten(), titles, extra_args, labels)):



        # #bad_bins=np.where(nPixels>150.0)

        # # #Get indices which correspond to the good bins
        # mask=np.isin(bins, bad_bins)
        image_copy=image.copy()
        if mask is not None:
            image_copy[mask]=np.nan
        # #image[mask]=np.nan
        # image_copy[~mask]=np.nan

        #Make velocities around 0
        if i==0:
            image_copy-=np.nanmedian(image_copy)
            print(np.nanmedian(image_copy))
            
        img=ax.imshow(np.rot90(image_copy.T), cmap=cm, **kwargs)
        x, y=np.indices((image_copy.shape[0], image.shape[1]))

        #ax.imshow(image_copy.T, cmap=cm, alpha=0.5, **kwargs)
        #Add the colorbar and label
        fig.colorbar(img, ax=ax, label=label)
        ax.set_title(title)
        ax.tick_params(axis='both', which='both', labelbottom='off', labeltop='off', labelleft='off', labelright='off')
        ax.set_xlabel('')
        ax.set_ylabel('')


    return fig, axs




#def make_final_plot(params, data, model,  errors, r_e):


def overplot_model_axis(PA, xc, yc, model):

    x=np.linspace(0.0, model.shape[1], 100)
    y=np.tan((PA+90)*np.pi/180.0)*x+(yc-np.tan((PA+90)*np.pi/180.0)*xc)

    min_y=0.0
    max_y=model.shape[0]

    mask=(y>min_y)&(y<max_y)

    return x[mask], y[mask]


def rotate_slit_get_gradient(model, x_p, y_p):



    shifted_model=ndi.shift(model.copy(), shift=[model.shape[0]/2-y_p, model.shape[1]/2-x_p], order=0, mode='nearest')

    thetas=np.linspace(0.0, 360.0, 360)

    n_iterations=len(thetas)
    grad=np.zeros(n_iterations)   

    for i, theta in enumerate(thetas):
        m=shifted_model.copy()
        rotated=ndi.rotate(m, theta, order=0, mode='nearest', reshape=False)
        #curve=np.nanmean(rotated[int(model.shape[0]/2):], axis=0)
        curve=rotated[int(model.shape[0]/2), :]
        tmp=curve[~np.isnan(curve)]


        #Now take the gradient of this
        #Sometimes there's only one element left, in which case we set grad[i] equal to 0
        try:
            grad[i]=np.mean(np.gradient(tmp))
        except ValueError:
            grad[i]=0.0

        #print(grad[i])
    
    


    #Fit a sinusoid to this
    guess_mean = np.mean(grad)
    guess_std = 3*np.std(grad)/(2**0.5)
    guess_phase = 0

    guess_amp = np.max(grad)

    from scipy.optimize import leastsq
    optimize_func = lambda x: x[0]*np.sin(thetas*np.pi/180.0+x[1]) + x[2] - grad
    est_amp, est_phase, est_mean = leastsq(optimize_func, [guess_amp, guess_phase, guess_mean])[0]

    fine_t=np.linspace(0.0, 360.0, 1000)
    sine_fit=est_amp*np.sin(fine_t*np.pi/180.0+est_phase)+est_mean
    
    best_angle=fine_t[np.argmax(sine_fit)]

    # fig, ax=plt.subplots()
    # ax.imshow(shifted_model, origin='lower')
    # ax=plot_slit(ax, best_angle, np.arange(30), 15, 15)
    # ax.set_ylim(0.0, 30.0)
    

    # plt.figure()
    # plt.plot(thetas, grad)
    # plt.plot(fine_t, sine_fit)
    # plt.show()

    # import ipdb; ipdb.set_trace()

    return best_angle 



def get_moments(model, x, y):

    m=model.copy()*0.0+1.0
    m=m.ravel()

    M_00=np.nansum(m)
    M_01=np.nansum(y*m)
    M_10=np.nansum(x*m)
    M_11=np.nansum(x*y*m)
    M_20=np.nansum(x**2*m)
    M_02=np.nansum(y**2*m)

    return M_00, M_01, M_10, M_11, M_20, M_02


def centre_of_mass(model, x, y):

    M_00, M_01, M_10, M_11, M_20, M_02=get_moments(model, x, y)

    xbar=M_10/M_00
    ybar=M_01/M_00

    return xbar, ybar


def find_theta(model, x, y):

    """https://en.wikipedia.org/wiki/Image_moment"""

    m=model.copy()*0.0+1.0
    m=m.ravel()



    M_00, M_01, M_10, M_11, M_20, M_02=get_moments(m, x, y)

    xbar, ybar=centre_of_mass(m, x, y)

    mu_11=M_11/M_00-xbar*ybar
    mu_20=M_20/M_00-xbar**2
    mu_02=M_02/M_00-ybar**2

    theta_ellipse=0.5*np.arctan2(2*mu_11, (mu_20-mu_02))*180.0/np.pi


    theta_map=rotate_slit_get_gradient(model, xbar, ybar)
    #print(mu_20, mu_02)


    
    
    return theta_map

def plot_slit(ax, PA, x_slit, xc, yc, plot_width=True, **kwargs):

    c=kwargs.pop('c', 'k')

    d=2
    x_offset=d*np.cos(np.pi/2.-PA*np.pi/180.0)
    y_offset=d*np.sin(np.pi/2.-PA*np.pi/180.0)

    y_slit=np.tan((PA)*np.pi/180.0)*x_slit+yc-np.tan((PA)*np.pi/180.0)*xc
    y_slit_lower=np.tan((PA)*np.pi/180.0)*(x_slit-x_offset)+yc-np.tan((PA)*np.pi/180.0)*xc - y_offset
    y_slit_higher=np.tan((PA)*np.pi/180.0)*(x_slit+x_offset)+yc-np.tan((PA)*np.pi/180.0)*xc + y_offset

    ax.plot(x_slit, y_slit, c=c, linewidth=2.0, **kwargs)
    if plot_width:
        ax.plot(x_slit, y_slit_lower, c=c, linestyle='dotted')
        ax.plot(x_slit, y_slit_higher, c=c, linestyle='dotted')

    return ax


def plot_model(params, data, errors, model, X, Y, bins, r_e, light_image, seeing_pixels, collapsed_cube, gaussian_fit, label, fit_to_light_result):#, stds):

    
    
    #Do this better!
    FWHM_seeing=0.5
    r22_disk=np.sqrt((1.3*r_e)**2 + (FWHM_seeing/2.35)**2)
    r3_disk=np.sqrt((1.8*r_e)**2 + (FWHM_seeing/2.35)**2)

    yc=params['yc'].value
    xc=params['xc'].value
    v0=params['v0']
    #PA_1=params['PA'].value

    max_y, max_x=data.shape

    #Make an unmasked one
    model_nomask=KF.make_binned_model(params, data, X, Y, bins, light_image, seeing_pixels, settings.oversample)
    #make an unbinned one
    #smooth_model=KF.velfield(params, data, settings.oversample).reshape(np.max(Y)+1, settings.oversample, np.max(X)+1, settings.oversample).mean(axis=-1).mean(axis=1)
    #smooth_model[np.isnan(data)]=np.nan

    try:
        mask=data.mask
    except AttributeError:
        mask=np.isnan(data)

    xbar, ybar=centre_of_mass(model, X, Y)

    PA=find_theta(model, X, Y)
    PA_ellipse=-1.*fit_to_light_result.params['ROTATION']*180.0/np.pi

    shift=np.array([max_y/2-ybar, max_x/2-xbar])

    rotated_data=KF.shift_rotate_velfield(data, shift, PA,reshape=False,order=0,cval=np.nan)
    rotated_errors=KF.shift_rotate_velfield(errors, shift,  PA,reshape=False,order=0,cval=np.nan)
    rotated_model=KF.shift_rotate_velfield(model, shift,  PA,reshape=False,order=0,cval=np.nan)
    rotated_residuals=KF.shift_rotate_velfield(data-model, shift,  PA,reshape=False,order=0,cval=np.nan)


    final_data=KF.shift_rotate_velfield(data, shift,0.0,reshape=False,order=0,cval=np.nan)
    final_errors=KF.shift_rotate_velfield(errors, shift, 0.0,reshape=False,order=0,cval=np.nan)
    final_model=KF.shift_rotate_velfield(model, shift, 0.0,reshape=False,order=0,cval=np.nan)
    final_residuals=KF.shift_rotate_velfield(data-model, shift, 0.0,reshape=False,order=0,cval=np.nan)

    
    #Smooth model which we then downsample
    smooth_model=KF.velfield(params, data, settings.oversample).reshape(np.max(Y)+1, settings.oversample, np.max(X)+1, settings.oversample).mean(axis=-1).mean(axis=1)
    rotated_smooth_model=KF.shift_rotate_velfield(smooth_model, shift,  PA,reshape=False,order=0,cval=np.nan)
    #rotated_smooth_model[np.isnan(final_model)]=np.nan


    v_profile_binned, v_profile_smooth, v_obs, v_err, [x_slit, y_slit]=KF.get_slit_profile(params=params, data=rotated_data, binned_model=rotated_model, smooth_model=rotated_smooth_model, noise=rotated_errors, stripe=5)
    
    #########################################################################################################
    #Plotting

    # min_vel=0.8*np.nanmin(final_model-v0)
    # assert min_vel<0.0, "Need to ensure we're around 0!"
    # max_vel=-1.0*min_vel
    min_vel=1.3*np.nanpercentile(v_profile_smooth, 5)
    max_vel=1.3*np.nanpercentile(v_profile_smooth, 95)
    #import pdb; pdb.set_trace()

    fig, axs=plt.subplots(nrows=1, ncols=5, figsize=(24,  5))   
    cbaxes = fig.add_axes([0.1, 0.1, 0.01, 0.8])


    img=axs[1].imshow(final_data, origin='lower', cmap=sauron, vmin=min_vel, vmax=max_vel)
    cbar=fig.colorbar(img, cax=cbaxes)
    cbar.set_label(label=r'$V_{\mathrm{rot}}$ (kms$^{-1}$)', fontsize=15)
        
    cbaxes.yaxis.set_label_position('left')
    cbaxes.yaxis.set_ticks_position('left')

    axs[2].imshow(final_model, origin='lower', cmap=sauron, vmin=min_vel, vmax=max_vel)
    axs[3].imshow(final_residuals, origin='lower', cmap=sauron, vmin=min_vel, vmax=max_vel)
    
    axs[3].tick_params(axis='both', which='both', labelbottom='off', labelleft='off')

    d=0.1*(x_slit-max_x/2)
    #d[:np.argmin(d)]=-1.0*d[:np.argmin(d)]
 
    #import ipdb; ipdb.set_trace()

    axs[4].plot(d, v_profile_binned, c='r')
    axs[4].plot(d, v_profile_smooth, c='b')
    

    
    axs[4].errorbar(d, v_obs, yerr=v_err, c='k', marker='o')
    axs[4].set_ylim([min_vel-40, max_vel+40])
  

    #axs[3].fill_between(d, v_profile - stds, v_profile +stds, facecolor='r', alpha=0.2)
   
    #2.2 and 3Re lines
    axs[4].axvline(r22_disk, linestyle='dotted', c='k')
    axs[4].axvline(r3_disk, linestyle='dashed', c='k')
    axs[4].axvline(-1.0*r22_disk, linestyle='dotted', c='k')
    axs[4].axvline(-1.0*r3_disk, linestyle='dashed', c='k')

    axs[4].annotate(r'2.2 R$_{\mathrm{d}}$', xy=(-1.0*r22_disk, 0.3*max_vel), xytext=(2, 10), textcoords='offset points', horizontalalignment='left', verticalalignment='top')
    axs[4].annotate(r'3 R$_{\mathrm{d}}$', xy=(-1.0*r3_disk, 0.6*max_vel), xytext=(2, 10), textcoords='offset points', horizontalalignment='left', verticalalignment='top')

    #axs[3].imshow(final_errors, origin='lower', cmap=sauron, vmin=min_vel, vmax=max_vel)
    #Titles
    axs[0].set_title(r'H$\alpha$', fontsize=25, loc='left')
    axs[1].set_title('Data', fontsize=25, loc='left')
    axs[2].set_title('Binned Model', fontsize=25, loc='left')
    axs[3].set_title('Residuals', fontsize=25, loc='left')

    axs[4].set_title('1-D rotation curve', fontsize=25, loc='left')
    axs[4].set_xlabel(r'$r (^{\prime\prime})$', fontsize=15)
    axs[4].set_ylabel(r'$V_{\mathrm{rot}}$ (kms$^{-1}$)', fontsize=15)

    axs[4].yaxis.set_label_position('right')
    axs[4].yaxis.set_ticks_position('right')

    #cube_image
    collapsed_cube[:2, :]=np.nan
    collapsed_cube[-2:, :]=np.nan
    collapsed_cube[:, :2]=np.nan
    collapsed_cube[:, -2:]=np.nan

    yp, xp=np.where(light_image==np.max(light_image))
    xp_shifted=max_x/2
    yp_shifted=max_y/2

    #import ipdb; ipdb.set_trace()
    # xp_rotated, yp_rotated=KF.rotate_coordinates(xp-max_x/2, yp-max_y/2, PA)
    # xp_rotated+=max_x/2
    # yp_rotated+=max_y/2

    cmap = plt.cm.hot
    cmap.set_bad('k')

    axs[0].imshow(collapsed_cube, cmap=cmap, vmin=np.nanpercentile(collapsed_cube, 10), vmax=np.nanpercentile(collapsed_cube, 90), origin='lower')
    #These values come from trial and error- np.sum(light[light>x*peak])/np.sum(light)=0.5, 0.8. 0.505 and 0.2202 work for x
    axs[0].contour(gaussian_fit, colors='k', linestyles=['dashed', 'dashed', 'solid'], levels=[settings.fraction_of_peak*np.max(gaussian_fit), 0.2202*np.max(gaussian_fit), 0.505*np.max(gaussian_fit)])


    #Add centre of Ha flux
    
    axs[1].scatter(xp_shifted, yp_shifted, c='w', marker='s', s=100, linewidths=2.0, edgecolors='k')
    axs[2].scatter(xp_shifted, yp_shifted, c='w', marker='s', s=100, linewidths=2.0, edgecolors='k')
    axs[3].scatter(xp_shifted, yp_shifted, c='w', marker='s', s=100, linewidths=2.0, edgecolors='k')
    axs[0].scatter(xp, yp, c='w', marker='s', s=100, linewidths=2.0, edgecolors='k')


    #Plot the slit
    for ax in axs[1:4]:

        ax=plot_slit(ax, PA, x_slit, xp_shifted, yp_shifted)
        ax.tick_params(axis='both', which='both', labelbottom='off', labelleft='off')
        ax.set_ylim([0.0, max_x])
        ax.set_xlim([0.0, max_y])

    #And now for the cube image- which goes through a different point
    for ax in [axs[0]]:

        ax=plot_slit(ax, PA, x_slit, xp, yp, c='0.1')
        #ax=plot_slit(ax, PA_ellipse, x_slit, xp, yp,  plot_width=False, c='0.5', alpha=0.8)

        ax.set_ylim([0.0, collapsed_cube.shape[0]])
        ax.set_xlim([0.0, collapsed_cube.shape[1]])



    #Add the circles
    for ax in axs[1:3]:
        circle1 = plt.Circle((max_x/2, max_y/2), r22_disk/0.1, facecolor='None', edgecolor='k', linestyle='dotted', alpha=0.8)
        circle2 = plt.Circle((max_x/2, max_y/2), r3_disk/0.1, facecolor='None', edgecolor='k', linestyle='dashed', alpha=0.8)
        ax.add_artist(circle1)
        ax.add_artist(circle2)
    for ax in [axs[0]]:
        circle1 = plt.Circle((xp, yp), r22_disk/0.1, facecolor='None', edgecolor='k', linestyle='dotted', alpha=0.8)
        circle2 = plt.Circle((xp, yp), r3_disk/0.1, facecolor='None', edgecolor='k', linestyle='dashed', alpha=0.8)
        ax.add_artist(circle1)
        ax.add_artist(circle2)

    #name
    fig.text(x=0.05, y=0.5, s='{}'.format(label), rotation='vertical', va='center', ha='center', fontsize=25) 
    
    for i, ax in enumerate(axs):
        x0,x1 = ax.get_xlim()
        y0,y1 = ax.get_ylim()

        if i in [0, 1, 2, 3]:
            ax.set_xticklabels([])
            ax.set_yticklabels([])
        
        ax.set_aspect((x1-x0)/(y1-y0)) 
        
    fig.subplots_adjust(hspace=0.2, wspace=0.16)


    return (fig, axs), np.nanmax(rotated_smooth_model)