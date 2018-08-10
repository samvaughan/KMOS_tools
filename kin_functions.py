
from scipy.special import iv, kv
import numpy as np 
import numpy.ma as ma
import lmfit_SPV as LMSPV
from scipy import ndimage as ndi
from scipy import signal
from astropy.convolution import convolve
from astropy.convolution import Gaussian2DKernel   

import settings
# #Likelihood function here: saves pickling the parameters dictionary
# def _lnprob(T, theta, var_names, bounds, data, errors, x, y, bins, a, s):

#     #Log prob function. T is an array of values

    
#     assert len(T)==len(var_names), 'Error! The number of variables and walker position shapes are different'

#     #Prior information comes from the parameter bounds now
#     if np.any(T > bounds[:, 1]) or np.any(T < bounds[:, 0]):
#         return -np.inf


#     #make theta from the emcee walker positions
#     for name, val in zip(var_names, T):
#         theta[name].value = val

#     ll=KHF.test_lnlike(theta, data, errors, x, y, bins, a, s)
#     return ll

def rotate_coordinates(x, y, theta):

    X=np.cos(theta)*x-np.sin(theta)*y
    Y=np.sin(theta)*x+np.cos(theta)*y

    return X, Y




#@profile(immediate=True)
def velfield(params, data, oversample):

    """
    Make a 2d array containing a velocity field. This Velocity field is oversampled 10x compared to the KMOS spaxel resolution.
    
    * Take a set of X,Y coordinates. These are _larger_ than the data we're trying to fit- we pad them such that we can shift the velocity map
    to the centre at the end.
    * Rotate these coordinates by angle PA degrees. 
    * Make a velocity map in terms of these rotated coordinates, with the centre at data.shape/2! NOT at the required xc, yc coordinates yet.
    * Finally, shift using ndi.shift
    * Crop away the padded 
    """

    #This is the 'angular eccentricity'
    #Shapes the flattening of the elliptical coordinates
    #cos(theta) is just b/a for the ellipse
    #sin(theta) is sqrt(1-b**2/a**2), or the eccentricity e
    #Should limit a=5b for reasonable galaxies

    assert type(oversample)==int, 'Oversample must be an integer'


    PA=params['PA'].value
    xc=params['xc'].value
    yc=params['yc'].value
    v0=params['v0'].value
    PA=params['PA'].value
    PA_rad=PA*np.pi/180.


    #Get coordinate axes
    max_shift=settings.max_centre_shift
    ys=np.linspace(0-max_shift, data.shape[0]+max_shift, oversample*(data.shape[0]+2*max_shift))
    xs=np.linspace(0-max_shift, data.shape[1]+max_shift, oversample*(data.shape[1]+2*max_shift))
    X_0, Y_0=np.meshgrid(xs, ys)


    #Shift things to the centre, rotate them by PA, then shift back
    centre_x=data.shape[0]/2.0
    centre_y=data.shape[1]/2.0

    X_r, Y_r=rotate_coordinates(X_0-centre_x, Y_0-centre_y, PA_rad)

    X=X_r+centre_x
    Y=Y_r+centre_y


    

    # X-=shift[0]
    # Y-=shift[1]






    # #Now rotate and move to the right place
    # #Get the integer shift and the float shift

    # xc_int=int(xc)
    # yc_int=int(yc)
    # xc_float=xc-xc_int
    # yc_float=yc-yc_int
    
    # #These shift units are in pixels of the *oversampled* array
    # #So we need to scale our shift by the value of oversample
    # shift=np.array([yc-np.mean(Y_old), xc-np.mean(X_old)])*oversample
    # PA=params['PA'].value
   
    # velfield_final=shift_rotate_velfield(velfield, shift, 0.0, order=0)
    


    # print(xc, yc)
    # print(centre_x, centre_y)
    # xc_prime, yc_prime=rotate_coordinates(xc, yc, PA_rad)
    # # xc_prime+=centre_x
    # # yc_prime+=centre_y
    # print(xc_prime, yc_prime)

    # import matplotlib.pyplot as plt 
    # import ipdb; ipdb.set_trace()

    #Intrinisc viewing angle of the disk
    theta=params['theta'].value
    theta_rad=theta*np.pi/180.

    

    #Get the simple axisymetric velfield, then scale by (X-centre_of_array)/R)
    R = np.sqrt((X-centre_x)**2 + ((Y-centre_y)/np.cos(theta_rad))**2)
    velfield= v_circ_exp_quick(R, params)*(X-centre_x)/(R*np.sin(theta_rad))






    #Shift the velfield to where it should be
    shift=np.array([yc-centre_y, xc-centre_x])*oversample
    velfield=ndi.shift(velfield, shift, order=0, mode='nearest', cval=np.nan)
    #velfield = rotateImage(velfield, PA, [xc, yc])
    #velfield_final = ndi.rotate(velfield,PA, order=0, mode='nearest', reshape=False)

    #Crop away everything which we don't need- larger than the original data
    m_x=(X_0>0)&(X_0<data.shape[1])
    m_y=(Y_0>0)&(Y_0<data.shape[0])
    m_t=(m_x)&(m_y)

   

    velfield_final=velfield[m_t].reshape(oversample*data.shape[0], oversample*data.shape[1])

    # import matplotlib.pyplot as plt 
    # import ipdb; ipdb.set_trace()

    return velfield_final


def rotateImage(img, angle, pivot):
    padX = [int(img.shape[1] - pivot[0]), int(pivot[0])]
    padY = [int(img.shape[0] - pivot[1]), int(pivot[1])]
    imgP = np.pad(img, [padY, padX], 'constant')
    imgR = ndi.rotate(imgP, angle, reshape=False)
    return imgR[padY[0] : -padY[1], padX[0] : -padX[1]]

def shift_rotate_velfield(velfield, shift,PA,**kwargs):

    order=kwargs.pop('order', 0)
    reshape=kwargs.pop('reshape', False)
    mode=kwargs.pop('mode', 'constant')
    cval=kwargs.pop('cval', np.nan)


    #Rotate to the right PA
    velfield_shifted = ndi.shift(velfield, shift, order=order, mode=mode, cval=cval)
    velfield_final = ndi.rotate(velfield_shifted,PA, order=order, mode=mode, cval=cval, reshape=reshape, **kwargs)
    

    return velfield_final




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
    log_R0=params['log_r0'].value
    R0 = 10**log_R0
    log_s0=params['log_s0'].value
    s0  = 10**log_s0
    # evaluate bessel functions (evaluated at 0.5aR; see Freeman70)

    half_a_R=(0.5*(R)/R0)

    #temp[temp>709.]=709.

    #Bessel Functions
    #Interpolate to speed up!
    I0K0 = settings.interpI0K0(half_a_R)
    I1K1 = settings.interpI1K1(half_a_R)

    #bsl  = I0K0 - I1K1

    

    # velocity curve
    V_squared  =  R*((np.pi*G*s0)*(I0K0 - I1K1)/R0)
    V=np.sqrt(V_squared)   

    return V



def get_slit_profile(params, data, binned_model, smooth_model, noise, stripe=5):


    import matplotlib.pyplot as plt 

    lower=np.floor(stripe/2.0).astype(int)
    upper=np.ceil(stripe/2.0).clip(1.0).astype(int) #If this is 0 then array[15:15] gives []
    

    PA=params['PA']
    xc=params['xc'].value
    yc=params['yc'].value
    max_y, max_x=data.shape
    

    #Slit along PA: y=mc+c, where m=tan(y/x), c=yc-xc*tan(theta)
    x_slit=np.arange(max_x).astype(int)
    y_slit=np.full_like(x_slit, max_y/2)
    # #y_slit=np.ones_like(x_slit)*max_y/2+1
    # y_slit=x_slit*np.tan(PA*np.pi/180.0)+yc-(xc*np.tan(PA*np.pi/180.0))
    # y_slit=y_slit.astype(int)



    # #mask so we don't go outside of the data
    # #subtract 0.5 to ensure that the last valye of y doesn't get rounded up and throw an error
    # mask=(y_slit>0)&(y_slit<max_y-0.5)



    #take a stripe along the PA axis of 5 pixels (0.5") and median/mean combine
    #with inverse variance weighting
    s=data.shape
    

    low_index=int(s[0]/2-(lower))
    high_index=int(s[0]/2+(upper))

    ivars=1./(noise[low_index:high_index, :]**2)
    vels=data[low_index:high_index, :]
    v_obs=np.nansum(ivars*vels, axis=0)/np.nansum(ivars, axis=0)

    if data.size!=smooth_model.size:

        low_index=int(s[0]/2-(lower*settings.oversample))
        high_index=int(s[0]/2+(upper*settings.oversample))

        v_profile_binned=np.nanmean(binned_model[low_index:high_index, :], axis=0)[::settings.oversample]
        v_profile_smooth=np.nanmean(smooth_model[low_index:high_index, :], axis=0)[::settings.oversample]
        x_slit=np.linspace(0.0, max_x, model.shape[0]).astype(int)
    else:
        v_profile_binned=np.nanmean(binned_model[low_index:high_index, :], axis=0)
        v_profile_smooth=np.nanmean(smooth_model[low_index:high_index, :], axis=0)
    #v_profile=np.nansum(model[s[0]/2-(lower):s[0]/2+(upper), :]*ivars, axis=0)/np.nansum(ivars, axis=0)
    #v_obs=np.nanmean(, axis=0)
    #v_err=np.sqrt(np.nansum(noise[s[0]/2-2:s[0]/2+3, :]**2, axis=0))
    v_err=np.sqrt(1./np.nansum(ivars, axis=0)) 


    #import ipdb; ipdb.set_trace()


    return v_profile_binned, v_profile_smooth, v_obs, v_err, [x_slit, y_slit]

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

def aggregate_bins(model, bins, light_image):




    vals=np.empty(len(np.unique(bins[bins>=0]))+1)
    for i in np.unique(bins[bins>=0]):
        mask=np.where(bins==i)
        vals[i]=np.nansum((model*light_image)[mask])/np.nansum(light_image[mask])
    #Sort out the fact that unbinned pixels have bin number -1
    vals[-1]=np.nan

    return vals




def get_fluxweighted_sigma(sigma_values, sigma_errors, light_image):


    #Need to also correct for beam smearing and the instrumental resolution!

    return np.nansum(sigma_values*light_image/sigma_errors**2)/np.nansum(light_image/sigma_errors**2)


# def objective_function(params, data, errors, x, y, bins, light_image, seeing_conv, oversample=1):

#     model=make_binned_model(params, data, x, y, bins, light_image=light_image, seeing_conv=seeing_conv, oversample=oversample)

#     residuals=((model[~data.mask]-data[~data.mask])/errors[~data.mask]).flatten()

#     return ma.getdata(residuals)

def lnprob(params, data, errors, x, y, bins, light_image, seeing_pixels, oversample):

    # a=np.exp(params['lnA'].value)
    # tau=np.exp(params['lnTau'].value)
    # gp = george.GP(a * kernels.Matern32Kernel(tau, ndim=2))

    # T=np.array([np.ravel(x[~data.mask]), np.ravel(y[~data.mask])]).T
    # E=errors[~data.mask].ravel()    
    # gp.compute(T, E)

    #print seeing_conv
    #Bin the model in the same way as the data
    model=make_binned_model(params, data, x, y, bins, light_image, seeing_pixels, oversample)
    
    residuals=(((model[~data.mask]-data[~data.mask])/errors[~data.mask]).flatten())**2

    likelihood=-0.5*np.sum(residuals) #- 0.5*np.sum(np.log(errors[~data.mask]))
    #print '{} seconds'.format(time.time()-T)
    
    
    return likelihood




def make_binned_model(params, data, x, y, bins, light_image, seeing_pixels, oversample):
    """
    Make an oversampled velocity map, which we then sample down to the KMOS bins
    """

    hires_model=velfield(params, data, oversample)



    if seeing_pixels is not None:    
 
        seeing_sigma=seeing_pixels*oversample/np.sqrt(8*np.log(2))
        #kernel = Gaussian2DKernel(seeing_sigma)
        #model=convolve(model, kernel)
        model=ndi.gaussian_filter(hires_model, seeing_sigma)
    
   
    
    model_reshaped=model.reshape(np.max(y)+1, oversample, np.max(x)+1, oversample)
    downsampled_model=np.nanmean(np.nanmean(model_reshaped, axis=-1), axis=1)

    binned_model=aggregate_bins(downsampled_model.ravel(), bins, light_image.ravel())
    binned_model_2d=display_binned_quantity(y, x, binned_model[bins])
  

    return binned_model_2d

# def lnlike_covariance(params, data, errors, x, y, bins):
    

#     a=np.exp(params['ln_a'])
#     s=np.exp(params['ln_s'])

#     model=velfield(params, data)

#     #Bin the model in the same way as the data
#     bin_mask=np.where(bins>=0)
#     binned_model=display_binned_quantity(x[bin_mask], y[bin_mask], model[bin_mask])

#     residuals=(binned_model[~data.mask]-data[~data.mask])

#     r2=(x[~data.mask, None]-x[None, ~data.mask])**2+(y[~data.mask, None]-y[None, ~data.mask])**2

#     C=np.diag(errors[~data.mask]**2) + a*np.exp(-0.5*r2/s*s)

#     factor, flag=cho_factor(C)

#     logdet=2*np.sum(np.log(np.diag(factor)))

#     lnlike=-0.5*(np.dot(residuals, cho_solve((factor, flag), residuals))+ logdet + len(x)*np.log(2*np.pi))


#     return lnlike






def get_errors_on_fit(params, data, errors, chain_samples, x, y, bins):

    yc=int(params['yc'].value)
    xc=int(params['xc'].value)
    PA=params['PA'].value
    max_y, max_x=data.shape
    nan_mask=~np.isfinite(data)

    #model=make_binned_model(params, data, x, y, bins)
    model=velfield(params, data)
    final_model=shift_rotate_velfield(model, [max_x/2-xc, max_y/2-yc], PA, order=0)

    v_profile, v_obs, v_err, [x_slit, y_slit]=get_slit_profile(params, data, final_model, errors)

    #Overlay error regions for the fit
    ndraws=1000
    variables=[v for v in list(params.keys()) if params[v].vary]
    fixed_vals=[v for v in list(params.keys()) if not params[v].vary]

    all_profiles=np.empty((len(v_profile), ndraws))

    for i, pars in enumerate(chain_samples[np.random.randint(len(chain_samples), size=ndraws)]):
        
        sample_pars=LMSPV.Parameters()
        for name in fixed_vals:
            sample_pars.add(name, value=params[name])
        for p, name in zip(pars, variables):
            sample_pars.add(name, value=p)
        sample_pars['PA'].set(params['PA'].value)
        
        # p_xc=sample_pars['xc']
        # p_yc=sample_pars['yc']

        #model=make_binned_model(sample_pars, data, x, y, bins)
        model=velfield(sample_pars, data)
        final_model=shift_rotate_velfield(model, [yc-max_y/2, xc-max_x/2], PA, order=0)
        final_model[nan_mask]=np.nan

        v_p, v_obs, v_err, [_, _]=get_slit_profile(sample_pars, data, final_model, errors)
        try:
            all_profiles[:, i]=v_p
        except ValueError:
            #Not sure why this is necessary?
            continue
        
    
    stds=2.0*np.nanstd((all_profiles-v_profile[:, None]), axis=1)

    return stds






