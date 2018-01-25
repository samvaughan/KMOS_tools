
from scipy.special import iv
from scipy.special import kv
from scipy import ndimage as ndi
import numpy as np 
import numpy.ma as ma
import lmfit_SPV as LMSPV
from scipy import ndimage as ndi


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
    #r0=params['r0'].value
    #s0=params['log_s0'].value
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


def aggregate_bins(model, bins):

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

    
    # xc=params['xc'].value
    # yc=params['xc'].value
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


def objective_function(params, data, errors, x, y, bins):

    model=make_binned_model(params, data, x, y, bins)

    residuals=((model[~data.mask]-data[~data.mask])/errors[~data.mask]).flatten()

    return ma.getdata(residuals)

def lnprob(params, data, errors, x, y, bins):

    # a=np.exp(params['lnA'].value)
    # tau=np.exp(params['lnTau'].value)
    # gp = george.GP(a * kernels.Matern32Kernel(tau, ndim=2))

    # T=np.array([np.ravel(x[~data.mask]), np.ravel(y[~data.mask])]).T
    # E=errors[~data.mask].ravel()    
    # gp.compute(T, E)

    
    #Bin the model in the same way as the data


    model=make_binned_model(params, data, x, y, bins)

    residuals=(((model[~data.mask]-data[~data.mask])/errors[~data.mask]).flatten())**2

    likelihood=-0.5*np.sum(residuals) #- 0.5*np.sum(np.log(errors[~data.mask]))

    
    return likelihood



def make_binned_model(params, data, x, y, bins):
    model=velfield(params, data)
    binned_model=aggregate_bins(model.ravel(), bins)
    binned_model_2d=display_binned_quantity(y, x, binned_model[bins])

    return binned_model_2d

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


def shift_rotate_velfield(velfield, centre,PA, order=3):


    shifted_data=ndi.shift(velfield, centre, mode='nearest', order=order)
    final_data=ndi.rotate(shifted_data,PA, reshape=False,mode='nearest', order=order)

    return final_data



def get_errors_on_fit(params, data, errors, chain_samples, x, y, bins):

    yc=int(params['yc'].value)
    xc=int(params['xc'].value)
    PA=params['PA'].value
    max_y, max_x=data.shape
    nan_mask=data< -9000

    #model=make_binned_model(params, data, x, y, bins)
    model=velfield(params, data)
    final_model=shift_rotate_velfield(model, [max_x/2-xc, max_y/2-yc], PA, order=0)

    v_profile, v_obs, v_err, [x_slit, y_slit]=get_slit_profile(params, data, final_model, errors)

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

        #model=make_binned_model(sample_pars, data, x, y, bins)
        model=velfield(sample_pars, data)
        final_model=shift_rotate_velfield(model, [max_x/2-xc, max_y/2-yc], PA, order=0)
        final_model[nan_mask]=np.nan

        v_p, v_obs, v_err, [_, _]=get_slit_profile(sample_pars, data, final_model, errors)
        all_profiles[:, i]=v_p
        #import pdb; pdb.set_trace()
    
    stds=2.0*np.nanstd((all_profiles-v_profile[:, None]), axis=1)

    return stds
