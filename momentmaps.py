from __future__ import print_function
import numpy as np

def filter_cube(data, fregion, siglevel=2):
    '''
    Filter the noise at a given sigma level, default to 2
    '''
    aa = fregion
    noise = data[:, aa[0]:aa[1], aa[2]:aa[3]]
    sig = np.median(np.median(noise, axis=2), axis=1)* siglevel

    sig = np.reshape(sig, (data.shape[0], 1, 1))
    new_data = data.copy()
    new_data[data<=sig] = 0.
    
    return new_data
def zero_moment(data, vel, vmin, vmax, calNHI=True, 
                dofilter=False, fregion=[0, 10, 0, 10]): 
    '''
    Integrate the cube over a [vmin, vmax] velocity range.

        NHI = 1.823e18 * init_vmin^vmax (Tv dv), where Tv is the pixel values 
        of the cube, and dv is the velocity step of the integration.
        check here: http://www.atnf.csiro.au/people/Tobias.Westmeier/tools_hihelpers.php#velocity
    Input: 
          data: 3D data cube with dimension of (velocity, coordinate1, coordinate2)
          vel: 1D velocity vector. same size as data.shape[0]
          vmin, vmax: the integration range. 
          
          dofilter: if True, for each velocity slice, set noise pixels to zero
          fregion: if do filter set true, this parameter has to be set. 
                   [ax1left, ax1right, ax2bottom, ax2top] the region to 
                   estimate the noise for each velocity slice. 2 sigma filter
    Return: 
          2D image of HI column density map
    History: updated as of 2016.10.03. Yong Zheng @ Columbia Astro
    '''

    k2cm2 = 1.823e18  # from K km/s to cm-2. Can be found in the ISM book, Draine 2004

    ind = np.all([vel>=vmin, vel<=vmax], axis=0)
    new_d = data[ind, :, :]
    new_v = vel[ind]

    delv = np.fabs(np.mean(new_v[1:]-new_v[:-1]))  # the velocity step

    if dofilter == True: new_d = filter_cube(new_d, fregion)

    colden = (new_d*delv).sum(0) 
    if calNHI == True: colden = colden*k2cm2

    return colden

def first_moment(data, vel, vmin, vmax, 
                 dofilter=False, fregion=[0, 10, 0, 10]):
    '''
    Flux-weighted velocity, calculated within vmin and vmax. 

        fluxwt_vel = init_vmin^vmax (Tv * vel *dv) / init_vmin^vmax(Tv * dv)
        check here: http://www.atnf.csiro.au/people/Tobias.Westmeier/tools_hihelpers.php#velocity
    Input: 
          data: 3D data cube with dimension of (velocity, coordinate1, coordinate2)
          vel: 1D velocity vector. same size as data.shape[0]
          vmin, vmax: the integration range. 
    Return: 
          2D image of flux-weighted velocity
    History: updated as of 2016.10.03. Yong Zheng @ Columbia Astro 
    '''

    ind = np.all([vel>=vmin, vel<=vmax], axis=0)
    new_d = data[ind, :, :]
    new_v = np.reshape(vel[ind], (vel[ind].size, 1, 1))

    if dofilter == True: new_d = filter_cube(new_d, fregion)
    # delv = np.fabs(np.mean(new_v[1:]-new_v[:-1]))  # the velocity step

    top_init = (new_v * new_d).sum(axis=0)  # both top/bottom needs a delv, but they cancel in the end
    bottom_init = new_d.sum(axis=0)       # so ignore delv here

    # to avoid 0 in the denominator
    ind0 = bottom_init==0.
    bottom_init[ind0] = 1.
    fluxwt_vel = top_init/bottom_init
    
    # get rid of pixels with peculiar velocities after the division
    fluxwt_vel[np.isnan(fluxwt_vel)] = -10000
    ind1 = fluxwt_vel<vmin
    ind2 = fluxwt_vel>vmax
        

    ind12 = np.any([ind1, ind2], axis=0)
    indnan = np.any([ind12, ind0], axis=0)

    fluxwt_vel[indnan] = np.nan

    return fluxwt_vel

def second_moment(data, vel, vmin, vmax, 
                  dofilter=False, fregion=[0, 10, 0, 10]):
    '''
    Velocity disperson map, flux-weighted square of the velocity. 

    It is the line width of the spectral line along the line of sightline, calculated within vmin and vmax. 
        fluxwt_sigV = sqrt(init_vmin^vmax [(vel-fluxwt_vel)^2 *dv]) / init_vmin^vmax(Tv * dv))
        check here: http://www.atnf.csiro.au/people/Tobias.Westmeier/tools_hihelpers.php#velocity
    Input: 
          data: 3D data cube with dimension of (velocity, coordinate1, coordinate2)
          vel: 1D velocity vector. same size as data.shape[0]
          vmin, vmax: the integration range. 
    Return: 
          2D image of flux-weighted velocity dispersion, i.e., line width
    History: updated as of 2016.10.03. Yong Zheng @ Columbia Astro 
             **************** Haven't fully tested *******************
             **************** use with caution ***********************
    '''

    ind = np.all([vel>=vmin, vel<=vmax], axis=0)
    new_d = data[ind]

    if dofilter == True: new_d = filter_cube(new_d, fregion)

    new_v = np.reshape(vel[ind], (vel[ind].size, 1, 1))
    delv = np.fabs(np.mean(new_v[1:]-new_v[:-1]))  # the velocity step
 						   # this both show up in numerator and denominator
						   # so ingore
   
    fluxwt_vel = first_moment(data, vel, vmin, vmax)  # first calculate the mean velocity along los 
    indnan = np.isnan(fluxwt_vel)     # to record those nan in the first moment map 
    fluxwt_vel[indnan] = 0.   # these nan pixels will be changed back to nan values in the end
    fluxwt_vel = np.reshape(fluxwt_vel, (1, fluxwt_vel.shape[0], fluxwt_vel.shape[1]))

    # the integral of the numerator and denominator    
    top_init = ((new_v - fluxwt_vel)**2 * new_d).sum(axis=0)
    bottom_init = new_d.sum(axis=0)

    # to avoid 0 in the denominator
    ind0 = bottom_init==0.
    bottom_init[ind0] = 1.
    sigV_sq = top_init / bottom_init

    # negative probably due to noise; so, non-physcial, make it nan
    indtop = top_init<0.
    indbot = bottom_init<0.

    # now incorporate all those nan values
    sigV_sq[ind0] = np.nan    # from denominator = 0.
    sigV_sq[indnan] = np.nan  # from first moment map
    sigV_sq[indtop] = np.nan  # from init top < 0.
    sigV_sq[indbot] = np.nan  # from init bot < 0.
   
    # find the dispersion
    sigV = np.sqrt(sigV_sq)
    return sigV



