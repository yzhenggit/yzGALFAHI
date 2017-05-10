def bootstrap_linearfit(xdata, ydata):
    import numpy as np
    import numpy.random as npr

    if np.where(xdata>0.5)[0].size == 0 or np.where(ydata>0.5)[0].size == 0:
	return (0., 0., 0., 0.) 
    else:
    	# generate N samples from the orginal dataset, N = yy2fit.size
	tofit_ind = xdata>0.5
	xx = xdata[tofit_ind]
	yy = ydata[tofit_ind]
    	n_samples = yy.size
    	inds = npr.randint(0, yy.size, (n_samples, yy.size))
    	samples_xx = xx[inds]
    	samples_yy = yy[inds]
    
    	# y = ax + b
    	a, b = np.zeros(n_samples), np.zeros(n_samples)
    	for i in range(n_samples):
            a[i], b[i] = np.polyfit(samples_xx[i, :], samples_yy[i, :], 1)        
    	return (a.mean(), np.std(a), b.mean(), np.std(b))

def least_square_fit(xdata, ydata):
    import numpy as np

    # linear model: y = ax+b, b should be 0
    y = np.transpose(np.asmatrix(ydata))
    x = np.transpose(np.asmatrix(xdata))
    desM = np.asmatrix(np.ones((xdata.size, 2)))
    desM[:, 0] = x
    desMM = np.dot(np.transpose(desM), desM)
    desMMi = np.linalg.inv(desMM)
    desM_y = np.dot(np.transpose(desM), y)
    ab = np.dot(desMMi, desM_y)

    ybar = np.dot(desM, ab)
    dely = y - ybar
    s_sq = np.dot(np.transpose(dely), dely)/(xdata.size-2)
    s_sq_a = s_sq*desMMi[0, 0]
    s_sq_b = s_sq*desMMi[1, 1]
    return (ab[0], np.sqrt(s_sq_a), ab[1], np.sqrt(s_sq_b))

def GALFAHI_cubeinfo(fitsfile):
    # This function yield the same RA/DEC/VLSR as using wcs to process
    from astropy.io import fits
    import numpy as np
    
    data = fits.getdata(fitsfile)
    header = fits.getheader(fitsfile)
    ra = header['CRVAL1'] + header['CDELT1'] * (
		np.arange(data.shape[2])+1 - header['CRPIX1'])             # degree
    dec = header['CRVAL2'] + header['CDELT2'] * (
		np.arange(data.shape[1])+1 - header['CRPIX2'])             # degree
    vlsr = (header['CRVAL3'] + header['CDELT3'] * (
		np.arange(data.shape[0])+1 - header['CRPIX3']))*(10**(-3)) # km/s
    delv = header['CDELT3']*(10**(-3))  				   # km/s
    return data, header, vlsr, dec, ra, delv

def relative_gain(cube1, cube2, vlsr, npixel=4, vlim=150):
    # gains = 0:  bad data, Tb<=0.5 K
    # gains nealy 1.: good data, with recognisable signal
    # gains = nan: no data
    import numpy as np
    if cube1.size != cube2.size:
	return 0

    gains = np.zeros((cube1.shape[1], cube1.shape[2]))
    velind = np.all([vlsr > -vlim, vlsr < vlim], axis=0)     	  # data beyond [-150, 150] mostly contains noise
    for i in np.mgrid[0:cube1.shape[2]:npixel]:
        for j in np.mgrid[0:cube1.shape[1]:npixel]:
            sub_cube1 = cube1[velind, j:j+npixel, i:i+npixel]
            sub_cube2 = cube2[velind, j:j+npixel, i:i+npixel]
            ds = sub_cube1.shape
            ispec1 = np.median(np.reshape(sub_cube1, (ds[0], ds[1]*ds[2])), axis=1)
            ispec2 = np.median(np.reshape(sub_cube2, (ds[0], ds[1]*ds[2])), axis=1)
        
            ispec1_allnan = np.where(np.isnan(ispec1)==True)[0].size == ispec1.size
            ispec2_allnan = np.where((np.isnan(ispec2)==True))[0].size == ispec2.size
            if ispec1_allnan or ispec2_allnan:
                gains[j:j+npixel, i:i+npixel] = np.nan
            else:
		xind5 = ispec1>=0.5	# ignore signals with T<0.5 K to avoid stray radiation 
    		yind5 = ispec2>=0.5	
    		ind5 = np.logical_and(xind5, yind5)
    		xdata = ispec1[ind5]
    		ydata = ispec2[ind5]
    		if xdata.size == 0 or ydata.size == 0:
       		    gains[j:j+npixel, i:i+npixel] = np.nan
		else:
                    gain, a, b, c = least_square_fit(xdata, ydata)
                    gains[j:j+npixel, i:i+npixel] = gain
                
    return gains

def find_median_relative_gain(gain21, cubeinfo1, cubeinfo2, yz_dir, time_cut=4, make_plot = True):
    import os 
    import numpy as np
    import matplotlib.pyplot as plt

    from scipy.optimize import curve_fit
    def gaussfit(x, A, mu, sigma):
	import numpy as np
    	return A*np.exp(-(x-mu)**2/(2.*sigma**2))
    
    non_nan = np.logical_not(np.isnan(gain21))
    time1 = cubeinfo1[-2][non_nan]
    time2 = cubeinfo2[-2][non_nan]
    min_time = np.array([time1, time2]).min(axis = 0)
    temp_gain = gain21[non_nan]
    # if the two cubes actually has no overlaps
    if temp_gain.size == 0.:
	gain_median_all = np.nan
    	sig_all = np.nan
    else:
    	gain_median_all = np.median(temp_gain)	
    	sig_all = np.std(temp_gain) 

    temp_gain_timecut = temp_gain[min_time >= time_cut]
    # if none of the overlap pixels have integration time larger than time_cut
    if temp_gain_timecut.size == 0:
	gain_median_timecut = np.nan
	sig_timecut = np.nan
    else:
    	gain_median_timecut = np.median(temp_gain_timecut)
    	sig_timecut = np.std(temp_gain_timecut)

    # from astroML import stats
    # sigmaG = stats.sigmaG(temp_gain_timecut)

    ## delete "if" block after testing
    if make_plot is False:
	coeff = [np.nan, np.nan, np.nan]    
    else:
        cube1 = cubeinfo1[0]
        cube2 = cubeinfo2[0]
        cubeshape = cubeinfo1[0].shape
        ra = cubeinfo1[4]
        dec = cubeinfo1[3]
        fig = plt.figure(figsize=(12, 8))
        imext = [ra.max(), ra.min(), dec.min(), dec.max()]
        # plot the two cubes together, only the velocity = 0 slices are shown
        ax0 = fig.add_axes([0.05, 0.55, 0.4, 0.4])
        ax0.imshow(cube1[cubeshape[0]/2, :, :], extent=imext, origin='lower', cmap=plt.cm.Greys)
        ax0.imshow(cube2[cubeshape[0]/2, :, :], extent=imext, origin='lower', cmap=plt.cm.Reds, alpha = 0.5)
        ax0.set_xlabel('RA')
        ax0.set_ylabel('DEC')
        ax0.set_title('%s & %s: Tb(v=0)' % (cubeinfo1[-1], cubeinfo2[-1]))

        # the gains between these two regions
        ax1 = fig.add_axes([0.05, 0.07, 0.4, 0.4])
        im1 = ax1.imshow(gain21, extent=imext, origin='lower', cmap=plt.cm.jet)
        ax1.set_xlabel('RA')
        ax1.set_ylabel('DEC')
        ax1.set_title('%s & %s: gains' % (cubeinfo1[-1], cubeinfo2[-1]))
        cax = fig.add_axes([0.39, 0.07, 0.02, 0.4])
        cb = fig.colorbar(im1, orientation='vertical', cax=cax)
        cb.set_clim(np.median(temp_gain)-0.1, np.median(temp_gain)+0.1)

        ###### plot the integration time of the image
	if temp_gain.size != 0:
            ax2 = fig.add_axes([0.5, 0.07, 0.3, 0.25])
            ax2.scatter(temp_gain, min_time, s=3, alpha=0.1, color='k')
            ax2.hlines(time_cut, 0.5, 1.5, color='r', linewidth=1.5, linestyle='--')
	    ax2.set_xlim(temp_gain.min(), temp_gain.max())
	    ax2.set_ylim(-1, min_time.max())
            # ax2.set_xlim(coeff[1]-coeff[2]*10, coeff[1]+coeff[2]*10)
            ax2.set_ylabel('Time (s)')

            ######### marginal histgram: right chart ##### 
            ax3 = fig.add_axes([0.81, 0.07, 0.08, 0.25], xticks=[], yticks=[])
            hist4, bins4 = np.histogram(min_time, bins=np.mgrid[-1:max(min_time):0.5])
            ax3.plot(hist4, bins4[:-1], '-k', lw=2)
            ax3.set_ylim(-1, min_time.max())

            ######## marginal histgram: top chart #######
            ax4 = fig.add_axes([0.5, 0.34, 0.3, 0.13], xticks=[], yticks=[])
            hist4, bins4 = np.histogram(temp_gain, bins=np.mgrid[temp_gain.min():temp_gain.max():0.005])
            ax4.plot(bins4[:-1], hist4, '-k', lw=2)
            # ax4.set_xlim(coeff[1]-coeff[2]*10, coeff[1]+coeff[2]*10)
	    ax4.set_xlim(temp_gain.min(), temp_gain.max())

	# make gaussfit for those data with integration time > time_cut
	if temp_gain_timecut.size != 0:
            # histgram of the gains
            ax5 = fig.add_axes([0.5, 0.55, 0.3, 0.4])
            hist, bins, patches = ax5.hist(temp_gain_timecut, normed=1, histtype='bar', color = 'k', 
				       bins=np.linspace(temp_gain_timecut.min(), temp_gain_timecut.max(), 200), 
                                       alpha=0.5, log=True)
            ind = np.arange(bins.size-1)
            bin_centers = (bins[ind] + bins[ind+1])/2.
	
	    p0 = [hist.max(), gain_median_timecut, sig_timecut]
            coeff, var_matrix = curve_fit(gaussfit, bin_centers, hist, p0=p0)
            x2fit = np.mgrid[0.5:1.5:0.001]
            hist_fit = gaussfit(x2fit, *coeff)
            ax5.plot(x2fit, hist_fit, lw=2, color='r')
            ax5.text(coeff[1]-coeff[2]*9, coeff[0]*0.7, r'${\rm med.t%ds=%.4f}$'%(4, gain_median_timecut),
                 	color='k', fontsize = 12)
            ax5.text(coeff[1]-coeff[2]*9, coeff[0]*0.4, r'${\rm sig.t%ds=%.4f}$'%(4, sig_timecut),
                 	color='k', fontsize = 12)
            ax5.text(coeff[1]+coeff[2]*3, coeff[0]*0.7, r'$\mu=%.4f$'%(coeff[1]),
                 	color='r', fontsize = 12)
            ax5.text(coeff[1]+coeff[2]*3, coeff[0]*0.4, r'$\sigma=%.4f$'%(coeff[2]),
                 	color='r', fontsize = 12)
            ax5.set_xlabel('Gain (T>=%ds)'%(time_cut))
            ax5.set_ylim(0.01, coeff[0]*2)
            ax5.set_xlim(coeff[1]-coeff[2]*10, coeff[1]+coeff[2]*10)   

	    # set the xlim of the time-gain plot
	    ax2.set_xlim(coeff[1]-coeff[2]*10, coeff[1]+coeff[2]*10)
	    ax4.set_xlim(coeff[1]-coeff[2]*10, coeff[1]+coeff[2]*10)
        else: 
            coeff = [np.nan, np.nan, np.nan]
	
        gain_dir = '%s/150206_relative_gain/gain_%s_%s' % (yz_dir, cubeinfo1[-1], cubeinfo2[-1])
        if not os.path.isdir(gain_dir): os.makedirs(gain_dir)
        figname = '%s/GALFA_HI_RA+DEC_%s_t%ds.png' % (gain_dir, cubeinfo1[1]['OBJECT'].split(' ')[-1], time_cut)
        fig.savefig(figname)
        plt.close()
        
    return gain_median_all, sig_all, gain_median_timecut, sig_timecut, coeff[1], coeff[2]

def match_cubes(area1, area2):
    # find the overlap cubes between two areas
    f1 = open(area1, 'r')
    f2 = open(area2, 'r')
    d1 = []
    d2 = []
    for line in f1.readlines():
        d1.append(line)
    for line in f2.readlines():
        d2.append(line)
    match = []
    for i in d1:
        for j in d2:
            if i == j:
                match.append(i[:-1])
    return match

# The biggest number of overlap cubes is 4, but it is impossible for one pixel to have 4 
# observations, since there must be one from s(n), and the other two from n(s), 
# and the last one from z. The 4 case only happens in z strip. 
# has a bug in line 228, need to fix later. 
def overlap_2cubes(cubeinfos, i, j, time_frac):
    import numpy as np
    cubei = cubeinfos[i][0]
    cubej = cubeinfos[j][0]
    timei = cubeinfos[i][-2]
    timej = cubeinfos[j][-2]
    slice_sp = (cubei.shape[1], cubei.shape[2])
    slicei = np.logical_not(np.isnan(cubei.mean(0)))
    slicej = np.logical_not(np.isnan(cubej.mean(0)))
    
    overlap_ij = np.logical_and(slicei, slicej)

    if np.where(overlap_ij == True)[0].size != 0:
        temp = np.zeros(slice_sp)
        randf = np.random.random(slice_sp)/2. + 0.5
        ind = np.where(overlap_ij == True)
        
        if np.median(timei[ind]) >= np.median(timej[ind]):
            time_frac[i, ind] = randf[ind]
            time_frac[j, ind] = 1. - randf[ind]
        else:
            time_frac[j, ind] = randf[ind]
            time_frac[i, ind] = 1. - randf[ind]
    return time_frac, overlap_ij
    
def overlap_3cubes(cubeinfos, i, j, k, time_frac):
    import numpy as np
    cubei = cubeinfos[i][0]
    cubej = cubeinfos[j][0]
    cubek = cubeinfos[k][0]

    slice_sp = (cubei.shape[1], cubei.shape[2])
    slicei = np.logical_not(np.isnan(cubei.mean(0)))
    slicej = np.logical_not(np.isnan(cubej.mean(0)))
    slicek = np.logical_not(np.isnan(cubek.mean(0)))
    
    overlap_ijk = np.logical_and(np.logical_and(slicei, slicej), slicek)
    if np.where(overlap_ijk == True)[0].size != 0:
        temp = np.zeros(slice_sp)
        randf1 = np.random.random(slice_sp)/2.
        randf2 = np.random.random(slice_sp)/2.
        ind = np.where(overlap_ijk == True)
        time_frac[i, ind] = randf1[ind]
        time_frac[j, ind] = randf2[ind]
        time_frac[k, ind] = 1 - randf1[ind] - randf2[ind]
    return time_frac, overlap_ijk

def nan2zero(data):
    import numpy as np
    isnan = np.isnan(data)
    ind = np.where(isnan==True)
    if ind[0].size != 0:
        data[ind] = 0.
    return data, isnan

def frac_zns(cubeinfos, extent_belt=False):
    import numpy as np

    shape = cubeinfos[0][-2].shape
    fracz = np.zeros(shape)
    fracn = np.zeros(shape)
    fracs = np.zeros(shape)
    aa = np.mgrid[0:(173-158):1]/(173.-158.)
    bb = np.mgrid[0:(353-339):1]/(353.-339.)
    fracz[158:173, :] = np.reshape(np.repeat(aa, 512), (aa.size, 512))
    fracz[173:339, :] = 1.
    fracz[339:353, :] = np.reshape(np.repeat(1-bb, 512), (bb.size, 512))
    fracn[339:353, :] = np.reshape(np.repeat(bb, 512), (bb.size, 512))
    fracn[353:, :] = 1.
    fracs[0:158, :] = 1.
    fracs[158:173, :] = np.reshape(np.repeat(1-aa, 512), (aa.size, 512))

    # deal with s/z cube, the trim edge
    if extent_belt == True:
    	for i in range(len(cubeinfos)):
            if cubeinfos[i][-1][2] == 'z':
            	z_nodata_s = np.zeros(shape, dtype=bool)
            	z_nodata_s[158:173, :] = np.isnan(cubeinfos[i][0][1024, 158:173, :])
                if np.where(z_nodata_s == True)[0].size != 0:
                    fracz[np.where(z_nodata_s==True)] = 0.
                    fracs[np.where(z_nodata_s==True)] = 1.

            	z_nodata_n = np.zeros(shape, dtype=bool)
            	z_nodata_n[339:353, :] = np.isnan(cubeinfos[i][0][1024, 339:353, :])
            	if np.where(z_nodata_n == True)[0].size != 0:
                    fracz[np.where(z_nodata_n==True)] = 0.
                    fracn[np.where(z_nodata_n==True)] = 1.

            elif cubeinfos[i][-1][2] == 'n':
	    	# data that covered by the n-belt
            	n_nodata = np.zeros(shape, dtype=bool)
            	n_nodata[339:353, :] = np.isnan(cubeinfos[i][0][1024, 339:353, :])
            	if np.where(n_nodata == True)[0].size != 0:
                    fracn[np.where(n_nodata==True)] = 0.
                    fracz[np.where(n_nodata==True)] = 1.

	    	# data that is not covered by the n-belt
	    	n_nodata = np.isnan(cubeinfos[i][0][1024])
	    	n_nodata[0:353, :] = False
	    	fracz[np.where(n_nodata==True)] = 1.
            else: # s
            	s_nodata = np.zeros(shape, dtype=bool)
            	s_nodata[158:173, :] = np.isnan(cubeinfos[i][0][1024, 158:173, :])
            	if np.where(s_nodata == True)[0].size !=0:
                    fracs[np.where(s_nodata==True)] = 0.
                    fracz[np.where(s_nodata==True)] = 1.

	    	# data that is not covered by the s-belt
            	s_nodata = np.isnan(cubeinfos[i][0][1024])
            	s_nodata[158:, ] = False
            	fracz[np.where(s_nodata==True)] = 1.

    return fracz, fracn, fracs

def check_redundancy(areas, cubeinfos):
    import numpy as np

    kickout = []
    for i in range(len(areas)):
        for j in np.mgrid[i+1:len(cubeinfos):1]:
            imi = np.logical_not(np.isnan(cubeinfos[i][0][1024]))
            imj = np.logical_not(np.isnan(cubeinfos[j][0][1024]))

            imij = np.logical_or(imi, imj)
            if np.where((imij==imi)==False)[0].size==0:
                print 'cube %s and cube %s completely overlap, kick out %s' % (areas[i], areas[j], areas[j])
                kickout.append(j)
            elif np.where((imij==imj)==False)[0].size==0:
                print 'cube %s and cube %s overlap, kick out %s' % (areas[i], areas[j], areas[i])
                kickout.append(i)

    new_cubeinfos = []
    new_areas = []
    for k in range(len(areas)):
        if k in kickout:
            print 'skip this one:', areas[k]
        else:
            new_cubeinfos.append(cubeinfos[k])
            new_areas.append(areas[k])

    return new_areas, new_cubeinfos

def coadd_time_weight(cubeinfos):
    import numpy as np

    ref_cube = cubeinfos[0][0]
    ref_time = cubeinfos[0][-2]
    for ic in np.arange(len(cubeinfos))[1:]:
        timei = ref_time
        cubei = ref_cube
        timej = cubeinfos[ic][-2]
        cubej = cubeinfos[ic][0]

        cubei, cisnani = nan2zero(cubei)
        cubej, cisnanj = nan2zero(cubej)

        time_tot = timei+timej
        ind0 = np.where(time_tot==0.)
        if ind0[0].size != 0:
            time_tot[ind0] = -1.
        fraci = np.float32(timei) / np.float32(time_tot)
        fracj = np.float32(timej) / np.float32(time_tot)
        fraci[ind0] = 0.
        fracj[ind0] = 0.

        coadd_2times = timei*fraci + timej*fracj
        coadd_2cubes = cubei*fraci + cubej*fracj

        still_nan_cube = np.logical_and(cisnani, cisnanj)
        if np.where(still_nan_cube==True)[0].size != 0:
            coadd_2cubes[np.where(still_nan_cube==True)] = np.nan

	ref_time = coadd_2times
	ref_cube = coadd_2cubes

    return coadd_2cubes, coadd_2times, still_nan_cube

def coadd_frac_zns(cubeinfos, extent_belt=False):
    import numpy as np

    fz,fn,fs = frac_zns(cubeinfos, extent_belt=extent_belt)
    data = []
    areas = []
    frac = []
    time = []
    isnan_cube = []
    for i in range(len(cubeinfos)):
	data.append(cubeinfos[i][0])
	isnan_cube.append(np.isnan(cubeinfos[i][0]))
        time.append(cubeinfos[i][-2])
        areas.append(cubeinfos[i][-1])
        if cubeinfos[i][-1][2] == 'z':
            frac.append(fz)
        elif cubeinfos[i][-1][2] == 'n':
            frac.append(fn)
        else:
            frac.append(fs)

    if len(areas) == 4:   # there is 2s or 2z or 2n, coadd these first
        # find the duplicate one
        for i in range(len(areas)):
            found = 0
            for j in np.mgrid[i+1:4:1]:
                if areas[i][2] == areas[j][2]:
                    found = 1
                    break
            if found == 1:
                    break

        sub_cubeinfos = [cubeinfos[i], cubeinfos[j]]
        print "First coadding %s and %s with time_weighted_alg." % (areas[i], areas[j])
        coadd_data, coadd_time, still_nan_cube = coadd_time_weight(sub_cubeinfos)
	
        del data[j]
        del data[i]
        data.append(coadd_data)

        del time[j]
        del time[i]
        time.append(coadd_time)

        del isnan_cube[j]
        del isnan_cube[i]
        isnan_cube.append(still_nan_cube)

        last = areas[j]
        del areas[j]
        del areas[i]
        areas.append(last)

        last = frac[i]
        del frac[j]
        del frac[i]
        frac.append(last)

    print "Now, coadding, ", areas, "with frac_zns_alg."
    for icube, jnan in zip(data, isnan_cube):	
	icube[jnan] = 0.

    coadd_cube = data[0]*frac[0] + data[1]*frac[1] + data[2]*frac[2]
    coadd_time = time[0]*frac[0] + time[1]*frac[1] + time[2]*frac[2]

    still_nan_cube = np.logical_and(isnan_cube[0], isnan_cube[1])
    still_nan_cube = np.logical_and(still_nan_cube, isnan_cube[2])
    if np.where(still_nan_cube==True)[0].size != 0:
	coadd_cube[np.where(still_nan_cube==True)] = np.nan

    return coadd_cube, coadd_time, still_nan_cube


def plt_image(n1, n2, coadd_cube, coadd_time, cube_ra, cube_dec, cube_vlsr, 
	      ra, dec, cubeinfos, figname='1.png', justplot=0):
    import numpy as np
    import matplotlib.pyplot as plt

    if n1 == n2:
        image = coadd_cube[n1]
    else:
        image = coadd_cube[min(n1, n2):max(n1, n2)].mean(0)

    pd = 0.02
    sk = 1.
    fig = plt.figure(figsize = (18, 12))
    ax1 = fig.add_axes([0.05, 0.55, 0.4, 0.4])
    image_temp = image[np.logical_not(np.isnan(image))]
    vminmax = [image_temp.min(), image_temp.max()]
    im1 = ax1.imshow(image, extent=[cube_ra.max(), cube_ra.min(),
                cube_dec.min(), cube_dec.max()], origin='lower',
                 vmin=vminmax[0], vmax=vminmax[1])
    cb1 = fig.colorbar(im1, pad=pd, shrink=sk)
    cb1.set_label('Tb (K km/s)', fontsize = 13)

    ax1.set_title('Weighted Tb @ Vel (%.1f, %.1f) km/s' % (cube_vlsr[n1], cube_vlsr[n2]))
    if justplot == 1:
        ax1.set_title('Vel (%.1f, %.1f) km/s, No image coadded, just plot for fun!' % (cube_vlsr[n1], cube_vlsr[n2]), color='r')

    ax2 = fig.add_axes([0.5, 0.55, 0.4, 0.4])
    im2 = ax2.imshow(coadd_time, extent = [cube_ra.max(), cube_ra.min(),
                        cube_dec.min(), cube_dec.max()], origin = 'lower',
                        cmap = plt.cm.jet)
    cb2 = fig.colorbar(im2, pad = pd, shrink = sk)
    cb2.set_label('Integration time  (s)', fontsize = 13)
    ax2.set_title('Coadded Integr. Time')
    for i in range(len(cubeinfos)):
        icube = cubeinfos[i][0]
        if n1 == n2:
            aa = icube[n1]
        else:
            aa = icube[min(n1, n2):max(n1, n2)].mean(0)

        axa = fig.add_axes([0.05+i*0.9/len(cubeinfos), 0.26, 0.2, 0.2])
        ima = axa.imshow(aa, extent=[cube_ra.max(), cube_ra.min(),
                cube_dec.min(), cube_dec.max()], origin='lower',
                vmin=vminmax[0], vmax=vminmax[1])
        axa.set_title(cubeinfos[i][-1])
        fig.colorbar(ima, pad = pd, shrink = sk)

        itime = cubeinfos[i][-2]
        axb = fig.add_axes([0.05+i*0.9/len(cubeinfos), 0.03, 0.2, 0.2])
        imb = axb.imshow(itime, extent=[cube_ra.max(), cube_ra.min(),
                cube_dec.min(), cube_dec.max()], origin='lower',
                vmin=itime.min(), vmax=itime.max())
        fig.colorbar(imb, pad = pd, shrink = sk)

    # fig.savefig('coadd_figs/G%06.2f+%05.2f_V%d_%d.eps'%(ra, dec, cube_vlsr[min(n1, n2)], cube_vlsr[max(n1, n2)]))
    fig.savefig(figname)
    plt.close()
    return 0

def gain_correction(cubeinfos):
    import numpy as np
    g_t2s3 = 1.056737       # check the overall_gain.txt, 20151120, yz
    # g_lab = 1/1.1937 # Josh, it is galfa = g_lab*lab
    g_lab = 1/1.1255   # yong 
 
    yz_dir = '/vpn/jansky/a/users/goldston/zheng'
    # original: f = open('%s/150206_relative_gain/overall_gain.txt' % (yz_dir), 'r')
    f = open('%s/150206_relative_gain/overall_gain.txt' % (yz_dir), 'r')
    areas = []
    ogain = []
    for iline in f.readlines():
        areas.append(iline.split()[0])
        ogain.append(np.float64(iline.split()[1]))

    bscales = np.zeros(len(cubeinfos))
    for i in range(len(cubeinfos)):
        cname = cubeinfos[i][-1]
        ic = areas.index(cname)
        ig = ogain[ic]

        cubeinfos[i][0] = cubeinfos[i][0]/(ig*g_lab/g_t2s3)
        chdr = cubeinfos[i][1]
        bscales[i] = chdr['BSCALE']/(ig*g_lab/g_t2s3)
    return cubeinfos, bscales

