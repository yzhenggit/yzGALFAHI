from __future__ import print_function
import sys
import astropy.wcs as wcs
import numpy as np


# a function to read in header and generate RA, DEC and VLSR array
def get_cubeinfo(header, returnHeader=False):
    '''
    A function created to parse the RA, DEC, (and velocity) information from the 2D (3D) header

    - This function has been tested with GALFA-HI/EBHIS cubes, and GALFA-HI 2D images. 
    - 		    also been tested with LAB cubes/images that are in (glon, glat) coordinates.
    - The input header can be 2D: NAXIS=2. NAXIS1 is RA (glon), 2 is DEC (glat)
    - 			   or 3D: NAXIS=3. NAXIS1 is RA (glon), 2 is DEC (glat), 3 is Velocity
    - Return: if GALFA-HI or EBHIS: 
                 	ra and dec in 2D, with shape of (dec.size, ra.size) or (NAXIS2, NAXIS1)
    -         	 	velocity in 1D.  
                        or (ra, dec, vlsr, header array)
    - 	      if LAB: 
			glon, glat in 2D, with shape of (glat.size, glon.size) or (NAXIS2, NAXIS1)
    - 			velocity in 1D. 
                        or (gl, gb, vlsr, header array)
    - History: updated as of 2016.10.03. Yong Zheng @ Columbia Astro.
    '''        

    #import sys
    #import astropy.wcs as wcs
    #import numpy as np

  
    hdrarrs = [] 
    if header['NAXIS'] == 2:
        hdr2d = header.copy()
        hdrarrs.append(hdr2d)
    elif header['NAXIS'] == 3:
        # create a 2D header (RA/DEC) to speed up the RA/DEC calculation using astropy.wcs
        hdr2d = header.copy()
        # we don't need the velocity (3) information in the header
        delkey = []
        for key in hdr2d.keys():
            if len(key) != 0 and key[-1] == '3': delkey.append(key)
        for i in delkey: del hdr2d[i]

        hdr2d['NAXIS'] = 2
        if 'WCSAXES' in hdr2d.keys(): hdr2d['WCSAXES']=2

        # create a 1D header (vel) to parse the velocity using astropy.wcs
        hdr1d = header.copy()
        # we don't need the RA/DEC keywords info in the header now.
        delkey = []
        for keya in hdr1d.keys():
            if len(keya) != 0 and keya[-1] in ['1', '2']: delkey.append(keya)
        for i in delkey: del hdr1d[i]
        delkey = []
        for keyb in hdr1d.keys():
            if len(keyb) != 0 and keyb[-1] == '3':
                hdr1d.append('%s1'%(keyb[:-1]))
                hdr1d['%s1'%(keyb[:-1])] = hdr1d[keyb]
                delkey.append(keyb)
        for i in delkey: del hdr1d[i]
        hdr1d['NAXIS'] = 1
        if 'WCSAXES' in hdr1d.keys(): hdr1d['WCSAXES']=1

        # save header arrays
        hdrarrs.append(hdr2d)
        hdrarrs.append(hdr1d)
    else:
        print("This code can only handle 2D or 3D data")
        sys.exit(1)

    return_arrays = []

    # calculate RA, DEC
    gwcsa = wcs.WCS(hdr2d)
    n1, n2 = hdr2d['NAXIS1'], hdr2d['NAXIS2']
    ax = np.reshape(np.mgrid[0:n1:1]+1, (1, n1))  # For FITS standard, origin = 1
    ay = np.reshape(np.mgrid[0:n2:1]+1, (n2, 1))  #   then for numpy standard, origin = 0
    coor1, coor2 = gwcsa.all_pix2world(ax, ay, 1) # coor1 = ra  or glon
    return_arrays.append(coor1) 		  # coor2 = dec or glat
    return_arrays.append(coor2)

    ## calculate VLSR
    if header['NAXIS'] == 3:
        gwcsb = wcs.WCS(hdr1d)
        n1 = hdr1d['NAXIS1']
        ax = np.mgrid[0:n1:1]+1
        # ax = np.linspace(0, n1, n1)  # nope, wrong
        vel = gwcsb.all_pix2world(ax, 1)[0]
        if 'CUNIT1' in hdr1d.keys():
            if hdr1d['CUNIT1'] in ['m/s', 'M/S', 'M/s', 'm/S']: 
                vel = vel/1e3
        else: vel = vel/1e3  # default is usually in m/s
        return_arrays.append(vel)
    
    if returnHeader == True: return_arrays.append(hdrarrs)
    return return_arrays

