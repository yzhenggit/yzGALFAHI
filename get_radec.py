import numpy as np

def get_radec(header):  # has been tested using wcs
    '''
    This is a very old funtion, hardcoded to use the header info to generate ra and dec
    It is very fast. It use only keywords on CAR projection, and yields the same result as the get_cubeinfo
    that uses wcs function to process the header info.
    '''
    ra = header['CRVAL1']+header['CDELT1']*(np.arange(header['NAXIS1'])+1-header['CRPIX1'])     # degree
    dec = header['CRVAL2'] + header['CDELT2']*(np.arange(header['NAXIS2'])+1-header['CRPIX2'])  # degree
    return ra, dec

