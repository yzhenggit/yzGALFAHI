import os
import datetime
import numpy as np
from astropy.table import Table
import astropy.io.fits as fits
from yzGALFAHI.get_cubeinfo import get_cubeinfo
from astropy.coordinates import SkyCoord

import warnings
warnings.filterwarnings('ignore')

import logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def find_the_cube(ra, dec, observation):
    '''
    Use the input (ra, dec) to decide which cube the data locates.

    observation: most of the time, it is HI4PI. 
    '''

    if observation == 'HI4PI': 
        datadir = '/Volumes/YongData2TB/'+observation
    elif observation == 'GALFA_HI': 
        datadir = '/Volumes/YongData2TB/GALFAHI_DR2/DR2W_RC5/DR2W_RC5/Wide'

    clt = Table.read('%s/%s_RADEC.dat'%(datadir, observation), format='ascii')
    cramin, cramax = clt['min_ra'], clt['max_ra']
    cdcmin, cdcmax = clt['min_dec'], clt['max_dec']
    indra = np.all([ra>cramin, ra<cramax], axis=0)
    inddc = np.all([dec>cdcmin, dec<cdcmax], axis=0)
    indall = np.where(np.all([indra, inddc], axis=0) == True)

    cubename = clt['cubename'][indall]
    if len(cubename)==0:
        logger.info("No corresponding HI cube in %s"%(observation))
        return '', ''
    else:
        cubename = cubename[0]
        if observation == 'EBHIS': cubedir = datadir+'/'+cubename+'.fit'
        else: cubedir = datadir+'/'+cubename+'.fits'

        return cubename, cubedir

def extract_LAB(tar_name, tar_gl, tar_gb, labfile, filedir='.', beam=1.0):
    '''
    To obtain the corresponding HI 21cm spec with certain beam of LAB. 

    beam: to decide within what diameter (in deg) the HI spec is averaged. 
    '''
    beam_radius = beam/2.

    # labfile = '/Users/Yong/Dropbox/databucket/LAB/labh_glue.fits'
    labdata = fits.getdata(labfile)
    labhdr = fits.getheader(labfile)
    gl, gb, cvel = get_cubeinfo(labhdr)
    cube_coord = SkyCoord(l=gl, b=gb, unit='deg', frame='galactic')

    tar_coord = SkyCoord(l=tar_gl, b=tar_gb, unit='deg', frame='galactic')
    dist = tar_coord.separation(cube_coord)
    
    dist_copy = dist.value.copy()
    within_beam_2d = dist_copy<=beam_radius
    within_beam_3d = np.asarray([within_beam_2d]*cvel.size)
    labdata_copy = labdata.copy()
    labdata_copy[np.logical_not(within_beam_3d)] = np.nan
    
    # I expect to see RuntimeWarnings in this block by taking nanmean of full nan arrays
    import warnings
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", category=RuntimeWarning)
        ispec = np.nanmean(np.nanmean(labdata_copy, axis=2), axis=1)

    return cvel, ispec

def save_LAB_fits(tar_name, tar_gl, tar_gb, tar_RA, tar_DEC, filedir='.', beam=1.):
    '''
    To obtain the corresponding HI 21cm spec with certain beam of LAB. 

    beam: to decide within what diameter (in deg) the HI spec is averaged. 
    '''

    labfile = '/Users/Yong/Dropbox/databucket/LAB/labh_glue.fits' 
    cvel, ispec = extract_LAB(tar_name, tar_gl, tar_gb, labfile, filedir=filedir, beam=beam)

    ### Create the primary header 
    labhdr = fits.getheader(labfile)
    prihdr = labhdr.copy()
    for ikey in ['NAXIS1', 'CTYPE1', 'CRVAL1', 'CRPIX1', 'CDELT1', 'CROTA1', 'CUNIT1', 
                 'NAXIS2', 'CTYPE2', 'CRVAL2', 'CRPIX2', 'CDELT2', 'CROTA2', 'CUNIT2',
                 'NAXIS3', 'CTYPE3', 'CRVAL3', 'CRPIX3', 'CDELT3', 'CROTA3', 'CUNIT3', 
                 'PROJ', 'SYSTEM', 'BUNIT', 'OBSTYP', 'DATAMIN', 'DATAMAX', 'COMMENT']:
        del prihdr[ikey]

    prihdr['HLSPNAME'] = ('COS-GAL', 'the COS Quasar Database for Galactic Absorption Lines')
    prihdr['HLSPLEAD'] = 'Yong Zheng'
    prihdr['EQUINOX'] = 2000.0
    prihdr['TARGNAME'] = (tar_name, 'Target name; HSLA(Peeples+2017)')
    prihdr['RA_TARG'] = (round(tar_RA, 4), 'Right Ascension (deg; J2000); HSLA(Peeples+2017)')
    prihdr['DEC_TARG'] = (round(tar_DEC, 4), 'Declination (deg; J2000); HSLA(Peeples+2017)')
    prihdr.comments['DATE'] = 'Date of LAB data cube final release' 
    prihdr['HISTORY'] = 'Extract HI21cm line from LAB within beam diameter size of %.2f degree. '%(beam)
    prihdr['HISTORY'] = 'This spectrum is generated on %s.'%(str(datetime.datetime.now()))
    prihdr['HISTORY'] = 'The LAB cube is from Kalberla et al. (2005). See there for more info.'
    prihdr['HISTORY'] = 'LAB cube data is in grid of 30 arcmin - violate Nyquist sampling.'
    prihdu = fits.PrimaryHDU(header=prihdr)

    ## Now save the table/header extension
    col1 = fits.Column(name='VLSR', format='D', array=cvel)
    col2 = fits.Column(name='FLUX', format='D', array=ispec)
    cols = fits.ColDefs([col1, col2])
    tbhdu = fits.BinTableHDU.from_columns(cols)
    tbhdu.header['TUNIT1'] = 'km/s' # (or whatever unit the "WAVE" column is).
    tbhdu.header['TUNIT2'] = 'K' # for the flux array or whatever unit that column is, etc. for the other columns.

    thdulist = fits.HDUList([prihdu, tbhdu])
    if os.path.isdir(filedir) is False: os.makedirs(filedir)
    hifile = '%s/hlsp_cos-gal_lab_lab_%s_v1_h-i-21cm-spec.fits.gz'%(filedir, tar_name.lower())

    thdulist.writeto(hifile, clobber=True)
    return hifile

def extract_HI4PI(target_info, filedir='.', beam=1.0):
    '''
    To obtain the corresponding HI 21cm spec with certain beam of HI4PI. 

    beam: to decide within what diameter (in deg) the HI spec is averaged. 
    '''
    target = target_info['NAME']
    beam_radius = beam/2.

    tar_coord = SkyCoord(ra=target_info['RA'], dec=target_info['DEC'], unit='deg')

    datadir = '/Volumes/YongData2TB/HI4PI'
    clt = Table.read('%s/HI4PI_RADEC.dat'%(datadir), format='ascii')

    # to find those cubes that have data within the beam 
    cubefiles = []
    for ic in range(len(clt)):
        cubefile = datadir+'/'+clt['cubename'][ic]+'.fits'
        cubehdr = fits.getheader(cubefile)
        cra, cdec, cvel = get_cubeinfo(cubehdr)
        cube_coord = SkyCoord(ra=cra, dec=cdec, unit='deg')
        dist_coord = tar_coord.separation(cube_coord)

        dist = dist_coord.value
        within_beam_2d = dist<=beam_radius
        if dist[within_beam_2d].size > 0:
            cubefiles.append(cubefile)

    specs = []
    for cubefile in cubefiles:
        cubehdr = fits.getheader(cubefile)
        cra, cdec, cvel = get_cubeinfo(cubehdr)
        cube_coord = SkyCoord(ra=cra, dec=cdec, unit='deg')
        dist_coord = tar_coord.separation(cube_coord)

        dist = dist_coord.value
        within_beam_2d = dist<=beam_radius
        within_beam_3d = np.asarray([within_beam_2d]*cvel.size)

        cubedata = fits.getdata(cubefile)
        cubedata[np.logical_not(within_beam_3d)] = np.nan

        import warnings
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", category=RuntimeWarning)
            jspec = np.nanmean(np.nanmean(cubedata, axis=2), axis=1)

        specs.append(jspec)

    ispec = np.mean(np.asarray(specs), axis=0)

    # save the spectrum 
    prihdr = fits.Header()
    prihdr['OBS'] = observation
    prihdr.comments['OBS'] = 'See %s publication.'%(observation)
    prihdr['CREATOR'] = "YZ"
    prihdr['COMMENT'] = "HI 21cm spectrum averaged within beam size of %.2f deg. "%(beam)
    import datetime as dt
    prihdr['DATE'] = str(dt.datetime.now())
    prihdu = fits.PrimaryHDU(header=prihdr)

    ## table 
    col1 = fits.Column(name='VLSR', format='D', array=cvel)
    col2 = fits.Column(name='FLUX', format='D', array=ispec)
    cols = fits.ColDefs([col1, col2])
    tbhdu = fits.BinTableHDU.from_columns(cols)
    thdulist = fits.HDUList([prihdu, tbhdu])

    if os.path.isdir(filedir) is False: os.makedirs(filedir)

    if beam >= 1.:
        hifile = '%s/%s_HI21cm_%s_Beam%ddeg.fits.gz'%(filedir, target, observation, beam)
    else:
        hifile = '%s/%s_HI21cm_%s_Beam%darcmin.fits.gz'%(filedir, target, observation, beam*60)
    thdulist.writeto(hifile, clobber=True)

      
def extract_HI21cm(target_info, filedir='.', observation='HI4PI', beam=1.):
    '''
    To obtain the corresponding HI data for the QSO sightlines. Can be used 
    to obtain from HI4PI (EBHIS+GASS) cubes. HI4PI has res of 10.8 arcmin, 
    each pixel has 3.25 arcmin. 

    beam: to decide within what diameter (in deg) the HI spec is averaged. 
    '''

    target = target_info['NAME']
    beam_radius = beam/2.

    if observation == 'LAB':
        labfile = '/Users/Yong/Dropbox/databucket/LAB/labh_glue.fits'
        labdata = fits.getdata(labfile)
        labhdr = fits.getheader(labfile)
        
        gl, gb, cvel = get_cubeinfo(labhdr)
        tar_coord = SkyCoord(l=target_info['l'], b=target_info['b'], unit='deg', frame='galactic')
        cube_coord = SkyCoord(l=gl, b=gb, unit='deg', frame='galactic')
        dist = tar_coord.separation(cube_coord)
        
        dist_copy = dist.value.copy()
        within_beam_2d = dist_copy<=beam_radius
        within_beam_3d = np.asarray([within_beam_2d]*cvel.size)
                 
        labdata_copy = labdata.copy()
        labdata_copy[np.logical_not(within_beam_3d)] = np.nan

        ispec = np.nanmean(np.nanmean(labdata_copy, axis=2), axis=1)

        # save the spectrum 
        prihdr = fits.Header()
        prihdr['OBS'] = observation
        prihdr.comments['OBS'] = 'See %s publication.'%(observation)
        prihdr['CREATOR'] = "YZ"
        prihdr['COMMENT'] = "HI 21cm spectrum averaged within beam size of %.2f deg. "%(beam)
        import datetime as dt
        prihdr['DATE'] = str(dt.datetime.now())
        prihdu = fits.PrimaryHDU(header=prihdr)

        ## table 
        col1 = fits.Column(name='VLSR', format='D', array=cvel)
        col2 = fits.Column(name='FLUX', format='D', array=ispec)
        cols = fits.ColDefs([col1, col2])
        tbhdu = fits.BinTableHDU.from_columns(cols)
        thdulist = fits.HDUList([prihdu, tbhdu])

        if os.path.isdir(filedir) is False: os.makedirs(filedir)
        if beam >= 1.:
            hifile = '%s/%s_HI21cm_%s_Beam%ddeg.fits.gz'%(filedir, target, observation, beam)
        else:
            hifile = '%s/%s_HI21cm_%s_Beam%darcmin.fits.gz'%(filedir, target, observation, beam*60)

        thdulist.writeto(hifile, clobber=True)
          
    elif observation in ['HI4PI', 'GALFA_HI']:
        tar_coord = SkyCoord(ra=target_info['RA'], dec=target_info['DEC'], unit='deg')

        ### use the input ra/dec to decide which cube to explore. 
        if observation == 'HI4PI': 
            datadir = '/Volumes/YongData2TB/'+observation
        else: 
            datadir = '/Volumes/YongData2TB/GALFAHI_DR2/DR2W_RC5/DR2W_RC5/Wide'

        clt = Table.read('%s/%s_RADEC.dat'%(datadir, observation), format='ascii')

        cubefiles = []
        for ic in range(len(clt)):
            cubefile = datadir+'/'+clt['cubename'][ic]+'.fits'
            cubehdr = fits.getheader(cubefile)
            cra, cdec, cvel = get_cubeinfo(cubehdr)
            cube_coord = SkyCoord(ra=cra, dec=cdec, unit='deg')
            dist_coord = tar_coord.separation(cube_coord)
        
            dist = dist_coord.value
            within_beam_2d = dist<=beam_radius
            if dist[within_beam_2d].size > 0:
                cubefiles.append(cubefile)
 
        specs = []
        for cubefile in cubefiles:
            cubehdr = fits.getheader(cubefile)
            cra, cdec, cvel = get_cubeinfo(cubehdr)
            cube_coord = SkyCoord(ra=cra, dec=cdec, unit='deg')
            dist_coord = tar_coord.separation(cube_coord)
        
            dist = dist_coord.value
            within_beam_2d = dist<=beam_radius
            within_beam_3d = np.asarray([within_beam_2d]*cvel.size)
        
            cubedata = fits.getdata(cubefile)
            cubedata[np.logical_not(within_beam_3d)] = np.nan
            
            ispec = np.nanmean(np.nanmean(cubedata, axis=2), axis=1)
            specs.append(ispec)
            
        ispec = np.mean(np.asarray(specs), axis=0)

        # save the spectrum 
        prihdr = fits.Header()
        prihdr['OBS'] = observation
        prihdr.comments['OBS'] = 'See %s publication.'%(observation)
        prihdr['CREATOR'] = "YZ"
        prihdr['COMMENT'] = "HI 21cm spectrum averaged within beam size of %.2f deg. "%(beam)
        import datetime as dt
        prihdr['DATE'] = str(dt.datetime.now())
        prihdu = fits.PrimaryHDU(header=prihdr)
        
        ## table 
        col1 = fits.Column(name='VLSR', format='D', array=cvel)
        col2 = fits.Column(name='FLUX', format='D', array=ispec)
        cols = fits.ColDefs([col1, col2])
        tbhdu = fits.BinTableHDU.from_columns(cols)
        thdulist = fits.HDUList([prihdu, tbhdu])
         
        if os.path.isdir(filedir) is False: os.makedirs(filedir)
       
        if beam >= 1.: 
            hifile = '%s/%s_HI21cm_%s_Beam%ddeg.fits.gz'%(filedir, target, observation, beam)
        else:
            hifile = '%s/%s_HI21cm_%s_Beam%darcmin.fits.gz'%(filedir, target, observation, beam*60)
        thdulist.writeto(hifile, clobber=True)

    else: 
        logger.info('%s are not in [LAB, HI4PI]'%(observation)) 
        hifile = ''
    return hifile
