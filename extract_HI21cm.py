import os
import numpy as np

## warning mainly from reading fits header. 
import warnings
warnings.filterwarnings('ignore')

import logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def find_the_cube(ra, dec, observation):
    from astropy.table import Table
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
        return False, False
    else:
        cubename = cubename[0]
        if observation == 'EBHIS': cubedir = datadir+'/'+cubename+'.fit'
        else: cubedir = datadir+'/'+cubename+'.fits'

        return cubename, cubedir

## =================================================================================================
def create_primary_header_HI(target_info, observation='HI4PI', beam=1.0):
    import astropy.io.fits as fits
    import datetime

    if observation not in ['HI4PI', 'LAB', 'GALFA-HI', 'GALFAHI']:
        logger.info('Do not recognize %s'%(observation))
        return False
    
    newhdr = fits.Header()
    newhdr['TARGNAME'] = (target_info['NAME'], 'Target name; HSLA (Peeples+2017)')
    newhdr['RA_TARG'] = (round(target_info['RA'], 4), 'Right Ascension (deg); HSLA (Peeples+2017)')
    newhdr['DEC_TARG'] = (round(target_info['DEC'], 4), 'Declination (deg); HSLA (Peeples+2017)')
    newhdr['HLSPNAME'] = ('COS-GAL', 'HSLP product name')
    newhdr['HLSPLEAD'] = 'Yong Zheng'
    newhdr['BEAMFWHM'] = (beam, 'Beam (deg) within which data are extracted')
    newhdr['RESTFRQ'] = (1420405751.77, 'Hz')
    newhdr['EQUINOX'] = 2000.
    
    if observation == 'LAB':
        newhdr['SURVEY'] = ('LAB', 'The Leiden/Argentine/Bonn (LAB) Survey')
        newhdr['TELESCOP'] = ('Dwingeloo - Villa Elisa', 'Telescope')
        newhdr['REFERENC'] = ('Kalberla et al. (2005) A&A 440, 775', 'LAB cube reference')
        newhdr['HISTORY'] = 'HI21cm line from LAB cube, averaged within %.3f deg beam.'%(beam)
        newhdr['HISTORY'] = 'This spectrum is generated on %s.'%(str(datetime.datetime.now()))
        newhdr['HISTORY'] = 'Note LAB cube data is in grid of 30 arcmin, no Nyquist sampling.'
    elif observation == 'HI4PI':
        newhdr['SURVEY'] = ('HI4PI', 'The HI 4-PI Survey')
        newhdr['TELESCOP'] = ('Effelsberg 100m RT; ATNF Parkes 64-m', 'Telescope')
        newhdr['REFERENC'] = ('HI4PI Collaboration 2016', 'HI4PI cubes reference')
        newhdr['HISTORY'] = 'HI21cm line from HI4PI cubes, averaged within %.3f deg beam.'%(beam)
        newhdr['HISTORY'] = 'This spectrum is generated on %s.'%(str(datetime.datetime.now()))
    else:  # this is for GALFA-HI
        newhdr['SURVEY'] = ('GALFA-HI DR2', 'The Galactic ALFA HI survey')
        newhdr['TELESCOP'] = ('Arecibo 305m', 'Telescope')
        newhdr['INSTRUME'] = ('Arecibo L-Band Feed Array (ALFA)', 'Instrument')
        newhdr['REFERENC'] = ('Peek et al. (2018), ApJS, 234, 2P', 'GALFA-HI cubes reference')
        newhdr['HISTORY'] = 'HI21cm line from GALFA-HI cubes, averaged within %.3f deg beam.'%(beam)
        newhdr['HISTORY'] = 'This spectrum is generated on %s.'%(str(datetime.datetime.now()))

    primary_hdu = fits.PrimaryHDU(header=newhdr)
    return primary_hdu

## =================================================================================================
def save_HIspec_fits(target_info, savedir='.', beam=1., observation='HI4PI', 
                     datadir='/Volumes/YongData2TB/HI4PI'):
    '''
    To obtain the corresponding HI 21cm spec with certain beam of LAB. 

    beam: to decide within what diameter (in deg) the HI spec is averaged. 
    datadir: for LAB data, you give directory and the cube name
             e.g., datadir='/Users/Yong/Dropbox/databucket/LAB/labh_glue.fits'
             for HI4PI and GALFA-HI data, you give the directory of the cubes
             e.g., datadir='/Volumes/YongData2TB/HI4PI'
                   datadir='/Volumes/YongData2TB/GALFAHI_DR2/RC5/Wide'
    '''
    
    import astropy.io.fits as fits
    import numpy as np
    import os

    if observation not in ['HI4PI', 'LAB', 'GALFA-HI', 'GALFAHI']:
        logger.info('Do not recognize %s'%(observation))
        return False

    ## create the primary header for this spectra 
    prihdu = create_primary_header_HI(target_info, observation=observation, beam=beam)

    ## extract HI spectra from certain HI survey, and create the fits data extension
    if observation == 'LAB':
        hivel, hispec = extract_LAB(target_info['l'], target_info['b'], beam=beam, labfile=datadir)
    else:  # for HI4PI or GALFA-HI
        hivel, hispec = extract_HI4PI_GALFAHI(target_info['RA'], target_info['DEC'], beam=beam, 
                                              observation=observation, datadir=datadir)
    ## this is mostly for GALFA-HI, which it only covers from DEC=-1 to 38 degree. 
    if type(hivel) == bool: 
        return False

    col1 = fits.Column(name='VLSR', format='D', array=hivel)
    col2 = fits.Column(name='FLUX', format='D', array=hispec)
    cols = fits.ColDefs([col1, col2])
    tbhdu = fits.BinTableHDU.from_columns(cols)
    tbhdu.header['TUNIT1'] = 'km/s' # (or whatever unit the "WAVE" column is).
    tbhdu.header['TUNIT2'] = 'K' # for the flux array or whatever unit that column is

    ## now save the data
    thdulist = fits.HDUList([prihdu, tbhdu])
    if os.path.isdir(savedir) is False: os.makedirs(savedir)
    obs_tag = observation.lower().replace('-', '')
    hifile = '%s/hlsp_cos-gal_%s_%s_%s_21cm_v1_h-i-21cm-spec-beam%.3fdeg.fits.gz'%(savedir, 
                                      obs_tag, obs_tag, target_info['NAME'].lower(), beam)
    thdulist.writeto(hifile, clobber=True)
    return hifile


## =================================================================================================
def cubes_within_beam(tar_RA, tar_DEC, datadir='.', beam=1.0, observation='HI4PI'):
    '''
    Find the cubes that all have data within the reqiured beam for HI4PI and GALFA-HI data
    
    '''
   
    from yzGALFAHI.get_cubeinfo import get_cubeinfo
    from astropy.coordinates import SkyCoord
    import astropy.io.fits as fits
    from astropy.table import Table
    import warnings
   
    tar_coord = SkyCoord(ra=tar_RA, dec=tar_DEC, unit='deg')
   
    surveytable = '/Users/Yong/Dropbox/GitRepo/yzGALFAHI/tables/%s_RADEC.dat'%(observation)
    clt = Table.read(surveytable, format='ascii')
    
    if observation == 'HI4PI':
        cubewth = 20   # deg 
    elif observation == 'GALFAHI':
        cubewth = 10   # deg 
    else:
        logger.info('Cannot recognize this observation: '%(observation))
   
    ## this warning would occur if there is nan in ra/dec read from the cubes
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", category=RuntimeWarning)

        cubefiles = []
        for ic in range(len(clt)):
            ## if the cube is really far away, then simply just let it go
            poss_ramax = abs(clt['cra'][ic]-tar_RA)
            poss_decmax = abs(clt['cdec'][ic]-tar_DEC)
            if poss_ramax>cubewth+beam or poss_decmax>cubewth+beam: # choose 20 because HI4PI cube size 
                continue
            else:
                ## if the cube is within the beam range, then do a serious search 
                cubefile = datadir+'/'+clt['cubename'][ic]+'.fits'
                cubehdr = fits.getheader(cubefile)
                cra, cdec, cvel = get_cubeinfo(cubehdr)
                cube_coord = SkyCoord(ra=cra, dec=cdec, unit='deg')
                dist_coord = tar_coord.separation(cube_coord)

                dist = dist_coord.value
                within_beam_2d = dist<=beam/2.
                if dist[within_beam_2d].size > 0:
                    cubefiles.append(cubefile)
    return cubefiles

## =================================================================================================
def extract_LAB(tar_gl, tar_gb, beam=1.0,
                labfile='/Users/Yong/Dropbox/databucket/LAB/labh_glue.fits'):
    '''
    To obtain the HI 21cm line averaged with certain beam of LAB. 

    beam: to decide within what diameter (in deg) the HI spec is averaged. 
          If input beam is less than 0.5, then force it to 0.5
    tar_gl: Galactic longitude of the line of sight. 
    tar_gb: Galactic latitude of the line of sight. 
    labfile: the LAB data cube to put in and extract data. 
    '''
    from yzGALFAHI.get_cubeinfo import get_cubeinfo
    from astropy.coordinates import SkyCoord
    import astropy.io.fits as fits

    if beam < 0.5: beam = 0.5 # LAB minimum pix size is 0.5 deg 
                              # extract 1 pix in such case 
    beam_radius = beam/2.

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


## =================================================================================================
def extract_HI4PI_GALFAHI(tar_RA, tar_DEC, beam=1.0, observation='HI4PI', 
                          datadir='/Volumes/YongData2TB/HI4PI/'):
    '''
    Extract HI4PI or GALFA-HI spectrum within certain beam size. 
    Spectra are averaged within the beam. 
    Note HI4PI has pixel size of 5.0 arcmin or 1/12 deg. 
    If input beam size is less than 1/12, force it to 1/12. 
    Note GALFA-HI has pixel size of 1 arcmin or 1/60 deg, 
    if input beam size is less than 1/60, force it to 1/60. 

    tar_RA: the right ascension of the line of sight. 
    tar_DEC: the declination of the line of sight. 
    '''

    from yzGALFAHI.get_cubeinfo import get_cubeinfo
    from astropy.coordinates import SkyCoord
    import astropy.io.fits as fits
    from astropy.table import Table
    import warnings

    observation = observation.replace('-', '')
    if observation == 'HI4PI':
        if beam < 1/12: beam = 1/12  # HI4PI minimum pix size is 1/12 deg 
        beam_radius = beam/2.
        # datadir = '/Volumes/YongData2TB/HI4PI/'
    elif observation == 'GALFAHI':
        if beam < 1/60: beam = 1/60  # GALFA-HI minimum pix size is 1/60 deg
        beam_radius = beam/2.
        # datadir = '/Volumes/YongData2TB/GALFAHI_DR2/RC5/Wide/'

        if tar_DEC < -1.3 or tar_DEC > -37.9: 
            logger.info('This target is not observed in GALFA-HI')
            return False, False
    else:
        logger.info('Cannot find this observation: '%(observation))
        return False, False

    # to find those cubes that have data within the beam 
    cubefiles = cubes_within_beam(tar_RA, tar_DEC, datadir=datadir, beam=beam, observation=observation)
    if len(cubefiles) == 0:
        logger.info('Do not have data in %s'%(observation))
        return False, False
    
    specs = []
    tar_coord = SkyCoord(ra=tar_RA, dec=tar_DEC, unit='deg')
    logger.info('Extract mean HI 21cm line from these cubes: ')
    for cubefile in cubefiles:
        logger.info(cubefile)
        cubehdr = fits.getheader(cubefile)
        cra, cdec, cvel = get_cubeinfo(cubehdr)
        cube_coord = SkyCoord(ra=cra, dec=cdec, unit='deg')
        dist_coord = tar_coord.separation(cube_coord)

        dist = dist_coord.value
        within_beam_2d = dist<=beam_radius
        within_beam_3d = np.asarray([within_beam_2d]*cvel.size)

        cubedata = fits.getdata(cubefile)
        cubedata[np.logical_not(within_beam_3d)] = np.nan

        with warnings.catch_warnings():
            warnings.simplefilter("ignore", category=RuntimeWarning)
            jspec = np.nanmean(np.nanmean(cubedata, axis=2), axis=1)

        specs.append(jspec)

    with warnings.catch_warnings():
        warnings.simplefilter("ignore", category=RuntimeWarning)
        ispec = np.mean(np.asarray(specs), axis=0)

    return cvel, ispec

## =================================================================================================
def extract_HI21cm(target_info, filedir='.', observation='HI4PI', beam=1.):
    '''
    To obtain the corresponding HI data for the QSO sightlines. Can be used 
    to obtain from HI4PI (EBHIS+GASS) cubes. HI4PI has res of 10.8 arcmin, 
    each pixel has 3.25 arcmin. 
    YZ noted on Mar 1, 2018: this func has been replaced by save_HIspec_fits, 
       create_primary_header_HI, and cubes_within_beam. 

    beam: to decide within what diameter (in deg) the HI spec is averaged. 
    '''

    from yzGALFAHI.get_cubeinfo import get_cubeinfo
    from astropy.coordinates import SkyCoord
    import astropy.io.fits as fits
    from astropy.table import Table

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
