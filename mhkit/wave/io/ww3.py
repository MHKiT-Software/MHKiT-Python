import urllib
import os
import time
import datetime


def _createUrl(sdate, edate, latBounds, lonBounds, varnames):
    """
    Formulates URL for data from WW3 global model
    https://pae-paha.pacioos.hawaii.edu/erddap/griddap/ww3_global.html
    
    Parameters
    ------------
    sdate: datetime.datetime
        Start date

    edate: datetime.datetime
        End date

    latBounds: list
        Latitude bounds, e.g., [0, 77.5] for northern hemisphere

    lonBounds: list
        Longitude bounds

    varnames: list
        Desired variable names (see https://pae-paha.pacioos.hawaii.edu/erddap/griddap/ww3_global.html)
    
    Returns
    ---------
    outUrl: string
        URL for download
    """
    docurl = 'https://pae-paha.pacioos.hawaii.edu/erddap/griddap/ww3_global.html'

    assert isinstance(sdate, datetime.datetime), 'sdate must be of type datetime.datetime'
    assert isinstance(edate, datetime.datetime), 'sdate must be of type datetime.datetime'
    valid_dates = [datetime.datetime(2010, 11, 7, 21), datetime.datetime.now()] # TODO - data is not actually uploaded continuously...
    assert valid_dates[0] <= sdate <= valid_dates[1], 'sdate not in valid range, see {docurl} for available dates'.format(docurl=docurl)
    assert valid_dates[0] <= edate <= valid_dates[1], 'edate not in valid range, see {docurl} for available dates'.format(docurl=docurl)

    assert isinstance(latBounds, list), 'latBounds must be of type float or list'
    assert isinstance(lonBounds, list), 'lonBounds must be of type float or list'
    assert len(latBounds) == 2, 'latBounds must be of length 2'
    assert len(lonBounds) == 2, 'lonBounds must be of length 2'
    assert latBounds[1] > latBounds[0], 'latBounds[1] must be greater than latBounds[1]'
    assert lonBounds[1] > lonBounds[0], 'lonBounds[1] must be greater than lonBounds[1]'
    assert isinstance(varnames, list), 'varnames must be of type float or list'

    valid_varnames = ['Tdir', 'Tper', 'Thgt', 'sdir', 'sper', 'shgt', 'wdir', 'wper', 'whgt']
    
    for vn in varnames:
        if vn not in valid_varnames:
            raise Exception('{mvn} is not a valid variable name, see list of valid variable names at {docurl}'.format(mvn=vn, docurl=docurl))

    baseUrl = 'https://coastwatch.pfeg.noaa.gov/erddap/griddap/NWW3_Global_Best.nc?'
    
    
    timeSeg = '[({sdate}):1:({edate})]'.format(sdate=sdate.strftime('%Y-%m-%d-%H'), 
        edate=edate.strftime('%Y-%m-%dT%H:%M:%Sz'))
    depthSeg = '[(0.0):1:(0.0)]'
    # latSeg = '[(-77.5):1:(77.5)]'
    # lonSeg = '[(0.0):1:(359.5)]'
    latSeg = '[({lat0}):1:({lat1})]'.format(lat0=latBounds[0],lat1=latBounds[1])
    lonSeg = '[({lon0}):1:({lon1})]'.format(lon0=lonBounds[0],lon1=lonBounds[1])
    
    vurl = timeSeg + depthSeg + latSeg + lonSeg

    outUrl = baseUrl
    for idx, varName in enumerate(varnames):
        if idx > 0:
            delim = ','
        else:
            delim = ''
            
        outUrl = outUrl + delim + varName + vurl
        
    return outUrl
    
def _reporthook(count, block_size, total_size):
    """
    Shows download progress
    
    Parameters
    ------------
    TODO
    """

    global start_time
    if count == 0:
        start_time = time.time()
        return
    duration = time.time() - start_time
    progress_size = count * block_size / (1024 ** 2)
    speed = progress_size / (duration)
    percent = count * block_size * 100 / total_size
    print('\t{size:.1f} MB, {speed:.3f} MB/s, {dur:.1f} s passed'.format(size=progress_size, 
        speed=speed, dur=duration), end='\r', flush=True)

def _downloadFile(url, outFileName):
    """
    Downloads data from global WW3 model 
    https://pae-paha.pacioos.hawaii.edu/erddap/griddap/ww3_global.html
    
    Parameters
    ------------
    url: string
        Download URL
    
    outFileName: string
        Name to save file locally (can include path)
    """
    assert isinstance(url, str), 'url must be of type str'
    assert isinstance(outFileName, str), 'outFileName must be of type str'

    print('\tDownloading...')
    print('\tURL: {url}'.format(url=url))
    print('\tFile name: {outFileName}'.format(outFileName=outFileName))
    attempts = 0
    while attempts < 10:
        try:
            urllib.request.urlretrieve(url, outFileName, _reporthook)
            break
        except Exception as e:
            attempts += 1
            print("\tAttempt {attempt:d} failed".format(attempt=attempts))
            print('\tError: {}'.format(e))
            for i in range(0,60):
                print("\tTrying again in {} seconds".format(60-i), end='\r')
                time.sleep(1)
    if attempts == 10:
        raise Exception("Error")

def request_data(start_date, hours, varnames, latBounds=None, lonBounds=None, savepath=None):
    """
    Downloads data from global WW3 model and saves locally as a NetCDF-3 binary 
    file with COARDS/CF/ACDD metadata.
    https://pae-paha.pacioos.hawaii.edu/erddap/griddap/ww3_global.html

    Individual file size limit is 2 GB
    
    Parameters
    ------------
    sdate: datetime.datetime
        Start date

    hours: float
        Number of hours after start date (23 = 1 day)

    latBounds: list
        Latitude bounds, e.g., [0, 77.5] for northern hemisphere

    lonBounds: list
        Longitude bounds

    varnames: list
        Desired variable names (see https://pae-paha.pacioos.hawaii.edu/erddap/griddap/ww3_global.html)

    savepath: string
        File path for saving downloaded data
    
    Returns
    ---------
    ofn: string
        Saved data file name
    """
    assert isinstance(start_date, datetime.datetime), 'start_date must be of type datetime.datetime'
    assert isinstance(hours, (int, float)), 'hours must be of type float'
    assert isinstance(varnames, list)
    

    if latBounds is not None:
        assert isinstance(latBounds, list), 'latBounds must be of type float or list'
        assert len(latBounds) == 2, 'latBounds must be of length 2'
        assert latBounds[1] > latBounds[0], 'latBounds[1] must be greater than latBounds[1]'
    else:
        latBounds = [-77.5, 77.5]
    if lonBounds is not None:
        assert isinstance(lonBounds, list), 'lonBounds must be of type float or list'
        assert len(lonBounds) == 2, 'lonBounds must be of length 2'
        assert lonBounds[1] > lonBounds[0], 'lonBounds[1] must be greater than lonBounds[1]'
    else:
        lonBounds = [0.0, 359.5]
    if savepath is not None:
        assert isinstance(savepath, str), 'savepath must be of type str'
    else:
        savepath = '.'

    end_date = start_date + datetime.timedelta(hours=hours)

    print('\n\n{date}'.format(date=start_date.strftime('%Y-%m-%d')))
    url = _createUrl(start_date, end_date, latBounds, lonBounds, varnames)
    outname = '{fn}.nc'.format(fn=start_date.strftime('%Y-%m-%d'))
    ofn = os.path.join(savepath, outname)
    _downloadFile(url, ofn)

    return ofn
