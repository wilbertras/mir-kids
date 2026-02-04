import numpy as np
import scipy
import matplotlib.pyplot as plt
from scipy.signal import savgol_filter

from .data import MKIDDataException


def oddify(n):
    return int(np.ceil(n/2.0)*2+1)


def find_peaks(freq, ampl, fc, smooth=None, threshold=None, minq=10000, maxratio=0.5, mindist=5e-3):
    """
    Search peaks in data.
    """
    dfreq = freq[1] - freq[0]
    if smooth is None:
        #smooth = 15#sigma*0.3
        #smooth = 30
        smooth = 15
        #print( smooth )

    deriv2 = savgol_filter(ampl, oddify(smooth), 3, 2, delta=dfreq)
    ampl_s = savgol_filter(ampl, oddify(smooth), 3, 0, delta=dfreq)

    if threshold is None:
        d2sigma   = np.std(deriv2)
        d2sigma   = np.std(deriv2[np.abs(deriv2)<2*d2sigma])
        #threshold = 5*d2sigma
        #threshold = 4*d2sigma
        threshold = 3*d2sigma

    # collect all peaks
    kid_indices = []
    for k in np.where(deriv2 > threshold)[0]:
        if k < len(deriv2)-1 and deriv2[k-1] <= deriv2[k] and deriv2[k] >= deriv2[k+1]:
            kid_indices.append(k)

    # traverse to zero-crossing
    kids = []
    nbadq = 0
    nbadd = 0
    if not kid_indices:
        return []

    for k in kid_indices:
        l, r = k, k
        while l > 0           and deriv2[l] > 0: l -= 1
        while r < len(deriv2) and deriv2[r] > 0: r += 1
        w = (r - l + 1) * dfreq
        w = w * 6.0 / np.sqrt(3) # convert to FWHM
        l = int((l-k) * 6.0 / np.sqrt(3) + k)
        r = int((r-k) * 6.0 / np.sqrt(3) + k)
        if l < 0:
            l = 0
        if r >= len(freq):
            r = len(freq) - 1

        q0 = freq[k] / w
        f1, q1, d1, bg = fitLorentzian(freq[l:r], ampl[l:r], freq[k], q0)
        ##### refitting by extending fit range
        if (bg-d1)/bg > maxratio:
            if l-10 >= 0 and r+10 < len(freq):
                f1, q1, d1, bg = fitLorentzian(freq[l-10:r+10], ampl[l-10:r+10], freq[k], q0)
                print( 'rough refitting with Lorentzian...' )
        #####
        if q1 < minq:
            nbadq += 1
            continue
        if (bg-d1)/bg > maxratio:
            nbadd += 1
            continue

        kids.append((f1, q1, d1, bg))
    del l, r, f1, q1, d1, bg
    if nbadq > 0:
        print( 'removed', nbadq, 'peaks with bad Q' )
    if nbadd > 0:
        print( 'removed', nbadd, 'peaks with bad S21min' )
    # sort by frequency
    kids.sort()

    ## pick up a peak which is closest to the carrier frequency (when fc in freq range)
    if len(kids)>0 and fc>=freq[0] and fc<=freq[-1]:
        idx = np.argmin( abs(np.array(kids).T[0]-fc) )
        #print( np.array(kids).T[0] )
        #print( kids )
        #print( len(kids), idx )
        kids = [kids[idx]]
        #print( len(kids) )
    #else:
    #    ## eliminate too close peaks
    #    nkill = 0
    #    if mindist > 0:
    #        # remove close kids weaker than preceding kids
    #        p = 0
    #        while p + 1 < len(kids):
    #            f0, q0, _, _ = kids[p]
    #            f1, q1, _, _ = kids[p+1]
    #            if f1 - f0 < mindist and q0 > q1:
    #                del kids[p+1]
    #                nkill += 1
    #            else:
    #                p += 1
    #        # remove close kids weaker than following kids
    #        p = len(kids)-1
    #        while p - 1 >= 0:
    #            f0, q0, _, _ = kids[p]
    #            f1, q1, _, _ = kids[p-1]
    #            if f0 - f1 < mindist and q0 > q1:
    #                del kids[p-1]
    #                nkill += 1
    #                p -= 1
    #            else:
    #                p -= 1

    for i, (f, q, depth, bg) in enumerate(kids):
        f0ind = np.argmin(abs(freq - f))
        w     = f/q
        dl    = w/2.0/dfreq
        dr    = w/2.0/dfreq
        bg_l  = ampl_s[max(int(f0ind - 3*dl), 0)]
        if int(f0ind-3*dl)<0:
            bg_r = ampl_s[0]
        else:
            bg_r  = ampl_s[min(int(f0ind - 3*dl), len(freq) - 1)]
        a_off = (bg_l + bg_r)/2.0
        a_on  = ampl_s[f0ind]
        kids[i] = {'Q': q, 'f0': f, 'f0ind': f0ind,
                  'dl': int(dl), 'dr': int(dr),
                   'a_off': a_off, 'a_on': a_on}
    
    return kids


#def search_peak(data, Q_search=10000, S21min=0.5):
def search_peak(data, fc=-1., Q_search=100, S21min=1):
    #peaks = find_peaks(data.x, data.amplitude, fc=-1., minq=Q_search, maxratio=S21min)
    peaks = find_peaks(data.x, data.amplitude, fc=fc, minq=Q_search, maxratio=S21min)
    if not peaks:
        raise MKIDDataException('peak find failure')
    center  = (data.x[0] + data.x[-1])/2.0
    minind  = 0
    mindist = abs(peaks[0]['f0'] - center)

    for i in range(1, len(peaks)):
        if mindist > abs(peaks[i]['f0'] - center):
            minind  = i
            mindist = abs(peaks[i]['f0'] - center)
    # print( peaks, center, minind, mindist )
    return peaks[minind]


### A simple Lorentzain fit
def fitLorentzian(freq, ampl, f0, q0):
    """
    Fit data with lorenzian curve.

    :param freq: a 1-D array of frequency
    :param ampl: a 1-D array of amplitude
    :param f0: initial parameter for center frequency
    :param q0: initial parameter for quality factor

    :return: (fc, q, d, bg)

    - **fc**: fit of center frequency
    - **q**: fit of quality factor
    - **d**: fit of amplitude for Lorentzian curve
    - **bg**: fit of constant background level
    """
    def f(x):
        (a, b, c, d) = x # background, amplitude, 2/FWHM, freq. center
        y = a + b / (((freq-d)*c)**2 + 1)
        return y - ampl
    def fprime(x):
        (a, b, c, d) = x
        g = np.zeros((len(freq)), 4)
        g[:, 0] = 1.0
        g[:, 1] = 1.0 / (1.0 + ((freq - d)*c)**2)
        g[:, 2] = -2.0 * b * c * (freq - d)**2 / (1.0 + ((freq - d)*c)**2)**2
        g[:, 3] =  2.0 * b * c**2 * (freq - d) / (1.0 + ((freq - d)*c)**2)**2

    a = np.median(ampl)
    b = -0.8 * a
    c = 2.0 * q0 / f0
    d = f0
    x0 = np.array([a, b, c, d])
    x1 = scipy.optimize.leastsq(f, x0)[0]
    (a, b, c, d) = x1
    #print( x1 )
    f = d
    q = abs(c * d / 2.0)

    return (f, q, -b, a)


