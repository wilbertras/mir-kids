import warnings
import os
from collections import Mapping, OrderedDict

import numpy as np
import matplotlib.pyplot as plt

from .data import SweepData, FixedData
from .files import read_kidslist, read_localsweep, hdf5_tods_fits
from .kidfit import fit_onepeak


class KIDs(Mapping):
    """
    hold all kids
    """
    def __init__(self, kidslist, sweeps_path, tods_path, **kws):
        kidslist = read_kidslist(kidslist)
        self._kidslist    = kidslist
        self.framelen     = kidslist[0].get('framelen')
        if not self.framelen:
            if kws.get('framelen'):
                self.framelen = kws['framelen']
            else:
                warnings.warn('framelen info not in kidslist: please set KIDs.framelen manually')
        
        self._sweeps_path = sweeps_path
        self._tods_path   = tods_path
        
        if 'Readout' in kws:
            self.Readout = kws['Readout']
        else:
            warnings.warn('please set Readout number')
            raise RuntimeError('error')
        if 'Channel' in kws:
            self.Channel = kws['Channel']
        else:
            warnings.warn('please set Channel number')
            raise RuntimeError('error')
        if 'Group' in kws:
            self.Group = kws['Group']
        else:
            warnings.warn('please set Group number')
            raise RuntimeError('error')
        
        # """read sweep file"""
        self._raw_sweeps = read_localsweep(self._sweeps_path, framelen=self.framelen)
    
        # """read tod hdf5 file"""
        self._raw_tods = hdf5_tods_fits(self._tods_path, kidslist[0], self.Readout, self.Channel, self.Group)

        # store kids information
        self._kids = [KID(self, i, k, l, r, self.powers[k]) for i, (k, l, r) in
                      enumerate(kids_with_both_blinds(kidslist, allow_without_blind=True))]

    @property
    def bins_kid(self):
        return self._kidslist[1]

    @property
    def bins_blind(self):
        return self._kidslist[2]

    @property
    def powers(self):
        return self._kidslist[3]

    @property
    def raw_sweeps(self):
        return self._raw_sweeps

    @property
    def raw_tods(self):
        return self._raw_tods

    @property
    def header(self):
        """
        header of bintable inside KID TOD fits file.
        """
        key = list(self._raw_tods.keys())[0]
        return self._raw_tods[key].info['header']

    ## methods to be like dict(), with inheritance of Mapping
    def __len__(self):
        return len(self._kidslist[1])
    def __getitem__(self, index):
        return self._kids[index]
    def __iter__(self):
        for i in range(len(self)):
            yield i
    def __contains__(self, item):
        return i in self._kids


class KID(object):
    def __init__(self, parent, index, freqbin, leftbin, rightbin, readpower):
        self._parent   = parent
        self._index    = index
        self.name      = 'kid%04d' %self._index
        self.bin       = freqbin
        self.bin_l     = leftbin
        self.bin_r     = rightbin
        self.enabled   = leftbin and rightbin
        self.readpower = readpower
        self.fitresult = None
        
    ## functions to retrieve raw data
    @property
    def raw_sweep(self):
        return self._parent.raw_sweeps[self.bin]

    @property
    def blind_tone_left(self):
        return self._parent.raw_tods[self.bin_l]

    @property
    def blind_tone_right(self):
        return self._parent.raw_tods[self.bin_r]

    @property
    def raw_tod(self):
        return self._parent.raw_tods[self.bin]

    @property
    def fcarrier(self):
        return self._parent.raw_tods[self.bin].frequency # GHz

    @property
    def fcenter(self):
        swp = self.raw_sweep
        return (swp.x[0] + swp.x[-1])/2. # GHz

    
    ### fit raw sweep
    def fit(self, nfwhm=5, fitter='gaolinbg', Q_search=1e+3):
        swp = self.raw_sweep
        #fc = self.fcarrier # GHz
        fc = self.fcenter # GHz
        err = None
        r = fit_onepeak(swp, fc, err, nfwhm, fitter=fitter, Q_search=Q_search)

        self.fitresult = r
    
    ### rewind tod according to fit result
    def rewind_tod(self, blindtone_calibration=True):
        tod = self.raw_tod
        r   = self.fitresult
        if r is None:
            print( '>>> KID::no fit result!!' )
            print( '>>> return raw tod' )
            return tod

        if blindtone_calibration:
            ### calibrate raw tod with blind tones
            cal = tod.calibrate_with_blind_tones(self.blind_tone_left,
                                                 self.blind_tone_right)
        else:
            cal = tod
        return r.rewind_data(cal)

    
    ### convert phase to df/fr or linearized phase
    def convert_to_fshift(self, phase, opt='phase'):
        r   = self.fitresult
        if r is None:
            print( '>>> KID::no fit result!!' )
            print( '>>> return phase' )
            return phase    
        
        swp = self.raw_sweep
        fr = r.params['fr'].value # GHz
        fc = self.fcarrier # GHz
        rw_f = r.rewind( swp.x, r.fitted(swp.x) )
        phase_f = -np.angle( -rw_f )
        ## spline interpolation
        import scipy.interpolate
        tck = scipy.interpolate.splrep(phase_f, swp.x, s=0)
        f = scipy.interpolate.splev(phase, tck, der=0)
        
        if opt=='phase':
            return phase
        elif opt=='fshift':
            #return (f-fc)/fr
            return (f-fr)/fr
        elif opt=='linphase':
            Qr = r.params['Qr'].value
            #return 4*Qr * (f-fc)/fr
            return 4*Qr * (f-fr)/fr
        else:
            print( '>>> KID::convert_to_fshift: Not supported option!!' )
            print( '>>> return phase' )
            return phase    


def kids_with_both_blinds(kidslist, tod=None, allow_without_blind=False):
    """
    search kids its nearest blind tones.

    kidslist :: a tuple of (info, kids, blinds, powers),
                that read_kidslist returns.
    tod      :: None or a OrderedDict.
                if not None, only search KID in `tod`.

    Return a list of list of form
      [[signal0, left0, right0],
       [signal1, left1, right1],
         :
         :                     
       [signalp, leftp, rightp]],
      where
        signalN : carrier bin number for KID,
        leftN   : carrier bin number for nearest left blind tone, 
        rightN  : carrier bin number for nearest right blind tone,
      respectively.
      this list is sorted by bin number in lowest-first order.
    """
    info, kids, blinds, powers = kidslist
    if tod:
        todkeys = tod.keys()
        kids    = np.intersect1d(kids, todkeys)

    allkeys = sorted(list(kids) + list(blinds))
    results = []
    for k in sorted(kids):
        pair = []
        pos = allkeys.index(k) - 1
        while pos >= 0:
            key = allkeys[pos]
            if (((key in blinds) and not tod) or
                ((key in blinds) and tod and (key in todkeys))):
                pair.append(allkeys[pos])
                break
            pos = pos - 1
        else:
            if allow_without_blind:
                pair.append(None)
            else:
                continue
        pos = allkeys.index(k) + 1
        while pos < len(allkeys):
            key = allkeys[pos]
            if (((key in blinds) and not tod) or
                ((key in blinds) and tod and (key in todkeys))):
                pair.append(key)
                break
            pos = pos + 1
        else:
            if allow_without_blind:
                pair.append(None)
            else:
                continue
        results.append([k] + pair)
        
    return np.array(results)

