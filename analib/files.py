import numpy as np
import h5py
from collections import Mapping, OrderedDict

#import mkid_data as md
from .data import SweepData, FixedData

"""
functions to read data files.
"""
def read_kidslist(fname):
    '''Read kidslist file.

    Return (info, kids, blinds, powers)
      where info   is a dict with keys 'LO' (LO freq in Hz) and
                      framelen (frame count);
            kids   is a 1D list of bin indices for KIDs;
            blinds is a 1D list of bin indices for blind tones;
            powers is a OrderedDict, whose key is bin indices 
                      and value is power for that bin.
    '''
    mode = ['header']
    kids   = []
    blinds = []
    powers = dict()
    def checkmode(line):
        if line[:5] == '#KIDs':
            mode[0] = 'kids'
            return True
        elif line[:7] == '#blinds':
            mode[0] = 'blinds'
            return True
        else:
            return False

    with open(fname) as f:
        info = dict()
        for l in f:
            l = l.strip()
            if not l:
                continue
            elif checkmode(l):
                continue
            if mode[0] == 'header':
                #print l
                key, val = l[1:].split(':')
                if key == 'LO':
                    info['LO'] = float(val)*1e6 # MHz to Hz
                elif key == 'framelen':
                    info['framelen'] = int(val)
                elif key == 'samplerate':
                    info['samplerate'] = float(val)
                else:
                    raise RuntimeError('error')
            else:
                values = l.split()
                nbin  = int(values[0])
                power = float(values[1])
                if mode[0] == 'kids':
                    kids.append(nbin)
                    powers[nbin] = power
                else:
                    blinds.append(nbin)
                    powers[nbin] = power
    return info, np.array(kids), np.array(blinds), powers


def read_localsweep(sweepfname, kidslistfname=None, framelen=None):
    """Read local sweep file.

    Returns a OrderedDict of SweepData's, keyed by bin number.
    It is sorted by bin number, in lowest-first order.
    """
    sweep    = _read_sweep(sweepfname)
    
    from collections import OrderedDict
    lofreqs, bins, data = sweep
    bins = np.array(bins)

    if framelen is None:
        kidslist = read_kidslist(kidslistfname)
        info, kids, blinds, powers  = kidslist
        framelen = info['framelen']

    sweeps = OrderedDict()
    dfreq    = 2e9 / (2**framelen)

    for b, d in zip(bins, data.T):
        swpdata = SweepData()
        super(SweepData, swpdata).__init__('I-Q', (np.real(d), np.imag(d)))
        swpdata._Hz = lofreqs + b * dfreq
        sweeps[b] = swpdata

    result = OrderedDict(sorted(sweeps.items(), key=lambda x: x[0])) # sort by bin
    return result


class hdf5_tods_fits(Mapping):
    def __init__(self, infile, info, Readout, Channel, Group):
        self.infile  = infile
        self.info = info
        self.R = Readout
        self.Ch = Channel
        self.Gr = Group
        self.header = dict()
        
        self.open()

    def set_header(self):
        #sysinfo = dict( self.hud.attrs )
        #chinfo = dict( self.hud['rack_%02d' %self.R]['channel_%02d' %self.Ch].attrs )
        dataconf = self.hud['rack_%02d' %self.R]['channel_%02d' %self.Ch]['configuration_group%d' %self.Gr][()]

        self.header['lofreq'] = self.info['LO'] # Hz
        self.header['npoints'] = self.info['framelen'] # 16 or 19
        self.header['fftgain'] = 2**15
        self.header['nbins']   = len(dataconf)
        self.header['framert'] = 1e+9/1024

        for i in range( len(dataconf) ):
            self.header['BIN%d' %i] = dataconf[i][0]
            self.header['CenterIQ%d' %i] = (dataconf[i][1], dataconf[i][2])
            self.header['BaseAngle%d' %i] = dataconf[i][3]
            self.header['Threshold%d' %i] = dataconf[i][4]
        
    def open(self):
        self.hud = h5py.File(self.infile, 'r')
        ### store information
        self.set_header()
        
        self.fftgain  = self.header['fftgain']
        self.framert  = self.header['framert']
        self.nbins    = self.header['nbins']
        self.npoints  = self.header['npoints']
        self.lofreq   = self.header['lofreq']

        self.bins = [_to_nbit_signed(self.header['BIN%d' %i], self.npoints)
                    for i in range(self.nbins)]
        self.if_freq = 2e+9 * np.array(self.bins) / 2**self.npoints
        self.carrier_freq = self.lofreq + self.if_freq # Hz

        ### store timestamp
        datalength = self.hud['rack_%02d' %self.R]['channel_%02d' %self.Ch]['continuous_group%d' %self.Gr].shape[0]
        self.timestamp = np.arange( datalength ) / self.framert # no absolute time stamp
        self.framenr = np.arange( datalength )
        self._read_bins = dict()

        #self.offset = -100 # remove last part of data (sometimes they behave bad)
        
    def close(self):
        self.hud.close()

    def __len__(self):
        return len(self.bins)
    
    def __contains__(self, key):
        return key in self.bins
    
    def __iter__(self):
        for key in sorted(self.bins):
            yield key
            
    def __getitem__(self, key):
        #print( 'get %s' % key )
        if key in self._read_bins:
            # return self.h5.root.
            return self._read_bins[key]
        else:
            ind = self.bins.index(key)
            name = 'kid_%04d' %ind
            rawdata = self.hud['rack_%02d' %self.R]['channel_%02d' %self.Ch]['continuous_group%d' %self.Gr][()]
            read_I_ = rawdata[:, 2*ind    ]
            read_Q_ = rawdata[:, 2*ind + 1]
            # remove last part of data (sometimes they behave bad)
            #read_I_ = rawdata[:self.offset, 2*ind    ]
            #read_Q_ = rawdata[:self.offset, 2*ind + 1]

            info = dict()
            info['bins']   = self.bins[ind]
            info['freqs']  = self.carrier_freq[ind]
            info['header'] = self.header
            d = FixedData('I-Q-Gain', (self.timestamp,), (read_I_, read_Q_, self.fftgain),
                              self.carrier_freq[ind], info=info)
            #d = FixedData('I-Q-Gain', (self.timestamp[:self.offset],), (read_I_, read_Q_, self.fftgain),
            #                 self.carrier_freq[ind], info=info)
            self._read_bins[key] = d
            return d
        
    def __getstate__(self):
        if hasattr(self, 'hud'):
            dic = self.__dict__.copy()
            dic['_read_bins'] = {}
            dic['hud'] = None
            return dic
        else:
            return self.__dict__

    def __setstate__(self, dic):
        self.__dict__ = dic.copy()
        if 'hud' in dic:
            self.open()


### private functions
def _read_sweep(fname):
    """Read local sweep file.

    Return (lofreqs, bins, data)
      where lofreqs is an 1D array of frequency in Hz;
            bins is an 1D array of bin indices;
            data is an 2D array of complex demodulated amplitudes,
            one row per LO frequency, one column per bin.
    """
    rawdata = np.loadtxt(fname)
    nrow, ncol = rawdata.shape
    
    #print nrow, ncol
    bins = list( map(int, rawdata[0, 1::3]) )
    data     = 1.0 * rawdata[:, 2::3] + 1.0j * rawdata[:, 3::3]
    lofreqs  = rawdata[:, 0]*1e6
    return lofreqs, bins, data

def _to_nbit_signed(x, n):
    if x > 2**(n-1):
        return -((~x & (2**n-1))+1)
    else:
        return x
    
