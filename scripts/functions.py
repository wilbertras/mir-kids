"""
functions.py

This script contains general functions for data analysis used in the top level analysis scripts.
"""

#--------------------------------------------------
# Import modules
# -------------------------------------------------
import re
import glob
import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import find_peaks, welch, windows, fftconvolve
from scipy.fft import fft, ifft
from scipy.stats import gaussian_kde
from scipy.optimize import curve_fit
from scipy.interpolate import interp1d
import scripts.utils.matplotlibcolors as matplotlibcolors
import pandas as pd
import scipy.constants as sc
import io


#--------------------------------------------------
# Define functions
# -------------------------------------------------
def get_files(dir_path, kid_nr, p_read, type='vis', chip=''):
    """
    This function returns the list of bin files and info files for a given kid in a directory at a certain readout power.
    """
    dir_path = dir_path.replace("\\", '/')
    txt = 'KID' + str(kid_nr) + '_' + str(p_read) + 'dBm_'+chip+'_TD' + str(type)
    info_path = dir_path + '/' + txt + '*_info.dat'
    bin_path = dir_path + '/' + txt + '*.bin'
    list_bin_files = glob.glob(bin_path)
    info_files = glob.glob(info_path)
    if not list_bin_files:
        raise Exception('Please correct folder path as no files were obtained using path:\n%s' % (bin_path))
    if not info_files:
        print('Please correct folder path as no files were obtained using path:\n%s' % (bin_path))
        info_files = []
    list_bin_files = sorted(list_bin_files, key=lambda s: int(re.findall(r'\d+', s)[-2]))
    return list_bin_files, info_files


def get_info(file_path):
    """
    This functions returns the important parameters of the measurement obtained from the info file
    """
    with open(file_path) as f:
        lines = f.readlines()
    f0 = float(re.findall("\d+\.\d+", lines[2])[0]) 
    Qs = re.findall("\d+\.\d+", lines[3])
    Qs = [float(Q) for Q in Qs]
    [Q, Qc, Qi, S21_min] = Qs
    dt = float(re.findall("\d+\.\d+", lines[4])[0])
    T = float(re.findall("\d+\.\d+", lines[7])[0])
    return f0, Q, Qc, Qi, S21_min, dt, T


def bin2mat(file_path):
    """
    This function reads the binary file containing the I and Q data and returns the radius/magnitude and phase response of the detector.
    """
    data = np.fromfile(file_path, dtype='>f8', count=-1)
    data = data.reshape((-1, 2))

    I = data[:, 0]
    Q = data[:, 1]

    # From I and Q data to Radius/Magnitude and Phase
    r = np.sqrt(I**2 + Q**2)
    I /= np.mean(r) # Normalize I to 1
    Q /= np.mean(r) # Normalize Q to 1
    R = np.sqrt(I**2 + Q**2)

    P = np.arctan2(Q, I) 
    P = np.pi - P % (2 * np.pi) # Convert phase to be taken from the negative I axis
    return R, P


def get_data(file_list, discard=True):
    """
    This concatenates the data from a list of files and returns the amplitude and phase response of the detector. If discard is True, files with phase <= -pi rad are discarded, otherwise they are included in the data with a 2*pi rad shift.
    """
    limit = -np.pi
    amp = []
    phase = []
    removed = 0
    saturated_Ts = []
    for i, file in enumerate(file_list):
        r, p = bin2mat(file)
        saturated = p <= limit
        if np.any(saturated):
            saturated_Ts.append(i)
            if discard:
                removed += 1
                append = 0
                fig, ax = plt.subplot_mosaic('ab', figsize=(6, 3), sharex=True, sharey=True, constrained_layout=True)
                ax['a'].set_title('phase')
                ax['a'].plot(p, lw=.5)
                # p[saturated] += 2 * np.pi
                ax['b'].set_title('amplitude')
                ax['b'].plot(r, lw=.5)
            else: 
                append = 1
        else:
            append = 1
        if append:
            amp.append(r)
            phase.append(p)
    amp = np.array(amp).flatten()
    phase = np.array(phase).flatten()
    nr_saturated = len(saturated_Ts)
    if nr_saturated:
        print('     WARNING: %d files found with phase <= %.1f pi rad (at T=' % (nr_saturated, limit/np.pi), saturated_Ts, 's)')
    return amp, phase, removed


def smith_coord(P, R):
    '''
    This function returns the phase and amplitude reponse in the Smith chart coordinate systam.
    '''
    # Normalised I and Q
    I_r = -np.cos(P) * R
    Q_r = np.sin(P) * R

    # SMith chart coordinate system
    G = I_r + 1j * Q_r
    z = (1 + G) / (1 - G)

    R_smith = np.real(z)
    X_smith = np.imag(z)
    R_smith -= np.mean(R_smith)
    X_smith -= np.mean(X_smith)
    return R_smith, X_smith


def coord_transformation(phase, amp, coord='smith', response='phase'):
    """
    This function returns the phase and amplitude response in the specified coordinate system (Smith or circle) and for the specified response type (amplitude or phase). For the Smith chart coordinate system, the amplitude and phase response are transformed to the real and imaginary part of the Smith chart, respectively. For the circle coordinate system, the amplitude and phase response are transformed to the distance from the mean value of the amplitude and phase response, respectively.
    """
    if coord == 'smith':
        amp, phase = smith_coord(phase, amp)
        if response == 'amp':
            return amp
        elif response == 'phase':
            return phase
        else:
            raise Exception('Please input a proper response type ("amp" or "phase")')
    elif coord == 'circle':
        if response == 'phase':
            return phase - np.mean(phase)
        elif response == 'amp':
            return (1 - amp) - np.mean(1 - amp)
        else:
            raise Exception('Please input a proper response type ("amp" or "phase")')
    else:
        raise Exception('Please input a proper coordinate system ("smith" or "circle")')   


def get_sigma(signal, window):
    """
    This functions returns the standard deviation of the (filtered) noise in the signal, which is used for peak finding. The noise is estimated from the negative part of the signal
    """
    if len(window):    
        signal = fftconvolve(signal, window, mode='valid')
    neg_signal = signal[signal<=0]
    std = np.std(np.hstack((neg_signal, np.absolute(neg_signal))))
    return np.round(std, decimals=3)


def find_pks(signal, ph, pp, window):
    """
    This function detects the peaks in the (filtered) signal using the scipy find_peaks function and minimal peak height and prominence thresholds
    """
    if len(window):    
        signal = fftconvolve(signal, window, mode='valid')
        window_offset = int(np.argmax(window[::-1]))
    else:
        signal = signal
        window_offset = 0

    # Find peaks in data
    locs, props = find_peaks(signal, height=ph, prominence=pp)
    locs = locs + window_offset
    return locs, props


def get_single_pulses(signal, locs, pw, rise_offset, args=[]):
    """
    This functions returns the single pulses in the signal. Single pulses are required to be seperated by at least a pulsewindow
    """
    pulses = []
    nr_peaks = len(locs)
    len_signal = len(signal)
    singles = np.zeros(nr_peaks, dtype=bool)
    if not len(args):
        args = np.arange(nr_peaks, dtype=int)
    for arg in args:
        loc = locs[arg]
        single = 1
        if  arg < nr_peaks - 1 and arg > 0: 
            prev_loc = locs[arg-1]
            next_loc = locs[arg+1]
            if (loc + pw >= next_loc or loc - pw - rise_offset <= prev_loc or loc + pw >= len_signal or loc - rise_offset < 0):
                single = 0
        elif arg == 0 and not arg == nr_peaks - 1:
            next_loc = locs[arg+1]
            if nr_peaks > 1:
                if (loc + pw >= next_loc or loc + pw >= len_signal or loc - rise_offset < 0):
                    single = 0
            else:
                if (loc + pw >= len_signal or loc - rise_offset < 0):
                    single = 0
        elif arg == nr_peaks - 1: 
            prev_loc = locs[arg-1]
            if (loc - pw - rise_offset <= prev_loc or loc + pw >= len_signal or loc - rise_offset < 0):
                single = 0 
        if single:                 
            singles[arg] = 1
            pulse = signal[loc-rise_offset:loc+pw]
            pulses.append(pulse)
    if np.sum(singles):
        pulses_aligned = np.array(pulses).reshape((-1, pw+rise_offset)) 
        return pulses_aligned, singles
    else:
        return  np.empty((1, pw+rise_offset)), singles
       


def get_single_noises(signal, locs, pw, nr_req_noises=50000):
    """
    This function returns a set of noise segments from the data. Noise segment containing pulses are discarded
    """
    noises = []
    len_signal = len(signal)
    nr_noises = 0
    t = 0
    while nr_noises < nr_req_noises and t+pw < len_signal:
        if np.any((locs >= t - pw) & (locs <= t+pw)):
            pass
        else:
            noise = signal[t:t+pw]
            noises.append(noise)
            nr_noises += 1  
        t += pw
    if nr_noises == 0:
        print('     WARNING: No noise windows could be obtained')
        return np.empty((1, pw))
    noises = np.array(noises).reshape((-1, pw))
    return noises


def optimal_filter(pulses, pulse_model, sf, ssf_model, nxx, exclude_dc=True, onesided=True):
    ''' 
    This function applies an optimal filter to the pulses to extract the optimal pulse heights. It also return the sigal-to-noise resolving power
    '''
    # Initialize important variables 
    len_pulses = pulses.shape[-1]
    len_model = len(pulse_model)
    if len_pulses < len_model:
        len_onesided = round(len_pulses / 2) + 1
        ssf_pulses = 1
    else:
        len_onesided = round(len_pulses / ssf_model / 2) + 1
        ssf_pulses = ssf_model

    # Compute normalized pulse model
    norm_pulse_model = pulse_model / np.amax(pulse_model)

    # Step 1: compute psd and fft of normalized peak-model
    # Mxx = psd(norm_pulse_model, sf*ssf_model)
    Mxx = welch(norm_pulse_model, fs=sf*ssf_model, window='hamming', nperseg=len_model, noverlap=None, nfft=None, return_onesided=False)[1]
    Mf = fft(norm_pulse_model) / ssf_model
    Mf_conj = Mf.conj()

    # Step 2: compute fft of all pulses
    Df = fft(pulses, axis=-1) / ssf_pulses
    # Dxx = psd(pulses, sf*ssf_pulses)
    Dxx = welch(pulses, fs=sf*ssf_pulses, window='hamming', nperseg=len_model, noverlap=None, nfft=None, return_onesided=False)[1]
    mean_Dxx = np.mean(Dxx, axis=0)

    # Step 3: obtain improved pulse height estimates
    numerator = Mf_conj * Df / nxx
    denominator = Mf_conj * Mf / nxx
    int_numerator = np.sum(numerator[:, exclude_dc:], axis=-1)
    int_denominator = np.sum(denominator[exclude_dc:], axis=-1)
    H = np.real(int_numerator / int_denominator)

    # Step 4: compute signal-to-noise resolving power
    NEP = (np.outer((1 / H)**2, (2 * nxx / Mxx)))**0.5
    dE = 2*np.sqrt(2*np.log(2)) * (np.sum(4 / NEP[:, exclude_dc:]**2, axis=-1))**-0.5
    R_sn = np.mean(1 / dE)

    error = (Df - np.outer(H, Mf))**2 / nxx
    chi_sq = np.sum(error[:, exclude_dc:], axis=-1)
    
    return H, R_sn, mean_Dxx, chi_sq


def resolving_power_sn(H, pulse_model, sf, ssf_model, nxx):
    """
    
    """
    norm_pulse_model = pulse_model / np.amax(pulse_model)
    len_model = len(pulse_model)
    Mxx = welch(norm_pulse_model, fs=sf*ssf_model, window='hamming', nperseg=len_model, noverlap=None, nfft=None, return_onesided=False)[1]
    Rsn = np.mean(H) / (2*np.sqrt(2*np.log(2))) * np.sqrt(np.sum(Mxx/nxx))
    return Rsn


def resolving_power(dist, histbin, range=None):
    ''' 
    This function obtains the resolving power of a distribution by means of a kernel density estimation
    '''
    
    # Limit the range of the KDE
    if range:
        if isinstance(range, (int, float)):
            dist = dist[dist>range]
        elif isinstance(range, (tuple, list, np.ndarray)):
            dist = dist[(dist > range[0]) & (dist < range[1])]
        else:
            raise Exception('Please input range as integer or array-like')
        
    # Check if distribution is not empty
    if dist.size == 0:
        raise Exception('Distribution is empty, check range')
    
    # Obtain pdf of distribution
    x = np.arange(np.amin(dist), np.amax(dist), histbin/10)
    nr_peaks = dist.size
    pdfkernel = gaussian_kde(dist, bw_method='scott')
    pdf = pdfkernel.evaluate(x)

    # Obtain index, value and x-position of the maximum of the distribution
    pdf_max = np.amax(pdf)
    pdf_max_idx = np.argmax(pdf)
    x_max = x[pdf_max_idx]

    # Find the left and right index and value of the pdf at half the maximum 
    hm = pdf_max / 2
   
    idx_right = (pdf > pdf_max / 4) & (x > x_max)
    pdf_right = pdf[idx_right]
    x_right = x[idx_right]
    if np.min(pdf_right) < hm < np.max(pdf_right):
        f_right = interp1d(pdf_right, x_right)(hm)
    else:
        print('Resolving power could not accurately be determined')
        f_right = np.max(pdf_right)
    
    idx_left = (pdf > pdf_max / 4) & (x < x_max) & (x > x_max - 2 * (f_right - x_max))
    x_left = x[idx_left]
    pdf_left = pdf[idx_left]
    if np.min(pdf_left) < hm < np.max(pdf_left):
        f_left = interp1d(pdf_left, x_left)(hm)
    else:
        print('Resolving power could not accurately be determined')
        f_left = np.min(pdf_left)

    # Compute the resolving power
    fwhm = f_right - f_left
    resolving_power = x_max / fwhm

    # Appropriately scale the pdf for plotting
    pdf = pdf * histbin * nr_peaks / (np.sum(pdf) * histbin/10)
    return resolving_power, pdf, x, x_max, fwhm


def fit_decaytime(pulse, fit_T, type='exp'):
    ''' 
    This function returns the quasiparticle regeneration time, tau_qp by fitting a function y=a*exp(-x/tau_qp) to the tail of the pulse
    '''

    if isinstance(fit_T, (int, float)):
        fit_pulse = pulse[fit_T:]
    elif isinstance(fit_T, (tuple, list, np.ndarray)):
        fit_pulse = pulse[fit_T[0]:fit_T[1]]
    else:
        raise Exception('Please input fit_T as integer or array-like') 
    fit_x = np.arange(len(fit_pulse))

    # Obtain the optimal parameters from fit
    popt, pcov = curve_fit(exp_decay, fit_x, fit_pulse)

    # Obtain 1 std error on parameters
    perr = np.sqrt(np.diag(pcov))

    # Obtain tau_qp and error
    tau_qp = 1 / popt[1]
    dtau_qp = perr[1]

    return tau_qp, dtau_qp, popt


def exp_decay(x, a, b):
    ''' 
    This is a one-term exponential function used for aluminium KIDs: y=a*exp(-b * x)
    '''
    return a * np.exp(-b * x)


def get_window(type, tau):
    if type == 'box':
        M = int(tau / 2)
        y = windows.boxcar(M, sym=False)
        y /= np.sum(y)
    elif type == 'exp':
        M = int(tau*3)
        y = windows.exponential(M, center=0, tau=tau, sym=False)
        y /= np.sum(y) 
    elif type == 'None':
        y = []
    else:
        raise Exception('Windowtype was given as %s. Please input a correct window type: "exp", "box" or "None"' % type)
    return y[::-1]


def fano(wl, Tc, J=0):
    """
    This function determines the Fano limit of the energy resolution of a KID for a given wavelength and critical temperature
    """
    Eph = sc.h * sc.c / wl
    F = 0.2
    Delta = 1.76 * sc.k * Tc
    eta_pb = 0.59
    return 1/(2*np.sqrt(2*np.log(2))) * np.sqrt(eta_pb * Eph / (Delta * (F+J)))


def logsmooth(fs, sxx):
    """
    This function smooths the power spectrum by binning the data in logarithmic bins and taking the mean value of the power spectrum in each bin. 
    """
    # Number of points per decade
    points_per_decade = 10

    # Define logarithmic bins
    log_min = np.log10(fs[1:].min())
    log_max = np.log10(fs.max())
    log_bins = np.logspace(log_min, log_max, int((log_max - log_min) * points_per_decade))

    # Bin the data
    binned_frequencies = []
    binned_powers = []

    for i in range(len(log_bins) - 1):
        # Find indices of frequencies within the current bin
        bin_indices = (fs >= log_bins[i]) & (fs < log_bins[i + 1])
        
        # Calculate the mean frequency and power spectrum value for the bin
        if np.any(bin_indices):
            binned_frequencies.append(np.mean(fs[bin_indices]))
            binned_powers.append(np.mean(sxx[bin_indices]))

    # Convert to arrays
    binned_frequencies = np.array(binned_frequencies)
    binned_powers = np.array(binned_powers)
    return binned_frequencies, binned_powers


def planck_wl(wl, t):
    """
    This functions returns Planck's spectrum for a given temperature and wavelength
    """
    return sc.h * sc.c**2 / (wl**5 * (np.exp(sc.h * sc.c / (wl * sc.k * t)) - 1))