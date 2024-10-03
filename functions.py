import re
import glob
import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import find_peaks, welch, windows, fftconvolve
from scipy.fft import fft, ifft
from scipy.stats import gaussian_kde
from scipy.optimize import curve_fit
from scipy.interpolate import interp1d
import os


def get_files(dir_path, kid_nr, p_read, type='vis'):
    txt = 'KID' + str(kid_nr) + '_' + str(p_read) + 'dBm__TD' + str(type)
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
    data = np.fromfile(file_path, dtype='>f8', count=-1)
    data = data.reshape((-1, 2))

    I = data[:, 0]
    Q = data[:, 1]

    # From I and Q data to Radius/Magnitude and Phase
    r = np.sqrt(I**2 + Q**2)
    R = r/np.mean(r) # Normalize radius to 1

    P = np.arctan2(Q, I) 
    P = np.pi - P % (2 * np.pi) # Convert phase to be taken from the negative I axis
    return R, P


def plot_bin(file_path):
    response = bin2mat(file_path)[1]
    time = np.arange(len(response)) * 20e-6
    info = get_info(file_path[:-4]+'_info.dat')
    print(info)
    fig, ax = plt.subplots(figsize=(10, 4), constrained_layout=True)
    ax.plot(time, response, lw=.2)
    ax.set_xlabel('t [s]')
    ax.set_ylabel('$\\theta$ [rad]')
    line_len = len(file_path) // 2
    ax.set_title(file_path[:line_len] + '\n' + file_path[line_len:])
    ax.set_xlim(time[0], time[-1])


def get_data(file_list, discard=True):
    limit = -0.5 * np.pi
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
    # Smooth timestream data for peak finding
    if len(window):    
        signal = fftconvolve(signal, window, mode='valid')
    neg_signal = signal[signal<=0]
    std = np.std(np.hstack((neg_signal, np.absolute(neg_signal))))
    return np.round(std, decimals=3)


def find_pks(signal, ph, pp, window):
    # Smooth timestream data for peak finding
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


def get_single_pulses(signal, locs, pw, rise_offset, args):
    pulses = []
    nr_peaks = len(locs)
    len_signal = len(signal)
    singles = np.zeros(nr_peaks, dtype=bool)
    for arg in args:
        loc = locs[arg]
        single = 1
        if  arg < nr_peaks - 1 and arg > 0: 
            prev_loc = locs[arg-1]
            next_loc = locs[arg+1]
            if (loc + pw >= next_loc or loc - pw - rise_offset <= prev_loc or loc + pw >= len_signal or loc - rise_offset < 0):
                single = 0
        elif arg == 0:
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
        return [], singles


def get_single_noises(signal, locs, pw):
    noises = []
    len_signal = len(signal)
    nr_noises = 0
    nr_req_noises = 1000
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
        raise Exception('No good noise segments found')
    noises = np.array(noises).reshape((-1, pw))
    return noises


def get_avg_psd(pulses, pw, sff, exclude_dc=True, onesided=True):
    freqs, sxx = welch(pulses, fs=int(sff*1e6), window='hamming', nperseg=pw, noverlap=None, nfft=None, return_onesided=onesided, axis=1)
    return freqs[exclude_dc:], np.mean(sxx, axis=0)[exclude_dc:]


def opt_filter(mean_pulse, noise_psd, exclude_dc=True):      
    """
    optimal_filter computes the optimal filter that needs to be uploaded to the mux
    :param norm_pulse_fft:  numpy.ndarray, 1D complex array of pulse model
    :param noise_psd:       numpy.ndarray, 1D complex array of noise model
    :param exclude_dc:      bool, whether to exclude the DC value of the fft
    :return filter:         numpy.ndarray, 1D comlex array of normalised optimal filter
    """
    norm_pulse = mean_pulse / np.amax(mean_pulse)
    norm_pulse_fft = fft(norm_pulse)
    filter = norm_pulse_fft.conj()/noise_psd
    return ifft(filter)


def optimal_filter(pulses, pulse_model, sf, ssf_model, nxx, exclude_dc=True, onesided=True):
    ''' 
    This function applies an optimal filter, i.e. a frequency weighted filter, to the pulses to extract a better estimate of the pulse heights
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


def fit_decaytime(pulse, pw, fit_T):
    ''' 
    This function returns the quasiparticle regeneration time, tau_qp by fitting a function y=a*exp(-x/tau_qp) to the tail of the pulse
    '''
    # Cut the tail from the pulse for fitting
    l = len(pulse)
    ssf = int(l / pw)
    t = np.linspace(0, pw, l)

    if isinstance(fit_T, (int, float)):
        fit_pulse = pulse[t>=fit_T]
        # fit_t = t[t>fit_T]
    elif isinstance(fit_T, (tuple, list, np.ndarray)):
        fit_pulse = pulse[(t>=fit_T[0]) & (t<fit_T[1])]
        # fit_t = t[(t>fit_T[0]) & (t<fit_T[1])]
    else:
        raise Exception('Please input fit_T as integer or array-like') 
    fit_x = np.arange(len(fit_pulse))

    # Obtain the optimal parameters from fit
    popt, pcov = curve_fit(exp_decay, fit_x, fit_pulse)

    # Obtain 1 std error on parameters
    perr = np.sqrt(np.diag(pcov))

    # Obtain tau_qp and error
    tau_qp = 1 / popt[1] / ssf
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