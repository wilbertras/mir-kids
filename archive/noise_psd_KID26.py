import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import fftconvolve, welch, csd
from scipy.fft import fft
import matplotlib as mpl
import pandas as pd
import os
import re
import sys
import pickle
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
import figures.functions as f
import KID_S21 as S21
from itertools import cycle
plt.style.use('figures/matplotlibrc')


def logsmooth(fs, sxx):
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


def get_noise_psd(dir, kid, pread, pw, filetype, nr_req_files, filter, tqp, mph, mpp, coord='circle', response='phase'):
    dir = dir.replace("\\", '/')
    if type(kid) is int:
        pulse_files, info_files = f.get_files(dir, kid, pread, type=filetype)
    elif type(kid) is str:
        pulse_files = [dir + '/' + kid + '.bin']
        info_files = [dir + '/' + kid + '_info.dat']
    f0, Q, Qc, Qi, S21_min, dt, T =  f.get_info(info_files[0])
    sff = 1 / dt / 1e6
    sw = round(tqp * sff)
    wl = round(pw * sff)
    nr_files = len(pulse_files)
    if nr_req_files >= nr_files:
        nr_req_files = nr_files
    amp, phase, _ = f.get_data(pulse_files[:nr_req_files])
    signal = f.coord_transformation(phase, amp, coord=coord, response=response)
    if filter:
        window = f.get_window(filter, sw)
        std = f.get_sigma(signal, window)
        ph = mph*std
        pp = mpp*std
        noise_locs, _ = f.find_pks(signal, ph, pp, window)
        signal_noises = f.get_single_noises(signal, noise_locs, wl)
        fxx, nxx = welch(signal_noises, fs=int(sff*1e6), window='flattop', nperseg=wl, return_onesided=True)
        print(nxx.shape)
        nxx = np.mean(nxx, axis=0)
    else:
        fxx, nxx = welch(signal, fs=int(sff*1e6), window='flattop', nperseg=wl, return_onesided=True)
    return logsmooth(fxx, nxx)


def get_Q(path, kid, plot=None):
    df = S21.loop_over_S21_files(path, kid, plot=plot)
    print(df)

# Q_bf_dark =  32554.641039207723
# Q_38um_off = 26130.694798
# Q_38um =  26155.015645914053
# Q_85um_off =  25888.428331481795
# Q_185um_3K =  53162.040560797424
# Q_185um_160K =  52979.36333999958
# Q_25um_dark =  28220.78914054096
# Q_25um_3K =  63829.27252722262
# Q_25um_24K =  63806.23296600575

# Q_bf_dark =  1
# Q_38um =  1
# Q_85um =  1
# Q_185um_3K =  1
# Q_185um_160K =1  
# Q_25um_dark = 1 
# Q_25um_3K = 1 
# Q_25um_24K = 1 
with open(r'KID26.pkl', 'rb') as f:
    kid_dict = pickle.load(f)

pw = 2000000
filetype = 'med'
mph = 500
mpp = mph
nr_req_files = 3
filter = None
lifetime = 50
exclude_dc = 1
name = 'KID26'

fig, axes = plt.subplot_mosaic('a;b', figsize=(18.5*2/3/2.54,9/2.54), sharex=True, sharey=True, constrained_layout=True)
ylabel_psd = '$S_{\\theta}/(4Q_l)^2$ [dBc/Hz]'
xlabel_psd = 'Frequency [Hz]'
ylim_psd = [-175, -145]
xlim_psd = [10, .5e6]
width = 1


ax = axes['a']

wl = '3.8um'
kid = kid_dict[wl]['kid']
pread = kid_dict[wl]['pread']
path_on  = r"D:\Data\LT218Chip1_BF_20221103_MIR3_8\12KIDs mono on 3800\TD_Power"
path_off  = r"D:\Data\LT218Chip1_BF_20221103_MIR3_8\12KIDs mono off\TD_Power"
path_dark = r"D:\Data\LT218Chip1_BF_20230208_dark\TD_Power"
freqs, sxx = get_noise_psd(path_on, kid, pread, pw, filetype, nr_req_files, filter, lifetime, mph, mpp)
_, nxx = get_noise_psd(path_off, kid, pread, pw, filetype, nr_req_files, filter, lifetime, mph, mpp)
_, dxx = get_noise_psd(path_dark, kid, pread, pw, filetype, nr_req_files, filter, lifetime, mph, mpp)
ax.semilogx(freqs[exclude_dc:],  10*np.log10(dxx[exclude_dc:]/(4*Q_bf_dark**2)), label='dark BF', c='k', lw=width)
ax.semilogx([],  [], label=' ', c='None', lw=width)
ax.semilogx([],  [], label=' ', c='None', lw=width)
ax.semilogx(freqs[exclude_dc:],  10*np.log10(nxx[exclude_dc:]/(4*Q_38um_off**2)), label='3.8 $\mu$m, lamp off', c='b', lw=width, ls='-')
ax.semilogx(freqs[exclude_dc:],  10*np.log10(sxx[exclude_dc:]/(4*Q_38um_off**2)), label='3.8 $\mu$m, lamp on', c='b', lw=width, ls='--')
ax.semilogx([],  [], label=' ', c='None', lw=width)
ax.set_ylabel(ylabel_psd)
ax.set_ylim(ylim_psd)
ax.set_xlim(xlim_psd)

kid = 24
pread = 113
path_off  = r"D:\Data\LT218Chip1_BF_20221025_MIR8_5\12KIDs LN2 load - long\TD_Power"
path_on  = r"D:\Data\LT218Chip1_BF_20221025_MIR8_5\12KIDs mono off long\TD_Power"
freqs, sxx = get_noise_psd(path_off, kid, pread, pw, filetype, nr_req_files, filter, lifetime, mph, mpp)
_, nxx = get_noise_psd(path_off, kid, pread, pw, filetype, nr_req_files, filter, lifetime, mph, mpp)
ax.semilogx(freqs[exclude_dc:],  10*np.log10(nxx[exclude_dc:]/(4*Q_85um_off**2)), label='8.5 $\mu$m, LN2 load', c='y', lw=width, ls='-')
ax.semilogx(freqs[exclude_dc:],  10*np.log10(sxx[exclude_dc:]/(4*Q_85um_off**2)), label='8.5 $\mu$m, lamp off', c='y', lw=width, ls='-')
ax.set_ylabel(ylabel_psd)

kid = 25
pread = 117
path_off  = r"D:\Data\LT218Chip1_BF_20240116_MIR18_5\12KIDs_185um_BB3K\TD_Power"
path_on  = r"D:\Data\LT218Chip1_BF_20240116_MIR18_5\12KIDs_185um_BB160K\TD_Power"
Toff = int(re.findall(r'BB(\d+)K', path_off)[0])
Ton = int(re.findall(r'BB(\d+)K', path_on)[0])
freqs, sxx = get_noise_psd(path_on, kid, pread, pw, filetype, nr_req_files, filter, lifetime, mph, mpp)
_, nxx = get_noise_psd(path_off, kid, pread, pw, filetype, nr_req_files, filter, lifetime, mph, mpp)
ax.semilogx(freqs[exclude_dc:],  10*np.log10(nxx[exclude_dc:]/(4*Q_185um_3K**2)), label='18.5 $\mu$m, $T_{bb}=%d\ K$' % Toff, c='o', ls='-', lw=width)
ax.semilogx(freqs[exclude_dc:],  10*np.log10(sxx[exclude_dc:]/(4*Q_185um_3K**2)), label='18.5 $\mu$m, $T_{bb}=%d\ K$' % Ton, c='o', lw=width, ls='--')
ax.set_ylabel(ylabel_psd)
ax.legend(loc='upper right', ncols=3, borderaxespad=0., frameon=False, handlelength=1)


ax = axes['b']

kid = 24
pread = 115
path_dark = r"D:\Data\LT218Chip1_ADR_20240605_MIR24_dark\12KIDs_3Pread_100s_1MHz\TD_Power"
path_off  = r'D:\Data\LT218Chip1_ADR_20240506_MIR24\12KIDs_3Pread_TD40s_1MHz_BB3K\TD_Power'
path_on  = r"D:\Data\LT218Chip1_ADR_20240506_MIR24\12KIDs_3Pread_TD40s_1MHz_BB26K\TD_Power"
Toff = int(re.findall(r'BB(\d+)K', path_off)[0])
Ton = int(re.findall(r'BB(\d+)K', path_on)[0])
freqs, sxx = get_noise_psd(path_on, kid, pread, pw, filetype, nr_req_files, filter, lifetime, mph, mpp)
_, nxx = get_noise_psd(path_off, kid, pread, pw, filetype, nr_req_files, filter, lifetime, mph, mpp)
pread = 113
_, dxx = get_noise_psd(path_dark, kid, pread, pw, filetype, nr_req_files, filter, lifetime, mph, mpp)
ax.semilogx(freqs[exclude_dc:],  10*np.log10(dxx[exclude_dc:]/(4*Q_25um_dark**2)), label='dark ADR', c='k', lw=width, ls='-')
ax.semilogx(freqs[exclude_dc:],  10*np.log10(nxx[exclude_dc:]/(4*Q_25um_3K**2)), label='25 $\mu$m, $T_{bb}=%d\ K$' % Toff, c='p', ls='-')
ax.semilogx(freqs[exclude_dc:],  10*np.log10(sxx[exclude_dc:]/(4*Q_25um_3K**2)), label='25 $\mu$m, $T_{bb}=%d\ K$' % Ton, c='p', lw=width, ls='--')
ax.set_ylabel(ylabel_psd)
ax.set_xlabel(xlabel_psd)
ax.legend(loc='upper right', ncols=3, borderaxespad=0., frameon=False, handlelength=1)
# plt.savefig('figures/noise psds KID26.pdf')
plt.show()