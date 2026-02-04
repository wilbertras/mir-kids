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
import figures.functions as ft
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
        pulse_files, info_files = ft.get_files(dir, kid, pread, type=filetype)
    elif type(kid) is str:
        pulse_files = [dir + '/' + kid + '.bin']
        info_files = [dir + '/' + kid + '_info.dat']
    f0, Q, Qc, Qi, S21_min, dt, T =  ft.get_info(info_files[0])
    sff = 1 / dt / 1e6
    sw = round(tqp * sff)
    wl = round(pw * sff)
    nr_files = len(pulse_files)
    if nr_req_files >= nr_files:
        nr_req_files = nr_files
    amp, phase, _ = ft.get_data(pulse_files[:nr_req_files])
    signal = ft.coord_transformation(phase, amp, coord=coord, response=response)
    if filter:
        window = ft.get_window(filter, sw)
        std = ft.get_sigma(signal, window)
        ph = mph*std
        pp = mpp*std
        noise_locs, _ = ft.find_pks(signal, ph, pp, window)
        signal_noises = ft.get_single_noises(signal, noise_locs, wl)
        fxx, nxx = welch(signal_noises, fs=int(sff*1e6), window='flattop', nperseg=wl, return_onesided=True)
        print(nxx.shape)
        nxx = np.mean(nxx, axis=0)
    else:
        fxx, nxx = welch(signal, fs=int(sff*1e6), window='flattop', nperseg=wl, return_onesided=True)
    return logsmooth(fxx, nxx)


def get_Q(path, kid, plot=None):
    df = S21.loop_over_S21_files(path, kid, plot=plot)
    print(df)


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

fig, axes = plt.subplot_mosaic('a;b', figsize=(18.5/1.85/2.54, 9/2.54), sharex=False, sharey=True, constrained_layout=True)
ylabel_psd = '$S_{\\theta}/(4Q_l)^2$ [dBc/Hz]'
xlabel_psd = 'Frequency [Hz]'
ylim_psd = [-175, -145]
xlim_psd = [10, 2e4]
xticks = np.logspace(1, 4, 4, endpoint=True)
width = 1

ax = axes['a']
ax.set_ylabel(ylabel_psd)
ax.set_xlim(xlim_psd)
ax.set_ylim(ylim_psd)
ax.set_xticklabels(xticks, minor=False)
# ax.set_xtick

wl = 'BF dark'
kid = kid_dict[name][wl]['kid']
pread = kid_dict[name][wl]['pread']
path = kid_dict[name][wl]['dir']
Q = kid_dict[name][wl]['Q']
freqs, nxx = get_noise_psd(path, kid, pread, pw, filetype, nr_req_files, filter, lifetime, mph, mpp)
ax.semilogx(freqs[exclude_dc:],  10*np.log10(nxx[exclude_dc:]/(4*Q**2)), label='dark DR', c='k', lw=width)
ax.semilogx([],  [], label=' ', c='None', lw=width)
ax.semilogx([],  [], label=' ', c='None', lw=width)

wl = '3.8um off'
kid = kid_dict[name][wl]['kid']
pread = kid_dict[name][wl]['pread']
path = kid_dict[name][wl]['dir']
Q = kid_dict[name][wl]['Q']
freqs, sxx = get_noise_psd(path, kid, pread, pw, filetype, nr_req_files, filter, lifetime, mph, mpp)
ax.semilogx(freqs[exclude_dc:],  10*np.log10(sxx[exclude_dc:]/(4*Q**2)), label='3.8 µm; $T_\mathrm{lab}=293\ K$', c='b', lw=width, ls='-')


wl = '3.8um'
kid = kid_dict[name][wl]['kid']
pread = kid_dict[name][wl]['pread']
path = kid_dict[name][wl]['dir']
Q = kid_dict[name][wl]['Q']
freqs, sxx = get_noise_psd(path, kid, pread, pw, filetype, nr_req_files, filter, lifetime, mph, mpp)
ax.semilogx(freqs[exclude_dc:],  10*np.log10(sxx[exclude_dc:]/(4*Q**2)), label='3.8 µm; $T_\mathrm{lab}=293\ K$, QTH', c='b', lw=width, ls='--')
ax.semilogx([],  [], label=' ', c='None', lw=width)
ax.set_ylabel(ylabel_psd)
ax.set_ylim(ylim_psd)
ax.set_xlim(xlim_psd)


# wl = '8.5um off'
# kid = kid_dict[name][wl]['kid']
# pread = kid_dict[name][wl]['pread']
# path = kid_dict[name][wl]['dir']
# Q = kid_dict[name][wl]['Q']
# freqs, sxx = get_noise_psd(path, kid, pread, pw, filetype, nr_req_files, filter, lifetime, mph, mpp)
# ax.semilogx(freqs[exclude_dc:],  10*np.log10(sxx[exclude_dc:]/(4*Q**2)), label='8.5 $\mu$m, LN2 load', c='y', lw=width, ls='--')
# ax.semilogx([],  [], label=' ', c='None', lw=width)

wl = '8.5um'
kid = kid_dict[name][wl]['kid']
pread = kid_dict[name][wl]['pread']
path = kid_dict[name][wl]['dir']
Q = kid_dict[name][wl]['Q']
freqs, sxx = get_noise_psd(path, kid, pread, pw, filetype, nr_req_files, filter, lifetime, mph, mpp)
ax.semilogx(freqs[exclude_dc:],  10*np.log10(sxx[exclude_dc:]/(4*Q**2)), label='8.5 µm; $T_\mathrm{lab}=293\ K$', c='y', lw=width, ls='--')

wl = '18.5um off'
kid = kid_dict[name][wl]['kid']
pread = kid_dict[name][wl]['pread']
path = kid_dict[name][wl]['dir']
Q = kid_dict[name][wl]['Q']
T = int(re.findall(r'BB(\d+)K', path)[0])
freqs, sxx = get_noise_psd(path, kid, pread, pw, filetype, nr_req_files, filter, lifetime, mph, mpp)
ax.semilogx(freqs[exclude_dc:],  10*np.log10(sxx[exclude_dc:]/(4*Q**2)), label='18.5 µm; $T_{bb}=%d\ K$' % T, c='o', lw=width, ls='-')

wl = '18.5um'
kid = kid_dict[name][wl]['kid']
pread = kid_dict[name][wl]['pread']
path = kid_dict[name][wl]['dir']
Q = kid_dict[name][wl]['Q']
T = int(re.findall(r'BB(\d+)K', path)[0])
freqs, sxx = get_noise_psd(path, kid, pread, pw, filetype, nr_req_files, filter, lifetime, mph, mpp)
ax.semilogx(freqs[exclude_dc:],  10*np.log10(sxx[exclude_dc:]/(4*Q**2)), label='18.5 µm; $T_{bb}=%d\ K$' % T, c='o', lw=width, ls='--')
ax.legend(loc='upper right', ncols=3, borderaxespad=0., frameon=False, handlelength=1, columnspacing=0.25)


ax = axes['b']
ax.set_xlabel(xlabel_psd)
ax.set_ylabel(ylabel_psd)
ax.set_ylim(ylim_psd)
ax.set_xlim(xlim_psd)
ax.set_xticklabels(xticks, minor=False)

wl = 'ADR dark'
kid = kid_dict[name][wl]['kid']
pread = kid_dict[name][wl]['pread']
path = kid_dict[name][wl]['dir']
Q = kid_dict[name][wl]['Q']
freqs, sxx = get_noise_psd(path, kid, pread, pw, filetype, nr_req_files, filter, lifetime, mph, mpp)
ax.semilogx(freqs[exclude_dc:],  10*np.log10(sxx[exclude_dc:]/(4*Q**2)), label='dark ADR', c='k', lw=width, ls='-')
ax.semilogx([],  [], label=' ', c='None', lw=width)

wl = '25um off'
kid = kid_dict[name][wl]['kid']
pread = kid_dict[name][wl]['pread']
path = kid_dict[name][wl]['dir']
Q = kid_dict[name][wl]['Q']
T = int(re.findall(r'BB(\d+)K', path)[0])
freqs, sxx = get_noise_psd(path, kid, pread, pw, filetype, nr_req_files, filter, lifetime, mph, mpp)
ax.semilogx(freqs[exclude_dc:],  10*np.log10(sxx[exclude_dc:]/(4*Q**2)), label='25 $\mu$m, $T_{bb}=%d\ K$' % T, c='p', lw=width, ls='-')
ax.semilogx([],  [], label=' ', c='None', lw=width)

wl = '25um'
kid = kid_dict[name][wl]['kid']
pread = kid_dict[name][wl]['pread']
path = kid_dict[name][wl]['dir']
Q = kid_dict[name][wl]['Q']
T = int(re.findall(r'BB(\d+)K', path)[0])
freqs, sxx = get_noise_psd(path, kid, pread, pw, filetype, nr_req_files, filter, lifetime, mph, mpp)
ax.semilogx(freqs[exclude_dc:],  10*np.log10(sxx[exclude_dc:]/(4*Q**2)), label='25 $\mu$m, $T_{bb}=%d\ K$' % T, c='p', lw=width, ls='--')
ax.semilogx([],  [], label=' ', c='None', lw=width)

ax.legend(loc='upper right', ncols=3, borderaxespad=0., frameon=False, handlelength=1, columnspacing=0.25)
plt.savefig('figures/noise psds KID26.pdf')
plt.show()