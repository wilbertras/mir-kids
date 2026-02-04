import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import fftconvolve, welch, csd
import matplotlib as mpl
import pandas as pd
import os
import sys
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


def get_circle(dirs, kid, pread):
    if type(dirs) is str:
        dirs = [dirs.replace('TD_Power', 'FFT\\Power')]
    elif type(dirs) is list:
        dirs = [dir.replace('TD_Power', 'FFT\\Power') for dir in dirs]
    files = [dir + '\\KID%d_%ddBm__S21.dat' % (kid, pread) for dir in dirs]
    for file in files:
        try:
            with open(file, 'r') as f:
                data = pd.read_csv(f, sep='\t', skiprows=9, header=None)
                fcircle, Re, Im = data[0], data[1], data[2]
        except:
            print('File %s not found' % file)
            fcircle, Re, Im = np.array([]), np.array([]), np.array([])
        try:
            with open(file.replace('S21', 'S21dB'), 'r') as f:
                data = pd.read_csv(f, sep='\t', skiprows=9, header=None)
                f21, dB, rad = data[0], data[1], data[2]
        except:
            print('File %s not found' % file)
            f21, dB, rad = np.array([]), np.array([]), np.array([])
        return fcircle, Im, Re, f21, dB, rad

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
        # amp_noises = f.get_single_noises(signal, noise_locs, wl)
        # fxx, nxx = csd(amp_noises, phase_noises, fs=int(sff*1e6), window='flattop', nperseg=wl, return_onesided=True, axis=1)
        fxx, nxx = welch(signal_noises, fs=int(sff*1e6), window='flattop', nperseg=wl, return_onesided=True)
        print(nxx.shape)
        nxx = np.mean(nxx, axis=0)
    else:
        nr_noises = len(signal) // wl 
        # fxx, nxx = csd(amp, phase, fs=int(sff*1e6), window='flattop', nperseg=wl, return_onesided=True)
        fxx, nxx = welch(signal, fs=int(sff*1e6), window='flattop', nperseg=wl, return_onesided=True)
        # _, nf = welch(phase, fs=int(sff*1e6), window='flattop', nperseg=wl, return_onesided=True)
        # fxx, nxx = welch(amp, fs=int(sff*1e6), window='flattop', nperseg=wl, return_onesided=True)
        # phase_noises = phase[:int(nr_noises*wl)].reshape((nr_noises, wl))
        # amp_noises = amp[:int(nr_noises*wl)].reshape((nr_noises, wl))
    
    return logsmooth(fxx, nxx)


def get_Q(path, kid, plot=None):
    df = S21.loop_over_S21_files(path, kid, plot=plot)
    print(df)

# kids = [3, 5, 8, 9, 10, 13, 16, 17, 20, 21, 23, 24]
# preads = [117, 113, 114, 118, 117, 120, 119, 116, 108, 103, 113, 113]
# kids = [ 21, 23, 24]
# preads = [103, 113, 113]
# path  = r"D:\Data\LT218Chip1_BF_20221103_MIR3_8\12KIDs mono off\TD_Power"
# # path = r"D:\Data\LT218Chip1_BF_20230208_dark\TD_Power"
# pw = 10000000
# filetype = 'med'
# mph = 3
# mpp = mph
# nr_req_files = 10
# filter = None
# lifetime = 250
# exclude_dc = 0
# width = 1
# styles = ['-', '--']
# colors = plt.rcParams['axes.prop_cycle'].by_key()['color']

# fig, axes = plt.subplot_mosaic('aa;bc', figsize=(18/2.54,7/2.54), constrained_layout=True)
# for i, kid in enumerate(kids[:1]):
#     p = preads[i]
#     ps = [p, p+2]
#     for j, pread in enumerate(ps):
#         color_cycler = cycle(colors)
#         color = next(color_cycler)
#         style = styles[j]
#         freqs, nc, na, nf = get_noise_psd(path, kid, pread, pw, filetype, nr_req_files, filter, lifetime, mph, mpp)
#         freqs_smooth, nc_smooth = logsmooth(freqs, nc)
#         freqs_smooth, na_smooth = logsmooth(freqs, na)
#         freqs_smooth, nf_smooth = logsmooth(freqs, nf)
#         axes['a'].semilogx(freqs_smooth[exclude_dc:],  10*np.log10(nc_smooth[exclude_dc:]), label='KID%d_P%d, cross' % (kid, pread), lw=width, c=color, ls=style)
#         axes['a'].semilogx(freqs_smooth[exclude_dc:],  10*np.log10(na_smooth[exclude_dc:]), label='KID%d_P%d, amp' % (kid, pread), lw=width, c=next(color_cycler), ls=style)
#         axes['a'].semilogx(freqs_smooth[exclude_dc:],  10*np.log10(nf_smooth[exclude_dc:]), label='KID%d_P%d, phase' % (kid, pread), lw=width, c=next(color_cycler), ls=style)
#         axes['a'].legend()
#         _, Im, Re, freqs, dB, rad = get_circle(path, kid, pread)
#         axes['b'].plot(Re, Im, c=color, ls=style)
#         axes['b'].set_aspect('equal')
#         axes['c'].plot(freqs/np.mean(freqs), dB, c=color, ls=style)
# plt.show()

kid = 5
pread = 113
pw = 500000
filetype = 'med'
mph = 50
mpp = mph
nr_req_files = 10
filter = 'exp'
lifetime = 250
exclude_dc = 0

fig, axes = plt.subplot_mosaic('a;b', figsize=(18.5*2/3/2.54,8/2.54), sharex=True, sharey=True, constrained_layout=True)
ylabel_psd = '$S_{\\theta}$ [dBc/Hz]'
xlabel_psd = 'Frequency [Hz]'
ylim_psd = [-80, -40]
xlim_psd = [7, 2e4]
width = 1
path_on  = r"D:\Data\LT218Chip1_BF_20221103_MIR3_8\12KIDs mono on 3800\TD_Power"
path_off  = r"D:\Data\LT218Chip1_BF_20221103_MIR3_8\12KIDs mono off\TD_Power"
path_dark = r"D:\Data\LT218Chip1_BF_20230208_dark\TD_Power"
freqs, sxx = get_noise_psd(path_on, kid, pread, pw, filetype, nr_req_files, filter, lifetime, mph, mpp)
# _, pxx = get_noise_psd(path_on, kid, pread, pw, filetype, nr_req_files, None, lifetime, mph, mpp)
_, nxx = get_noise_psd(path_off, kid, pread, pw, filetype, nr_req_files, filter, lifetime, mph, mpp)
_, dxx = get_noise_psd(path_dark, kid, pread, pw, filetype, nr_req_files, filter, lifetime, mph, mpp)
# plot_circle([path_dark, path_off, path_on], kid, pread)
ax = axes['a']
ax.semilogx(freqs[exclude_dc:],  10*np.log10(dxx[exclude_dc:]), label='dark BF', c='k', lw=width)
ax.semilogx(freqs[exclude_dc:],  10*np.log10(nxx[exclude_dc:]), label='3.8 $\mu$m, lamp off', c='b', lw=width, ls='-')
# ax.semilogx(freqs[exclude_dc:],  10*np.log10(nxx[exclude_dc:]), label=' ', c='None', ls='None', lw=width)
# ax.semilogx(freqs[exclude_dc:],  10*np.log10(nxx[exclude_dc:]), label=' ', c='None', ls='None', lw=width)
ax.semilogx(freqs[exclude_dc:],  10*np.log10(sxx[exclude_dc:]), label='3.8 $\mu$m, lamp on', c='b', lw=width, ls='--')
# ax.semilogx(freqs[exclude_dc:],  10*np.log10(pxx[exclude_dc:]), label='3.8 $\mu$m, lamp on, with pulses', c='b', lw=width)
ax.set_ylabel(ylabel_psd)
# ax.set_xlabel(xlabel_psd)
ax.set_ylim(ylim_psd)
ax.set_xlim(xlim_psd)

path_on  = r"D:\Data\LT218Chip1_BF_20221025_MIR8_5\12KIDs mono off long\TD_Power"
path_off  = r"D:\Data\LT218Chip1_BF_20221025_MIR8_5\12KIDs LN2 load - long\TD_Power"
freqs, sxx = get_noise_psd(path_off, kid, pread, pw, filetype, nr_req_files, filter, lifetime, mph, mpp)
_, nxx = get_noise_psd(path_off, kid, pread, pw, filetype, nr_req_files, filter, lifetime, mph, mpp)
# plot_circle([path_dark, path_off, path_on], kid, pread)
ax = axes['a']
# ax.semilogx(freqs[exclude_dc:],  10*np.log10(sxx[exclude_dc:]/Q_signal**2), label='on', c='o')
# ax.semilogx(freqs[exclude_dc:],  10*np.log10(pxx[exclude_dc:]), label='8.5 $\mu$m, lamp off, with pulses', c='y', lw=width)
ax.semilogx(freqs[exclude_dc:],  10*np.log10(sxx[exclude_dc:]), label='8.5 $\mu$m, lamp off', c='y', lw=width, ls='-')
ax.set_ylabel(ylabel_psd)
# ax.set_ylim(ylim_psd)


path_on  = r"D:\Data\LT218Chip1_BF_20240116_MIR18_5\12KIDs_185um_BB160K\TD_Power"
path_off  = r"D:\Data\LT218Chip1_BF_20240116_MIR18_5\12KIDs_185um_BBoff\TD_Power"
freqs, sxx = get_noise_psd(path_on, kid, pread, pw, filetype, nr_req_files, filter, lifetime, mph, mpp)
# _, pxx = get_noise_psd(path_on, kid, pread, pw, filetype, nr_req_files, None, lifetime, mph, mpp)
_, nxx = get_noise_psd(path_off, kid, pread, pw, filetype, nr_req_files, filter, lifetime, mph, mpp)
ax = axes['a']
ax.semilogx(freqs[exclude_dc:],  10*np.log10(nxx[exclude_dc:]), label='18.5 $\mu$m, $T_{bb}=3\ K$', c='o', ls='-', lw=width)
ax.semilogx(freqs[exclude_dc:],  10*np.log10(sxx[exclude_dc:]), label='18.5 $\mu$m, $T_{bb}=160\ K$', c='o', lw=width, ls='--')
# ax.semilogx(freqs[exclude_dc:],  10*np.log10(pxx[exclude_dc:]), label='18.5 $\mu$m, $T_{bb}=160\ K$', c='o', ls='--', lw=width)
ax.set_ylabel(ylabel_psd)
# ax.set_ylim(ylim_psd)
ax.legend(loc='upper right', ncols=3, borderaxespad=0., frameon=False, handlelength=1)



ax = axes['b']
# path_on  = r"D:\Data\LT218Chip1_ADR_20240506_MIR24\12KIDs_3Pread_TD40s_1MHz_BB24K\TD_Power"
path_on = r"D:\Data\LT218Chip1_ADR_20240506_MIR24\12KIDs_3Pread_TD80s_50KHz_BB3-28K"
path_off  = path_on
path_dark = r"D:\Data\LT218Chip1_ADR_20240605_MIR24_dark\3KIDs_1Pread_100s_50kHz\TD_Power"
kid = 'KID5_113dBm_Tchip0.13_TDmed_TmK24016'
freqs, sxx = get_noise_psd(path_on, kid, pread, pw, filetype, nr_req_files, filter, lifetime, mph, mpp)
# _, pxx = get_noise_psd(path_on, kid, pread, pw, filetype, nr_req_files, None, lifetime, mph, mpp)
kid = 'KID5_113dBm_Tchip0.13_TDmed_TmK3255'
_, nxx = get_noise_psd(path_off, kid, pread, pw, filetype, nr_req_files, filter, lifetime, mph, mpp)
path_s21_dark = r'D:\Data\LT218Chip1_ADR_20240605_MIR24_dark\3KIDs_1Pread_100s_50kHz\FFT\Power'
kid = 'KID5_111dBm__TDmed_TmK130'
_, dxx = get_noise_psd(path_dark, kid, pread, pw, filetype, nr_req_files, filter, lifetime, mph, mpp)
# ax = axes['a']
# ax.semilogx(freqs[exclude_dc:],  10*np.log10(nxx[exclude_dc:]/Q_noise**2), label='25 $\mu$m, $T_{bb}=3\ K$', c='p')
ax.semilogx(freqs[exclude_dc:],  10*np.log10(dxx[exclude_dc:]), label='dark ADR', c='k', lw=width, ls='-')
ax.semilogx(freqs[exclude_dc:],  10*np.log10(nxx[exclude_dc:]), label=' ', c='None', ls='None', lw=width)
ax.semilogx(freqs[exclude_dc:],  10*np.log10(nxx[exclude_dc:]), label='25 $\mu$m, $T_{bb}=3\ K$', c='p', ls='-')
ax.semilogx(freqs[exclude_dc:],  10*np.log10(sxx[exclude_dc:]), label='25 $\mu$m, $T_{bb}=24\ K$', c='p', lw=width, ls='--')
# ax.semilogx(freqs[exclude_dc:],  10*np.log10(pxx[exclude_dc:]), label='25 $\mu$m, $T_{bb}=24\ K$', c='p', ls='--', lw=width)
# ax.set_ylim(ylim_psd)
ax.set_ylabel(ylabel_psd)
ax.set_xlabel(xlabel_psd)
# ax.legend(bbox_to_anchor=(0., 1, 1., .102), loc='lower left', ncols=2, mode="expand", borderaxespad=0.)
# handles, labels = axes['a'].get_legend_handles_labels()
# handles_b, labels_b = ax.get_legend_handles_labels()
# handles.extend(handles_b)
# labels.extend(labels_b)
ax.legend(loc='upper right', ncols=3, borderaxespad=0., frameon=False, handlelength=1)
# fig.legend(handles, labels, ncol=5, loc='lower left', bbox_to_anchor=(0.075, .95))
# plt.savefig('figures/noise_psds3.pdf')
plt.show()