import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import fftconvolve, welch
import matplotlib as mpl
import pandas as pd
import os
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
import figures.functions as f
import KID_S21 as S21
plt.style.use('figures/matplotlibrc')


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
    noise = f.coord_transformation(phase, amp, coord=coord, response=response)
    if filter:
        window = f.get_window(filter, sw)
        std = f.get_sigma(noise, window)
        ph = mph*std
        pp = mpp*std
        noise_locs, _ = f.find_pks(noise, ph, pp, window)
        noises = f.get_single_noises(noise, noise_locs, wl)
    else:
        nr_noises = len(noise) // wl 
        noises = noise[:int(nr_noises*wl)].reshape((nr_noises, wl))
    fxx, nxx = f.get_avg_psd(noises, wl, sff, exclude_dc=True, onesided=True)
    return fxx, nxx 

def get_Q(path, kid, plot=None):
    df = S21.loop_over_S21_files(path, kid, plot=plot)
    print(df)

kid = 5
pread = 113
pw = 50000
filetype = 'med'
mph = 3
mpp = mph
nr_req_files = 10
filter = 'exp'
lifetime = 250

fig, axes = plt.subplot_mosaic('ab', figsize=(18.58/2.54,7/2.54), sharex=True, sharey=True, constrained_layout=True)
ylabel_psd = '$S_{\\theta\\theta}$ [dBc/Hz]'
xlabel_psd = 'Frequency [Hz]'
ylim_psd = [-80, -50]
xlim_psd = [1e2, 25e3]
width = 1
path_on  = r"D:\Data\LT218Chip1_BF_20221103_MIR3_8\12KIDs mono on 3800\TD_Power"
path_off  = r"D:\Data\LT218Chip1_BF_20221103_MIR3_8\12KIDs mono off\TD_Power"
path_dark = r"D:\Data\LT218Chip1_BF_20230208_dark\TD_Power"
freqs, sxx = get_noise_psd(path_on, kid, pread, pw, filetype, nr_req_files, filter, lifetime, mph, mpp)
_, pxx = get_noise_psd(path_on, kid, pread, pw, filetype, nr_req_files, None, lifetime, mph, mpp)
_, nxx = get_noise_psd(path_off, kid, pread, pw, filetype, nr_req_files, filter, lifetime, mph, mpp)
_, dxx = get_noise_psd(path_dark, kid, pread, pw, filetype, nr_req_files, filter, lifetime, mph, mpp)
ax = axes['a']
ax.semilogx(freqs[1:],  10*np.log10(dxx[1:]), label='dark BF', c='k', lw=width)
ax.semilogx(freqs[1:],  10*np.log10(nxx[1:]), label='3.8 $\mu$m, lamp off', c='b', lw=width, ls='--')
ax.semilogx(freqs[1:],  10*np.log10(sxx[1:]), label='3.8 $\mu$m, lamp on', c='b', lw=width, ls='-')
# ax.semilogx(freqs[1:],  10*np.log10(pxx[1:]), label='3.8 $\mu$m, lamp on, with pulses', c='b', lw=width)
ax.set_ylabel(ylabel_psd)
ax.set_xlabel(xlabel_psd)
ax.set_ylim(ylim_psd)
ax.set_xlim(xlim_psd)

path_on  = r"D:\Data\LT218Chip1_BF_20221025_MIR8_5\12KIDs mono off long\TD_Power"
path_off  = r"D:\Data\LT218Chip1_BF_20221025_MIR8_5\12KIDs LN2 load - long\TD_Power"
freqs, pxx = get_noise_psd(path_off, kid, pread, pw, filetype, nr_req_files, None, lifetime, mph, mpp)
freqs, nxx = get_noise_psd(path_off, kid, pread, pw, filetype, nr_req_files, filter, lifetime, mph, mpp)
ax = axes['a']
# ax.semilogx(freqs[1:],  10*np.log10(sxx[1:]/Q_signal**2), label='on', c='o')
# ax.semilogx(freqs[1:],  10*np.log10(pxx[1:]), label='8.5 $\mu$m, lamp off, with pulses', c='y', lw=width)
ax.semilogx(freqs[1:],  10*np.log10(nxx[1:]), label='8.5 $\mu$m, lamp off', c='y', lw=width, ls='-')
ax.set_ylabel(ylabel_psd)
# ax.set_ylim(ylim_psd)


path_on  = r"D:\Data\LT218Chip1_BF_20240116_MIR18_5\12KIDs_185um_BB160K\TD_Power"
path_off  = r"D:\Data\LT218Chip1_BF_20240116_MIR18_5\12KIDs_185um_BBoff\TD_Power"
freqs, sxx = get_noise_psd(path_on, kid, pread, pw, filetype, nr_req_files, filter, lifetime, mph, mpp)
_, pxx = get_noise_psd(path_on, kid, pread, pw, filetype, nr_req_files, None, lifetime, mph, mpp)
_, nxx = get_noise_psd(path_off, kid, pread, pw, filetype, nr_req_files, filter, lifetime, mph, mpp)
ax = axes['a']
ax.semilogx(freqs[1:],  10*np.log10(nxx[1:]), label='18.5 $\mu$m, $T_{bb}=3\ K$', c='o', ls='--', lw=width)
ax.semilogx(freqs[1:],  10*np.log10(sxx[1:]), label='18.5 $\mu$m, $T_{bb}=160\ K$', c='o', lw=width, ls='-')
# ax.semilogx(freqs[1:],  10*np.log10(pxx[1:]), label='18.5 $\mu$m, $T_{bb}=160\ K$, with pulses', c='o', lw=width)
ax.set_ylabel(ylabel_psd)
# ax.set_ylim(ylim_psd)
ax.legend(loc='upper left', ncols=3, mode="expand", borderaxespad=0., frameon=False, handlelength=1)



ax = axes['b']
# path_on  = r"D:\Data\LT218Chip1_ADR_20240506_MIR24\12KIDs_3Pread_TD40s_1MHz_BB24K\TD_Power"
path_on = r"D:\Data\LT218Chip1_ADR_20240506_MIR24\12KIDs_3Pread_TD80s_50KHz_BB3-28K"
path_off  = path_on
path_s21 = path_on + '\\Noise_TBB'
s21 = 'KID5_113dBm_Tchip0.13_S21dB'
df = S21.loop_over_S21_files(path_s21, s21)
print(df)
# path_off  = r"D:\Data\LT218Chip1_ADR_20240506_MIR24\12KIDs_3Pread_TD40s_1MHz_BB3K\TD_Power"
# path_dark = r"D:\Data\LT218Chip1_ADR_20240605_MIR24_dark\12KIDs_3Pread_100s_1MHz\TD_Power"
path_dark = r"D:\Data\LT218Chip1_ADR_20240605_MIR24_dark\3KIDs_1Pread_100s_50kHz\TD_Power"
kid = 'KID5_113dBm_Tchip0.13_TDmed_TmK24016'
freqs, sxx = get_noise_psd(path_on, kid, pread, pw, filetype, nr_req_files, filter, lifetime, mph, mpp)
_, pxx = get_noise_psd(path_on, kid, pread, pw, filetype, nr_req_files, None, lifetime, mph, mpp)
Qs = df['Ql'].values[6]
kid = 'KID5_113dBm_Tchip0.13_TDmed_TmK3255'
_, nxx = get_noise_psd(path_off, kid, pread, pw, filetype, nr_req_files, filter, lifetime, mph, mpp)
Qn = df['Ql'].values[0]
# noise, Q_noise = get_filtered_response(path_off, kid, pread, nr_req_files, filter, lifetime, file='KID5_109dBm_Tchip0.13_TDmed_TmK3243')
path_s21_dark = r'D:\Data\LT218Chip1_ADR_20240605_MIR24_dark\3KIDs_1Pread_100s_50kHz\FFT\Power'
s21 = 'KID5_112dBm__S21dB'
df = S21.loop_over_S21_files(path_s21_dark, s21)
print(df)
kid = 'KID5_111dBm__TDmed_TmK130'
_, dxx = get_noise_psd(path_dark, kid, pread, pw, filetype, nr_req_files, filter, lifetime, mph, mpp)
Qd = df['Ql'].values[0]
# ax = axes['a']
# ax.semilogx(freqs[1:],  10*np.log10(nxx[1:]/Q_noise**2), label='25 $\mu$m, $T_{bb}=3\ K$', c='p')
ax.semilogx(freqs[1:],  10*np.log10(dxx[1:]), label='dark ADR', c='k', lw=width, ls='--')
ax.semilogx(freqs[1:],  10*np.log10(nxx[1:]), label='25 $\mu$m, $T_{bb}=3\ K$', c='p', ls='--')
ax.semilogx(freqs[1:],  10*np.log10(sxx[1:]), label='25 $\mu$m, $T_{bb}=24\ K$', c='p', lw=width, ls='-')
# ax.semilogx(freqs[1:],  10*np.log10(pxx[1:]), label='25 $\mu$m, $T_{bb}=24\ K$ with pulses', c='p', lw=width)
# ax.set_ylim(ylim_psd)
# ax.set_ylabel(ylabel_psd)
ax.set_xlabel(xlabel_psd)
# ax.legend(bbox_to_anchor=(0., 1, 1., .102), loc='lower left', ncols=2, mode="expand", borderaxespad=0.)
handles, labels = axes['a'].get_legend_handles_labels()
handles_b, labels_b = ax.get_legend_handles_labels()
handles.extend(handles_b)
labels.extend(labels_b)
ax.legend(loc='upper left', ncols=3, mode="expand", borderaxespad=0., frameon=False, handlelength=2)
# fig.legend(handles, labels, ncol=5, loc='lower left', bbox_to_anchor=(0.075, .95))
# plt.savefig('figures/noise_psds2.pdf')
plt.show()