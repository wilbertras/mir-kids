import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import fftconvolve, welch
import matplotlib as mpl
import os
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
import functions as f
import matplotlibcolors


def get_filtered_response(dir, kid, pread, nr_req_files, filter, lifetime, file=None, coord='circle', response='phase'):
    dir = dir.replace("\\", '/')
    if not file:
        pulse_files, info_files = f.get_files(dir, kid, pread, type='med')
    else:
        pulse_files = [dir + '/' + file + '.bin']
        info_files = [dir + '/' + file + '_info.dat']
    f0, Q, Qc, Qi, S21_min, dt, T =  f.get_info(info_files[0])
    rc = Q/(2*Qc)
    nr_files = len(pulse_files)
    if nr_req_files >= nr_files:
        nr_req_files = nr_files
    amp, phase, _ = f.get_data(pulse_files[:nr_req_files])
    signal = f.coord_transformation(phase, amp, coord=coord, response=response)
    if filter:
        window = f.get_window(filter, lifetime)
        filtered_signal = fftconvolve(signal, window, mode='valid')
        return signal, filtered_signal
    else:
        return signal, rc

def noise_psd(signal, sf=1e6):
    return welch(signal, fs=int(sf), window='hamming')

fig, axes = plt.subplot_mosaic('ab', figsize=(6,4), sharex=True, sharey=False, constrained_layout=True)
ylabel_psd = '$S_{xx}/Q^2$ [dBc/Hz]'
xlabel_psd = 'Frequency [Hz]'
ylim_psd = [-160, -130]
xlim_psd = [20, 2.5e3]


path_on  = r"D:\Data\LT218Chip1_BF_20221103_MIR3_8\12KIDs mono on 3800\TD_Power"
path_off  = r"D:\Data\LT218Chip1_BF_20221103_MIR3_8\12KIDs mono off\TD_Power"
path_dark = r"D:\Data\LT218Chip1_BF_20230208_dark\TD_Power"
kid = 5
pread = 113
nr_req_files = 10
filter = False
lifetime = 250
# signal = get_filtered_response(path_on, kid, pread, nr_req_files, filter, lifetime)
noise, rc_noise = get_filtered_response(path_off, kid, pread, nr_req_files, filter, lifetime)
dark, rc_dark = get_filtered_response(path_dark, kid, pread, nr_req_files, filter, lifetime)
ax = axes['a']
# freqs, sxx = noise_psd(signal)
freqs, nxx = noise_psd(noise, 5e3)
_, dxx = noise_psd(dark, 5e3)
# ax.semilogx(freqs[1:],  10*np.log10(sxx[1:]/Q_signal**2), label='3.8 $\mu$m, source on', c='b')
ax.semilogx(freqs[1:],  10*np.log10(dxx[1:]/rc_dark**2), label='dark BF', c='k')
ax.semilogx(freqs[1:],  10*np.log10(nxx[1:]/rc_noise**2), label='3.8 $\mu$m, monochromator off', c='b')
ax.set_ylabel(ylabel_psd)
ax.set_xlabel(xlabel_psd)
# ax.set_ylim(ylim_psd)
ax.set_xlim(xlim_psd)

path_on  = r"D:\Data\LT218Chip1_BF_20221025_MIR8_5\12KIDs mono off long\TD_Power"
path_off  = r"D:\Data\LT218Chip1_BF_20221025_MIR8_5\12KIDs LN2 load - long\TD_Power"
kid = 5
pread = 113
nr_req_files = 10
filter = False
lifetime = 250
# signal = get_filtered_response(path_on, kid, pread, nr_req_files, filter, lifetime)
noise, rc_noise = get_filtered_response(path_off, kid, pread, nr_req_files, filter, lifetime)
ax = axes['a']
# freqs, sxx = noise_psd(signal)
_, nxx = noise_psd(noise, 5e3)
# ax.semilogx(freqs[1:],  10*np.log10(sxx[1:]/Q_signal**2), label='on', c='o')
ax.semilogx(freqs[1:],  10*np.log10(nxx[1:]/rc_noise**2), label='8.5 $\mu$m, monochromator off', c='y')
ax.set_ylabel(ylabel_psd)
# ax.set_ylim(ylim_psd)


# path_on  = r"D:\Data\LT218Chip1_BF_20240116_MIR18_5\12KIDs_185um_BB160K\TD_Power"
# path_off  = r"D:\Data\LT218Chip1_BF_20240116_MIR18_5\12KIDs_185um_BBoff\TD_Power"
# kid = 5
# pread = 113
# nr_req_files = 10
# filter = False
# lifetime = 250
# # signal = get_filtered_response(path_on, kid, pread, nr_req_files, filter, lifetime)
# noise, Q_noise = get_filtered_response(path_off, kid, pread, nr_req_files, filter, lifetime)
# ax = axes['a']
# # freqs, sxx = noise_psd(signal)
# _, nxx = noise_psd(noise, 5e3)
# # ax.semilogx(freqs[1:],  10*np.log10(sxx[1:]/Q_signal**2), label='on')
# ax.semilogx(freqs[1:],  10*np.log10(nxx[1:]), label='18.5 $\mu$m, $T_{bb}=3\ K$', c='o')
# ax.set_ylabel(ylabel_psd)
# # ax.set_ylim(ylim_psd)

# path_on  = r"D:\Data\LT218Chip1_ADR_20240506_MIR24\12KIDs_3Pread_TD40s_1MHz_BB24K\TD_Power"
ax = axes['b']
path_off  = r"D:\Data\LT218Chip1_ADR_20240506_MIR24\12KIDs_3Pread_TD80s_50KHz_BB3-28K"
path_dark = r"D:\Data\LT218Chip1_ADR_20240605_MIR24_dark\12KIDs_3Pread_100s_1MHz\TD_Power"
path_dark = r"D:\Data\LT218Chip1_ADR_20240605_MIR24_dark\3KIDs_1Pread_100s_50kHz\TD_Power"
kid = 5
pread = 111
nr_req_files = 10
filter = False
lifetime = 250
noise, Q_noise = get_filtered_response(path_off, kid, pread, nr_req_files, filter, lifetime, file='KID5_113dBm_Tchip0.13_TDmed_TmK3255')
# noise, Q_noise = get_filtered_response(path_off, kid, pread, nr_req_files, filter, lifetime, file='KID5_109dBm_Tchip0.13_TDmed_TmK3243')
dark, Q_dark = get_filtered_response(path_dark, kid, pread, nr_req_files, filter, lifetime)
# ax = axes['a']
freqs, nxx = noise_psd(noise, 5e3)
_, dxx = noise_psd(dark, 5e3)
# ax.semilogx(freqs[1:],  10*np.log10(nxx[1:]/Q_noise**2), label='25 $\mu$m, $T_{bb}=3\ K$', c='p')
# ax.semilogx(freqs[1:],  10*np.log10(dxx[1:]/Q_dark**2), label='dark ADR', c='g')
ax.semilogx(freqs[1:],  10*np.log10(nxx[1:]*Q_noise**2), label='25 $\mu$m, $T_{bb}=3\ K$', c='p')
ax.semilogx(freqs[1:],  10*np.log10(dxx[1:]*Q_dark**2), label='dark ADR', c='k')
ax.set_ylabel(ylabel_psd)
ax.set_xlabel(xlabel_psd)
# fig.savefig('figures/noise_psds.pdf')
ax.legend()
plt.show()