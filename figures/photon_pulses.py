import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import fftconvolve, welch
import matplotlib as mpl
import os
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
import figures.functions as f
plt.style.use('figures/matplotlibrc')


def get_filtered_response(dir, kid, pread, nr_req_files, filter, lifetime, coord='circle', response='phase', temp=''):
    dir = dir.replace("\\", '/')
    if type(kid) is int:
        pulse_files, _ = f.get_files(dir, kid, pread, type='vis')
    elif type(kid) is str:
        pulse_files = [dir + '/' + kid + '.bin']
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
        return signal, signal

def noise_psd(signal):
    return welch(signal, fs=int(1e6), window='hamming', nperseg=1000)



fig, axes = plt.subplot_mosaic('abcde', figsize=(18.5/2.54,6/2.54), sharex=True, sharey=True, constrained_layout=True)
ylim = [-.5, 1]
width = .1
a = 1
nr_req_files = 3
xlabel_response = 'Time [s]'
ylabel_response = 'Smoothed $\\theta$ [a.u.]'
std_ls = '-'
std_c = 'r'
std_lw = 1
step = .25
kid = 5
lifetime = 250
yticks = np.arange(ylim[0], ylim[1]+step, step)
ncols = 2
filter = 'exp'
pread = 113

path_on  = r"D:\Data\LT218Chip1_BF_20221103_MIR3_8\12KIDs mono on 3800\TD_Power"
path_off  = r"D:\Data\LT218Chip1_BF_20221103_MIR3_8\12KIDs mono off\TD_Power"
path_dark = r"D:\Data\LT218Chip1_BF_20230208_dark\TD_Power"
signal, filtered_signal = get_filtered_response(path_on, kid, pread, nr_req_files, filter, lifetime)
noise, filtered_noise = get_filtered_response(path_off, kid, pread, nr_req_files, filter, lifetime)
dark, filtered_dark = get_filtered_response(path_dark, kid, pread, nr_req_files, filter, lifetime)
time_filtered = np.linspace(0, nr_req_files, len(filtered_signal))
time = np.linspace(0, nr_req_files, len(signal))
ax = axes['a']
ax.plot(time_filtered, filtered_dark, lw=width, c='k', alpha=a)
ax.axhline(3*f.get_sigma(filtered_dark, window=[]), ls=std_ls, c=std_c, lw=std_lw)
ax.plot([], [], label='Closed setup', lw=1, c='k')
ax.plot([], [], label='3$\sigma$', lw=1, c='r')
ax.set_xlim([0, nr_req_files])
ax.set_ylim(ylim)
ax.set_xlabel(xlabel_response)
ax.set_ylabel(ylabel_response)
ax.set_title('Dark')
ax.set_yticks(yticks)
# ax.legend(loc='upper center', ncol=ncols, frameon=False)

ax = axes['b']
# ax.plot(time, signal, lw=width, alpha=.1)
ax.plot(time_filtered, filtered_noise-.3, lw=width, c='gray', alpha=a)
ax.plot([], [], label='Lamp off', lw=1, c='gray')
ax.plot(time_filtered, filtered_signal, lw=width, c='b', alpha=a)
ax.plot([], [], label='Lamp on', lw=1, c='b')
ax.axhline(3*f.get_sigma(filtered_signal, window=[]), ls=std_ls, c=std_c, lw=std_lw)
ax.plot([], [], label='3$\sigma$', lw=1, c='r')
ax.set_xlim([0, nr_req_files])
ax.set_ylim(ylim)
ax.set_xlabel(xlabel_response)
ax.set_title('$3.8$ $\mu m$')

path_on  = r"D:\Data\LT218Chip1_BF_20221025_MIR8_5\12KIDs mono off long\TD_Power"
path_off  = r"D:\Data\LT218Chip1_BF_20221025_MIR8_5\12KIDs LN2 load - long\TD_Power"
signal, filtered_signal = get_filtered_response(path_on, kid, pread, nr_req_files, filter, lifetime)
noise, filtered_noise = get_filtered_response(path_off, kid, pread, nr_req_files, filter, lifetime)
time_filtered = np.linspace(0, nr_req_files, len(filtered_signal))
time = np.linspace(0, nr_req_files, len(signal))
ax = axes['c']
# ax.plot(time, signal, lw=width, alpha=.1, c='b')
ax.plot(time_filtered, filtered_signal, lw=width, c='y', alpha=a)
ax.plot([], [], label=' Lamp off', lw=1, c='y')
ax.axhline(3*f.get_sigma(filtered_signal, window=[]), ls=std_ls, c=std_c, lw=std_lw)
ax.plot([], [], label='3$\sigma$', lw=1, c='r')
ax.set_xlim([0, nr_req_files])
ax.set_ylim(ylim)
ax.set_xlabel(xlabel_response)
ax.set_title('$8.5$ $\mu m$')


path_on  = r"D:\Data\LT218Chip1_BF_20240116_MIR18_5\12KIDs_185um_BB160K\TD_Power"
path_off  = r"D:\Data\LT218Chip1_BF_20240116_MIR18_5\12KIDs_185um_BBoff\TD_Power"
signal, filtered_signal = get_filtered_response(path_on, kid, pread, nr_req_files, filter, lifetime)
noise, filtered_noise = get_filtered_response(path_off, kid, pread, nr_req_files, filter, lifetime)
time_filtered = np.linspace(0, nr_req_files, len(filtered_signal))
time = np.linspace(0, nr_req_files, len(signal))
ax = axes['d']
ax.plot(time_filtered, filtered_noise-0.3, lw=width, c='gray', alpha=a)
ax.plot([], [], label='$T_{bb}=3\ K$', lw=1, c='gray')
ax.plot(time_filtered, filtered_signal, lw=width, c='o', alpha=a)
ax.plot([], [], label='$T_{bb}=160\ K$', lw=1, c='o')
ax.axhline(3*f.get_sigma(filtered_signal, window=[]), ls=std_ls, c=std_c, lw=std_lw)
ax.plot([], [], label='3$\sigma$', lw=1, c='r')
ax.set_xlim([0, nr_req_files])
ax.set_ylim(ylim)
ax.set_xlabel(xlabel_response)
ax.set_title('$18.5$ $\mu m$')

# path_on  = r"D:\Data\LT218Chip1_ADR_20240506_MIR24\12KIDs_3Pread_TD40s_1MHz_BB24K\TD_Power"
# path_off  = r"D:\Data\LT218Chip1_ADR_20240506_MIR24\12KIDs_3Pread_TD40s_1MHz_BB3K\TD_Power"
# pread = 109
# filter = 'exp'
# lifetime = 250
# signal, filtered_signal = get_filtered_response(path_on, kid, pread, nr_req_files, filter, lifetime)
# noise, filtered_noise = get_filtered_response(path_off, kid, pread, nr_req_files, filter, lifetime)

path_on  = r"D:\Data\LT218Chip1_ADR_20240506_MIR24\12KIDs_3Pread_TD80s_50KHz_BB3-28K"
path_off  = path_on
filter = 'exp'
lifetime = 12.5
kid = 'KID5_113dBm_Tchip0.13_TDmed_TmK24016'
signal, filtered_signal = get_filtered_response(path_on, kid, pread, nr_req_files, filter, lifetime)
signal = signal[:int(nr_req_files/20e-6)]
filtered_signal = filtered_signal[:int(nr_req_files/20e-6)]
kid = 'KID5_113dBm_Tchip0.13_TDmed_TmK3255'
noise, filtered_noise = get_filtered_response(path_off, kid, pread, nr_req_files, filter, lifetime)
noise = noise[:int(nr_req_files/20e-6)]
filtered_noise = filtered_noise[:int(nr_req_files/20e-6)]


time_filtered = np.linspace(0, nr_req_files, len(filtered_signal))
time = np.linspace(0, nr_req_files, len(signal))
ax = axes['e']
ax.plot(time_filtered, filtered_noise-0.3, lw=width, c='gray')
ax.plot([], [], label='$T_{bb}=3\ K$', lw=1, c='gray')
ax.plot(time_filtered, filtered_signal, lw=width, c='p')
ax.plot([], [], label='$T_{bb}=24\ K$', lw=1, c='p')
ax.axhline(3*f.get_sigma(filtered_signal, window=[]), lw=1, c='r')
ax.plot([], [], label='3$\sigma$', lw=1, c='r')

ax.set_xlim([0, nr_req_files])
ax.set_ylim(ylim)
ax.set_xlabel(xlabel_response)
ax.set_title('$25$ $\mu m$')

for i, ax in enumerate(axes.values()):
    if i == 0:
        ncols = 1    
    else:
        ncols = 2
    ax.legend(loc='upper center', ncols=ncols, mode="expand", borderaxespad=0., frameon=False, handlelength=1.5)
plt.savefig('figures/photon pulses.pdf')
plt.show()