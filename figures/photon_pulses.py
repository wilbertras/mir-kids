import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import fftconvolve, welch
import matplotlib as mpl
import os
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
import functions as f
import matplotlibcolors


def get_filtered_response(dir, kid, pread, nr_req_files, filter, lifetime, coord='circle', response='phase'):
    dir = dir.replace("\\", '/')
    pulse_files, _ = f.get_files(dir, kid, pread, type='vis')
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
        return signal

def noise_psd(signal):
    return welch(signal, fs=int(1e6), window='hamming', nperseg=1000)

fig, axes = plt.subplot_mosaic('abcd', figsize=(12,3), sharex=True, sharey=True, constrained_layout=True)
ylim_response = [-.2, .8]
ylim_psd = [-180, -150]
width = .2
xlabel_response = 'Time [s]'
ylabel_response = 'Phase [rad]'

path_on  = r"D:\Data\LT218Chip1_BF_20221103_MIR3_8\12KIDs mono on 3800\TD_Power"
path_off  = r"D:\Data\LT218Chip1_BF_20221103_MIR3_8\12KIDs mono off\TD_Power"
path_dark = r"D:\Data\LT218Chip1_BF_20230208_dark\TD_Power"
kid = 5
pread = 113
nr_req_files = 1
filter = 'exp'
lifetime = 250
signal, filtered_signal = get_filtered_response(path_on, kid, pread, nr_req_files, filter, lifetime)
noise, filtered_noise = get_filtered_response(path_off, kid, pread, nr_req_files, filter, lifetime)
dark, filtered_dark = get_filtered_response(path_dark, kid, pread, nr_req_files, filter, lifetime)
time = np.linspace(0, nr_req_files, len(filtered_signal))
ax = axes['a']
ax.plot(time, filtered_dark, label='dark', lw=width)
ax.set_xlim([0, nr_req_files])
ax.set_ylim(ylim_response)
ax.set_xlabel(xlabel_response)
ax.set_ylabel(ylabel_response)
ax.legend()
ax = axes['b']
ax.plot(time, filtered_signal, label='3.8 $\mu$m, monochromator on', lw=width)
ax.set_xlim([0, nr_req_files])
ax.set_ylim(ylim_response)
ax.set_xlabel(xlabel_response)
ax.legend()

path_on  = r"D:\Data\LT218Chip1_BF_20221025_MIR8_5\12KIDs mono off long\TD_Power"
path_off  = r"D:\Data\LT218Chip1_BF_20221025_MIR8_5\12KIDs LN2 load - long\TD_Power"
kid = 5
pread = 113
nr_req_files = 1
filter = 'exp'
lifetime = 250
signal, filtered_signal = get_filtered_response(path_on, kid, pread, nr_req_files, filter, lifetime)
noise, filtered_noise = get_filtered_response(path_off, kid, pread, nr_req_files, filter, lifetime)
time = np.linspace(0, nr_req_files, len(filtered_noise))
ax = axes['c']
ax.plot(time, filtered_signal, label='8.5 $\mu$m, monochromator off', lw=width)
ax.set_xlim([0, nr_req_files])
ax.set_ylim(ylim_response)
ax.set_xlabel(xlabel_response)
ax.legend()

path_on  = r"D:\Data\LT218Chip1_BF_20240116_MIR18_5\12KIDs_185um_BB160K\TD_Power"
path_off  = r"D:\Data\LT218Chip1_BF_20240116_MIR18_5\12KIDs_185um_BBoff\TD_Power"
kid = 5
pread = 113
nr_req_files = 1
filter = 'exp'
lifetime = 250
signal, filtered_signal = get_filtered_response(path_on, kid, pread, nr_req_files, filter, lifetime)
noise, filtered_noise = get_filtered_response(path_off, kid, pread, nr_req_files, filter, lifetime)
time = np.linspace(0, nr_req_files, len(filtered_noise))
ax = axes['d']
ax.plot(time, filtered_signal, label='18.5 $\mu$m, $T_{bb}=160\ K$', lw=width)
ax.set_xlim([0, nr_req_files])
ax.set_ylim(ylim_response)
ax.set_xlabel(xlabel_response)
ax.legend()

plt.savefig('figures/photon_pulses.pdf')
plt.show()