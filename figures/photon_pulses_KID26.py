import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import fftconvolve, welch
import matplotlib as mpl
import os
import re
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
import figures.functions as f
plt.style.use('figures/matplotlibrc')


def get_filtered_response(dir, kid, pread, nr_req_files, filter, lifetime, coord='circle', response='phase', temp='', offset=0):
    dir = dir.replace("\\", '/')
    if type(kid) is int:
        pulse_files, _ = f.get_files(dir, kid, pread, type='vis')
    elif type(kid) is str:
        pulse_files = [dir + '/' + kid + '.bin']
    nr_files = len(pulse_files)

    if nr_req_files >= nr_files:
        nr_req_files = nr_files
    amp, phase, _ = f.get_data(pulse_files[offset:offset+nr_req_files])
    signal = f.coord_transformation(phase, amp, coord=coord, response=response)
    if filter:
        window = f.get_window(filter, lifetime)
        filtered_signal = fftconvolve(signal, window, mode='valid')
        return signal, filtered_signal
    else:
        return signal, signal

def noise_psd(signal):
    return welch(signal, fs=int(1e6), window='hamming', nperseg=1000)


Q_bf_dark =  32554.641039207723
Q_38um =  26155.015645914053
Q_85um =  25888.428331481795
Q_185um_3K =  53162.040560797424
Q_185um_160K =  52979.36333999958
Q_25um_3K =  63829.27252722262
Q_25um_24K =  63806.23296600575

fig, axes = plt.subplot_mosaic('abcde', figsize=(18.5/2.54,6/2.54), sharex=True, sharey=True, constrained_layout=True)
ylim = [-.75, 1.5]
width = .1
a = 1
nr_req_files = 1
xlabel_response = 'Time [s]'
ylabel_response = 'Smoothed $\\theta$ [a.u.]'
std_ls = '-'
std_c = 'r'
std_lw = 1
step = .25
kid = 24
lifetime = 50
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
ax.plot([], [], label='Closed setup', lw=1, c='k')
ax.set_xlim([0, nr_req_files])
ax.set_ylim(ylim)
ax.set_xlabel(xlabel_response)
ax.set_ylabel(ylabel_response)
ax.set_title('Dark')
ax.set_yticks(yticks)

ax = axes['b']
ax.plot(time_filtered, filtered_noise/(Q_38um/Q_bf_dark)-0.5, lw=width, c='gray', alpha=a)
ax.plot([], [], label='Lamp off', lw=1, c='gray')
ax.plot(time_filtered, filtered_signal/(Q_38um/Q_bf_dark), lw=width, c='b', alpha=a)
ax.plot([], [], label='Lamp on', lw=1, c='b')
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
# ax.plot(time_filtered, filtered_noise/(Q_85um/Q_bf_dark)-0.5, lw=width, c='gray', alpha=a, label='LN2 load')
ax.plot(time_filtered, filtered_signal/(Q_85um/Q_bf_dark), lw=width, c='y', alpha=a, label=' Lamp off')
ax.plot([], [], label=' Lamp off', lw=1, c='y')
ax.set_xlim([0, nr_req_files])
ax.set_ylim(ylim)
ax.set_xlabel(xlabel_response)
ax.set_title('$8.5$ $\mu m$')


kid = 25
pread = 119
path_on  = r"D:\Data\LT218Chip1_BF_20240116_MIR18_5\12KIDs_185um_BB160K\TD_Power"
path_off  = r"D:\Data\LT218Chip1_BF_20240116_MIR18_5\12KIDs_185um_BB3K\TD_Power"
Toff = int(re.findall(r'BB(\d+)K', path_off)[0])
Ton = int(re.findall(r'BB(\d+)K', path_on)[0])
signal, filtered_signal = get_filtered_response(path_on, kid, pread, nr_req_files, filter, lifetime, offset=6)
noise, filtered_noise = get_filtered_response(path_off, kid, pread, nr_req_files, filter, lifetime, offset=6)
time_filtered = np.linspace(0, nr_req_files, len(filtered_signal))
time = np.linspace(0, nr_req_files, len(signal))
ax = axes['d']
ax.plot(time_filtered, filtered_noise/(Q_185um_3K/Q_bf_dark)-0.5, lw=width, c='gray', alpha=a)
ax.plot([], [], label='$T_{bb}=%d\ K$' % Toff, lw=1, c='gray')
ax.plot(time_filtered, filtered_signal/(Q_185um_160K/Q_bf_dark), lw=width, c='o', alpha=a)
ax.plot([], [], label='$T_{bb}=%d\ K$' % Ton, lw=1, c='o')
ax.set_xlim([0, nr_req_files])
ax.set_ylim(ylim)
ax.set_xlabel(xlabel_response)
ax.set_title('$18.5$ $\mu m$')

kid = 24
pread = 115
path_on  = r"D:\Data\LT218Chip1_ADR_20240506_MIR24\12KIDs_3Pread_TD40s_1MHz_BB24K\TD_Power"
path_off  = r"D:\Data\LT218Chip1_ADR_20240506_MIR24\12KIDs_3Pread_TD40s_1MHz_BB3K\TD_Power"
filter = 'exp'
lifetime = 50
signal, filtered_signal = get_filtered_response(path_on, kid, pread, nr_req_files, filter, lifetime)
noise, filtered_noise = get_filtered_response(path_off, kid, pread, nr_req_files, filter, lifetime)

time_filtered = np.linspace(0, nr_req_files, len(filtered_signal))
time = np.linspace(0, nr_req_files, len(signal))
ax = axes['e']
ax.plot(time_filtered, filtered_noise/(Q_25um_3K/Q_bf_dark)-0.5, lw=width, c='gray')
Toff = int(re.findall(r'BB(\d+)K', path_off)[0])
Ton = int(re.findall(r'BB(\d+)K', path_on)[0])
ax.plot([], [], label='$T_{bb}=%d\ K$' % Toff, lw=1, c='gray')
ax.plot(time_filtered, filtered_signal/(Q_25um_24K/Q_bf_dark), lw=width, c='p')
ax.plot([], [], label='$T_{bb}=%d\ K$' % Ton, lw=1, c='p')
ax.set_xlim([0, nr_req_files])
ax.set_ylim(ylim)
ax.set_xlabel(xlabel_response)
ax.set_title('$25$ $\mu m$')

for i, ax in enumerate(axes.values()):
    ncols = 1
    ax.legend(loc='upper center', ncols=ncols, mode="expand", borderaxespad=0., frameon=False, handlelength=1.5)
# plt.savefig('figures/photon pulses KID26.pdf')
plt.show()