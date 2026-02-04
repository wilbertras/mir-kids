import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import fftconvolve, welch
import matplotlib as mpl
import pickle
import os
import re
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
import figures.functions as ft
plt.style.use('figures/matplotlibrc')


def get_filtered_response(dir, kid, pread, nr_req_files, filter, lifetime, coord='circle', response='phase', temp='', offset=0):
    dir = dir.replace("\\", '/')
    if type(kid) is int:
        pulse_files, _ = ft.get_files(dir, kid, pread, type='vis')
    elif type(kid) is str:
        pulse_files = [dir + '/' + kid + '.bin']
    nr_files = len(pulse_files)

    if nr_req_files >= nr_files:
        nr_req_files = nr_files
    amp, phase, _ = ft.get_data(pulse_files[offset:offset+nr_req_files])
    signal = ft.coord_transformation(phase, amp, coord=coord, response=response)
    if filter:
        window = ft.get_window(filter, lifetime)
        filtered_signal = fftconvolve(signal, window, mode='valid')
        return signal, filtered_signal
    else:
        return signal, signal

def noise_psd(signal):
    return welch(signal, fs=int(1e6), window='hamming', nperseg=1000)

with open(r'KID26.pkl', 'rb') as f:
    kid_dict = pickle.load(f)

name = 'KID26'
lifetime = 50
filter = 'exp'
nr_req_files = 1
offset_off = 0.5

fig, axes = plt.subplot_mosaic('abcde', figsize=(18.5/2.54,6/2.54), sharex=True, sharey=True, constrained_layout=True)
ylabel_response = 'Smoothed $\\theta$ [rad]'
xlabel_response = 'Time [s]'
ylim = [-.75, 1.5]
width = .1
std_ls = '-'
std_c = 'r'
std_lw = 1
step = .25
ncols = 2
a = 1
yticks = np.arange(ylim[0], ylim[1]+step, step)

wl = 'BF dark'
kid = kid_dict[name][wl]['kid']
pread = kid_dict[name][wl]['pread']
path = kid_dict[name][wl]['dir']
Q_bf_dark = kid_dict[name][wl]['Q']
dark, filtered_dark = get_filtered_response(path, kid, pread, nr_req_files, filter, lifetime)

ax = axes['a']
time_filtered = np.linspace(0, nr_req_files, len(filtered_dark))
ax.plot(time_filtered, filtered_dark, lw=width, c='k', alpha=a)
ax.plot([], [], label='Closed setup', lw=1, c='k')
ax.set_xlabel(xlabel_response)
ax.set_ylabel(ylabel_response)
ax.set_xlim([0, nr_req_files])
ax.set_yticks(yticks)
ax.set_ylim(ylim)
ax.set_title('Dark')


wl = '3.8um off'
kid = kid_dict[name][wl]['kid']
pread = kid_dict[name][wl]['pread']
path = kid_dict[name][wl]['dir']
Q = kid_dict[name][wl]['Q']
noise, filtered_noise = get_filtered_response(path, kid, pread, nr_req_files, filter, lifetime)
ax = axes['b']
ax.plot(time_filtered, filtered_noise/(Q/Q_bf_dark)-offset_off, lw=width, c='gray', alpha=a)
ax.plot([], [], label='$T_\mathrm{lab}=293\ K$', lw=1, c='gray')

wl = '3.8um'
kid = kid_dict[name][wl]['kid']
pread = kid_dict[name][wl]['pread']
path = kid_dict[name][wl]['dir']
Q = kid_dict[name][wl]['Q']
signal, filtered_signal = get_filtered_response(path, kid, pread, nr_req_files, filter, lifetime)
ax.plot(time_filtered, filtered_signal/(Q/Q_bf_dark), lw=width, c='b', alpha=a)
ax.plot([], [], label='$T_\mathrm{lab}=293\ K$, QTH', lw=1, c='b')
ax.set_xlim([0, nr_req_files])
ax.set_ylim(ylim)
ax.set_xlabel(xlabel_response)
ax.set_title('$3.8$ $\mu m$')


wl = '8.5um'
kid = kid_dict[name][wl]['kid']
pread = kid_dict[name][wl]['pread']
path = kid_dict[name][wl]['dir']
Q = kid_dict[name][wl]['Q']
signal, filtered_signal = get_filtered_response(path, kid, pread, nr_req_files, filter, lifetime)
time_filtered = np.linspace(0, nr_req_files, len(filtered_signal))
time = np.linspace(0, nr_req_files, len(signal))
ax = axes['c']
ax.plot(time_filtered, filtered_signal/(Q/Q_bf_dark), lw=width, c='y', alpha=a)
ax.plot([], [], label='$T_\mathrm{lab}=293\ K$', lw=1, c='y')
ax.set_xlim([0, nr_req_files])
ax.set_ylim(ylim)
ax.set_xlabel(xlabel_response)
ax.set_title('$8.5$ $\mu m$')


wl = '18.5um off'
kid = kid_dict[name][wl]['kid']
pread = kid_dict[name][wl]['pread']
path = kid_dict[name][wl]['dir']
Q = kid_dict[name][wl]['Q']
T = int(re.findall(r'BB(\d+)K', path)[0])
noise, filtered_noise = get_filtered_response(path, kid, pread, nr_req_files, filter, lifetime, offset=6)
ax = axes['d']
time_filtered = np.linspace(0, nr_req_files, len(filtered_signal))
ax.plot(time_filtered, filtered_noise/(Q/Q_bf_dark)-offset_off, lw=width, c='gray', alpha=a)
ax.plot([], [], label='$T_{bb}=%d\ K$' % T, lw=1, c='gray')

wl = '18.5um'
kid = kid_dict[name][wl]['kid']
pread = kid_dict[name][wl]['pread']
path = kid_dict[name][wl]['dir']
Q = kid_dict[name][wl]['Q']
T = int(re.findall(r'BB(\d+)K', path)[0])
signal, filtered_signal = get_filtered_response(path, kid, pread, nr_req_files, filter, lifetime, offset=6)
time = np.linspace(0, nr_req_files, len(signal))
ax.plot(time_filtered, filtered_signal/(Q/Q_bf_dark), lw=width, c='o', alpha=a)
ax.plot([], [], label='$T_{bb}=%d\ K$' % T, lw=1, c='o')
ax.set_xlim([0, nr_req_files])
ax.set_ylim(ylim)
ax.set_xlabel(xlabel_response)
ax.set_title('$18.5$ $\mu m$')


wl = '25um off'
kid = kid_dict[name][wl]['kid']
pread = kid_dict[name][wl]['pread']
path = kid_dict[name][wl]['dir']
Q = kid_dict[name][wl]['Q']
T = int(re.findall(r'BB(\d+)K', path)[0])
noise, filtered_noise = get_filtered_response(path, kid, pread, nr_req_files, filter, lifetime)
time_filtered = np.linspace(0, nr_req_files, len(filtered_signal))
ax = axes['e']
ax.plot(time_filtered, filtered_noise/(Q/Q_bf_dark)-offset_off, lw=width, c='gray')
ax.plot([], [], label='$T_{bb}=%d\ K$' % T, lw=1, c='gray')

wl = '25um'
kid = kid_dict[name][wl]['kid']
pread = kid_dict[name][wl]['pread']
path = kid_dict[name][wl]['dir']
Q = kid_dict[name][wl]['Q']
T = int(re.findall(r'BB(\d+)K', path)[0])
signal, filtered_signal = get_filtered_response(path, kid, pread, nr_req_files, filter, lifetime)
ax.plot(time_filtered, filtered_signal/(Q/Q_bf_dark), lw=width, c='p')
ax.plot([], [], label='$T_{bb}=%d\ K$' % T, lw=1, c='p')
ax.set_xlim([0, nr_req_files])
ax.set_ylim(ylim)
ax.set_xlabel(xlabel_response)
ax.set_title('$25$ $\mu m$')

for i, ax in enumerate(axes.values()):
    ncols = 1
    ax.legend(loc='upper center', ncols=ncols, mode="expand", borderaxespad=0., frameon=False, handlelength=1.5)
plt.savefig('figures/photon pulses KID26.pdf')
plt.show()