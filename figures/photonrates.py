import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import find_peaks, peak_widths, fftconvolve
import scipy.constants as sc
from scipy.optimize import curve_fit
import os
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
import figures.functions as f
from main import pulse_analysis
import figures.matplotlibcolors as matplotlibcolors
from filters.filters import load_all_filters


kid = 5
pread = 113
file_type = 'vis'
pw = 1500
pw_offset = 100
filter = 'exp'
lifetime = 250
tqp = [200, 600]
iterate = 0
nr_req_files = 40
mph = np.array([5, 20])
mpp = mph[0] 
dir = r'D:\Data\LT218Chip1_BF_20240116_MIR18_5\KID5_all_temps'
temps = [3,30,50,85,100,135,142,150,160,180]
Eph = sc.h*sc.c/(18.5e-6)

filters = load_all_filters()
wl = filters['wl']*1e6
bp185 = filters['bp185']
znse = filters['znse']
nd1 = filters['nd1']
nd3 = filters['nd3']
ger = filters['ger'] 
bp25 = filters['bp25']
sp_a = filters['sp_a']
sp_b = filters['sp_b']
lp = filters['lp']
si = filters['si']

A = (1e-3)**2
omega = np.pi*(10e-3)**2/(300e-3)**2
tot185 = bp185**5*nd1*nd3*znse**2*A*omega
tot25 = bp25**3*lp*sp_a**3*sp_b*si*A*omega


def count_pulses(dir, kid, pread, nr_req_files, filter, lifetime, mph, mpp, coord='circle', response='phase'):
    dir = dir.replace("\\", '/')
    pulse_files, _ = f.get_files(dir, kid, pread, type='vis')
    nr_files = len(pulse_files)

    if nr_req_files >= nr_files:
        nr_req_files = nr_files
    amp, phase, _ = f.get_data(pulse_files[:nr_req_files])
    signal = f.coord_transformation(phase, amp, coord=coord, response=response)
    if filter:
        window = f.get_window(filter, lifetime)
    std = f.get_sigma(signal, window)
    ph = mph*std
    pp = mpp*std
    locs, _ = f.find_pks(signal, ph[0], pp, window)
    total_pulses = len(locs)
    return total_pulses, nr_req_files


def planck_wl(wl, t):
    return sc.h * sc.c**2 / (wl**5 * (np.exp(sc.h * sc.c / (wl * sc.k * t)) - 1))


def get_index(x, val, round):
    idx = np.argmin(np.absolute(x-val))
    if round=='floor':
        if x[idx] > val:
            idx -= 1
    elif round=='ceil':
        if x[idx] < val:
            idx += 1
    return idx


def int_pow(y, x, x1, x2):
    idx = np.arange(get_index(x, x1*1e-6, 'floor'), get_index(x, x2*1e-6, 'ceil')+1, 1)
    pow = np.trapz(y[idx], x[idx])
    return pow


def get_bp(theta, mph, plot=False):
    locs, _ = find_peaks(theta, height=mph, prominence=mph)
    width_full = peak_widths(theta, locs, rel_height=.99)
    bp = width_full[2:]
    idx = np.arange(np.floor(bp[0]), np.ceil(bp[1]), dtype=int)
    if plot:
        fig, ax = plt.subplots()
        ax.plot(theta)
        ax.axvline(locs[0], c='k')
        ax.hlines(*width_full[1:], color='r')
        for edge in bp:   
            ax.axvline(edge, c='tab:red')
    return idx


def counts_vs_temp():
    Nph = []
    fig, ax = plt.subplot_mosaic('a')

    for temp in temps:
        path = dir + '/' + str(temp) + 'K'
        # pulses = pulse_analysis(path, kid, pread, file_type, 20,2, pw, pw_offset, filter, lifetime, mph, mpp, iterate=False, exclude_dc=True, plot=False, tmax=5, coord='circle', response='phase', fit_tqp=None)
        # print(pulses.shape[0])
        nr_pulses, nr_analyzed_files = count_pulses(path, kid, pread, nr_req_files, filter, lifetime, mph, mpp, coord='circle', response='phase')
        Nph.append(nr_pulses/(nr_analyzed_files))
        print('Temp: %d K, Nph: %.1f' % (temp, Nph[-1]))
    ax['a'].scatter(temps, Nph)
    Nph = np.array(Nph)
    np.save('figures/Nph_KID5_113dBm_40s.npy', Nph)
    plt.show()

def get_log_power(temp, a):
    inband = get_bp(bp185, .5)
    bb = planck_wl(wl*1e-6, np.array(temp).reshape((-1,1)))
    radiance = tot185*bb
    return np.log10(a) + np.log10(np.trapezoid(radiance[:,inband], wl[inband]))


def get_power(temp, a):
    inband = get_bp(tot185, 1e-19, plot=False)
    bb = planck_wl(wl*1e-6, np.array(temp).reshape((-1,1)))
    radiance = tot185*bb
    return a*np.trapezoid(radiance[:,inband], wl[inband])


def fit_efficiency():
    Nph = np.load('figures/Nph_KID5_113dBm_40s.npy')
    absorbed_powers = Nph*Eph
    fit_idx = 4
    # [a], pcov = curve_fit(get_log_power, temps[fit_idx:], np.log10(absorbed_powers[fit_idx:]))
    [a], pcov = curve_fit(get_power, temps[fit_idx:], absorbed_powers[fit_idx:])

    fig, ax = plt.subplots(figsize=(5,4), constrained_layout=True)
    # ax.scatter(temps[fit_idx:], absorbed_powers[fit_idx:], marker='o', facecolor='w', edgecolor='o', linewidth=2, label='In-band pulses')
    yerr = np.sqrt(Nph)*Eph
    ax.errorbar(temps, absorbed_powers, yerr=yerr, linestyle='None', ecolor='o', elinewidth=1, marker='o', markerfacecolor='o', markeredgecolor='o', linewidth=2, label='5$\sigma$ counts')
    # ax.errorbar(temps[:fit_idx], absorbed_powers[:fit_idx], yerr=yerr[:fit_idx], linestyle='None', ecolor='o', elinewidth=1, marker='o', markerfacecolor='w', markeredgecolor='o', linewidth=2, label='Dark counts')
    # ax.scatter(temps[:fit_idx], absorbed_powers[:fit_idx], marker='o', facecolor='w', edgecolor='o', linewidth=2)
    secax = ax.secondary_yaxis('right', functions=(lambda x: x / Eph, lambda x: x * Eph))
    secax.set_ylabel('Count rate [s$^{-1}$]')
    t = np.linspace(3,200,100)
    # ax.semilogy(t, 10**get_log_power(t, a), ls='--', c='k', label='Fit, a=%.2e' % a)
    ax.semilogy(t, get_power(t, a), ls='--', c='k', label='Fit of eq. (x), $\eta$=%.1e' % a)
    ax.axhline(np.mean(absorbed_powers[:fit_idx]), ls='-.', c='k', label='Dark count rate')
    ax.set_ylim(1e-21,1e-18)
    ax.set_xlim(0,200)
    ax.set_ylabel('Power [W]')
    ax.set_xlabel('Temperature [K]')
    ax.legend()
    plt.savefig('figures/photonrates.pdf')
    plt.show()


fit_efficiency()