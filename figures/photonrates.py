import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import find_peaks, peak_widths, fftconvolve
import scipy.constants as sc
from scipy.optimize import curve_fit
import re
import os
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
import figures.functions as f
from main import pulse_analysis
import figures.matplotlibcolors as matplotlibcolors
from filters.filters import load_all_filters
plt.style.use('figures/matplotlibrc')


filters = load_all_filters()
wl = filters['wl']
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



def count_pulses(dir, kid, pread, nr_req_files, filter, lifetime, mph, mpp, coord='circle', response='phase', file_type='vis'):
    dir = dir.replace("\\", '/')
    pulse_files, info_files = f.get_files(dir, kid, pread, type=file_type)
    f0, Q, Qc, Qi, S21_min, dt, T = f.get_info(info_files[0])
    sff = round(1 / dt) / 1e6
    nr_files = len(pulse_files)
    if nr_req_files >= nr_files:
        nr_req_files = nr_files
    amp, phase, _ = f.get_data(pulse_files[:nr_req_files])
    signal = f.coord_transformation(phase, amp, coord=coord, response=response)
    time = len(signal) * dt
    if filter:
        window = f.get_window(filter, lifetime)
    std = f.get_sigma(signal, window)
    ph = mph*std
    pp = mpp*std
    locs, _ = f.find_pks(signal, ph[0], pp, window)
    total_pulses = len(locs)
    return total_pulses, time


def count_pulses_med(dir, kid, pread, filter, lifetime, mph, mpp, coord='circle', response='phase', file_type='med'):   
    dir = dir.replace("\\", '/')
    pulse_files, info_files = f.get_files(dir, kid, pread, type=file_type, chip='Tchip0.13')
    f0, Q, Qc, Qi, S21_min, dt, T = f.get_info(info_files[0])
    sff = round(1 / dt) / 1e6
    lifetime = round(lifetime * sff)
    temps = []
    rates = []
    if filter:
        window = f.get_window(filter, lifetime)
    for i, file in enumerate(pulse_files):
        amp, phase, _ = f.get_data([file], discard=False)
        T = int(re.findall('\d+', file)[-1])*1e-3
        signal = f.coord_transformation(phase, amp, coord=coord, response=response)
        time = len(signal) * dt
        std = f.get_sigma(signal, window)
        ph = mph*std
        pp = mpp*std
        locs, _ = f.find_pks(signal, ph[0], pp, window)
        total_pulses = len(locs)
        rates.append(total_pulses / time)
        temps.append(T)
        print('Temp: %d K, Nph: %.1f' % (T, rates[-1]))
    np.save('figures/Nph_%sum_KID%d_%ddBm_%ds_med.npy' % (str(name), kid, pread, time), np.vstack((temps, rates)))
    


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


def get_bp(theta, mph, plot=True):
    locs, _ = find_peaks(theta, height=mph, prominence=mph)
    widths, heights, lefts, rights = peak_widths(theta, locs, rel_height=.999)
    if len(locs) == 1:
        idx = np.arange(np.floor(lefts[0]), np.ceil(rights[0]), dtype=int)
        if plot:
            fig, ax = plt.subplots()
            ax.loglog(wl*1e6, theta)
            ax.set_xlim([1, 1e2])
            ax.plot(wl[idx]*1e6, theta[idx], c='r')
        return idx
    else:
        raise ValueError('More than one peak found')


def counts_vs_temp(dir, temps):
    Nph = []
    fig, ax = plt.subplot_mosaic('a')
    for temp in temps:
        path = dir % temp
        nr_pulses, time = count_pulses(path, kid, pread, nr_req_files, filter, lifetime, mph, mpp, coord='circle', response='phase')
        Nph.append(nr_pulses/time)
        print('Temp: %d K, Nph: %.1f' % (temp, Nph[-1]))
    ax['a'].scatter(temps, Nph)
    Nph = np.array(Nph)
    np.save('figures/Nph_%sum_KID%d_%ddBm_%ds.npy' % (str(name), kid, pread, time), np.vstack((temps, Nph)))
    plt.show()


def get_log_power(temp, a):
    inband = get_bp(theta, .5)
    bb = planck_wl(wl, np.array(temp).reshape((-1,1)))
    radiance = theta*bb / Eph
    return np.log10(a) + np.log10(np.trapezoid(radiance[:,inband], wl[inband]))


def get_power(temp, a):
    inband = get_bp(theta, bp_mph, plot=False)
    bb = planck_wl(wl, np.array(temp).reshape((-1,1)))
    radiance = theta*bb / Eph
    return a*np.trapezoid(radiance[:,inband], wl[inband])


def fit_efficiency(path, fit_idx, xlim, color, title):
    data = np.load(path)
    temps = data[0,:]
    Nph = data[1,:]
    absorbed_rates = Nph
    [eta], pcov = curve_fit(get_power, temps[fit_idx[0]:fit_idx[1]], absorbed_rates[fit_idx[0]:fit_idx[1]])
    print('Fit efficiency: %.1f$\pm$%.2f' % (eta*100, np.sqrt(np.diag(pcov))[0]*100))
    fig, ax = plt.subplots(figsize=(18.5/2/2.54, 7/2.54), constrained_layout=True)
    yerr = np.sqrt(Nph)
    ax.errorbar(temps[:fit_idx[1]], absorbed_rates[:fit_idx[1]], yerr=yerr[:fit_idx[1]], linestyle='None', ecolor=color, elinewidth=1, marker='p', markerfacecolor=color, markeredgecolor=color, linewidth=2, label='%d $\sigma$ counts' % mph[0])
    # secax = ax.secondary_yaxis('right', functions=(lambda x: x / Eph, lambda x: x * Eph))
    ax.set_ylabel('Count rate [$Hz$]')
    t = np.linspace(xlim[0],xlim[1])
    ax.semilogy(t, get_power(t, eta), ls='--', c='k', label='Fit eq. (3)')
    Nbg = np.mean(absorbed_rates[:fit_idx[0]])
    std_Nbg = np.std(absorbed_rates[:fit_idx[0]])
    print('Background rate: %.2f$\pm$%.3f' % (Nbg, std_Nbg))
    ax.axhline(Nbg, ls='-.', c='k', label='$N_{\mathrm{bg}}$')
    ax.set_ylim([1e-1, 1e2])
    ax.set_xlim(xlim)
    # secax.set_ylabel('Power [W]')
    ax.set_xlabel('Radiator Temperature [K]')
    ax.legend(bbox_to_anchor=(0., 1, 1., .102), loc='lower left',
        ncols=3, mode="expand", borderaxespad=0.)
    fig.suptitle(title, fontsize=10)
    # plt.savefig(path[:-4] + 'v2.pdf')

name = 18.5
kid = 5
pread = 113
file_type = 'vis'
pw = 1500
pw_offset = 100
filter = 'exp'
lifetime = 215
nr_req_files = 40
mph = np.array([5, 20])
mpp = mph[0] 
dir = r'D:\Data\LT218Chip1_BF_20240116_MIR18_5\KID5_all_temps\%dK'
temps = [3,30,50,85,100,135,142,150,160,180,200]
Eph = sc.h*sc.c/(name*1e-6)
theta = bp185**5*nd1*nd3*znse**2*A*omega
bp_mph=1e-19

# counts_vs_temp(dir, temps)
fit_efficiency(r'figures\Nph_18.5um_KID5_113dBm_40s.npy', fit_idx=[4,-2], xlim=[0,225], color='o', title='$18$ $\mu$m')


# name = 25
# kid = 5
# pread = 109
# file_type = 'vis'
# pw = 1500
# pw_offset = 100
# filter = 'exp'
# lifetime = 270
# nr_req_files = 40
# mph = np.array([4, 20])
# mpp = mph[0] 
# dir = r'D:\Data\LT218Chip1_ADR_20240506_MIR24\12KIDs_3Pread_TD40s_1MHz_BB%dK\TD_Power'
# temps = [3,18,20,22,24,26]
# Eph = sc.h*sc.c/(name*1e-6)
# theta = bp25**3*lp*sp_a**3*sp_b*si*A*omega
# bp_mph=4e-10

# # counts_vs_temp(dir, temps)
# fit_efficiency(r'figures\Nph_25um_KID5_109dBm_40s.npy', fit_idx=[3,None], xlim=[0,30])

name = 25
kid = 5
pread = 113
file_type = 'med'
pw = 1500
pw_offset = 100
filter = 'exp'
lifetime = 250
nr_req_files = 1
mph = np.array([5, 20])
mpp = mph[0] 
dir = r'D:\Data\LT218Chip1_ADR_20240506_MIR24\12KIDs_3Pread_TD80s_50KHz_BB3-28K'
Eph = sc.h*sc.c/(name*1e-6)
theta = bp25**3*lp*sp_a**3*sp_b*si*A*omega
bp_mph=4e-10

# # count_pulses_med(dir, kid, pread, filter, lifetime, mph, mpp, coord='circle', response='phase', file_type='med')
fit_efficiency(r'figures\Nph_25um_KID5_113dBm_80s_med.npy', fit_idx=[4,-2], xlim=[0,30], color='p', title='$25$ $\mu$m')
    
plt.show()