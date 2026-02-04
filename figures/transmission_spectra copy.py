import matplotlib.pyplot as plt
import numpy as np
import scipy.constants as sc
from scipy.signal import savgol_filter, medfilt
from scipy.integrate import cumulative_trapezoid
from scipy.interpolate import interp1d
import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from filters.filters import load_all_filters
# plt.style.use('matplotlibrc')
import figures.matplotlibcolors as matplotlibcolors
plt.style.use('figures/matplotlibrc')

def planck_wl(wl, t):
    return sc.h * sc.c**2 / (wl**5 * (np.exp(sc.h * sc.c / (wl * sc.k * t)) - 1))

def res_power(wl, rad):
    max = np.amax(rad)
    hm = max / 2
    peak_idx = np.argmax(rad)
    cwl = wl[peak_idx]
    left_idx = np.where(rad[:peak_idx] < hm)[0][-1]
    right_idx = np.where(rad[peak_idx:] < hm)[0][0] + peak_idx
    left_wl = wl[left_idx]
    right_wl = wl[right_idx]
    d_wl = right_wl - left_wl
    R_filter = left_wl * right_wl / (d_wl * cwl)    
    print('R=%.2f, cwl=%.2f, dwl=%.2f' % (R_filter, cwl, d_wl))
    fig, ax = plt.subplots()
    ax.plot(wl, rad)
    ax.axvline(left_wl)
    ax.axvline(right_wl)
    ax.axhline(hm)
    ax.scatter(cwl, max)
    return R_filter


wl = np.linspace(0, 100, 1000)*1e-6
filters = load_all_filters(wl)

wl = filters['wl']*1e6
bp38 = filters['bp38']
bp85 = filters['bp85']
bp185 = filters['bp185']
caf2 = filters['caf2']
znse = filters['znse']
nd1 = filters['nd1']
nd2 = filters['nd2']
nd3 = filters['nd3']
ger = filters['ger'] 
bp25 = filters['bp25']
sp_a = filters['sp_a']
sp_b = filters['sp_b']
lp = filters['lp']
si = filters['si']


fig, axes = plt.subplot_mosaic('a;d;e;f', figsize=(18.5/2/2.54, 12/2.54), constrained_layout=True, sharex=True)
xlim = [1, 100]
ax = axes['a']
ax.plot(wl, savgol_filter(bp38, 6, 5), label='BP$_{3.8}$', c='b')
# ax.fill_between(wl, bp38, 0, where=(wl < 45), alpha=0.3)
ax.legend()
ax.set_ylim([0,1])
ax.set_xlim(xlim)
# ax.set_xlabel('Wavelength [µm]')
ax.set_ylabel('Transmission [-]')

# ax = axes['b']
ax.plot(wl, medfilt(bp85), label='BP$_{8.5}$', zorder=1, c='y')
# ax.fill_between(wl, bp85, 0, where=(wl < 45), alpha=0.3)
# ax.loglog(wl, filters['caf2'], label='$CaF_2$', zorder=0)
# ax.fill_between(wl, caf2, 0, where=(wl < 16), alpha=0.3)
ax.legend()
ax.set_ylim([0,1])
ax.set_xlim(xlim)
# ax.set_xlabel('Wavelength [µm]')
# ax.set_ylabel('Transmission [-]')
# ax.set_yticklabels([])

# ax = axes['c']
ax.plot(wl, medfilt(bp185), label='BP$_{18.5}$', zorder=1, c='o')
# ax.fill_between(wl, bp185, 0, where=(wl < 40), alpha=0.3)
# ax.loglog(wl, znse, label='ZnSe', zorder=0)
# ax.fill_between(wl, znse, 0, where=(wl < 25), alpha=0.3)
ax.legend()
ax.set_ylim([0,1])
ax.set_xlim(xlim)
# ax.set_xlabel('Wavelength [µm]')
ax.set_ylabel('Transmission [-]')

ax = axes['d']
ax.loglog(wl, bp25, label='BP$_{25}$', zorder=3, c='p')
# ax.fill_between(wl, bp25, 0, where=((wl < 77) & (wl > 1.67)), alpha=0.3)
ax.loglog(wl, sp_a, label='SP$_{A}$', zorder=2, c='b')
# ax.fill_between(wl, sp_a, 0, where=((wl < 77)&(wl > 1.67)), alpha=0.3)
ax.loglog(wl, sp_b, label='SP$_{B}$', zorder=1, c='y')
# ax.fill_between(wl, sp_b, 0, where=((wl < 77)&(wl > 1.67)), alpha=0.3)
ax.loglog(wl, lp, label='LP', zorder=0, c='o')
# ax.fill_between(wl, lp, 0, where=((wl < 82)&(wl > 1.67)), alpha=0.3)
ax.legend()
ax.set_ylim([0,1])
ax.set_xlim(xlim)
# ax.set_xlabel('Wavelength [µm]')
# ax.set_ylabel('Transmission [-]')
# ax.set_yticklabels([])

ax = axes['e']
ax.plot(wl, znse, label='ZnSe')
ax.plot(wl, si, label='Si')
ax.plot(wl, caf2, label='CaF$_2$')
# ax.fill_between(wl, si, 0, alpha=0.3)
ax.legend()
ax.set_ylim([0,1])
ax.set_xlim(xlim)
ax.set_xlabel('Wavelength [µm]')
ax.set_ylabel('Transmission [-]')
ax.set_yticklabels([])

ax = axes['f']
ax.loglog(wl, nd1, label='ND1')
# ax.fill_between(wl, nd1, 0, where=(wl < 18), alpha=0.3)
ax.loglog(wl, nd2, label='ND2')
# ax.fill_between(wl, nd2, 0, where=(wl < 18), alpha=0.3)
ax.loglog(wl, nd3, label='ND3')
# ax.fill_between(wl, nd3, 0, where=(wl < 18), alpha=0.3)
ax.legend()
ax.set_ylim([1e-4, 1e0])
ax.set_xlim(xlim)
ax.set_xlabel('Wavelength [µm]')
# ax.set_ylabel('Transmission [-]')
# plt.savefig('figures/transmission_spectra.pdf')



A = (1e-3)**2
omegaBF = np.pi*(10e-3)**2/(300e-3)**2
omegaADR = np.pi*(10e-3)**2/(65e-3)**2
print(np.rad2deg(np.arctan(10/300)))
print(np.rad2deg(np.arctan(10/65)))
bp38 = savgol_filter(bp38, 21, 1)
bp85 = savgol_filter(bp85, 21, 1)
bp185 = savgol_filter(bp185, 21, 1)
bp25 = savgol_filter(bp25, 21, 1)
tot38 = bp38**4*caf2**4*ger*nd2*nd3
tot85 = bp85**4*caf2**4*ger*nd2*nd3
tot185 = bp185**5*nd1*nd3*znse**2
tot25 = bp25**3*lp*sp_a**3*sp_b
# np.save('25um_filterstack.npy', np.stack((wl, tot25), axis=0))
Tbb38 = 293
Tbb185 = 160
Tbb25 = 24
bb3K = planck_wl(wl*1e-6, 3)*1e-6
bb38 = planck_wl(wl*1e-6, Tbb38)*1e-6
bb185 = planck_wl(wl*1e-6, Tbb185)*1e-6
bb25 = planck_wl(wl*1e-6, Tbb25)*1e-6


fig, axes = plt.subplot_mosaic('a', figsize=(18.5/3/2.54, 8/2.54), constrained_layout=True, sharey=True, sharex=True)
ax = axes['a']
ax.plot(wl, tot38*bb38*A*omegaBF, label='3.8 µm, lamp off', color='b')
ax.loglog(wl, tot85*bb38*A*omegaBF, label='8.5 µm, lamp off', color='y')
ax.loglog(wl, tot185*bb185*A*omegaBF, label='18.5 µm, $T_{bb}=160\ K$', color='o', ls='--')
# ax.loglog(wl, tot185*bb3K*A*omega, label='18.5 µm', color='o', ls='--')
ax.loglog(wl, tot25*bb25*A*omegaADR, label='25 µm, $T_{bb}=24\ K$', color='p', ls='--')
# ax.loglog(wl, tot25*bb25*A*omega, label='25 µm', color='p', ls='--')
ax.set_ylim([1e-28, 1e-12])
ax.set_xlim([1,100])
ax.set_xlabel('Wavelength [µm]')
ax.set_ylabel('Spectral radiance [$W sr^{-1}m^{-2}\mu m^{-1}$]')
# ax.legend(bbox_to_anchor=(0., 1, 1., .102), loc='lower left',
#         ncols=2, mode="expand", borderaxespad=0.)
ax.legend(loc='upper center', ncols=1, mode="expand", borderaxespad=0., frameon=False, handlelength=1)
# res_power(wl, tot38*bb38*A*omega)
# res_power(wl, tot85*bb38*A*omega)
# res_power(wl, tot185*bb185*A*omega)
# res_power(wl, tot25*bb25*A*omega)


# # plt.savefig('figures/total_transmission.pdf')
plt.show()