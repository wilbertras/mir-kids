import matplotlib.pyplot as plt
import numpy as np
import scipy.constants as sc
from scipy.signal import savgol_filter
from scipy.integrate import cumulative_trapezoid
import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from filters.filters import load_all_filters
# plt.style.use('matplotlibrc')
import figures.matplotlibcolors as matplotlibcolors
plt.style.use('figures/matplotlibrc')

def planck_wl(wl, t):
    return sc.h * sc.c**2 / (wl**5 * (np.exp(sc.h * sc.c / (wl * sc.k * t)) - 1))

filters = load_all_filters()

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


fig, axes = plt.subplot_mosaic('ab;cd;ef', figsize=(18.5/2.54,8/2.54), constrained_layout=True, sharex=True)
xlim = [1, 100]
ax = axes['a']
ax.semilogx(wl, bp38, label='BP38')
ax.fill_between(wl, bp38, 0, where=(wl < 45), alpha=0.3)
ax.legend()
ax.set_ylim([0,1])
ax.set_xlim(xlim)
# ax.set_xlabel('Wavelength [µm]')
ax.set_ylabel('Transmission [-]')

ax = axes['b']
ax.semilogx(wl, bp85, label='BP85', zorder=1)
ax.fill_between(wl, bp85, 0, where=(wl < 45), alpha=0.3)
ax.semilogx(wl, filters['caf2'], label='$CaF_2$', zorder=0)
ax.fill_between(wl, caf2, 0, where=(wl < 16), alpha=0.3)
ax.legend()
ax.set_ylim([0,1])
ax.set_xlim(xlim)
# ax.set_xlabel('Wavelength [µm]')
# ax.set_ylabel('Transmission [-]')
# ax.set_yticklabels([])

ax = axes['c']
ax.semilogx(wl, bp185, label='BP185', zorder=1)
ax.fill_between(wl, bp185, 0, where=(wl < 40), alpha=0.3)
ax.semilogx(wl, znse, label='ZnSe', zorder=0)
ax.fill_between(wl, znse, 0, where=(wl < 25), alpha=0.3)
ax.legend()
ax.set_ylim([0,1])
ax.set_xlim(xlim)
# ax.set_xlabel('Wavelength [µm]')
ax.set_ylabel('Transmission [-]')

ax = axes['d']
ax.semilogx(wl, bp25, label='BP25', zorder=3)
ax.fill_between(wl, bp25, 0, where=((wl < 77) & (wl > 1.67)), alpha=0.3)
ax.semilogx(wl, sp_a, label='SPA', zorder=2)
ax.fill_between(wl, sp_a, 0, where=((wl < 77)&(wl > 1.67)), alpha=0.3)
ax.semilogx(wl, sp_b, label='SPB', zorder=1)
ax.fill_between(wl, sp_b, 0, where=((wl < 77)&(wl > 1.67)), alpha=0.3)
ax.semilogx(wl, lp, label='LP', zorder=0)
ax.fill_between(wl, lp, 0, where=((wl < 82)&(wl > 1.67)), alpha=0.3)
ax.legend()
ax.set_ylim([0,1])
ax.set_xlim(xlim)
# ax.set_xlabel('Wavelength [µm]')
# ax.set_ylabel('Transmission [-]')
# ax.set_yticklabels([])

ax = axes['e']
ax.plot(wl, si, label='Si')
ax.fill_between(wl, si, 0, alpha=0.3)
ax.legend()
ax.set_ylim([0,1])
ax.set_xlim(xlim)
ax.set_xlabel('Wavelength [µm]')
ax.set_ylabel('Transmission [-]')
# ax.set_yticklabels([])

ax = axes['f']
ax.loglog(wl, nd1, label='ND1')
ax.fill_between(wl, nd1, 0, where=(wl < 18), alpha=0.3)
ax.loglog(wl, nd2, label='ND2')
ax.fill_between(wl, nd2, 0, where=(wl < 18), alpha=0.3)
ax.loglog(wl, nd3, label='ND3')
ax.fill_between(wl, nd3, 0, where=(wl < 18), alpha=0.3)
ax.legend()
ax.set_ylim([1e-4, 1e0])
ax.set_xlim(xlim)
ax.set_xlabel('Wavelength [µm]')
# ax.set_ylabel('Transmission [-]')
# plt.savefig('figures/transmission_spectra.pdf')



A = (1e-3)**2
omega = np.pi*(10e-3)**2/(300e-3)**2
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
bb38 = planck_wl(wl*1e-6, Tbb38)*1e-6
bb185 = planck_wl(wl*1e-6, Tbb185)*1e-6
bb25 = planck_wl(wl*1e-6, Tbb25)*1e-6
fig, axes = plt.subplot_mosaic('a', figsize=(8/2.54, 6/2.54), constrained_layout=True, sharey=True, sharex=True)
ax = axes['a']
ax.plot(wl, tot38*bb38*A*omega, label='3.8 µm', color='b')
# ax.fill_between(wl, tot38*bb38, 0, alpha=0.3, color='b')
# ax.loglog(wl, bb38* A * omega, color='k')
# ax.annotate('Planck 300 K', xy=(10,0.01), xycoords='data', size=8)
# cumulative_tot38_bb38 = np.cumsum(tot38 * bb38*np.diff(wl, prepend=wl[0]))
# cumulative_tot38_bb38 = cumulative_trapezoid(tot38 * bb38, wl, initial=None)
# ax.loglog(wl, cumulative_tot38_bb38, label='Cumulative power')
ax.legend()
ax.set_ylim([1e-30, 1e-14])
ax.set_xlim([1,100])
# ax.set_xlabel('Wavelength [µm]')
ax.set_ylabel('$I$ [$W sr^{-1}m^{-2}\mu m^{-1}$]')

ax = axes['a']
ax.loglog(wl, tot85*bb38*A*omega, label='8.5 µm', color='y')
# ax.fill_between(wl, tot85*bb38, 0, alpha=0.3, color='y')
# ax.loglog(wl, bb38* A * omega, color='k')
# ax.annotate('Planck 300 K', xy=(10,0.01), xycoords='data', size=8)
# cumulative_tot85_bb38 = cumulative_trapezoid(tot85 * bb38, wl, initial=None)
# ax.loglog(wl[1:], cumulative_tot85_bb38, label='Cumulative power')
ax.legend()
# ax.set_xlabel('Wavelength [µm]')
# ax.set_ylabel('$B_\lambda$ [$W sr^{-1}m^{-2}\mu m^{-1}$]')

# ax = axes['c']
ax.loglog(wl, tot185*bb185*A*omega, label='18.5 µm', color='o')
# ax.fill_between(wl, tot185*bb185, 0, alpha=0.3, color='r')
# ax.loglog(wl, bb185* A * omega, color='k')
# ax.annotate('Planck 160 K', xy=(18.5,0.001), xycoords='data', size=8)
# cumulative_tot185_bb185 = cumulative_trapezoid(tot185 * bb185, wl, initial=None)
# ax.loglog(wl[1:], cumulative_tot185_bb185, label='Cumulative power')
ax.legend()
ax.set_xlabel('Wavelength [µm]')
ax.set_ylabel('Spectral radiance [$W sr^{-1}m^{-2}\mu m^{-1}$]')

# ax = axes['c']
ax.loglog(wl, tot25*bb25*A*omega, label='25 µm', color='p')
# ax.fill_between(wl,  tot25*bb25, 0, alpha=0.3, color='p')
# ax.loglog(wl, bb25* A * omega, color='k')
# ax.annotate('Planck 40 K', xy=(25,0.001), xycoords='data', size=8)
# cumulative_tot25_bb25 = cumulative_trapezoid(tot25 * bb25, wl, initial=None)
# ax.loglog(wl[1:], cumulative_tot25_bb25, label='Cumulative power')
ax.legend()
ax.set_xlabel('Wavelength [µm]')
# plt.savefig('figures/transmission_power.pdf')

# fig, ax = plt.subplots()
# ax.semilogx(wl, cumulative_tot38_bb38, label='3.8')
# ax.semilogx(wl[1:], cumulative_tot85_bb38, label='8.5')
# ax.semilogx(wl[1:], cumulative_tot185_bb185, label='18.5')
# # ax.semilogx(wl[1:], cumulative_tot25_bb25, label='25')
# ax.legend()
# ax.set_ylabel('$B_\lambda$ [$W sr^{-1}m^{-2}\mu m^{-1}$]')
ax.legend(bbox_to_anchor=(0., 1, 1., .102), loc='lower left',
        ncols=2, mode="expand", borderaxespad=0.)
# plt.savefig('figures/transmission_power_combined.pdf')
plt.show()