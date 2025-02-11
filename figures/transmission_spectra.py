import matplotlib.pyplot as plt
import numpy as np
import scipy.constants as sc
from scipy.integrate import cumulative_trapezoid
import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from filters.filters import load_all_filters

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

tot38 = bp38**4*caf2**4*ger*nd2*nd3
tot85 = bp85**4*caf2**4*ger*nd2*nd3
tot185 = bp185**5*nd1*nd3*znse**2
tot25 = bp25**3*lp*sp_a**3*sp_b

bb300 = planck_wl(wl*1e-6, 300)*1e-6
bb160 = planck_wl(wl*1e-6, 160)*1e-6
bb40 = planck_wl(wl*1e-6, 40)*1e-6

fig, axes = plt.subplot_mosaic('ab;cd;ef', figsize=(12,6), constrained_layout=True)
xlim = [1, 100]
ax = axes['a']
ax.semilogx(wl, bp38, label='BP38')
ax.fill_between(wl, bp38, 0, where=(wl < 45), alpha=0.3)
ax.legend()
ax.set_ylim([0,1])
ax.set_xlim(xlim)
ax.set_xlabel('Wavelength [µm]')
ax.set_ylabel('Transmission [-]')

ax = axes['b']
ax.semilogx(wl, bp85, label='BP85', zorder=1)
ax.fill_between(wl, bp85, 0, where=(wl < 45), alpha=0.3)
ax.semilogx(wl, filters['caf2'], label='CaF_2', zorder=0)
ax.fill_between(wl, caf2, 0, where=(wl < 16), alpha=0.3)
ax.legend()
ax.set_ylim([0,1])
ax.set_xlim(xlim)
ax.set_xlabel('Wavelength [µm]')
ax.set_ylabel('Transmission [-]')

ax = axes['c']
ax.semilogx(wl, bp185, label='BP185', zorder=1)
ax.fill_between(wl, bp185, 0, where=(wl < 40), alpha=0.3)
ax.semilogx(wl, znse, label='ZnSe', zorder=0)
ax.fill_between(wl, znse, 0, where=(wl < 25), alpha=0.3)
ax.legend()
ax.set_ylim([0,1])
ax.set_xlim(xlim)
ax.set_xlabel('Wavelength [µm]')
ax.set_ylabel('Transmission [-]')

ax = axes['d']
ax.semilogx(wl, bp25, label='BP25', zorder=3)
ax.fill_between(wl, bp25, 0, where=((wl < 77) & (wl > 1.67)), alpha=0.3)
ax.semilogx(wl, sp_a, label='SP1', zorder=2)
ax.fill_between(wl, sp_a, 0, where=((wl < 77)&(wl > 1.67)), alpha=0.3)
ax.semilogx(wl, sp_b, label='SP2', zorder=1)
ax.fill_between(wl, sp_b, 0, where=((wl < 77)&(wl > 1.67)), alpha=0.3)
ax.semilogx(wl, lp, label='LP', zorder=0)
ax.fill_between(wl, lp, 0, where=((wl < 82)&(wl > 1.67)), alpha=0.3)
ax.legend()
ax.set_ylim([0,1])
ax.set_xlim(xlim)
ax.set_xlabel('Wavelength [µm]')
ax.set_ylabel('Transmission [-]')

ax = axes['e']
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
ax.set_ylabel('Transmission [-]')


fig, axes = plt.subplot_mosaic('ab;cd', figsize=(8,5), constrained_layout=True, sharey=True, sharex=True)
ax = axes['a']
ax.loglog(wl, tot38*bb300, label='3.8 µm')
ax.fill_between(wl, tot38*bb300, 0, alpha=0.3)
ax.loglog(wl, bb300, c='k', label='Planck 300 K')
cumulative_tot38_bb300 = np.cumsum(tot38 * bb300*np.diff(wl, prepend=wl[0]))
# cumulative_tot38_bb300 = cumulative_trapezoid(tot38 * bb300, wl, initial=None)
ax.loglog(wl, cumulative_tot38_bb300, label='Cumulative power')
ax.legend()
ax.set_ylim([1e-20, 1e1])
ax.set_xlim([1,100])
ax.set_xlabel('Wavelength [µm]')
ax.set_ylabel('$B_\lambda$ [$W sr^{-1}m^{-2}\mu m^{-1}$]')

ax = axes['b']
ax.loglog(wl, tot85*bb300, label='8.5 µm')
ax.fill_between(wl, tot85*bb300, 0, alpha=0.3)
ax.loglog(wl, bb300, c='k', label='Planck 300 K')
cumulative_tot85_bb300 = cumulative_trapezoid(tot85 * bb300, wl, initial=None)
ax.loglog(wl[1:], cumulative_tot85_bb300, label='Cumulative power')
ax.legend()
ax.set_xlabel('Wavelength [µm]')
ax.set_ylabel('$B_\lambda$ [$W sr^{-1}m^{-2}\mu m^{-1}$]')

ax = axes['c']
ax.loglog(wl, tot185*bb160, label='18.5 µm')
ax.fill_between(wl, tot185*bb160, 0, alpha=0.3)
ax.loglog(wl, bb160, c='k', label='Planck 160 K')
cumulative_tot185_bb160 = cumulative_trapezoid(tot185 * bb160, wl, initial=None)
ax.loglog(wl[1:], cumulative_tot185_bb160, label='Cumulative power')
ax.legend()
ax.set_xlabel('Wavelength [µm]')
ax.set_ylabel('$B_\lambda$ [$W sr^{-1}m^{-2}\mu m^{-1}$]')

ax = axes['d']
ax.loglog(wl, tot25*bb40, label='18.5 µm')
ax.fill_between(wl,  tot25*bb40, 0, alpha=0.3)
ax.loglog(wl, bb40, c='k', label='Planck 40 K')
cumulative_tot25_bb40 = cumulative_trapezoid(tot25 * bb40, wl, initial=None)
ax.loglog(wl[1:], cumulative_tot25_bb40, label='Cumulative power')
ax.legend()
ax.set_xlabel('Wavelength [µm]')
ax.set_ylabel('$B_\lambda$ [$W sr^{-1}m^{-2}\mu m^{-1}$]')
plt.show()