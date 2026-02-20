from scipy.ndimage import label
import matplotlib.pyplot as plt
from natsort import natsorted
from itertools import groupby
from natsort import natsorted
from scipy.fft import fft
from glob import glob
import numpy as np
import pickle
import os


def load_dark_counts(path2data, name, wl, from_scratch=False):
    path2kid_dict = r'%skid_dict.pkl' % path2data
    with open(path2kid_dict, 'rb') as f:
        kid_dict = pickle.load(f)
    print("Loaded file %s" % path2kid_dict)

    path2mux_dict = r'%smux_dict.pkl' % path2data    # file path to the pickle file containing the processed multiplexed dark measurement data
    if os.path.exists(path2mux_dict) and not from_scratch:
        with open(path2mux_dict, 'rb') as f:
            mux_dict = pickle.load(f)
        print("Loaded file %s" % path2mux_dict)
    else:
        dir = kid_dict[name][wl]['dir']                                                     # dir to multiplexed dark measurement data
        pkl_files = natsorted(glob(dir[:-7] + '*.pkl'))                                     # make a list of all the .pkl files per analysed segment
        for i, file in enumerate(pkl_files):
            with open(file, 'rb') as f:
                if i == 0:
                    mux_dict = pickle.load(f)
                else:
                    seg_pkl = pickle.load(f)
                    for kid, item in mux_dict.items():
                        if len(seg_pkl[kid]['t']) == len(seg_pkl[kid]['pulses']):
                                item['t'].extend(seg_pkl[kid]['t'])                         # timestamps of pulses in this segment, between 0-10s
                                item['pulses'].extend(seg_pkl[kid]['pulses'])               # individual pulses
                                item['T'] += seg_pkl[kid]['T']                              # timestamp of this segment by cumulatively summing the length of every individual segment
                        else:
                            print('ERROR: Mismatch in lengths for nr of pulses and timestamps in %s' % kid)

        with open(path2mux_dict, 'wb') as f:
            pickle.dump(mux_dict, f)
            print('Generated %s, which contains the dark counts for the detectors %s' % (path2mux_dict, mux_dict.keys()))  


def analyse_coincidences(path2data, name, coincidence_dt):
    path2mux_dict = r'%smux_dict.pkl' % path2data
    with open(path2mux_dict, 'rb') as f:
        mux_dict = pickle.load(f)

    del mux_dict['coincidence_dt'], mux_dict['diffs'], mux_dict['multiplicities']   
    # concatenate all timestamps
    ts = []
    ids = []
    i = 0
    for kid, item in mux_dict.items():                        
        mux_dict.items()
        t = np.array(item['t'])
        ts.extend(t)
        ids.extend(np.ones(len(t))*i)
        i += 1
    ts = np.array(ts)
    ids = np.array(ids)

    # sort the timestamps identify the single events
    argsort = np.argsort(ts)                                   
    rev_argsort = np.argsort(argsort)
    sorted_ts = ts[argsort]
    diffs = np.diff(sorted_ts)
    too_close = (diffs<=coincidence_dt)                              
    good = np.ones(ts.shape, dtype=int)
    good[1:] -= too_close
    good[:-1] -= too_close
    good = (good == True)

    # determine the multiplicity (= number of coincident pulses) of each event
    labeled_array, num_features = label(~good)
    multiplicity = np.zeros_like(good, dtype=int)
    for label_id in range(1, num_features + 1):                     # group 
        multiplicity[labeled_array == label_id] = np.sum(labeled_array == label_id)
    multiplicities = [sum(1 for _ in group) for value, group in groupby(~good) if value]

    # identify the coincident pulses per detector 
    coincident = np.sum(~good)
    rev_good = good[rev_argsort]
    rev_multiplicty = multiplicity[rev_argsort]
    i = 0
    for kid, item in mux_dict.items():
        item['single events'] = rev_good[ids==i].astype(bool)
        item['multiplicity'] = rev_multiplicty[ids==i].astype(int)
        i += 1
    mux_dict['coincidence_dt'] = coincidence_dt
    mux_dict['diffs'] = diffs
    mux_dict['multiplicities'] = multiplicities    
    
    # save meta data in mux_dict.pkl
    with open(path2mux_dict, 'wb') as f:
        pickle.dump(mux_dict, f)
    print("Updated file %s" % path2mux_dict)

    fig, axes = plt.subplot_mosaic('a', figsize=(18.5/2/2.54, 4/2.54), constrained_layout=True)
    ax = axes['a']
    ax.hist(mux_dict['diffs']*1e6, bins=np.logspace(0, 4, 100), facecolor='k', alpha=0.7)
    ax.axvline(mux_dict['coincidence_dt']*1e6, color='r', linestyle='--')
    ax.set_xscale('log')
    ax.set_yscale('log')
    ax.set_xlabel('Time between consecutive events [µs]')
    ax.set_ylabel('Counts')
    plt.savefig('figures/%s_coincidences.pdf' % (name))

    fig, axes = plt.subplot_mosaic('a', figsize=(18.5/2/2.54, 4/2.54), constrained_layout=True)
    ax = axes['a']
    ax.hist(mux_dict['multiplicities'], bins=np.arange(-.5, 7, 1), facecolor='k', alpha=0.7)
    ax.set_xlabel('Multiplicity of coincident events')
    ax.set_ylabel('Counts')
    print('total dark counts = %d \n%d coincident pulses across all detectors in %d events' % (len(ts), coincident, len(mux_dict['multiplicities'])))


def analyse_dark_countrates(path2data, name, wl, confidence_intervals):
    path2kid_dict = r'%skid_dict.pkl' % path2data
    with open(path2kid_dict, 'rb') as f:
        kid_dict = pickle.load(f)
    print("Loaded file %s" % path2kid_dict)
    path2mux_dict = r'%smux_dict.pkl' % path2data
    with open(path2mux_dict, 'rb') as f:
        mux_dict = pickle.load(f)
    
    Q_mux = mux_dict[name]['Q']
    nxx = mux_dict[name]['Nxx']
    single_event_mask = mux_dict[name]['single events']
    print('coincident events removed for %s: %d out of %d' % (name, np.sum(~single_event_mask), len(single_event_mask)))

    confidence_intervals = [3, 4, 5]
    colors = ['b', 'y', 'o', 'p']
    wls = ['3.8um', '8.5um', '18.5um', '25um']

    fig, axes = plt.subplot_mosaic('a;b;c;d', figsize=(18.5/2/2.54, 12/2.54), constrained_layout=True, sharey=True)
    axids = 'abcd'
    for nr in confidence_intervals:
        for i, wl in enumerate(wls):
            if nr == confidence_intervals[0]:
                kid_dict[name][wl]['dcr'] = {}
            bins = np.arange(0, 4, 0.05)
            Q_st = kid_dict[name][wl]['Q']
            template = kid_dict[name][wl]['pulse template'] 
            std = kid_dict[name][wl]['stds'][0]
            norm_template = template / np.max(template)
            L = round(len(norm_template)/2+1)
            pulses = np.array(mux_dict[name]['pulses'])[single_event_mask]
            neg_pulses = np.array(mux_dict[name]['neg pulses'])
            norm_fft = fft(norm_template[:len(nxx)])
            opt_filter = norm_fft.conj() / nxx
            normalisation = np.sum((np.abs(norm_fft)**2 / nxx))
            pulses_fft = fft(pulses, axis=1)
            H_mux = np.real(np.sum((opt_filter*pulses_fft), axis=1) / normalisation)
            H_st = kid_dict[name][wl]['Hopts'][-1] *(Q_mux/Q_st)
            x, y = kid_dict[name][wl]['kde']
            Ropt = kid_dict[name][wl]['Ropt']
            mu = x[np.argmax(y)]
            std = mu/Ropt / 2.355
            lims = np.array([mu - nr*std, mu + nr*std])*(Q_mux/Q_st)
            dcr = np.sum((H_mux >=lims[0])&(H_mux <= lims[1])) / mux_dict[name]['T'] * 1e3
            kid_dict[name][wl]['dcr'][str(nr)] = dcr
            if nr == confidence_intervals[1]:
                ax = axes[axids[i]]
                ax.hist(H_st, bins=bins, alpha=.75, color=colors[i], zorder=2)
                _ = ax.hist(H_mux, bins=bins, label='dark counts', color='k', alpha=0.5, zorder=1)
                ax.fill_between(lims, 1e4, color='r', alpha=.3, linestyle= '-', lw=1.5, label='%d$\sigma$ conf. interval' % nr, zorder=0)
                axes['a'].hist([], bins=bins, alpha=.75, label='%s $\mu$m' % (wl[:-2]), color=colors[i], zorder=2)
                ax.set_yscale('log')
                ax.set_ylabel('Counts')
                ax.set_ylim(1e0,1e4)
                yticks = np.logspace(0,4,3)
                ax.set_yticks(yticks)
                ax.set_xlim(0,np.pi)
                ax.grid(True, which='major')
                ax.grid(False, which='minor')
    axes['a'].legend(bbox_to_anchor=(0., 1, 1., .102), loc='lower left',ncols=3, mode="expand", borderaxespad=0., handlelength=2)
    axes['d'].set_xlabel('Pulse height [rad]')

    plt.savefig('figures/%s_dark_counts.pdf' % (name))

    with open(path2kid_dict, 'wb') as f:
        pickle.dump(kid_dict, f)

    fig, axes = plt.subplot_mosaic('a', figsize=(18.5/2/2.54, 7/2.54), constrained_layout=True)
    Ndarks = []
    lambdas = []
    for wl in wls:
        item= kid_dict[name][wl]
        lambdas.append(float(wl[:-2]))
        Ndarks.append([item['dcr']['3'], item['dcr']['4'], item['dcr']['5']])
    lambdas = np.array(lambdas)
    Ndarks = np.array(Ndarks)
    print('dark count rates = \n', Ndarks)
    ax = axes['a']
    ax.plot(lambdas, Ndarks[:, 1], 'p-', c='k', label='$N_\mathrm{dark}$ 4 $\sigma$', linewidth=1, markerfacecolor='k', markeredgecolor='k', zorder=-3, markeredgewidth=2)
    # ax.plot(wls, Ndarks[1], 'p-', c='k', label='$N_\mathrm{dark}$', linewidth=1, markerfacecolor='None', markeredgecolor='k', zorder=-3)
    ax.fill_between(lambdas, Ndarks[:, 0], Ndarks[:, 2], color='k', alpha=0.2, label='3-5 $\sigma$', zorder=-4)
    ax.set_ylabel('Dark count rate [mHz]')
    ax.set_xlabel('Wavelength [µm]')
    ax.set_yscale('log')
    ax.set_xscale('log')
    ax.set_xlim([3,30])
    ax.set_ylim([1,100])
    xticks = [3, 4, 5, 6, 7, 8, 9, 10, 20, 30]
    ax.set_xticks(xticks)
    ax.set_xticklabels(xticks)
    yticks = [1, 2, 5, 10, 20, 50, 100]
    ax.set_yticks(yticks)
    _ = ax.set_yticklabels(yticks)

    plt.savefig('figures/%s_dark_count_rates.pdf' % (name))
