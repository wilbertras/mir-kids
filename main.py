import numpy as np
import matplotlib.pyplot as plt
import figures.functions as ft
from scipy.signal import convolve, fftconvolve, find_peaks, medfilt
from scipy.optimize import curve_fit
from scipy.fft import fft, ifft
from mpl_toolkits.axes_grid1.inset_locator import inset_axes
import copy
import figures.matplotlibcolors as matplotlibcolors
plt.style.use('figures/matplotlibrc')


def pulse_analysis(dir, kid, pread, file_type, chuncksize, nr_chuncks, pw, pw_offset, filter, lifetime, mph, mpp, iterate=True, exclude_dc=True, plot=False, tmax=5, coord='smith', response='phase', fit_tqp=None):
    colors = plt.rcParams['axes.prop_cycle'].by_key()['color']
    b, o, y, p, g, lb, r = colors
    pulse_color = b
    noise_color = 'tab:gray'
    smooth_color = o
    single_color = 'tab:green'
    not_single_color = r
    mp_color = r
    
    dir = dir.replace("\\", '/')
    pulse_files, info_files = ft.get_files(dir, kid, pread, type=file_type)
    folders = dir.split('/')
    title = [folder for folder in folders if folder.startswith('LT')][0]
    nr_files = len(pulse_files)
    if info_files:
        f0, Q, Qc, Qi, S21_min, dt, T = ft.get_info(info_files[0])
    else:
        dt = 0.000001
        print('No info file found: taking default values dt=1e-6')

    sff = round(1 / dt) / 1e6

    pw = round(pw * sff)
    pw_offset = int(pw_offset * sff)
    sw = round(lifetime * sff)
    if not fit_tqp:
        fit_tqp = [2*pw_offset, 3*pw_offset]
    else: 
        fit_tqp = [round(fit_tqp[0]*sff), round(fit_tqp[1]*sff)]

    if nr_chuncks:
        nr_req_files = np.amin((chuncksize * nr_chuncks, nr_files))
    else:
        nr_req_files = nr_files

    window = ft.get_window(filter, sw)
    analysed_files = 0
    made_copy = 0
    while analysed_files < nr_req_files:
        if analysed_files + chuncksize > nr_files:
            chuncksize = nr_files - analysed_files
        amp, phase, _ = ft.get_data(pulse_files[analysed_files:analysed_files+chuncksize])
        signal = ft.coord_transformation(phase, amp, coord=coord, response=response)

        if analysed_files == 0:
            std = ft.get_sigma(signal, window)
            ph = mph*std
            pp = mpp*std
            pulses = []
            single_idx = []
            too_high_idx = []
            single_locs = []
            len_file = round(len(signal) / chuncksize)

        locs, props = ft.find_pks(signal, ph[0], pp, window)
        too_high_chunck = props['peak_heights'] >= ph[1]
        args = np.argwhere(~too_high_chunck).flatten()
        pulses_chunck, single_idx_chunck = ft.get_single_pulses(signal, locs, pw, pw_offset, args)
        if len(pulses_chunck):
            pulses.append(pulses_chunck)
            single_idx.append(single_idx_chunck)
            too_high_idx.append(too_high_chunck)
            single_locs.append(locs[single_idx_chunck] + len_file*analysed_files)
        if analysed_files==0:
            noises = ft.get_single_noises(signal, locs, pw+pw_offset)
            _, noise_psd = ft.get_avg_psd(noises, pw+pw_offset, sff, exclude_dc=False, onesided=False)
        analysed_files += chuncksize
        print('Analysed %d out of %d files' % (analysed_files, nr_req_files), end='\r')
        if iterate:
            mean_pulse = np.mean(pulses_chunck, axis=0)
            copy_window = copy.copy(window)
            window = ft.opt_filter(mean_pulse, noise_psd, exclude_dc=False).real
            window /= np.sum(window)
            analysed_files = 0
            iterate -= 1
            copy_locs = copy.copy(locs)
            copy_std = copy.copy(std)
            copy_heights = copy.copy(props['peak_heights'])
            made_copy = 1
        if made_copy and analysed_files:
            new_locs = np.setdiff1d(locs, copy_locs)
            removed_locs = np.setdiff1d(copy_locs, locs)
            new_heights = props['peak_heights']
            fig, axes = plt.subplot_mosaic('ab', constrained_layout=True, figsize=(8, 4)) 
            t = np.linspace(0, (len(signal)-1)*dt, len(signal))
            window_offset = int(np.argmax(window[::-1]))
            t_smooth = t[window_offset:-len(window)+window_offset+1]
            smoothed_signal = fftconvolve(signal, window, mode='valid')
            ax = axes['a']
            ax.plot(t, signal, lw=.1, c=pulse_color, zorder=0, alpha=.5, label='response')
            ax.plot(t_smooth, smoothed_signal, lw=.5, zorder=1, c=smooth_color, label='smoothed response')
            ax.scatter(t[new_locs], smoothed_signal[new_locs-window_offset], facecolor='None', edgecolor=single_color, marker='v', zorder=2, label='%d added' % len(new_locs))
            ax.axhline(mph[0]*std, zorder=2, c=mp_color, lw=.5, label='mph=%d$\\sigma$' % (mph[0]))
            ax.legend(bbox_to_anchor=(0., 1, 1., .102), loc='lower left',
                ncols=6, mode="expand", borderaxespad=0., fontsize=9)
            ax = axes['b']
            window_offset = int(np.argmax(copy_window[::-1]))
            t_smooth = t[window_offset:-len(copy_window)+window_offset+1]
            smoothed_signal = fftconvolve(signal, copy_window, mode='valid')
            ax.plot(t, signal, lw=.1, c=pulse_color, zorder=0, alpha=.5, label='response')
            ax.plot(t_smooth, smoothed_signal, lw=.5, zorder=1, c=smooth_color, label='smoothed response')
            ax.scatter(t[removed_locs], smoothed_signal[removed_locs-window_offset], facecolor='None', edgecolor=not_single_color, marker='o', zorder=2, label='%d removed' % len(removed_locs))
            ax.axhline(mph[0]*copy_std, zorder=2, c=mp_color, lw=.5, label='mph=%d$\\sigma$' % (mph[0]))
            ax.legend(bbox_to_anchor=(0., 1, 1., .102), loc='lower left',
                ncols=6, mode="expand", borderaxespad=0., fontsize=9)
            made_copy = 0

    print('Analysed %d out of %d files' % (analysed_files, nr_req_files))

    t_file = len(signal)*dt / chuncksize
    t_files = analysed_files*t_file
    single_idx = np.hstack(single_idx)
    pulses = np.vstack(pulses)
    too_high_idx = np.hstack(too_high_idx)
    single_locs = np.hstack(single_locs)
    
    nr_total = len(single_idx)
    nr_single = np.sum(single_idx)
    nr_too_high = np.sum(too_high_idx)
    nr_too_close = np.sum(~single_idx) - nr_too_high

    Nph_total = nr_total / t_files
    Nph_single = nr_single / t_files
    Nph_too_close = nr_too_close / t_files
    Nph_too_high = nr_too_high / t_files

    perc_single = nr_single/nr_total * 100
    perc_too_close = nr_too_close/nr_total * 100
    perc_too_high = nr_too_high/nr_total * 100
    print('Found %d single pulses (%d%% of all detected pulses)' % (nr_single, perc_single))
    freqs, pulse_psd = ft.get_avg_psd(pulses, pw+pw_offset, sff, exclude_dc=False, onesided=False)
    pulse_template = np.mean(pulses, axis=0)
    H_opt, R_sn, _, _ = ft.optimal_filter(pulses, pulse_template, sff*1e6, 1, noise_psd, exclude_dc)
    binedges = np.histogram_bin_edges(H_opt, bins='auto')
    pulse_binsize = binedges[1] - binedges[0]
    H_0, _, _, _ = ft.optimal_filter(noises, pulse_template, sff*1e6, 1, noise_psd)
    R_opt, pdf_y, pdf_x, _, _ = ft.resolving_power(H_opt, pulse_binsize)

    tqp, _, popt = ft.fit_decaytime(pulse_template, fit_tqp)
    tqp /= sff
    tqp_x = np.linspace(round(fit_tqp[0]/sff), round(fit_tqp[1]/sff), int((fit_tqp[1] - fit_tqp[0])), endpoint=False)
    fit_x = np.arange(len(tqp_x))
    tqp_y = ft.exp_decay(fit_x, *popt)

    if plot:
        if tmax > t_files:
            tmax = t_files
        max = round(tmax/dt)+1
        plot_signal = signal[:max]
        t = np.linspace(0, tmax, len(plot_signal))
        if len(window):
            smoothed_plot_signal = fftconvolve(plot_signal, window, mode='valid')
            window_offset = int(np.argmax(window[::-1]))
            t_smooth = t[window_offset:-len(window)+window_offset+1]
        t_pulse = np.linspace(0, (pw+pw_offset)*dt, len(pulse_template)) * 1e6
        single_locs = locs[single_idx_chunck]
        too_close_locs = locs[(~single_idx_chunck) & (~too_high_chunck)]
        too_high_locs = locs[too_high_chunck]

        single_locs = single_locs[single_locs < max]
        too_close_locs = too_close_locs[too_close_locs < max]
        too_high_locs = too_high_locs[too_high_locs < max]
        title += '\n KID%d, -%d dBm' % (kid, pread)

        fig, axes = plt.subplot_mosaic('bbbb;faec', figsize=(12, 6), constrained_layout=True)
        fig.suptitle(title)
        
        ax = axes['a']
        ax.plot(t_pulse, pulse_template, c=pulse_color, label='avg. pulse', zorder=2)
        ax.plot(t_pulse, np.mean(noises, axis=0), c=noise_color, label='avg. noise', zorder=1)
        ax.set_xlabel('t [$\mu$s]')
        ax.set_ylabel(str(response))
        ax.set_xlim([t_pulse[0], t_pulse[-1]])
        ax.legend(bbox_to_anchor=(0., 1, 1., .102), loc='lower left',
                ncols=2, mode="expand", borderaxespad=0., fontsize=9)

        if sw:
            ax = axes['f']
            ax.plot(window[::-1], c=pulse_color, label='avg. pulse', zorder=2)
        
        ax = axes['e']
        ax.semilogy(t_pulse, pulse_template, c=pulse_color, label='avg. pulse', zorder=2)
        ax.semilogy(tqp_x, tqp_y, c=smooth_color, ls='--', label='$\\tau_{qp}$=%d $\mu$s' % tqp, zorder=3)
        ax.set_xlabel('t [$\mu$s]')
        ax.set_ylabel(str(response))
        ax.set_ylim([1e-3, (np.amax(pulse_template)//.1+1)*.1])
        ax.set_xlim([t_pulse[0], t_pulse[-1]])
        ax.legend(bbox_to_anchor=(0., 1, 1., .102), loc='lower left',
                ncols=2, mode="expand", borderaxespad=0., fontsize=9)

        ax = axes['b']
        if len(window):
            ax.plot(t, plot_signal, lw=.1, c=pulse_color, zorder=0, alpha=.5, label='response')
            ax.plot(t_smooth, smoothed_plot_signal, lw=.5, zorder=1, c=smooth_color, label='smoothed response')
            ax.scatter(t[single_locs], smoothed_plot_signal[single_locs-window_offset], facecolor='None', edgecolor=single_color, marker='v', zorder=2, label='single pulses $N_{ph}$=%d cps' % Nph_single)
            ax.scatter(t[too_close_locs], smoothed_plot_signal[too_close_locs-window_offset], facecolor='None', edgecolor=not_single_color, marker='o', zorder=2, label='%d%% too close' % (perc_too_close))
            ax.scatter(t[too_high_locs], smoothed_plot_signal[too_high_locs-window_offset], facecolor='None', edgecolor=not_single_color, marker='v', zorder=2, label='%d%% too high' % (perc_too_high))
        else:
            ax.plot(t, plot_signal, lw=.5, c=pulse_color, zorder=0, alpha=1, label='response')
            ax.scatter(t[single_locs], plot_signal[single_locs], facecolor='None', edgecolor=single_color, marker='v', zorder=2, label='single pulses $N_{ph}$=%d cps' % Nph_single)
            ax.scatter(t[too_close_locs], plot_signal[too_close_locs], facecolor='None', edgecolor=not_single_color, marker='o', zorder=2, label='%d%% too close' % (perc_too_close))
            ax.scatter(t[too_high_locs], plot_signal[too_high_locs], facecolor='None', edgecolor=not_single_color, marker='v', zorder=2, label='%d%% too high' % (perc_too_high))
        if isinstance(mph, np.ndarray):
            ax.axhline(mph[0]*std, zorder=2, c=mp_color, lw=.5)
            ax.axhline(mph[1]*std, zorder=2, c=mp_color, lw=.5, label='mph=[%d, %d]$\\sigma$' % (mph[0], mph[1]))
        else:
            ax.axhline(mph*std, zorder=2, c=mp_color, lw=.5, label='mph=%d$\\sigma$' % (mph))
        if isinstance(mpp, np.ndarray):
            ax.axhline(mpp[0]*std, zorder=2, c=mp_color, lw=.5, ls='--')
            ax.axhline(mpp[1]*std, zorder=2, c=mp_color, lw=.5, ls='--', label='mpp=[%d, %d]$\\sigma$' % (mpp[0], mpp[1]))
        else:
            ax.axhline(mpp*std, zorder=2, c=mp_color, lw=.5, ls='--', label='mpp=%d$\\sigma$' % (mpp))
        ax.axhline(mpp*std, zorder=2, c=mp_color, ls='--', lw=.5)
        ax.set_xlabel('t [s]')
        ax.set_ylabel(str(response))
        ax.set_xlim([t[0], t[-1]])
        ax.legend(bbox_to_anchor=(0., 1, 1., .102), loc='lower left',
                ncols=6, mode="expand", borderaxespad=0., fontsize=9)

        ax = axes['c']
        len_oneside = round(len(pulse_template) / 2) + 1
        ax.semilogx(np.absolute(freqs[exclude_dc:len_oneside]), 10*np.log10(pulse_psd[exclude_dc:len_oneside]), c=pulse_color,label='avg. pulse', zorder=1)
        ax.semilogx(np.absolute(freqs[exclude_dc:len_oneside]), 10*np.log10(noise_psd[exclude_dc:len_oneside]), c=noise_color, label='noise', zorder=0)
        ax.set_xlabel('f [Hz]')
        ax.set_ylabel('$S_{xx}$ [dBc/Hz]')
        ax.set_xlim([freqs[exclude_dc:len_oneside][0], sff*1e6/2])
        ax.grid(which='major', lw=0.5)
        ax.grid(which='minor', lw=0.2)
        ax.xaxis.get_major_locator().set_params(numticks=99)
        ax.xaxis.get_minor_locator().set_params(numticks=99, subs=np.arange(2, 10, 2)*.1)
        ax.legend(bbox_to_anchor=(0., 1, 1., .102), loc='lower left',
                ncols=2, mode="expand", borderaxespad=0., fontsize=9)

        fig, axes = plt.subplot_mosaic('d', constrained_layout=True, figsize=(6, 6))
        fig.suptitle(title)
        ax = axes['d']
        ax.hist(H_0, bins='auto', facecolor='None', edgecolor=noise_color, zorder=0, label='noise heights')
        ax.hist(H_opt, bins=binedges, facecolor='None', edgecolor=pulse_color, zorder=1, label='pulse heights')
        ax.plot(pdf_x, pdf_y, c=pulse_color, label='R=%.1f, $R_{sn}$=%.1f' % (R_opt, R_sn))
        ax.set_ylabel('counts')
        ax.set_xlabel(str(response))
        ax.legend(bbox_to_anchor=(0., 1, 1., .102), loc='lower left',
                ncols=3, mode="expand", borderaxespad=0., fontsize=9)
        
    return pulses

def get_pulses(pulse_files, ph, pp, pw, pw_offset, chuncksize, nr_chuncks, window):
    nr_files = len(pulse_files)
    if nr_chuncks:
        nr_req_files = np.amin((chuncksize * nr_chuncks, nr_files))
    else:
        nr_req_files = nr_files
    analysed_files = 0
    pulses = []
    single_idx = []
    single_locs = []
    neg_pulses = []
    neg_single_idx = []
    neg_single_locs = []
    while analysed_files < nr_req_files:
        if analysed_files + chuncksize > nr_files:
            chuncksize = nr_files - analysed_files
        amp, phase, _ = ft.get_data(pulse_files[analysed_files:analysed_files+chuncksize])
        signal = ft.coord_transformation(phase, amp, coord='circle', response='phase')
        medsignal = medfilt(signal)
        len_file = int(len(signal) / chuncksize)
        locs, _ = ft.find_pks(medsignal, ph, pp, window)
        neg_locs, _ = ft.find_pks(-medsignal, ph, pp, window)
        pulses_chunck, single_idx_chunck = ft.get_single_pulses(signal, locs, pw, 
                                                                pw_offset)
        neg_pulses_chunck, neg_single_idx_chunck = ft.get_single_pulses(signal, neg_locs, pw, 
                                                                pw_offset)
        if len(pulses_chunck):
            pulses.append(pulses_chunck)
            single_idx.append(single_idx_chunck)
            single_locs.append(locs[single_idx_chunck] + len_file*analysed_files)
        if len(neg_pulses_chunck):
            neg_pulses.append(neg_pulses_chunck)
            neg_single_idx.append(neg_single_idx_chunck)
            neg_single_locs.append(neg_locs[neg_single_idx_chunck] + len_file*analysed_files)
        analysed_files += chuncksize
        print('Analysed %d out of %d files' % (analysed_files, nr_req_files), end='\r')
    single_idx = np.hstack(single_idx)
    pulses = np.vstack(pulses)
    single_locs = np.hstack(single_locs)
    neg_single_idx = np.hstack(neg_single_idx)
    neg_pulses = np.vstack(neg_pulses)
    neg_single_locs = np.hstack(neg_single_locs)
    return pulses, neg_pulses


def filter_pulses(pulses, sw, outlier_filter, doubles_fraction):
    sw = int(sw)
    box = np.ones(sw)/sw
    if doubles_fraction == 1:
        mask_doubles = np.ones(len(pulses), dtype=bool)
    else:
        mph = np.amax(convolve(np.mean(pulses, axis=0), box , mode='full'))/doubles_fraction
        mask_doubles = np.ones(len(pulses), dtype=bool)
        for i, pulse in enumerate(pulses):
            filtered_pulse = convolve(pulse, box, mode='full')
            peaks, _ = find_peaks(filtered_pulse, height=mph, prominence=mph)
            if len(peaks)>1:
                mask_doubles[i] = 0
    mean = np.mean(pulses, axis=0)
    std = np.std(pulses, axis=0)   
    mask_outlier = np.all(np.abs(pulses - mean) < (outlier_filter * std), axis=1)
    mask = mask_doubles & mask_outlier
    print('Secondaries/Outliers/Total = %d/%d/%d' % (np.sum(~mask_doubles), np.sum(~mask_outlier), len(mask)))
    return mask


def peaks_vs_thresholds(dir, kid, pread, file_type, noise_mph, pulse_mph, pw, pw_offset, lifetime, nr_stds, chuncks, chuncksize, filter_type, fit_tqp, exclude_dc=True, outlier_filter=4.5, doubles_fraction=10, binsize=0.025):
    pulse_files, info_files = ft.get_files(dir, kid, pread, type=file_type)
    nr_files = len(pulse_files)

    amp, phase, _ = ft.get_data(pulse_files[:chuncksize])
    signal = ft.coord_transformation(phase, amp, coord='circle', response='phase')

    f0, Q, Qc, Qi, S21_min, dt, T = ft.get_info(info_files[0])
    sf = round(1 / dt)
    sff = sf / 1e6
    sw = round(lifetime * sff)
    exp_filter = ft.get_window(filter_type, sw)
    std = ft.get_sigma(signal, exp_filter)
    std_raw = ft.get_sigma(signal, [])

    ph = noise_mph*std
    pp = ph
    noise_locs, _ = ft.find_pks(signal, ph, pp, exp_filter)
    medsignal = medfilt(signal)
    filtered_signal = fftconvolve(medsignal, exp_filter, mode='valid')
    noises = ft.get_single_noises(signal, noise_locs, pw+pw_offset)

    ph = pulse_mph*std
    pp = ph
    locs, props = ft.find_pks(medsignal, ph, pp, exp_filter)
    pulses, single_idx = ft.get_single_pulses(signal, locs, pw, pw_offset)
    mask = filter_pulses(pulses, lifetime/2, outlier_filter, doubles_fraction)
    pulses = pulses[mask]
    pulse_template = np.mean(pulses, axis=0)

    norm_pulse = pulse_template / np.amax(pulse_template)
    N = len(norm_pulse)
    L = round(N/2+1)
    df = sf/N
    T = 1/(sf*N)
    freqs = np.arange(0, sf/2+df, df)
    norm_fft = fft(norm_pulse)
    pulse_psd = T * np.abs(fft(pulse_template))**2
    pulses_psd = T * np.abs(fft(pulses))**2
    noises_psd = T * np.abs(fft(noises))**2
    avg_psd_noises = np.mean(noises_psd, axis=0)
    avg_psd_pulses = np.mean(pulses_psd, axis=0)
    normalisation = np.sum((np.abs(norm_fft)**2 / avg_psd_noises)[exclude_dc:])
    sigma = np.real(np.sqrt(1 / (T * normalisation)))
    fwhm = 2.355 * sigma

    opt_filter = norm_fft.conj() / avg_psd_noises / normalisation

    def tau_qp(x, a, tqp):
        return a*np.exp(-x/tqp)
    
    print('Optimal filter constructed with %d pulses and %d noise segments' % (pulses.shape[0], noises.shape[0]))
    fig, axes = plt.subplot_mosaic('abc;def', constrained_layout=True, figsize=(10, 5))
    ax = axes['a']
    t = np.arange(0, pw+pw_offset, 1)
    ax.plot(t, pulses.T, alpha=.2, lw=.1, c='y', label='_nolegend_')
    if len(pulses):
        ax.plot(t, pulses[0], lw=.1, c='y', label='%d pulses' % len(pulses))
    ax.plot(t, pulse_template, label='mean pulse')
    ax.set_xlim([t[0], t[-1]])
    ax.set_xlabel('Time [$\mu$s]')
    ax.set_ylabel('$\\theta$ [rad]')
    _ = ax.legend()
    popt, pcov = curve_fit(tau_qp, t[fit_tqp[0]:fit_tqp[1]], pulse_template[fit_tqp[0]:fit_tqp[1]], p0=[1, 200])
    tqp = popt[1]
    x = np.arange(fit_tqp[0], fit_tqp[1])
    ax.plot(x, tau_qp(x, *popt), ls='--', lw=2, label='$\\tau_{qp}$=%d $\mu$s' % tqp, zorder=3)
    ax.legend(bbox_to_anchor=(0., 1, 1., .102), loc='lower left',
            ncols=2, mode="expand", borderaxespad=0.)
    ax_inset = inset_axes(ax, width="50%", height="50%", loc='upper right')
    ax_inset.plot(t, pulse_template, label='mean pulse')
    ax_inset.set_xlim([t[0], t[-1]])
    ax_inset.set_yscale('log')
    ax_inset.semilogy(x, tau_qp(x, *popt), ls='--', lw=2, label='$\\tau_{qp}$=%d $\mu$s' % tqp, zorder=3)
    ax = axes['b']
    ax.hist(filtered_signal, bins=np.arange(-10*std, 20*std, 0.25*std))
    ax.axvline(noise_mph*std, c='r', lw=1, label='Noise mph=%d$\\sigma$' % noise_mph)
    ax.set_yscale('log')
    xticks = np.arange(-10, 21, 1)
    ax.set_xticks(xticks*std)
    ax.set_xticklabels(xticks)
    ax.set_xlabel('sigma')

    H_opts = []
    neg_H_opts = []
    for i, nr_std in enumerate(nr_stds):
        ph = nr_std * std
        pp = nr_std * std
        pulses, neg_pulses = get_pulses(pulse_files, ph, pp, pw, pw_offset, chuncksize, chuncks[i], exp_filter)
        pulses_fft = fft(pulses, axis=1)
        neg_pulses_fft = fft(neg_pulses, axis=1)
        H_opt = np.sum((opt_filter*pulses_fft)[:, exclude_dc:], axis=1).real
        neg_H_opt = np.sum((opt_filter*neg_pulses_fft)[:, exclude_dc:], axis=1).real
        H_opts.append(H_opt)
        neg_H_opts.append(neg_H_opt)
        print('\nAnalysed chunck %d out of %d' % (i+1, len(chuncks)), end='\r')

    mask = filter_pulses(pulses, lifetime/2, outlier_filter, doubles_fraction)
    pulses = pulses[mask]
    pulses_fft = pulses_fft[mask]
    H_opts[-1] = H_opts[-1][mask]
    Rsn = np.mean(H_opt) / fwhm

    ax = axes['c']
    opt_filter_td = ifft(opt_filter[exclude_dc:]).real
    ax.plot(opt_filter_td)
    ax = axes['d']
    pulse_psd_onesided = pulse_psd[:L]
    pulse_psd_onesided[1:L-1] *= 2
    avg_psd_pulses_onesided = avg_psd_pulses[:L]
    avg_psd_pulses_onesided[1:L-1] *= 2
    avg_psd_noises_onesided = avg_psd_noises[:L]
    avg_psd_noises_onesided[1:L-1] *= 2
    ax.semilogx(freqs, 10*np.log10(pulse_psd_onesided), label='Single Pulse')
    ax.semilogx(freqs, 10*np.log10(avg_psd_pulses_onesided), label='Average Pulse')
    ax.semilogx(freqs, 10*np.log10(avg_psd_noises_onesided), label='Average Noise')
    ax = axes['e']
    bins=np.arange(-.5, 2, binsize)
    # noises = np.array([signal[i:i+pw+pw_offset] for i in range(0, len(signal)-(pw+pw_offset), pw+pw_offset)])
    noises_fft = fft(noises, axis=1)
    optimal_noise = np.sum((opt_filter*noises_fft)[:, exclude_dc:], axis=1).real
    ax.hist(optimal_noise, bins=bins, facecolor='gray', label='Noise', zorder=0)
    # ax.hist(-neg_H_opts[0], bins=bins, facecolor='r', alpha=.5, label='Neg. Pulses', zorder=1)
    print('Nr of neg pulses at lowest threshold: %d' % len(neg_H_opts[0]))
    # ax.set_yscale('log')
    colors = ['b', 'y', 'o'] 
    for i, H_opt in enumerate(H_opts):
        _ = ax.hist(H_opt, bins=bins, facecolor=colors[i], alpha=.75, label='%d$\sigma$' % (nr_stds[i]), zorder=i+1)

    max_y = np.amax(np.histogram(H_opts[-1], bins=bins)[0])
    ax.set_ylim([0, max_y*1.5])
    ax.set_xlim([-.25,2])
    ax.set_ylabel('Counts')
    ax.set_xlabel('Pulse heights [rad]')

    H_opt = H_opts[-1]
    c = colors[len(H_opts)-1]

    ax = axes['f']
    mean_pulse = np.mean(pulses, axis=0)    
    std_pulse = np.std(pulses, axis=0)
    ax.plot(t, mean_pulse, c=c, label='Mean pulse main')
    ax.fill_between(t, mean_pulse - std_pulse, mean_pulse + std_pulse, color=c, alpha=.2, label='$\pm1$ std')
    popt, pcov = curve_fit(tau_qp, t[fit_tqp[0]:fit_tqp[1]], mean_pulse[fit_tqp[0]:fit_tqp[1]], p0=[1, 200])
    tqp = popt[1]
    perr = np.sqrt(np.diag(pcov))
    y = tau_qp(x, *popt)
    ax.plot(x, y, c='k', ls='--', lw=1.5, label='fit $\\tau_{qp}$', zorder=3)
    xticks = np.arange(t[0], t[-1], 250)
    ax_inset = inset_axes(ax, width="50%", height="50%", loc='upper right')
    ax_inset.semilogy(t, mean_pulse, c=c, label='mean pulse')
    ax_inset.semilogy(x, y, c='k', ls='--', lw=1, label='fit $\\tau_{qp}$', zorder=3)
    ax_inset.set_xlim([t[0], t[-1]])
    ax_inset.set_ylim([1e-3, 2])
    ax_inset.set_xticks(xticks)
    ax_inset.set_xticklabels(xticks)
    ax.set_xticks(xticks)
    ax.set_xticklabels(xticks)
    ax.set_xlim([t[0], t[-1]])
    ax.set_ylim([-.25, 2])
    ax.set_ylabel('Pulse heights [rad]')
    ax.set_xlabel('Time [$\mu$s]')
    ax.legend(bbox_to_anchor=(0., 1, 1., .102), loc='lower left',ncols=3, mode="expand", borderaxespad=0., handlelength=1.5)

    ax = axes['e']
    R_opt, pdf_y, pdf_x, _, _ = ft.resolving_power(H_opt, binsize)
    ax.plot(pdf_x, pdf_y, c='k', ls='--', label='KDE', lw=1)
    ax.legend(bbox_to_anchor=(0., 1, 1., .102), loc='lower left',
            ncols=4, mode="expand", borderaxespad=0.)
    print('\nR = %.2f, Rsn = %.2f, %d pulses' % (R_opt, Rsn, len(pulses)))
    print('tqp = %d +- %.1f us' % (tqp, perr[1]))

    kid_object = {}
    kid_object['dir'] = dir
    kid_object['kid'] = kid
    kid_object['pread'] = pread
    kid_object['pulse template'] = norm_pulse
    kid_object['pulses'] = pulses
    kid_object['opt_filter'] = opt_filter
    kid_object['Hopts'] = H_opts
    kid_object['Hopt noise'] = optimal_noise
    kid_object['stds'] = nr_stds
    kid_object['std'] = std
    kid_object['std raw'] = std_raw
    kid_object['tqp'] = tqp
    kid_object['fit'] = [x, y]
    kid_object['Ropt'] = R_opt
    kid_object['Rsn'] = Rsn
    kid_object['kde'] = [pdf_x, pdf_y]
    kid_object['binsize'] = binsize
    kid_object['nxx'] = [freqs, avg_psd_noises_onesided]
    kid_object['dxx'] = [freqs, avg_psd_pulses_onesided]
    kid_object['mxx'] = [freqs, pulse_psd_onesided]
    kid_object['f0'] = f0
    return kid_object