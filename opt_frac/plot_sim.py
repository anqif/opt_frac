from collections import defaultdict

import numpy as np
import matplotlib.pyplot as plt

from opt_frac.simulation import primer_simulation, EQD2_simulation
from warnings import warn

###########################################################################
# <<<<<<<<<<<<<<<<<<<<<<<< EQD2_model for lung SBRT >>>>>>>>>>>>>>>>>>>>>>>
###########################################################################
# This function reports radiobiologically equivalent dose in 2Gy-weekday 
# fractionation (EQD2 model) for a specific SBRT fractionation regime.
# Specify the SBRT fractional dose and fractionation schedule below and run
# the simulation to get the EQD2 model value.

# Input variables: lung SBRT fractional dose and fractionation schedule.
# gf_in: growth fraction.
# fx_in: fractional dose in Gy on each day.
#        (e.g., three different fractions: [10, 12, 14]; same fraction all days: 10).
# schedule_in: SBRT fractionation schedule. 
#              (e.g., three fractions with every other day: [1, 3, 5]; single fx: [1]).
###########################################################################

def create_schedule(dose, delta_t = 15, only_nonzero = True, zero_cutoff = 1e-6):
    delta_day = int(24*60/delta_t)    # Number of time steps per day.
    n_days = len(dose) // delta_day
    
    # Pad out an extra day if necessary.
    if len(dose) > n_days*delta_day:
        dose = np.concatenate([dose, np.zeros(len(dose) - n_days*delta_day)])
        n_days = n_days + 1
    dose_split = np.split(dose, n_days)

    # Sum up dose delivered in each day.
    fx_in = []
    schedule_in = []
    for j in range(n_days):
        d_day_sum = np.sum(dose_split[j])
        if not only_nonzero or (d_day_sum > zero_cutoff and only_nonzero):   # Only save days when positive dose is delivered, if requested.
            fx_in.append(d_day_sum)
            schedule_in.append(j + 1)   # 1-index days in schedule.
    fx_in = np.array(fx_in)
    schedule_in = np.array(schedule_in)
    if len(schedule_in) == 0:
        return np.array([0]), np.array([0])
    else:
        return fx_in, schedule_in

def plot_schedule_line(fx, schedule, gf_in = 0.25, clf_in = 0.92, delta_t = 15, T_days = None, figsize = (12,8),
                       label = None, newfig = True, show = True, fileprefix = None):
    # delta_t = 15
    if T_days is None:
        T_days = np.max(schedule)
    delta_day = int(24*60 / delta_t)  # Number of time steps per day.

    # Consolidate and plot dose schedule.
    max_day = T_days if len(schedule) == 0 else np.max(schedule)
    days_vec = np.arange(max_day * delta_day)
    days_vec_plus_one = days_vec / delta_day + 1
    fx_vec = np.zeros(len(days_vec))
    for j in range(len(schedule)):
        s = schedule[j]
        fx_vec[(s - 1) * delta_day:s * delta_day] = fx[j]

    if newfig:
        fig = plt.figure(figsize=figsize)
        plt.title("Final Dose Schedule for GF = {0}, CLF = {1}".format(gf_in, clf_in))
        plt.xlabel("Day")
        plt.ylabel("Dose (Gy)")

    if label is not None:
        plt.plot(days_vec_plus_one, fx_vec, label=label)
    else:
        plt.plot(days_vec_plus_one, fx_vec)
    plt.ylim(bottom=0)
    if newfig and (label is not None):
        plt.legend()

    if show:
        plt.show()
    if fileprefix is not None:
        if newfig:
            fig.savefig(fileprefix + "-dose.jpg", bbox_inches="tight", dpi=300)
        else:
            plt.savefig(fileprefix + "-dose.jpg", bbox_inches="tight", dpi=300)
    plt.close()
    return fx, schedule

def plot_schedule_bar(fx, schedule, gf_in = 0.25, clf_in = 0.92, T_days = None, figsize = (12,8), label = None,
                      width = 0.75, x_delta = 0.5, newfig = True, show = True, fileprefix = None):
    if T_days is None:
        T_days = np.max(schedule)

    # Consolidate and plot dose schedule.
    max_day = T_days if len(schedule) == 0 else np.max(schedule)
    days_bar_plus_one = np.arange(max_day) + 1
    fx_bar = np.zeros(max_day)
    for j in range(len(schedule)):
        s = schedule[j]
        fx_bar[s - 1] = fx[j]

    if newfig:
        fig = plt.figure(figsize=figsize)
        plt.title("Final Dose Schedule for GF = {0}, CLF = {1}".format(gf_in, clf_in))
        plt.xlabel("Day")
        plt.ylabel("Dose (Gy)")

    if label is not None:
        plt.bar(days_bar_plus_one, fx_bar, width=width, label=label)
    else:
        plt.bar(days_bar_plus_one, fx_bar, width=width)
    # plt.xticks(list(plt.xticks()[0]) + [1, max_day])
    plt.xticks(np.arange(1, max_day + 1, 1))
    plt.xlim([1 - x_delta, max_day + x_delta])
    plt.ylim(bottom=0)
    if newfig and (label is not None):
        plt.legend()

    if show:
        plt.show()
    if fileprefix is not None:
        if newfig:
            fig.savefig(fileprefix + "-dose.jpg", bbox_inches="tight", dpi=300)
        else:
            plt.savefig(fileprefix + "-dose.jpg", bbox_inches="tight", dpi=300)
    plt.close()
    return fx, schedule

def plot_schedule(fx, schedule, gf_in = 0.25, clf_in = 0.92, figsize = (12,8), label = None, width = 0.75,
                  x_delta = 0.5, line = False, newfig = True, show = True, fileprefix = None):
    if line:
        return plot_schedule_line(fx, schedule, gf_in=gf_in, clf_in=clf_in, figsize=figsize, label=label, newfig=newfig,
                                  show=show, fileprefix=fileprefix)
    else:
        return plot_schedule_bar(fx, schedule, gf_in=gf_in, clf_in=clf_in, figsize=figsize, label=label, width=width,
                                 x_delta=x_delta, newfig=newfig, show=show, fileprefix=fileprefix)

def plot_dose_line(dose, gf_in = 0.25, clf_in = 0.92, delta_t = 15, figsize = (12,8), label = None, newfig = True,
                   show = True, fileprefix = None):
    # T_days = int(len(dose)*delta_t/(24*60))
    # delta_day = int(24*60/delta_t)    # Number of time steps per day.
    fx_in, schedule_in = create_schedule(dose, delta_t=delta_t)
    return plot_schedule_line(fx_in, schedule_in, gf_in=gf_in, clf_in=clf_in, figsize=figsize, label=label,
                              newfig=newfig, show=show, fileprefix=fileprefix)

def plot_dose_bar(dose, gf_in = 0.25, clf_in = 0.92, delta_t = 15, figsize = (12, 8), label = None, width = 0.75,
                  x_delta = 0.5, newfig = True, show = True, fileprefix = None):
    # T_days = int(len(dose) * delta_t / (24 * 60))
    # delta_day = int(24 * 60 / delta_t)  # Number of time steps per day.
    fx_in, schedule_in = create_schedule(dose, delta_t=delta_t)
    return plot_schedule_bar(fx_in, schedule_in, gf_in=gf_in, clf_in=clf_in, figsize=figsize, label=label, width=width,
                             x_delta=x_delta, newfig=newfig, show=show, fileprefix=fileprefix)

def plot_dose(dose, gf_in = 0.25, clf_in = 0.92, delta_t = 15, figsize = (12,8), label = None, width = 0.75,
              x_delta = 0.5, line = False, newfig = True, show = True, fileprefix = None):
    if line:
        return plot_dose_line(dose, gf_in=gf_in, clf_in=clf_in, delta_t=delta_t, figsize=figsize, label=label,
                              newfig=newfig, show=show, fileprefix=fileprefix)
    else:
        return plot_dose_bar(dose, gf_in=gf_in, clf_in=clf_in, delta_t=delta_t, figsize=figsize, label=label,
                             width=width, x_delta=x_delta, newfig=newfig, show=show, fileprefix=fileprefix)

def EQD2_primer_sim_step_sched(fx, schedule, gf_in = 0.25, clf_in = 0.92, alpha_p_ori = 0.305, a_over_b = 2.8, oer_i = 1.7,
                               oer_h = 1.37, plot_survival = True, figsize = (12,8), show = True, filename = None):
    sur_frac, s_sbrt, sf_sbrt = primer_simulation(fx, schedule, gf_in=gf_in, clf_in=clf_in, alpha_p_ori=alpha_p_ori,
                                                  a_over_b=a_over_b, oer_i=oer_i, oer_h=oer_h)
    eqd2, tcp, n_frac = EQD2_simulation(s_sbrt, gf_in=gf_in, clf_in=clf_in, alpha_p_ori=alpha_p_ori, a_over_b=a_over_b,
                                        oer_i=oer_i, oer_h=oer_h)

    # Plot SF over time for SBRT schedule.
    if plot_survival:
        fig = plt.figure(figsize=figsize)
        plt.semilogy(sur_frac[:,0], sur_frac[:,1], label="Total Compartment")
        plt.semilogy(sur_frac[:,0], sur_frac[:,2], label="P Compartment")
        plt.semilogy(sur_frac[:,0], sur_frac[:,3], label="I Compartment")
        plt.semilogy(sur_frac[:,0], sur_frac[:,4], label="H Compartment")
        plt.legend()
        plt.xlim([0, np.max(schedule) + 1])
        plt.ylim([1e-12, 1])
        plt.xlabel("Days")
        plt.ylabel("Surviving Fraction")
        if np.isscalar(fx):
            plt.title(
                "Cell Survival for {0} Gy x {1} (GF = {2}, CLF = {3})".format(fx, len(schedule), gf_in, clf_in))
        else:
            plt.title("Cell Survival for {0} Fractions (GF = {1}, CLF = {2})".format(len(schedule), gf_in, clf_in))
        if show:
            plt.show()
        if filename is not None:
            fig.savefig(filename, bbox_inches="tight", dpi=300)
        plt.close()
    return sur_frac, eqd2, tcp, fx, schedule

def EQD2_primer_sim_step(dose, gf_in = 0.25, clf_in = 0.92, alpha_p_ori = 0.305, a_over_b = 2.8, oer_i = 1.7, oer_h = 1.37,
                         delta_t = 15, plot_survival = True, figsize = (12,8), show = True, filename = None):
    fx_in, schedule_in = create_schedule(dose, delta_t = delta_t)
    return EQD2_primer_sim_step_sched(fx_in, schedule_in, gf_in=gf_in, clf_in=clf_in, alpha_p_ori=alpha_p_ori, a_over_b=a_over_b,
                                      oer_i=oer_i, oer_h=oer_h, plot_survival=plot_survival, figsize=figsize, show=show, filename=filename)

def plot_schedule_sf_stacked(fx, schedule, gf_in = 0.25, clf_in = 0.92, alpha_p_ori = 0.305, a_over_b = 2.8, oer_i = 1.7,
                             oer_h = 1.37, delta_t = 15, T_days = None, figsize = (8,12), title = None, days_lim = None,
                             dose_lim = None, sf_lim = None, line = False, width = 0.75, color_dict = dict(), show = True,
                             filename = None, show_subtitle = True, show_legend = True, show_xlabel = True, show_ylabel = True,
                             leg_loc = "best", xtick_step = 1, xtick_max = None, ytick_step = None, ytick_max = None):
    delta_day = int(24 * 60 / delta_t)  # Number of time steps per day.
    if np.isscalar(fx):
        fx = np.repeat(fx, len(schedule))
    if T_days is None:
        T_days = np.max(schedule)

    # Simulate cell survival using dose.
    sur_frac, s_sbrt, sf_sbrt = primer_simulation(fx, schedule, gf_in=gf_in, clf_in=clf_in, alpha_p_ori=alpha_p_ori,
                                                  a_over_b=a_over_b, oer_i=oer_i, oer_h=oer_h)
    eqd2, tcp, n_frac = EQD2_simulation(s_sbrt, gf_in=gf_in, clf_in=clf_in, alpha_p_ori=alpha_p_ori, a_over_b=a_over_b,
                                        oer_i=oer_i, oer_h=oer_h)

    # Plot survival fraction and dose schedule.
    fig, axs = plt.subplots(2, 1, figsize=figsize, sharex=True)
    for j, key in zip(np.arange(1,5), ["Total", "P", "I", "H"]):
        if key in color_dict:
            axs[0].semilogy(sur_frac[:, 0], sur_frac[:, j], label="{0} Compartment".format(key), color=color_dict[key])
        else:
            axs[0].semilogy(sur_frac[:, 0], sur_frac[:, j], label="{0} Compartment".format(key))

    # axs[0].semilogy(sur_frac[:, 0], sur_frac[:, 1], label="Total Compartment")
    # axs[0].semilogy(sur_frac[:, 0], sur_frac[:, 2], label="P Compartment")
    # axs[0].semilogy(sur_frac[:, 0], sur_frac[:, 3], label="I Compartment")
    # axs[0].semilogy(sur_frac[:, 0], sur_frac[:, 4], label="H Compartment")

    if show_legend:
        axs[0].legend(loc=leg_loc)
        # axs[0].legend(fontsize="large")
    if show_subtitle:
        axs[0].set_title("Cell Survival")
    if days_lim:
        axs[0].set_xlim(days_lim)
    else:
        axs[0].set_xlim([0, np.max(schedule) + 1])
    if sf_lim:
        axs[0].set_ylim(sf_lim)
    else:
        axs[0].set_ylim([1e-12, 1])
    if show_ylabel:
        axs[0].set_ylabel("Survival Fraction")
        # axs[0].set_ylabel("Survival Fraction", fontsize="x-large")

    # Consolidate dose schedule.
    max_day = T_days if len(schedule) == 0 else np.max(schedule)
    days_bar_plus_one = np.arange(max_day) + 1
    fx_bar = np.zeros(max_day)
    days_vec = np.arange(max_day * delta_day)
    days_vec_plus_one = days_vec / delta_day + 1
    fx_vec = np.zeros(len(days_vec))
    for j in range(len(schedule)):
        s = schedule[j]
        fx_bar[s - 1] = fx[j]
        fx_vec[(s - 1) * delta_day:s * delta_day] = fx[j]

    days_vec_pad_zero = np.concatenate([np.arange(delta_day) / delta_day, days_vec_plus_one])
    fx_vec_pad_zero = np.concatenate([np.zeros(delta_day), fx_vec])

    if line:
        axs[1].plot(days_vec_pad_zero, fx_vec_pad_zero)
    else:
        axs[1].bar(days_bar_plus_one, fx_bar, width=width)
        if days_lim:
            xtick_max_label = int(days_lim[1]) if xtick_max is None else xtick_max
            axs[1].set_xticks(np.arange(max(days_lim[0], 1), xtick_max_label + xtick_step, xtick_step))
        else:
            xtick_max_label = max_day if xtick_max is None else max_day
            axs[1].set_xticks(np.arange(1, xtick_max_label + xtick_step, xtick_step))
            # axs[1].set_xticks(np.concatenate([np.array([1]), np.arange(2, max_day + 1, xtick_step), np.array([max_day])]))

        if ytick_step is not None:
            max_dose = np.max(fx_bar) if ytick_max is None else ytick_max
            axs[1].set_yticks(np.arange(0, max_dose + ytick_step, ytick_step))
    if show_subtitle:
        axs[1].set_title("Dose Schedule")
    if show_ylabel:
        axs[1].set_ylabel("Dose (Gy)")
        # axs[1].set_ylabel("Dose (Gy)", fontsize="x-large")
    if dose_lim:
        axs[1].set_ylim(dose_lim)
    else:
        axs[1].set_ylim(bottom=0)

    axs[1].tick_params(axis="x")
    axs[0].tick_params(axis="y")
    axs[1].tick_params(axis="y")
    # axs[1].tick_params(axis="x", labelsize="large")
    # axs[0].tick_params(axis="y", labelsize="large")
    # axs[1].tick_params(axis="y", labelsize="large")

    if title is not None:
        fig.suptitle(title)
    if show_xlabel:
        fig.supxlabel("Days")
        # fig.supxlabel("Days", fontsize="x-large")
    fig.tight_layout()

    if show:
        plt.show()
    if filename is not None:
        fig.savefig(filename, bbox_inches="tight", dpi=300)
    plt.close()

def plot_dose_sf_stacked(dose, gf_in = 0.25, clf_in = 0.92, alpha_p_ori = 0.305, a_over_b = 2.8, oer_i = 1.7,
                         oer_h = 1.37, delta_t = 15, figsize = (8,12), title = None, dose_lim = None,
                         sf_lim = None, line = False, width = 0.75, color_dict = dict(), show = True, filename = None,
                         show_subtitle = True, show_legend = True, show_xlabel = True, show_ylabel = True):
    T_days = int(len(dose)*delta_t/(24*60))
    fx_in, schedule_in = create_schedule(dose, delta_t = delta_t)
    plot_schedule_sf_stacked(fx_in, schedule_in, gf_in=gf_in, clf_in=clf_in, alpha_p_ori=alpha_p_ori, a_over_b=a_over_b,
                             oer_i=oer_i, oer_h=oer_h, delta_t=delta_t, T_days=T_days, figsize=figsize, title=title,
                             dose_lim=dose_lim, sf_lim=sf_lim, line=line, width=width, color_dict=color_dict, show=show,
                             filename=filename, show_subtitle=show_subtitle, show_legend=show_legend,
                             show_xlabel=show_xlabel, show_ylabel=show_ylabel)

def EQD2_primer_sim_comp(d_list, gf_list = 0.25, clf_list = 0.92, ab_ratio_N = 3, delta_t = 15, label_list = None,
                         figsize = (12,8), show = True, fileprefix = None):
    # delta_t = 15
    delta_day = int(24*60/delta_t)    # Number of time steps per day.
    
    K = len(d_list)
    if not np.isscalar(gf_list) and len(gf_list) != K:
        raise ValueError("gf_list must be a scalar or vector of length {0}".format(K))
    if not np.isscalar(clf_list) and len(clf_list) != K:
        raise ValueError("clf_list must be a scalar or vector of length {0}".format(K))
    
    if label_list is None:
        label_list = []
        for k in range(K):
            label = []
            if not np.isscalar(gf_list):
                label += ["GF = {0}".format(gf_list[k])]
            if not np.isscalar(clf_list):
                label += ["CLF = {0}".format(clf_list[k])]
            label += ["BED3 = {:.2f}"]
            label = ", ".join(map(str, label))
            label_list.append(label)
    if len(label_list) != K:
        raise ValueError("label_list must be a list of length {0}".format(K))
    
    title_detail = []
    if np.isscalar(gf_list):
        title_detail += ["GF = {0}".format(gf_list)]
        gf_list = K*[gf_list]
    if np.isscalar(clf_list):
        title_detail += ["CLF = {0}".format(clf_list)]
        clf_list = K*[clf_list]
    title_detail = ", ".join(map(str, title_detail))
    
    if len(gf_list) != K:
        raise ValueError("gf_list must be a list of length {0}".format(K))
    if len(clf_list) != K:
        raise ValueError("clf_list must be a list of length {0}".format(K))

    days_list = []    
    fx_list = []
    fx_vec_list = []
    schedule_list = []
    nbed_list = []
    for dose in d_list:
        # Generate dose schedule by day.
        if isinstance(dose, dict):
            fx_in = np.array(dose["fx"])
            schedule_in = np.array(dose["schedule"])
        else:
            fx_in, schedule_in = create_schedule(dose, delta_t = delta_t)
        nbed = np.sum(fx_in*(1 + fx_in/ab_ratio_N))
         
        # Consolidate dose schedule.
        max_day = np.max(schedule_in)
        days_vec = np.arange(max_day*delta_day)
        days_vec_plus_one = days_vec/delta_day + 1
        fx_vec = np.zeros(len(days_vec))
        for j in range(len(schedule_in)):
            s = schedule_in[j]
            fx_vec[(s - 1)*delta_day:s*delta_day] = fx_in[j]
        
        days_list.append(days_vec_plus_one)
        fx_list.append(fx_in)
        fx_vec_list.append(fx_vec)
        schedule_list.append(schedule_in)
        nbed_list.append(nbed)
    schedule_max = np.max(np.concatenate(schedule_list))
    
    # Plot and compare dose schedules.
    fig = plt.figure(figsize = figsize)
    for k in range(K):
        # plt.plot(days_list[k], fx_vec_list[k], label = "GF = {0}, CLF = {1}, BED3 = {2:.2f}".format(gf_list[k], clf_list[k], nbed_list[k]))
        plt.plot(days_list[k], fx_vec_list[k], label = label_list[k].format(nbed_list[k]))
    plt.legend()
    if len(title_detail) == 0:
        plt.title("Comparison of Final Dose Schedules")
    else:
        plt.title("Comparison of Final Dose Schedules ({0})".format(title_detail))
    plt.xlabel("Day")
    plt.ylabel("Dose (Gy)")
    if show:
        plt.show()
    if fileprefix is not None:
        fig.savefig(fileprefix + "-dose.jpg", bbox_inches = "tight", dpi = 300)

    # Lung patient simulation with primer shot schedule.
    sur_frac_list = []
    for k in range(K):
        sur_frac, s_sbrt, sf_sbrt = primer_simulation(fx_list[k], schedule_list[k], gf_in = gf_list[k], clf_in = clf_list[k])
        sur_frac_list.append(sur_frac)
    
    # Plot SF over time for different compartments.
    comp_idxs = [1, 2, 3, 4]
    comp_names = ["Total Compartment", "P Compartment", "I Compartment", "H Compartment"]
    file_suffixes = ["sf_total", "sf_p", "sf_i", "sf_h"]
    
    for idx, name, suffix in zip(comp_idxs, comp_names, file_suffixes):
        fig = plt.figure(figsize = figsize)
        for k in range(K):
            # plt.semilogy(sur_frac_list[k][:,0], sur_frac_list[k][:,idx], label = "GF = {0}, CLF = {1}, BED3 = {2:.2f}".format(gf_list[k], clf_list[k], nbed_list[k]))
            plt.semilogy(sur_frac_list[k][:,0], sur_frac_list[k][:,idx], label = label_list[k].format(nbed_list[k]))
        plt.legend()
        if len(title_detail) == 0:
            plt.title("Comparison of Cell Survival Rates for {0}".format(name))
        else:
            plt.title("Comparison of Cell Survival Rates for {0} ({1})".format(name, title_detail))
        plt.xlim([0, schedule_max + 1])
        plt.ylim([1e-12, 1])
        plt.xlabel("Days")
        plt.ylabel("Surviving Fraction")
        if show:
            plt.show()
        if fileprefix is not None:
            fig.savefig(fileprefix + "-{0}.jpg".format(suffix), bbox_inches = "tight", dpi = 300)
    plt.close()
