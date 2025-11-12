from warnings import warn

import numpy as np
import matplotlib.pyplot as plt
from copy import deepcopy

def scalar_to_vec(x, T):
    if np.isscalar(x):
        return np.repeat(x, T)
    else:
        if len(x) == T:
            return x
        else:
            raise ValueError("x must be a scalar or vector of length {0}".format(T))

def scalar_to_vec_list(x_list, T):
    return [scalar_to_vec(x, T) for x in x_list]

def scalar_to_vec_dict(x_dict, T):
    return {k: scalar_to_vec(v, T) for k, v in x_dict.items()}

def convert_dose_delta(d_in, T_days, delta_t_in, delta_t_out):
    if delta_t_out < delta_t_in:
        raise NotImplementedError

    if delta_t_in == delta_t_out:
        return deepcopy(d_in)
    
    delta_step = int(delta_t_out/delta_t_in)
    T_in = int((T_days*24*60)/delta_t_in)
    T_out = int((T_days*24*60)/delta_t_out)
    
    # d_out = np.zeros(T_out)
    # idx = 0
    # for t in range(T_out):
    #     idx_end = np.min([idx + delta_step, T_in])
    #     d_out[t] = np.sum(d_in[idx:(idx + delta_step)])
    #     idx = idx + delta_step
    
    nrow_reshape = int(T_in/delta_step)
    size_reshape = nrow_reshape*delta_step
    d_in_reshape = np.reshape(d_in[:size_reshape], (nrow_reshape, delta_step))
    d_out = np.sum(d_in_reshape, axis = 1)
    if len(d_in) > size_reshape:
        d_sum_last = np.sum(d_in[size_reshape:])
        d_out = np.concatenate([d_out, np.array([d_sum_last])])
    return d_out

def calc_normal_bed(d, T_days, delta_t, ab_ratio_N = 3):
    T_days = int(T_days)
    delta_day = int(24 * 60 / delta_t)  # Number of time steps per day.
    d_per_day = np.array([np.sum(d[t * delta_day:(t + 1) * delta_day]) for t in range(T_days)])
    # nbed = np.sum(d_per_day) + np.sum(d_per_day**2) / ab_ratio_N
    nbed = calc_normal_bed_vec(d_per_day, ab_ratio_N)
    return nbed

def calc_normal_bed_vec(d_per_day, ab_ratio_N = 3):
    return np.sum(d_per_day) + np.sum(d_per_day**2) / ab_ratio_N

def calc_normal_bed_const(d_const, num_fracs, ab_ratio_N = 3):
    # return num_fracs * d_const * (1 + d_const / ab_ratio_N)
    return num_fracs * d_const + (num_fracs * d_const**2) / ab_ratio_N

def calc_normal_bed_sched(fx, schedule, ab_ratio_N = 3):
    if np.isscalar(fx):
        return calc_normal_bed_const(fx, len(schedule), ab_ratio_N)
    else:
        return calc_normal_bed_vec(np.array(fx), ab_ratio_N)

def calc_cell_dynamics(d, N0_P, N0_I, f_pro_P, T_C, delta_t, k_m, alpha_P, beta_P, alpha_I, beta_I, recomp = True):
    T = len(d)
    N = np.zeros((T+1,4))
    N_tld = np.zeros((T,4))
    N_tld_tot = np.zeros(T)
    z = np.zeros(T)

    alpha_P, alpha_I = scalar_to_vec_list([alpha_P, alpha_I], T)
    beta_P, beta_I = scalar_to_vec([beta_P, beta_I], T)
    
    c1 = np.exp(f_pro_P*(np.log(2)/T_C)*delta_t)
    c2 = c1**(2*k_m - 1)
    # c1 = (2)**(f_pro_P*delta_t/T_C)
    # c2 = (2)**(f_pro_P*(2*k_m - 1)*delta_t/T_C)
    
    N[0,0] = N0_P
    N[0,2] = N0_I
    for t in range(T):
        # Change in f_pro_P (k_p) as blood supply improves.
        # f_pro_P = 1 - 0.5*(N[t,0] + N[t,1])/N0_P
    
        # Intermediate cell update.
        N_tld[t,0] = N[t,0]*c1*np.exp(-alpha_P[t]*d[t] - beta_P[t]*d[t]**2)
        N_tld[t,1] = c2*(N[t,1] + N[t,0] - N_tld[t,0]/c1)
        N_tld[t,2] = N[t,2]*np.exp(-alpha_I[t]*d[t] - beta_I[t]*d[t]**2)
        N_tld[t,3] = N[t,3] + N[t,2] - N_tld[t,2]
        N_tld_tot[t] = np.sum(N_tld[t,:])
        
        # Recompartmentalization.
        if recomp:
            NP_tot = min(N_tld_tot[t], N0_P)
            NI_tot = max(N_tld_tot[t] - N0_P, 0)

            # Maintain ratio of doomed to viable cells in each compartment.
            # Note: In the initial update, we split the cells evenly.
            # P compartment.
            if t == 0 or N[t,0] == N[t,1]:
                N[t+1,0] = 0.5*NP_tot
                N[t+1,1] = 0.5*NP_tot
            elif N[t,0] == 0:
                N[t+1,0] = 0
                N[t+1,1] = NP_tot
            else:
                NP_ratio = N[t,1]/N[t,0]   # N_{t-1}^{P,d}/N_{t-1}^{P,v}.
                N[t+1,0] = NP_tot/(1 + NP_ratio)
                N[t+1,1] = NP_ratio*N[t+1,0]
            
            # I compartment.
            if t == 0 or N[t,2] == N[t,3]:
                N[t+1,2] = 0.5*NI_tot
                N[t+1,3] = 0.5*NI_tot
            elif N[t,2] == 0:
                N[t+1,2] = 0
                N[t+1,3] = NI_tot
            else:
                NI_ratio = N[t,3]/N[t,2]   # N_{t-1}^{I,d}/N_{t-1}^{I,v}.
                N[t+1,2] = NI_tot/(1 + NI_ratio)
                N[t+1,3] = NI_ratio*N[t+1,2]
        else:
            N[t+1,:] = N_tld[t,:]
        
        # Indicator of whether all cells are in P compartment (no spillover into I).
        z[t] = int(N_tld_tot[t] <= N0_P)
    return N, N_tld, N_tld_tot, z

def calc_cell_dynamics_three(d, N0, alpha, beta, f_pro_P, T_C, T_loss, delta_t, k_m, recomp = True):
    T = len(d)
    N = np.zeros((T+1,6))
    N_tld = np.zeros((T,6))
    N_tld_tot = np.zeros(T)
    NH_hat = np.zeros(T)

    N0_P, N0_I, N0_H = N0
    alpha_P, alpha_I, alpha_H = scalar_to_vec_list(alpha, T)
    beta_P, beta_I, beta_H = scalar_to_vec_list(beta, T)
    
    # State indicator z_{t,s} = I(cells are in state s at time t),
    # where s = 1 (all cells in P), 2 (P full and H empty), or 3 (P and I full).
    z = np.zeros((T,3))
    
    c1 = np.exp(f_pro_P*(np.log(2)/T_C)*delta_t)
    c2 = c1**(2*k_m - 1)
    c3 = np.exp(-(np.log(2)/T_loss)*delta_t)
    # c1 = (2)**(f_pro_P*delta_t/T_C)
    # c2 = (2)**(f_pro_P*(2*k_m - 1)*delta_t/T_C)
    # c3 = (2)**(-delta_t/T_hloss)
    
    # N_0 = (N_0^P, N_0^I, N_0^H).
    N[0,0] = N0_P
    N[0,2] = N0_I
    N[0,4] = N0_H
    for t in range(T):
        # Change in f_pro_P (k_p) as blood supply improves.
        # f_pro_P = 1 - 0.5*(N[t,0] + N[t,1])/N0_P
    
        # Intermediate cell update.
        # N_t = (N_t^{P,v}, N_t{P,d}, N_t^{I,v}, N_t{I,d}, N_t^{H,v}, N_t{H,d}).
        N_tld[t,0] = N[t,0]*c1*np.exp(-alpha_P[t]*d[t] - beta_P[t]*d[t]**2)
        N_tld[t,1] = c2*(N[t,1] + N[t,0] - N_tld[t,0]/c1)

        N_tld[t,2] = N[t,2]*np.exp(-alpha_I[t]*d[t] - beta_I[t]*d[t]**2)
        N_tld[t,3] = N[t,3] + N[t,2] - N_tld[t,2]

        N_tld[t,4] = 0.5*c3*(N[t,4] + N[t,5])*np.exp(-alpha_H[t]*d[t] - beta_H[t]*d[t]**2)
        N_tld[t,5] = c3*(N[t,4] + N[t,5]) - N_tld[t,4]

        N_tld_tot[t] = np.sum(N_tld[t,:])
        NH_hat[t] = 0.5*c3*(N[t,4] + N[t,5])
        
        # Recompartmentalization.
        if recomp:
            if N_tld_tot[t] <= N0[0]:   # All cells in P compartment.
                NP_tot = N_tld_tot[t]
                NI_tot = 0
                NH_tot = 0
                z[t,0] = 1
            elif N_tld_tot[t] > N0[0] and N_tld_tot[t] <= N0[0] + N0[1]:   # P compartment full, H compartment empty.
                NP_tot = N0[0]
                NI_tot = N_tld_tot[t] - N0[0]
                NH_tot = 0
                z[t,1] = 1
            elif N_tld_tot[t] > N0[0] + N0[1]:   # P and I compartments full.
                NP_tot = N0[0]
                NI_tot = N0[1]
                NH_tot = N_tld_tot[t] - N0[0] - N0[1]
                z[t,2] = 1

            # Maintain ratio of doomed to viable cells in each compartment.
            # Note: In the initial update, we split the cells evenly.
            # P compartment.
            if t == 0 or N[t,0] == N[t,1]:
                N[t+1,0] = 0.5*NP_tot
                N[t+1,1] = 0.5*NP_tot
            elif N[t,0] == 0:
                N[t+1,0] = 0
                N[t+1,1] = NP_tot
            else:
                NP_ratio = N[t,1]/N[t,0]   # N_{t-1}^{P,d}/N_{t-1}^{P,v}.
                N[t+1,0] = NP_tot/(1 + NP_ratio)
                N[t+1,1] = NP_ratio*N[t+1,0]
            
            # I compartment.
            if t == 0 or N[t,2] == N[t,3]:
                N[t+1,2] = 0.5*NI_tot
                N[t+1,3] = 0.5*NI_tot
            elif N[t,2] == 0:
                N[t+1,2] = 0
                N[t+1,3] = NI_tot
            else:
                NI_ratio = N[t,3]/N[t,2]   # N_{t-1}^{I,d}/N_{t-1}^{I,v}.
                N[t+1,2] = NI_tot/(1 + NI_ratio)
                N[t+1,3] = NI_ratio*N[t+1,2]
            
            # H compartment.
            if t == 0 or N[t,4] == N[t,5]:
                N[t+1,4] = 0.5*NH_tot
                N[t+1,5] = 0.5*NH_tot
            elif N[t,4] == 0:
                N[t+1,4] = 0
                N[t+1,5] = NH_tot
            else:
                NH_ratio = N[t,5]/N[t,4]   # N_{t-1}^{H,d}/N_{t-1}^{H,v}.
                N[t+1,4] = NH_tot/(1 + NH_ratio)
                N[t+1,5] = NH_ratio*N[t+1,4]
        else:
            N[t+1,:] = N_tld[t,:]
    return N, N_tld, N_tld_tot, NH_hat, z

def calc_frac_const(T_days, nbed, ab_ratio_N = 3):
    # Normal tissue BED (NBED) = sum_{t=1}^T (d_t + beta_N/alpha_N d_t^2).
    # For constant fractions, we solve the quadratic equation:
    #   beta_N/alpha_N*T*d_const^2 + T*d_const - NBED = 0
    #   d_const^2 + alpha_N/beta_N*d_const - (alpha_N/beta_N)*(NBED/T) = 0
    if T_days == 0:
        return 0

    fx_const = (-ab_ratio_N + np.sqrt(ab_ratio_N**2 + 4*ab_ratio_N*nbed/T_days))/2
    if fx_const < 0:
        warn("No positive root detected. Setting constant fraction dose to zero.")
    return np.maximum(fx_const, 0)

def calc_frac_const_sched(sched, nbed, ab_ratio_N = 3):
    return calc_frac_const(len(sched), nbed, ab_ratio_N)

def plot_plan_results(d_opt, N_opt, T, delta_t, N0_P, N0_I, model_name = "(P,I) Model", figsize = (12,8), show = True, file_prefix = None):
    prop_cycle = plt.rcParams['axes.prop_cycle']
    colors_def = prop_cycle.by_key()['color']

    # Time steps in days.
    t_days = (delta_t/(24*60))*np.arange(1,T+1)
    t_days_with_zero = np.concatenate((np.array([0]), t_days))

    # Plot optimal dose.
    fig = plt.figure(figsize = figsize)
    plt.plot(t_days, d_opt)
    plt.title("Optimal Dose for {0}".format(model_name))
    plt.xlabel("Time (days)")
    plt.ylabel("Dose (Gy)")
    if show:
        plt.show()
    if file_prefix is not None:
        fig.savefig(file_prefix + "-dose.jpg", bbox_inches = "tight", dpi = 300)
    
    # Plot survival fraction.
    Nv_opt_vec = N_opt[:,0] + N_opt[:,2]   # N_t^{P,v} + N_t^{I,v}.
    sf_opt_vec = Nv_opt_vec/(N0_P + N0_I)
    
    fig = plt.figure(figsize = figsize)
    # plt.plot(t_days_with_zero, sf_opt_vec)
    plt.semilogy(t_days_with_zero, N_opt[:,0]/(N0_P + N0_I), label = "P Compartment", color = colors_def[0], linestyle = "-")
    plt.semilogy(t_days_with_zero, N_opt[:,2]/(N0_P + N0_I), label = "I Compartment", color = colors_def[1], linestyle = "-")
    plt.semilogy(t_days_with_zero, sf_opt_vec, label = "All Compartments", color = colors_def[2], linestyle = "-")
    # plt.ylim([0, 1.1])
    plt.title("Survival Fraction for {0}".format(model_name))
    plt.legend()
    plt.xlabel("Time (days)")
    plt.ylabel("Fraction of Viable Cells")
    if show:
        plt.show()
    if file_prefix is not None:
        fig.savefig(file_prefix + "-sf.jpg", bbox_inches = "tight", dpi = 300)

    # Plot total cells in each compartment.
    # P_tot_vec = N_opt[:,0] + N_opt[:,1]
    # I_tot_vec = N_opt[:,2] + N_opt[:,3]
    # PI_max = np.max([P_tot_vec, I_tot_vec])
    
    # fig = plt.figure(figsize = figsize)
    # plt.plot(t_days_with_zero, P_tot_vec, label = "Total P Cells", color = colors_def[0], linestyle = "-")
    # plt.plot(t_days_with_zero, I_tot_vec, label = "Total I Cells", color = colors_def[1], linestyle = "-")
    # plt.axhline(N0_P, color = "gray", linestyle = ":", label = "P Capacity")
    # plt.ylim([0, 1.1*PI_max])
    # plt.title("Total Tumor Cells for {0}".format(model_name))
    # plt.legend()
    # plt.xlabel("Time (days)")
    # plt.ylabel("Number of Cells in Compartment")
    # if show:
    #     plt.show()
    # if file_prefix is not None:
    #     fig.savefig(file_prefix + "-cell_total.jpg", bbox_inches = "tight", dpi = 300)
    
    # Plot number of viable cells in each compartment.
    Nv_stack = np.column_stack([N_opt[:,0], N_opt[:,2]])
    
    fig = plt.figure(figsize = figsize)
    plt.plot(t_days_with_zero, N_opt[:,0], label = "P Compartment", color = colors_def[0], linestyle = "-")
    plt.plot(t_days_with_zero, N_opt[:,2], label = "I Compartment", color = colors_def[1], linestyle = "-")
    plt.axhline(N0_P, color = "gray", linestyle = ":", label = "P Capacity")
    plt.ylim([0, 1.1*np.max(Nv_stack)])
    plt.title("Tumor Cells by Compartment for {0}".format(model_name))
    plt.legend()
    plt.xlabel("Time (days)")
    plt.ylabel("Number of Viable Cells")
    if show:
        plt.show()
    if file_prefix is not None:
        fig.savefig(file_prefix + "-cell_comp.jpg", bbox_inches = "tight", dpi = 300)
    
    # Plot number of viable/doomed cells in each compartment.
    # colors = [colors_def[0], colors_def[0], colors_def[1], colors_def[1]]
    # lstyles = ["-", "--", "-", "--"]
    # labels = ["P (viable)", "P (doomed)", "I (viable)", "I (doomed)"]
    
    # fig = plt.figure(figsize = figsize)
    # for j in range(len(labels)):
    #     plt.plot(t_days_with_zero, N_opt[:,j], label = labels[j], color = colors[j], linestyle = lstyles[j])
    # plt.axhline(N0_P, color = "gray", linestyle = ":", label = "P Capacity")
    # plt.ylim([0, 1.1*np.max(N_opt)])
    # plt.title("Tumor Cells by Compartment for {0}".format(model_name))
    # plt.legend()
    # plt.xlabel("Time (days)")
    # plt.ylabel("Number of Cells in Compartment")
    # if show:
    #     plt.show()
    # if file_prefix is not None:
    #     fig.savefig(file_prefix + "-cell_comp.jpg", bbox_inches = "tight", dpi = 300)