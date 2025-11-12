import numpy as np
import pickle
import cvxpy as cvx
import cvxpy.settings as cvxpy_s

from cvxpy import *
from time import time

from opt_frac.problem import fun_objective, construct_problem
from opt_frac.utilities import calc_cell_dynamics_three

class InfeasibleError(RuntimeError):
    def __init__(self, status, iteration = None):
        self.status = status
        self.iteration = iteration
        if iteration is None:
            self.message = "Solver failed with status {0}".format(status)
        else:
            self.message = "Solver failed at iteration {0} with status {1}".format(iteration, status)

    def __str__(self):
        return self.message

def init_objective(nt, T_days, N_init, d_init, delta_t, ab_ratio_N = 3, lam_bed = 0, recomp = False):
    return fun_objective(nt, T_days, N_init, delta_t, slack_dyn = None, slack_rec = None, d = d_init, ab_ratio_N = ab_ratio_N, lam_bed = lam_bed, recomp = recomp).value

def init_parms(parm_dict, T_days, N0, alpha, beta, f_pro_P, T_C, T_loss, delta_t, k_m, d_init = None, ab_ratio_N = 3, M_bed = 146.67, n_scale = 1, treat_break = 0, treat_len = 1):
    T = int((T_days*24*60)/delta_t)    # Total time steps.
    delta_day = int(24*60/delta_t)     # Number of time steps per day.
    N0_P, N0_I, N0_H = N0
    M_tld_N = ab_ratio_N*M_bed + T*(0.5*ab_ratio_N)**2
    c3 = np.exp(-(np.log(2)/T_loss)*delta_t)
    
    treat_break = int(treat_break)
    treat_len = int(treat_len)
    
    if d_init is None:
        df = np.max([np.sqrt(M_tld_N/T) - 0.5*ab_ratio_N, 0])   # Optimal (constant) fraction with only P compartment.
        df = np.min([df, 1e6])   # TODO: What to do if M_bed is infinite, rendering df infinite?

        if treat_break > 0:
            t = 0
            d_init = np.zeros(T)
            for s in range(T_days // (treat_break + treat_len)):
                d_init[t*delta_day:(t + treat_len)*delta_day] = 1
                t = t + treat_len + treat_break
            if t < T_days:
                idx_end = np.min([(t + treat_len)*delta_day, T])
                d_init[t*delta_day:idx_end] = 1
            n_treat = np.sum(d_init)
            df_break = (T*df)/n_treat
            d_init = df_break*d_init
        else:
            d_init = np.repeat(df, T)
    if d_init.shape not in [(T,), (T,1)]:
        raise ValueError("d_init must be a vector of length {0}".format(T))
    if np.any(d_init < 0):
        raise ValueError("d_init must be non-negative")

    N_init, N_tld_init, N_tld_tot_init, NH_hat_init, z_init = calc_cell_dynamics_three(d_init, N0, alpha, beta, f_pro_P, T_C, T_loss, delta_t, k_m, recomp = True)
    # d_per_day_init = np.array([np.sum(d_init[t*delta_day:(t+1)*delta_day]) for t in range(T_days)])
    # bed_N_init = np.sum(d_per_day_init) + np.sum(d_per_day_init**2)/ab_ratio_N
    # obj_init = np.sum(N_init[1:,:])/nt + lam_bed*bed_N_init/T
    
    parm_dict["d_lin"].value = d_init
    parm_dict["NH_hat_norm_lin"].value = np.concatenate((0.5*c3*np.array([N0_H]), NH_hat_init))/n_scale
    parm_dict["Nv_norm_lin"].value = np.column_stack((N_init[:,0], N_init[:,2], N_init[:,4]))/n_scale
    if "Nv_tld_norm_lin" in parm_dict:
        parm_dict["Nv_tld_norm_lin"].value = np.row_stack((np.array([N0_P, N0_I, N0_H]),   # First row is filler that isn't used in constraints (only for consistent indexing).
                                                           np.column_stack((N_tld_init[:,0], N_tld_init[:,2], N_tld_init[:,4]))
                                                          ))/n_scale
    return N_init, d_init, z_init

def solve_ccp(nt, T_days, N0, alpha, beta, f_pro_P, T_C, T_loss, delta_t, k_m, d_init = None, ab_ratio_N = 3, M_bed = 146.67, frac_max_end = 1e-8, d_max_day = np.inf,
              lam_bed = 0, lam_l1 = 0, n_scale = 1, R_rec = None, has_slack_dyn = True, has_slack_rec = True, weekend_break = False, treat_break = 0, treat_len = 1, constant_dose = False,
              recomp = False, filename = None, *args, **kwargs):
    max_iter = kwargs.pop("max_iter", 1000)
    delta_stop = kwargs.pop("delta_stop", 1e-3)
    verbose = kwargs.get("verbose", False)
    
    if verbose:
        print("Constructing problem...")
    prob, var_dict, parm_dict, expr_dict = construct_problem(nt, T_days, N0, alpha, beta, f_pro_P, T_C, T_loss, delta_t, k_m,
                                                ab_ratio_N = ab_ratio_N, M_bed = M_bed, frac_max_end = frac_max_end, d_max_day = d_max_day,
                                                lam_bed = lam_bed, lam_l1 = lam_l1, n_scale = n_scale, R_rec = R_rec, has_slack_dyn = has_slack_dyn,
                                                has_slack_rec = has_slack_rec, weekend_break = weekend_break, treat_break = treat_break,
                                                treat_len = treat_len, constant_dose = constant_dose, recomp = recomp)
    
    if verbose:
        print("Initializing parameters...")
    N_init, d_init, z_init = init_parms(parm_dict, T_days, N0, alpha, beta, f_pro_P, T_C, T_loss, delta_t, k_m, d_init, ab_ratio_N, M_bed, n_scale, treat_break, treat_len)
    obj_prev = init_objective(nt, T_days, N_init, d_init, delta_t, ab_ratio_N, lam_bed, recomp)
    
    if verbose:
        print("Starting CCP loop...")
        
    k = 0
    finished = False
    obj_diff = obj_prev
    res_hist_list = []
    start_time = time()
    while not finished:
        # if k % 10 == 0 and verbose:
        # if recomp or (k % 10 == 0 and not recomp):
        #     print("CCP iteration: {0}, Objective change: {1}".format(k, obj_diff))
        print("CCP iteration: {0}, Objective change: {1}".format(k, obj_diff))
    
        # Solve linearized problem.
        prob.solve(*args, **kwargs)
        if prob.status not in cvxpy_s.SOLUTION_PRESENT:
            # raise RuntimeError("Solver failed with status {0}".format(prob.status))
            raise InfeasibleError(prob.status, k)
        
        # print("d_cons = {0}, sum(z_cons_ind) = {1}, avg(slack) = {2}".format(var_dict["d_cons"].value, np.sum(var_dict["z_cons_ind"].value), np.sum(var_dict["slack_dyn"].value)/var_dict["slack_dyn"].size))
        # print("d_per_day = {0}, avg(slack) = {1}".format(expr_dict["d_per_day"].value, np.sum(var_dict["slack_dyn"].value)/var_dict["slack_dyn"].size))
        
        # Update linearization point.
        N_norm_val = var_dict["N_norm"].value
        parm_dict["d_lin"].value = var_dict["d"].value
        parm_dict["NH_hat_norm_lin"].value = var_dict["NH_hat_norm"].value[:,0]
        parm_dict["Nv_norm_lin"].value = np.column_stack((N_norm_val[:,0], N_norm_val[:,2], N_norm_val[:,4]))
        if recomp:
            N_tld_norm_val = var_dict["N_tld_norm"].value
            parm_dict["Nv_tld_norm_lin"].value = np.column_stack((N_tld_norm_val[:,0], N_tld_norm_val[:,2], N_tld_norm_val[:,4]))
        
        # Check stopping criterion.
        obj_diff = np.abs(obj_prev - prob.value)
        obj_prev = prob.value
        
        # Save results to file.
        if filename is not None:
            res_iter = {var_name: var_cvx.value for var_name, var_cvx in var_dict.items()}
            res_iter["iteration"] = k
            res_iter["objective"] = prob.value
            # res_iter["obj_diff"] = obj_diff
            
            res_hist_list.append(res_iter)
            with open(filename, "wb") as handle:
                pickle.dump(res_hist_list, handle, protocol = pickle.HIGHEST_PROTOCOL)
        
        finished = (k + 1) >= max_iter or obj_diff <= delta_stop
        k = k + 1
    stop_time = time()
    run_time = stop_time - start_time
    
    result = {"obj": prob.value, "d": var_dict["d"].value, "N": expr_dict["N"].value, "slack_dyn": var_dict["slack_dyn"].value,
              "d_per_day": expr_dict["d_per_day"].value, "normal_bed": expr_dict["normal_bed"].value, "obj_diff": obj_diff,
              "iterations": k, "run_time": run_time}
    if recomp:
        result["z"] = expr_dict["z"].value
        result["slack_rec"] = var_dict["slack_rec"].value
    
    # if verbose:
    #     print_result(result)
    return result

def print_result(result):
    print("Optimal objective:", result["obj"])
    # print("Optimal dose vector:", result["d"])
    # print("Optimal slack:", result["slack"])
    print("Optimal dose per day:", np.squeeze(result["d_per_day"]))
    print("Average optimal slack in dynamics (over time):", np.sum(result["slack_dyn"])/result["slack_dyn"].shape[0])
    if "slack_rec" in result:
        print("Average optimal slack in recompartmentalization (over time):", np.sum(result["slack_rec"])/result["slack_rec"].shape[0])
    print("\nOptimal cell count")
    print("\tP viable = {0}, P doomed = {1}".format(result["N"][-1,0], result["N"][-1,1]))
    print("\tI viable = {0}, I doomed = {1}".format(result["N"][-1,2], result["N"][-1,3]))
    print("\tH viable = {0}, H doomed = {1}".format(result["N"][-1,4], result["N"][-1,5]))
    print("\nNormal tissue BED:", result["normal_bed"])
    print("Absolute change in objective:", result["obj_diff"])
    print("Total iterations:", result["iterations"])
    print("Elapsed runtime:", result["run_time"])
