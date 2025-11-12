import numpy as np
import cvxpy as cvx

from cvxpy import Constant, Variable, Parameter, Problem, Minimize

from opt_frac.utilities import scalar_to_vec_list


def fun_objective(nt, T_days, N, delta_t, slack_dyn = None, slack_rec = None, d = None, ab_ratio_N = 3, lam_bed = 0, lam_l1 = 0, recomp = False):
    if recomp:
        return fun_objective_rec(nt, T_days, N, delta_t, slack_dyn, slack_rec, d, ab_ratio_N, lam_bed, lam_l1)
    else:
        return fun_objective_norec(nt, T_days, N, delta_t, slack_dyn, d, ab_ratio_N, lam_bed, lam_l1)

def fun_objective_norec(nt, T_days, N, delta_t, slack = None, d = None, ab_ratio_N = 3, lam_bed = 0, lam_l1 = 0):
    T = int((T_days*24*60)/delta_t)   # Total time steps.
    delta_day = int(24*60/delta_t)    # Number of time steps per day.
    
    obj = cvx.sum(N[1:,:])/nt
    # obj = (cvx.sum(N[1:,0]) + cvx.sum(N[1:,2]) + cvx.sum(N[1:,4]))/nt   # Just penalize the viable cells.
    if slack is not None:
        # obj = obj + cvx.sum(slack)/T
        obj = obj + cvx.sum(slack)/slack.size
    # if d is not None and lam_bed != 0:
    # NOTE: Include normal tissue BED penalty term even if its value is zero (lam_bed = 0) because it helps with CCP convergence.
    if d is not None:
        # obj = obj + lam_bed*cvx.sum_squares(d + 0.5*ab_ratio_N)/T
        d_per_day = cvx.vstack([cvx.sum(d[t*delta_day:(t+1)*delta_day]) for t in range(T_days)])
        bed_N = cvx.sum(d_per_day) + cvx.sum_squares(d_per_day)/ab_ratio_N
        # obj = obj + lam_bed*bed_N/T
        # TODO: How to encourage sparsity in dose schedule without solving a MIP? Try penalizing just \sum_{i,t} d_{i,t}/T.
        obj = obj + lam_bed*bed_N/T + lam_l1*cvx.sum(d)/T
    return obj

def fun_objective_rec(nt, T_days, N, delta_t, slack_dyn = None, slack_rec = None, d = None, ab_ratio_N = 3, lam_bed = 0, lam_l1 = 0):
    T = int((T_days*24*60)/delta_t)   # Total time steps.
    delta_day = int(24*60/delta_t)    # Number of time steps per day.
    
    obj = cvx.sum(N[1:,:])/nt
    # obj = (cvx.sum(N[1:,0]) + cvx.sum(N[1:,2]) + cvx.sum(N[1:,4]))/nt   # Just penalize the viable cells.
    if slack_dyn is not None:
        obj = obj + cvx.sum(slack_dyn)/slack_dyn.size
    if slack_rec is not None:
        obj = obj + cvx.sum(slack_rec)/slack_rec.size
    
    if d is not None:
        d_per_day = cvx.vstack([cvx.sum(d[t*delta_day:(t+1)*delta_day]) for t in range(T_days)])
        bed_N = cvx.sum(d_per_day) + cvx.sum_squares(d_per_day)/ab_ratio_N
        # obj = obj + lam_bed*bed_N/T
        obj = obj + lam_bed*bed_N/T + lam_l1*cvx.sum(d)/T
    return obj

def construct_problem(nt, T_days, N0, alpha, beta, f_pro_P, T_C, T_loss, delta_t, k_m, ab_ratio_N = 3, M_bed = 146.67, frac_max_end = 1e-8, d_max_day = np.inf, lam_bed = 0, 
                      lam_l1 = 0, n_scale = 1, R_rec = None, has_slack_dyn = True, has_slack_rec = True, weekend_break = False, treat_break = 0, treat_len = 1, constant_dose = False,
                      recomp = False):
    if recomp:
        return construct_problem_rec(nt, T_days, N0, alpha, beta, f_pro_P, T_C, T_loss, delta_t, k_m, ab_ratio_N = ab_ratio_N, M_bed = M_bed, frac_max_end = frac_max_end, 
                                     d_max_day = d_max_day, lam_bed = lam_bed, lam_l1 = lam_l1, n_scale = n_scale, R_rec = R_rec, has_slack_dyn = has_slack_dyn, has_slack_rec = has_slack_rec,
                                     weekend_break = weekend_break, treat_break = treat_break, treat_len = treat_len, constant_dose = constant_dose)
    else:
        return construct_problem_norec(nt, T_days, N0, alpha, beta, f_pro_P, T_C, T_loss, delta_t, k_m, ab_ratio_N = ab_ratio_N, M_bed = M_bed, frac_max_end = frac_max_end, 
                                       d_max_day = d_max_day, lam_bed = lam_bed, lam_l1 = lam_l1, n_scale = n_scale, has_slack = has_slack_dyn, weekend_break = weekend_break,
                                       treat_break = treat_break, treat_len = treat_len, constant_dose = constant_dose)

def construct_problem_norec(nt, T_days, N0, alpha, beta, f_pro_P, T_C, T_loss, delta_t, k_m, ab_ratio_N = 3, M_bed = 146.67, frac_max_end = 1e-8, d_max_day = np.inf, lam_bed = 0, 
                            lam_l1 = 0, n_scale = 1, has_slack = True, weekend_break = False, treat_break = 0, treat_len = 1, constant_dose = False):
    # Problem constants.
    T = int((T_days*24*60)/delta_t)   # Total time steps.
    delta_day = int(24*60/delta_t)    # Number of time steps per day.
    treat_break = int(treat_break)
    treat_len = int(treat_len)
    
    if treat_break < 0:
        raise ValueError("treat_break must be a nonnegative integer (number of consecutive days of zero dose)")
    if treat_len <= 0:
        raise ValueError("treat_len must be an integer greater than or equal to 1 (number of consecutive days of treatment)")
    
    N0_P, N0_I, N0_H = N0
    alpha_P, alpha_I, alpha_H = scalar_to_vec_list(alpha, T)
    beta_P, beta_I, beta_H = scalar_to_vec_list(beta, T)
    
    c1 = np.exp(f_pro_P*(np.log(2)/T_C)*delta_t)
    c2 = c1**(2*k_m - 1)
    c3 = np.exp(-(np.log(2)/T_loss)*delta_t)
    
    # Define variables.
    d = Variable(T, nonneg = True)
    # N = Variable((T+1,6), nonneg = True)            # N_t = (N_t^{P,v}, N_t^{P,d}, N_t^{I,v}, N_t^{I,d}, N_t^{H,v}, N_t^{H,d}).
    N_norm = Variable((T+1,6), nonneg = True)         # N_t^{norm} = N_t/n_scale.
    # NH_hat = Variable((T+1,2), nonneg = True)       # \hat N_t^H = (\hat N_t^{H,v}, \hat N_t^{H,d}).
    NH_hat_norm = Variable((T+1,2), nonneg = True)    # \hat N_t^{norm,H} = \hat N_t^H/n_scale.
    
    if has_slack:
        slack = Variable((T,6), nonneg = True)
    else:
        slack = Constant(value = np.zeros((T,6)))
    
    # Define linearization parameters.
    d_lin = Parameter(T, nonneg = True)               # d_t^{(k)} for t = 1,...,T.
    # Nv_lin = Parameter((T+1,3), pos = True)         # N_t^{v,(k)} = (N_t^{P,v,(k)}, N_t^{I,v,(k)}, N_t^{H,v,(k)}).
    Nv_norm_lin = Parameter((T+1,3), pos = True)      # N_t^{norm,v,(k)} = N_t^{v,(k)}/n_scale.
    # NH_hat_lin = Parameter(T+1, pos = True)         # \hat N_t^{H,v,(k)}.
    NH_hat_norm_lin = Parameter(T+1, pos = True)      # \hat N_t^{norm,H,(k)} = \hat N_t^{H,v,(k)}/n_scale.
    
    N = N_norm*n_scale                                # N_t = (N_t^{P,v}, N_t^{P,d}, N_t^{I,v}, N_t^{I,d}, N_t^{H,v}, N_t^{H,d}).
    Nv_lin = Nv_norm_lin*n_scale                      # N_t^{v,(k)} = (N_t^{P,v,(k)}, N_t^{I,v,(k)}, N_t^{H,v,(k)}).
    
    # Define expressions.
    d_per_day = cvx.vstack([cvx.sum(d[t*delta_day:(t+1)*delta_day]) for t in range(T_days)])
    d_per_day = d_per_day[:,0]   # Flatten into single dimensional vector.
    bed_N = cvx.sum(d_per_day) + cvx.sum_squares(d_per_day)/ab_ratio_N
    
    # Define objective.
    # obj = cvx.sum(N[1:,:])/nt + cvx.sum(slack)/T + cvx.sum(d)/T
    # obj = cvx.sum(N[1:,:])/nt + cvx.sum(slack)/T + lam_bed*cvx.sum_squares(d + 0.5*ab_ratio_N)/T
    # obj = cvx.sum(N[1:,:])/nt + cvx.sum(slack)/T + lam_bed*bed_N/T
    obj = fun_objective_norec(nt, T_days, N, delta_t, slack, d, ab_ratio_N, lam_bed, lam_l1)
    
    # Define constraints.
    constr = [N_norm[0,0] == N0_P/n_scale, N_norm[0,1] == 0, N_norm[0,2] == N0_I/n_scale, N_norm[0,3] == 0, N_norm[0,4] == N0_H/n_scale, N_norm[0,5] == 0]
    for t in range(T):
        # Linear cell dynamics.
        constr += [N_norm[t+1,1] == c2*(N_norm[t,0] + N_norm[t,1]) - (c2/c1)*N_norm[t+1,0],
                   N_norm[t+1,3] == N_norm[t,3] + N_norm[t,2] - N_norm[t+1,2],
                   # N_norm[t+1,5] == c3*(N_norm[t,4] + N_norm[t,5]) - N_norm[t+1,4],
                   NH_hat_norm[t+1,0] + NH_hat_norm[t+1,1] == c3*(N_norm[t,4] + N_norm[t,5]),
                   N_norm[t+1,5] == NH_hat_norm[t+1,0] + NH_hat_norm[t+1,1] - N_norm[t+1,4]]
        
        # Nonlinear cell dynamics, with CCP linearization.
        # N_{t+1}^{P,v} = N_t^{P,v}*c1*exp(-\alpha_P*d_t - \beta_P*d_t^2).
        constr += [(alpha_P[t]*d[t] + beta_P[t]*d[t]**2 - cvx.log(N_norm[t,0]) - np.log(c1)) + (cvx.log(Nv_norm_lin[t+1,0]) + (N_norm[t+1,0] - Nv_norm_lin[t+1,0])/Nv_norm_lin[t+1,0]) <= slack[t,0],
                   (cvx.log(N_norm[t+1,0]) + alpha_P[t]*d[t] - np.log(c1)) + (-cvx.log(Nv_norm_lin[t,0]) + beta_P[t]*d_lin[t]**2 - (N_norm[t,0] - Nv_norm_lin[t,0])/Nv_norm_lin[t,0] + 2*beta_P[t]*d_lin[t]*(d[t] - d_lin[t])) >= -slack[t,1]]
        
        # N_{t+1}^{I,v} = N_t^{I,v}*exp(-\alpha_I*d_t - \beta_I*d_t^2).
        constr += [(alpha_I[t]*d[t] + beta_I[t]*d[t]**2 - cvx.log(N_norm[t,2])) + (cvx.log(Nv_norm_lin[t+1,1]) + (N_norm[t+1,2] - Nv_norm_lin[t+1,1])/Nv_norm_lin[t+1,1]) <= slack[t,2],
                   (cvx.log(N_norm[t+1,2]) + alpha_I[t]*d[t]) + (-cvx.log(Nv_norm_lin[t,1]) + beta_I[t]*d_lin[t]**2 - (N_norm[t,2] - Nv_norm_lin[t,1])/Nv_norm_lin[t,1] + 2*beta_I[t]*d_lin[t]*(d[t] - d_lin[t])) >= -slack[t,3]]
        
        # N_{t+1}^{H,v} = \hat N_{t+1}^{H,v}*exp(-\alpha_H*d_t - \beta_H*d_t^2).
        constr += [(alpha_H[t]*d[t] + beta_H[t]*d[t]**2 - cvx.log(NH_hat_norm[t+1,0])) + (cvx.log(Nv_norm_lin[t+1,2]) + (N_norm[t+1,4] - Nv_norm_lin[t+1,2])/Nv_norm_lin[t+1,2]) <= slack[t,4],
                   (cvx.log(N_norm[t+1,4]) + alpha_H[t]*d[t]) + (-cvx.log(NH_hat_norm_lin[t+1]) + beta_H[t]*d_lin[t]**2 - (NH_hat_norm[t+1,0] - NH_hat_norm_lin[t+1])/NH_hat_norm_lin[t+1] + 2*beta_H[t]*d_lin[t]*(d[t] - d_lin[t])) >= -slack[t,5]]
    
    # Final viable tumor cell constraint.
    # constr += [cvx.sum(N[-1,:])/nt <= 0.01]
    # constr += [(N[-1,0] + N[-1,2] + N[-1,4])/nt <= 1e-6]
    # constr += [(N[-delta_day-1,0] + N[-delta_day-1,2] + N[-delta_day-1,4])/nt <= 1e-4]
    # constr += [N_norm[-delta_day:,0] + N_norm[-delta_day:,2] + N_norm[-delta_day:,4] <= 1e-4*nt/n_scale]
    # constr += [N_norm[11*delta_day,2] <= 1e-6*nt/n_scale]
    if np.isfinite(frac_max_end):
        constr += [N_norm[-1,0] + N_norm[-1,2] + N_norm[-1,4] <= frac_max_end*nt/n_scale]
    
    # Normal tissue BED constraint.
    if np.isfinite(M_bed):
        constr += [bed_N <= M_bed]
    
    # Maximum dose (per day) constraint.
    if np.isfinite(d_max_day):
        # constr += [cvx.sum(d[t*delta_day:(t+1)*delta_day]) <= d_max_day for t in range(T_days)]
        constr += [d_per_day <= d_max_day]

    # Weekend break assuming we start on Monday.
    if weekend_break:
        # constr += [d[((t+1)*7-2)*delta_day:(t+1)*7*delta_day] == 0 for t in range(T_days // 7)]
        constr += [d_per_day[((t+1)*7-2):(t+1)*7] == 0 for t in range(T_days // 7)]
    
    # Consecutive dose(s) alternating with regular treatment breaks.
    if treat_break > 0:
        t = 0
        for s in range(T_days // (treat_break + treat_len)):
            # t = t + treat_len
            # constr += [d_per_day[t:(t + treat_break)] == 0]
            # t = t + treat_break
            constr += [d_per_day[(t + treat_len):(t + treat_len + treat_break)] == 0]
            t = t + treat_len + treat_break
        if t < T_days and treat_len < (T_days - t):
            # t = t + treat_len
            # constr += [d_per_day[t:] == 0]
            constr += [d_per_day[(t + treat_len):] == 0]
    
    # Dose must be a constant or zero.
    if constant_dose:
        M_cons = 1000*d_max_day if np.isfinite(d_max_day) else 1e8   # Upper limit on feasible dose per day (fraction).
        d_cons = Variable(nonneg = True)
        z_cons_ind = Variable(T, boolean = True)
        constr += [d <= z_cons_ind*M_cons, d >= d_cons - (1 - z_cons_ind)*M_cons, d <= d_cons]
        # z_cons_ind = Variable(T_days, boolean = True)   # z_t^{cons} = 1 if d_t = d^{cons}, 0 if d_t = 0.
        # constr += [d_per_day <= z_cons_ind*M_cons, d_per_day >= d_cons - (1 - z_cons_ind)*M_cons, d_per_day <= d_cons]
    
    prob = Problem(Minimize(obj), constr)
    var_dict = {"d": d, "slack_dyn": slack, "N_norm": N_norm, "NH_hat_norm": NH_hat_norm}
    if constant_dose:
        var_dict.update({"d_cons": d_cons, "z_cons_ind": z_cons_ind})
    parm_dict = {"d_lin": d_lin, "Nv_norm_lin": Nv_norm_lin, "NH_hat_norm_lin": NH_hat_norm_lin}
    expr_dict = {"N": N, "Nv_lin": Nv_lin, "d_per_day": d_per_day, "normal_bed": bed_N,
                 "NH_hat": NH_hat_norm*n_scale, "NH_hat_lin": NH_hat_norm_lin*n_scale}
    return prob, var_dict, parm_dict, expr_dict

def construct_problem_rec(nt, T_days, N0, alpha, beta, f_pro_P, T_C, T_loss, delta_t, k_m, ab_ratio_N = 3, M_bed = 146.67, frac_max_end = 1e-8, d_max_day = np.inf, lam_bed = 0, 
                          lam_l1 = 0, n_scale = 1, R_rec = None, has_slack_dyn = True, has_slack_rec = True, weekend_break = False, treat_break = 0, treat_len = 1, constant_dose = False):
    # Problem constants.
    T = int((T_days*24*60)/delta_t)   # Total time steps.
    delta_day = int(24*60/delta_t)    # Number of time steps per day.
    treat_break = int(treat_break)
    treat_len = int(treat_len)
    
    if treat_break < 0:
        raise ValueError("treat_break must be a nonnegative integer (number of consecutive days of zero dose)")
    if treat_len <= 0:
        raise ValueError("treat_len must be an integer greater than or equal to 1 (number of consecutive days of treatment)")
    
    N0_P, N0_I, N0_H = N0
    alpha_P, alpha_I, alpha_H = scalar_to_vec_list(alpha, T)
    beta_P, beta_I, beta_H = scalar_to_vec_list(beta, T)
    
    c1 = np.exp(f_pro_P*(np.log(2)/T_C)*delta_t)
    c2 = c1**(2*k_m - 1)
    c3 = np.exp(-(np.log(2)/T_loss)*delta_t)
    
    if R_rec is None:
        R_rec = c1**T
    # R_norm = R_rec/n_scale
    R_norm = R_rec
    
    # Define variables.
    d = Variable(T, nonneg = True)
    
    # Cell compartment configuration indicators.
    # z_{t,0} = 1{all cells in P compartment}. 
    # z_{t,1} = 1{P full, H empty, excess cells in I compartment}.
    # z_{t,2} = 1 - z_{t,0} - z_{t,1} = 1{P and I compartments full}.
    z_ind = Variable((T+1,2), boolean = True)
    z_PI = 1 - z_ind[:,0] - z_ind[:,1]
    z = cvx.vstack([z_ind[:,0], z_ind[:,1], z_PI]).T
    
    # N = Variable((T+1,6), nonneg = True)             # N_t = (N_t^{P,v}, N_t^{P,d}, N_t^{I,v}, N_t^{I,d}, N_t^{H,v}, N_t^{H,d}).
    N_norm = Variable((T+1,6), nonneg = True)          # N_t^{norm} = N_t/n_scale.
    # N_tld = Variable((T+1,6), nonneg = True)         # \tilde N_t = (\tilde N_t^{P,v}, \tilde N_t^{P,d}, \tilde N_t^{I,v}, \tilde N_t^{I,d}, \tilde N_t^{H,v}, \tilde N_t^{H,d}).
    N_tld_norm = Variable((T+1,6), nonneg = True)      # \tilde N_t^{norm} = \tilde N_t/n_scale.
    # NH_hat = Variable((T+1,2), nonneg = True)        # \hat N_t^H = (\hat N_t^{H,v}, \hat N_t^{H,d}).
    NH_hat_norm = Variable((T+1,2), nonneg = True)     # \hat N_t^{norm,H} = \hat N_t^H/n_scale.
    
    # Slack for cell dynamics constraints.
    if has_slack_dyn:
        slack_dyn = Variable((T,6), nonneg = True)
    else:
        slack_dyn = Constant(value = np.zeros((T,6)))
    
    # Slack for cell recompartmentalization constraints.
    if has_slack_rec:
        slack_rec = Variable((T,22), nonneg = True)
    else:
        slack_rec = Constant(value = np.zeros((T,22)))

    # Define linearization parameters.
    d_lin = Parameter(T, nonneg = True)                # d_t^{(k)} for t = 1,...,T.
    # Nv_lin = Parameter((T+1,3), pos = True)          # N_t^{v,(k)} = (N_t^{P,v,(k)}, N_t^{I,v,(k)}, N_t^{H,v,(k)}).
    Nv_norm_lin = Parameter((T+1,3), pos = True)       # N_t^{norm,v,(k)} = N_t^{v,(k)}/n_scale.
    # Nv_tld_lin = Parameter((T+1,3), pos = True)      # \tilde N_t^{v,(k)} = (\tilde N_t^{P,v,(k)}, \tilde N_t^{I,v,(k)}, \tilde N_t^{H,v,(k)}).
    Nv_tld_norm_lin = Parameter((T+1,3), pos = True)   # \tilde N_t^{norm,v,(k)} = \tilde N_t^{v,(k)}/n_scale.
    # NH_hat_lin = Parameter(T+1, pos = True)          # \hat N_t^{H,v,(k)}.
    NH_hat_norm_lin = Parameter(T+1, pos = True)       # \hat N_t^{norm,H,(k)} = \hat N_t^{H,v,(k)}/n_scale.
    
    N = N_norm*n_scale                                 # N_t = (N_t^{P,v}, N_t^{P,d}, N_t^{I,v}, N_t^{I,d}, N_t^{H,v}, N_t^{H,d}).
    N_tld = N_tld_norm*n_scale                         # \tilde N_t = (\tilde N_t^{P,v}, \tilde N_t^{P,d}, \tilde N_t^{I,v}, \tilde N_t^{I,d}, \tilde N_t^{H,v}, \tilde N_t^{H,d}). 
    Nv_lin = Nv_norm_lin*n_scale                       # N_t^{v,(k)} = (N_t^{P,v,(k)}, N_t^{I,v,(k)}, N_t^{H,v,(k)}).
    # N_tld_tot = cvx.sum(N_tld, axis = 1)             # \tilde N_t^{tot} = \tilde N_t^{P,v} + \tilde N_t^{P,d} + \tilde N_t^{I,v} + \tilde N_t^{I,d} + \tilde N_t^{H,v} + \tilde N_t^{H,d}.
    N_tld_norm_tot = cvx.sum(N_tld_norm, axis = 1)     # \tilde N_t^{norm,tot} = \tilde N_t^{tot}/n_scale.
    
    # Define expressions.
    d_per_day = cvx.vstack([cvx.sum(d[t*delta_day:(t+1)*delta_day]) for t in range(T_days)])
    d_per_day = d_per_day[:,0]   # Flatten into single dimensional vector.
    bed_N = cvx.sum(d_per_day) + cvx.sum_squares(d_per_day)/ab_ratio_N
    
    # Define objective.
    # obj = cvx.sum(N[1:,:])/nt + cvx.sum(slack_dyn)/T + cvx.sum(slack_rec)/T
    # obj = cvx.sum(N[1:,:])/nt + cvx.sum(slack_dyn)/slack_dyn.size + cvx.sum(slack_rec)/slack_rec.size
    # obj = cvx.sum(N[1:,:])/nt + cvx.sum(slack_dyn)/slack_dyn.size + cvx.sum(slack_rec)/slack_rec.size + lam_bed*cvx.sum_squares(d + 0.5*ab_ratio_N)/T
    obj = fun_objective_rec(nt, T_days, N, delta_t, slack_dyn, slack_rec, d, ab_ratio_N, lam_bed, lam_l1)
    
    # Define constraints.
    # Initial cell compartment configuration.
    if N0_P > 0 and N0_I == 0 and N0_H == 0:
        constr = [z_ind[0,0] == 1, z_ind[0,1] == 0]
    elif N0_P > 0 and N0_I > 0 and N0_H == 0:
        constr = [z_ind[0,0] == 0, z_ind[0,1] == 1]
    elif N0_P > 0 and N0_I > 0 and N0_H > 0:
        constr = [z_ind[0,0] == 0, z_ind[0,1] == 0]
    elif N0_P == 0 and N0_I == 0 and N0_H == 0:
        raise ValueError("All cell compartments empty. No treatment needed")
    else:
        raise ValueError("Invalid cell compartment configuration")
        
    constr += [z_ind[:,0] + z_ind[:,1] <= 1]   # Cell compartment configurations are mutually exclusive.
    
    constr += [N_norm[0,0] == N0_P/n_scale, N_norm[0,1] == 0, 
               N_norm[0,2] == N0_I/n_scale, N_norm[0,3] == 0, 
               N_norm[0,4] == N0_H/n_scale, N_norm[0,5] == 0]
    
    for t in range(T):
        # Linear cell dynamics.
        # \tilde N_{t+1}^{P,d} = c_2*(N_t^{P,v} + N_t^{P,d}) - (c_2/c_1)*\tilde N_{t+1}^{P,v}.
        # \tilde N_{t+1}^{I,d} = N_t^{I,v} + N_t^{I,d} - \tilde N_{t+1}^{I,v}.
        # \hat N_{t+1}^{H,v} + \hat N_{t+1}^{H,d} = c_3*(N_t^{H,v} + N_t^{H,d}).
        # \tilde N_{t+1}^{H,d} = \hat N_{t+1}^{H,v} + \hat N_{t+1}^{H,d} - \tilde N_{t+1}^{H,v}
        constr += [N_tld_norm[t+1,1] == c2*(N_norm[t,0] + N_norm[t,1]) - (c2/c1)*N_tld_norm[t+1,0],
                   N_tld_norm[t+1,3] == N_norm[t,3] + N_norm[t,2] - N_tld_norm[t+1,2],
                   NH_hat_norm[t+1,0] + NH_hat_norm[t+1,1] == c3*(N_norm[t,4] + N_norm[t,5]),
                   N_tld_norm[t+1,5] == NH_hat_norm[t+1,0] + NH_hat_norm[t+1,1] - N_tld_norm[t+1,4]]
        
        # Nonlinear cell dynamics, with CCP linearization.
        # \tilde N_{t+1}^{P,v} = N_t^{P,v}*c1*exp(-\alpha_P*d_t - \beta_P*d_t^2).
        constr += [(alpha_P[t]*d[t] + beta_P[t]*d[t]**2 - cvx.log(N_norm[t,0]) - np.log(c1)) + (cvx.log(Nv_tld_norm_lin[t+1,0]) + (N_tld_norm[t+1,0] - Nv_tld_norm_lin[t+1,0])/Nv_tld_norm_lin[t+1,0]) <= slack_dyn[t,0],
                   (cvx.log(N_tld_norm[t+1,0]) + alpha_P[t]*d[t] - np.log(c1)) + (-cvx.log(Nv_norm_lin[t,0]) + beta_P[t]*d_lin[t]**2 - (N_norm[t,0] - Nv_norm_lin[t,0])/Nv_norm_lin[t,0] + 2*beta_P[t]*d_lin[t]*(d[t] - d_lin[t])) >= -slack_dyn[t,1]]
        
        # \tilde N_{t+1}^{I,v} = N_t^{I,v}*exp(-\alpha_I*d_t - \beta_I*d_t^2).
        constr += [(alpha_I[t]*d[t] + beta_I[t]*d[t]**2 - cvx.log(N_norm[t,2])) + (cvx.log(Nv_tld_norm_lin[t+1,1]) + (N_tld_norm[t+1,2] - Nv_tld_norm_lin[t+1,1])/Nv_tld_norm_lin[t+1,1]) <= slack_dyn[t,2],
                   (cvx.log(N_tld_norm[t+1,2]) + alpha_I[t]*d[t]) + (-cvx.log(Nv_norm_lin[t,1]) + beta_I[t]*d_lin[t]**2 - (N_norm[t,2] - Nv_norm_lin[t,1])/Nv_norm_lin[t,1] + 2*beta_I[t]*d_lin[t]*(d[t] - d_lin[t])) >= -slack_dyn[t,3]]
        
        # \tilde N_{t+1}^{H,v} = \hat N_{t+1}^{H,v}*exp(-\alpha_H*d_t - \beta_H*d_t^2).
        constr += [(alpha_H[t]*d[t] + beta_H[t]*d[t]**2 - cvx.log(NH_hat_norm[t+1,0])) + (cvx.log(Nv_tld_norm_lin[t+1,2]) + (N_tld_norm[t+1,4] - Nv_tld_norm_lin[t+1,2])/Nv_tld_norm_lin[t+1,2]) <= slack_dyn[t,4],
                   (cvx.log(N_tld_norm[t+1,4]) + alpha_H[t]*d[t]) + (-cvx.log(NH_hat_norm_lin[t+1]) + beta_H[t]*d_lin[t]**2 - (NH_hat_norm[t+1,0] - NH_hat_norm_lin[t+1])/NH_hat_norm_lin[t+1] + 2*beta_H[t]*d_lin[t]*(d[t] - d_lin[t])) >= -slack_dyn[t,5]]
        
        # Recompartmentalization.
        # 1. All cells in P compartment, I and H compartments empty.
        # N_t^{P,v} + N_t^{P,d} = \tilde N_t^{tot}, \tilde N_t^{tot} <= N_0^P, N_t^{I,v} + N_t^{I,d} = 0, N_t^{H,v] + N_t^{H,d} = 0.
        constr += [N_norm[t+1,0] + N_norm[t+1,1] - N_tld_norm_tot - (1 - z[t+1,0])*R_norm <= slack_rec[t,0], 
                   N_norm[t+1,0] + N_norm[t+1,1] - N_tld_norm_tot + (1 - z[t+1,0])*R_norm >= -slack_rec[t,1],
                   N_tld_norm_tot[t+1] - N0_P/n_scale - (1 - z[t+1,0])*R_norm <= slack_rec[t,2],
                   N_norm[t+1,2] + N_norm[t+1,3] - (1 - z[t+1,0])*R_norm <= slack_rec[t,3],
                   # N_norm[t+1,2] + N_norm[t+1,3] + (1 - z[t+1,0])*R_norm >= -slack_rec[t,4] already satisfied by N_norm >= 0,
                   N_norm[t+1,4] + N_norm[t+1,5] - (1 - z[t+1,0])*R_norm <= slack_rec[t,5]]
                   # N_norm[t+1,4] + N_norm[t+1,5] + (1 - z[t+1,0])*R_norm >= -slack_rec[t,6] already satisfied by N_norm >= 0]
        
        # 2. P compartment full, H compartment empty, excess cells in I compartment.
        # N_t^{P,v} + N_t^{P,d} = N_0^P, \tilde N_t^{tot} >= N_0^P, \tilde N_t^{tot} <= N0_P + N0_I, N_t^{I,v} + N_t^{I,d} = \tilde N_t^{tot} - N_0^P, N_t^{H,v} + N_t^{H,d} = 0.
        constr += [N_norm[t+1,0] + N_norm[t+1,1] - N0_P/n_scale - (1 - z[t+1,1])*R_norm <= slack_rec[t,7],
                   N_norm[t+1,0] + N_norm[t+1,1] - N0_P/n_scale + (1 - z[t+1,1])*R_norm >= -slack_rec[t,8],
                   N0_P/n_scale - N_tld_norm_tot[t+1] - (1 - z[t+1,1])*R_norm <= slack_rec[t,9],
                   N_tld_norm_tot[t+1] - N0_P/n_scale - N0_I/n_scale - (1 - z[t+1,1])*R_norm <= slack_rec[t,10],
                   N_norm[t+1,2] + N_norm[t+1,3] - N_tld_norm_tot[t+1] + N0_P/n_scale - (1 - z[t+1,1])*R_norm <= slack_rec[t,11],
                   N_norm[t+1,2] + N_norm[t+1,3] - N_tld_norm_tot[t+1] + N0_P/n_scale + (1 - z[t+1,1])*R_norm >= -slack_rec[t,12],
                   N_norm[t+1,4] + N_norm[t+1,5] - (1 - z[t+1,1])*R_norm <= slack_rec[t,13]]
                   # N_norm[t+1,4] + N_norm[t+1,5] - (1 - z[t+1,1])*R_norm >= -slack_rec[t,14] already satisfied by N_norm >= 0]
        
        # 3. P and I compartments full, excess cells in H compartment.
        # N_t^{P,v} + N_t^{P,d} = N_0^P, N_t^{I,v} + N_t^{I,d} = N_0^I, \tilde N_t^{tot} >= N_0^P + N_0^I, N_t^{H,v} + N_t^{H,d} = \tilde N_t^{tot} - N_0^P - N_0^I.
        constr += [N_norm[t+1,0] + N_norm[t+1,1] - N0_P/n_scale - (1 - z_PI[t+1])*R_norm <= slack_rec[t,15],
                   N_norm[t+1,0] + N_norm[t+1,1] - N0_P/n_scale + (1 - z_PI[t+1])*R_norm >= -slack_rec[t,16],
                   N_norm[t+1,2] + N_norm[t+1,3] - N0_I/n_scale - (1 - z_PI[t+1])*R_norm <= slack_rec[t,17],
                   N_norm[t+1,2] + N_norm[t+1,3] - N0_I/n_scale + (1 - z_PI[t+1])*R_norm >= -slack_rec[t,18],
                   N0_P/n_scale + N0_I/n_scale - N_tld_norm_tot[t+1] - (1 - z_PI[t+1])*R_norm <= slack_rec[t,19],
                   N_norm[t+1,4] + N_norm[t+1,5] - N_tld_norm_tot[t+1] + N0_P/n_scale + N0_I/n_scale - (1 - z_PI[t+1])*R_norm <= slack_rec[t,20],
                   N_norm[t+1,4] + N_norm[t+1,5] - N_tld_norm_tot[t+1] + N0_P/n_scale + N0_I/n_scale + (1 - z_PI[t+1])*R_norm >= -slack_rec[t,21]]
    
    # Final viable tumor cell constraint.
    if np.isfinite(frac_max_end):
        constr += [N_norm[-1,0] + N_norm[-1,2] + N_norm[-1,4] <= frac_max_end*nt/n_scale]
    
    # Normal tissue BED constraint.
    if np.isfinite(M_bed):
        constr += [bed_N <= M_bed]
    
    # Maximum dose (per day) constraint.
    if np.isfinite(d_max_day):
        constr += [cvx.sum(d[t*delta_day:(t+1)*delta_day]) <= d_max_day for t in range(T_days)]
    
    # Weekend break assuming we start on Monday.
    if weekend_break:
        constr += [d[((t+1)*7-2)*delta_day:(t+1)*7*delta_day] == 0 for t in range(T_days // 7)]
    
    # Consecutive dose(s) alternating with regular treatment breaks.
    if treat_break > 0:
        t = 0
        for s in range(T_days // (treat_break + treat_len)):
            # t = t + treat_len
            # constr += [d_per_day[t:(t + treat_break)] == 0]
            # t = t + treat_break
            constr += [d_per_day[(t + treat_len):(t + treat_len + treat_break)] == 0]
            t = t + treat_len + treat_break
        if t < T_days and treat_len < (T_days - t):
            # t = t + treat_len
            # constr += [d_per_day[t:] == 0]
            constr += [d_per_day[(t + treat_len):] == 0]
    
    # Dose must be a constant or zero.
    if constant_dose:
        M_cons = 1000*d_max_day if np.isfinite(d_max_day) else 1e10   # Upper limit on feasible dose per fraction.
        d_cons = Variable(nonneg = True)
        # z_cons_ind = Variable(T, boolean = True)
        # constr += [d <= z_cons_ind*M_cons, d >= d_cons - (1 - z_cons_ind)*M_cons, d <= d_cons]
        z_cons_ind = Variable(T_days, boolean = True)
        constr += [d_per_day <= z_cons_ind*M_cons, d_per_day >= d_cons - (1 - z_cons_ind)*M_cons, d_per_day <= d_cons]
    
    prob = Problem(Minimize(obj), constr)
    var_dict = {"d": d, "z_ind": z_ind, "slack_dyn": slack_dyn, "slack_rec": slack_rec, "N_norm": N_norm, "N_tld_norm": N_tld_norm, 
                "NH_hat_norm": NH_hat_norm}
    if constant_dose:
        var_dict.update({"d_cons": d_cons, "z_cons_ind": z_cons_ind})
    parm_dict = {"d_lin": d_lin, "Nv_norm_lin": Nv_norm_lin, "Nv_tld_norm_lin": Nv_tld_norm_lin, "NH_hat_norm_lin": NH_hat_norm_lin}
    expr_dict = {"N": N, "N_tld": N_tld, "Nv_lin": Nv_lin, "N_tld_norm_tot": N_tld_norm_tot, "NH_hat": NH_hat_norm*n_scale, 
                 "NH_hat_lin": NH_hat_norm_lin*n_scale, "z": z, "d_per_day": d_per_day, "normal_bed": bed_N}
    return prob, var_dict, parm_dict, expr_dict
