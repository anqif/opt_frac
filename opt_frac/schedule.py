import numpy as np
import cvxpy as cvx
import cvxpy.settings as cvxpy_s
import pickle

from time import time
from cvxpy import Constant, Variable, Parameter, Problem, Minimize

from opt_frac.optimization import init_objective, InfeasibleError
from opt_frac.problem import fun_objective_norec, fun_objective_rec
from opt_frac.utilities import calc_cell_dynamics_three, scalar_to_vec_list

# Default.
NH_HAT_NORM_LIN_INIT_OFFSET = 0
NV_NORM_LIN_INIT_OFFSET = np.array([0, 0, 0])
NV_TLD_NORM_LIN_INIT_OFFSET = np.array([0, 0, 0])

# test_paper_sims.test_nbed_high: Scaling causes Nv_norm_lin[:,2] and NH_hat_norm_lin to be too close to zero for
#   cvx.log to handle in initial solve, so add a small positive constant to compensate.
# NH_HAT_NORM_LIN_INIT_OFFSET = 1e-4
# NV_NORM_LIN_INIT_OFFSET = np.array([0, 0, 1e-4])
# NV_TLD_NORM_LIN_INIT_OFFSET = np.array([0, 0, 0])

# TODO: Solve for best plan with constant fractions delivered daily + weekend break,
#  e.g., T_days = 14, schedule = [1, 2, 3, 4, 5, 8, 9, 10, 11, 12].

# Allow user to define arbitrary treatment schedule, e.g., [1, 3, 5], (but with potentially different fractions).
def construct_problem_sched(nt, T_days, N0, alpha, beta, f_pro_P, T_C, T_loss, delta_t, k_m, ab_ratio_N = 3,
                            M_bed = 146.67, frac_max_end = 1e-8, d_max_day = np.inf, lam_bed = 0,  lam_l1 = 0,
                            n_scale = 1, R_rec = None, has_slack_dyn = True, has_slack_rec = True, schedule = None,
                            constant_dose = False, recomp = False):
    if recomp:
        return construct_problem_sched_rec(nt, T_days, N0, alpha, beta, f_pro_P, T_C, T_loss, delta_t, k_m,
                                           ab_ratio_N = ab_ratio_N, M_bed = M_bed, frac_max_end = frac_max_end,
                                           d_max_day = d_max_day, lam_bed = lam_bed, lam_l1 = lam_l1, n_scale = n_scale,
                                           R_rec = R_rec, has_slack_dyn = has_slack_dyn, has_slack_rec = has_slack_rec,
                                           schedule = schedule, constant_dose = constant_dose)
    else:
        return construct_problem_sched_norec(nt, T_days, N0, alpha, beta, f_pro_P, T_C, T_loss, delta_t, k_m,
                                             ab_ratio_N = ab_ratio_N, M_bed = M_bed, frac_max_end = frac_max_end,
                                             d_max_day = d_max_day, lam_bed = lam_bed, lam_l1 = lam_l1, n_scale = n_scale,
                                             has_slack = has_slack_dyn, schedule = schedule, constant_dose = constant_dose)

def construct_problem_sched_norec(nt, T_days, N0, alpha, beta, f_pro_P, T_C, T_loss, delta_t, k_m, ab_ratio_N=3,
                                  M_bed=146.67, frac_max_end=1e-8, d_max_day=np.inf, lam_bed=0, lam_l1=0, n_scale=1,
                                  has_slack=True, schedule=None, constant_dose=False):
    # Schedule must be an array of positive integers indicating treatment days (rest of days have dose zero).
    if schedule is None:
        schedule = np.arange(0, T_days, 1) + 1   # Default to treating every day.
    if isinstance(schedule, list):
        schedule = np.array(schedule)
    if np.any(schedule <= 0) or np.any(schedule > T_days):
        raise ValueError("schedule must consist of positive integers in [1, {0}]".format(T_days))
    schedule = schedule.astype(int)   # Cast to integer.

    # Problem constants.
    T = int((T_days * 24 * 60) / delta_t)  # Total time steps.
    delta_day = int(24 * 60 / delta_t)  # Number of time steps per day.

    N0_P, N0_I, N0_H = N0
    alpha_P, alpha_I, alpha_H = scalar_to_vec_list(alpha, T)
    beta_P, beta_I, beta_H = scalar_to_vec_list(beta, T)

    c1 = np.exp(f_pro_P * (np.log(2) / T_C) * delta_t)
    c2 = c1**(2 * k_m - 1)
    c3 = np.exp(-(np.log(2) / T_loss) * delta_t)

    # Define variables.
    d = Variable(T, nonneg=True)
    # N = Variable((T+1,6), nonneg = True)            # N_t = (N_t^{P,v}, N_t^{P,d}, N_t^{I,v}, N_t^{I,d}, N_t^{H,v}, N_t^{H,d}).
    N_norm = Variable((T + 1, 6), nonneg=True)  # N_t^{norm} = N_t/n_scale.
    # NH_hat = Variable((T+1,2), nonneg = True)       # \hat N_t^H = (\hat N_t^{H,v}, \hat N_t^{H,d}).
    NH_hat_norm = Variable((T + 1, 2), nonneg=True)  # \hat N_t^{norm,H} = \hat N_t^H/n_scale.

    if has_slack:
        slack = Variable((T, 6), nonneg=True)
    else:
        slack = Constant(value=np.zeros((T, 6)))

    # Define linearization parameters.
    d_lin = Parameter(T, nonneg=True)  # d_t^{(k)} for t = 1,...,T.
    # Nv_lin = Parameter((T+1,3), pos = True)         # N_t^{v,(k)} = (N_t^{P,v,(k)}, N_t^{I,v,(k)}, N_t^{H,v,(k)}).
    Nv_norm_lin = Parameter((T + 1, 3), pos=True)  # N_t^{norm,v,(k)} = N_t^{v,(k)}/n_scale.
    # NH_hat_lin = Parameter(T+1, pos = True)         # \hat N_t^{H,v,(k)}.
    NH_hat_norm_lin = Parameter(T + 1, pos=True)  # \hat N_t^{norm,H,(k)} = \hat N_t^{H,v,(k)}/n_scale.

    N = N_norm * n_scale  # N_t = (N_t^{P,v}, N_t^{P,d}, N_t^{I,v}, N_t^{I,d}, N_t^{H,v}, N_t^{H,d}).
    Nv_lin = Nv_norm_lin * n_scale  # N_t^{v,(k)} = (N_t^{P,v,(k)}, N_t^{I,v,(k)}, N_t^{H,v,(k)}).

    # Define expressions.
    d_per_day = cvx.vstack([cvx.sum(d[t * delta_day:(t + 1) * delta_day]) for t in range(T_days)])   # Total dose each day.
    d_per_day = d_per_day[:, 0]  # Flatten into single dimensional vector.
    bed_N = cvx.sum(d_per_day) + cvx.sum_squares(d_per_day) / ab_ratio_N

    # Define objective.
    # obj = cvx.sum(N[1:,:])/nt + cvx.sum(slack)/T + cvx.sum(d)/T
    # obj = cvx.sum(N[1:,:])/nt + cvx.sum(slack)/T + lam_bed*cvx.sum_squares(d + 0.5*ab_ratio_N)/T
    # obj = cvx.sum(N[1:,:])/nt + cvx.sum(slack)/T + lam_bed*bed_N/T
    obj = fun_objective_norec(nt, T_days, N, delta_t, slack, d, ab_ratio_N, lam_bed, lam_l1)

    # Define constraints.
    constr = [N_norm[0, 0] == N0_P / n_scale, N_norm[0, 1] == 0, N_norm[0, 2] == N0_I / n_scale, N_norm[0, 3] == 0,
              N_norm[0, 4] == N0_H / n_scale, N_norm[0, 5] == 0]
    for t in range(T):
        # Linear cell dynamics.
        constr += [N_norm[t + 1, 1] == c2 * (N_norm[t, 0] + N_norm[t, 1]) - (c2 / c1) * N_norm[t + 1, 0],
                   N_norm[t + 1, 3] == N_norm[t, 3] + N_norm[t, 2] - N_norm[t + 1, 2],
                   # N_norm[t+1,5] == c3*(N_norm[t,4] + N_norm[t,5]) - N_norm[t+1,4],
                   NH_hat_norm[t + 1, 0] + NH_hat_norm[t + 1, 1] == c3 * (N_norm[t, 4] + N_norm[t, 5]),
                   N_norm[t + 1, 5] == NH_hat_norm[t + 1, 0] + NH_hat_norm[t + 1, 1] - N_norm[t + 1, 4]]

        # Nonlinear cell dynamics, with CCP linearization.
        # N_{t+1}^{P,v} = N_t^{P,v}*c1*exp(-\alpha_P*d_t - \beta_P*d_t^2).
        constr += [(alpha_P[t]*d[t] + beta_P[t]*d[t]**2 - cvx.log(N_norm[t, 0]) - np.log(c1)) +
                   (cvx.log(Nv_norm_lin[t + 1, 0]) + (N_norm[t + 1, 0] - Nv_norm_lin[t + 1, 0])/Nv_norm_lin[t + 1, 0]) <= slack[t, 0],
                   (cvx.log(N_norm[t + 1, 0]) + alpha_P[t]*d[t] - np.log(c1)) +
                        (-cvx.log(Nv_norm_lin[t, 0]) + beta_P[t]*d_lin[t]**2 - (N_norm[t, 0] - Nv_norm_lin[t, 0])/Nv_norm_lin[t, 0] +
                        2*beta_P[t] * d_lin[t] * (d[t] - d_lin[t])) >= -slack[t, 1]]

        # N_{t+1}^{I,v} = N_t^{I,v}*exp(-\alpha_I*d_t - \beta_I*d_t^2).
        constr += [(alpha_I[t]*d[t] + beta_I[t]*d[t]**2 - cvx.log(N_norm[t, 2])) + (
                    cvx.log(Nv_norm_lin[t + 1, 1]) + (N_norm[t + 1, 2] - Nv_norm_lin[t + 1, 1])/Nv_norm_lin[t + 1, 1]) <= slack[t, 2],
                   (cvx.log(N_norm[t + 1, 2]) + alpha_I[t]*d[t]) +
                        (-cvx.log(Nv_norm_lin[t, 1]) + beta_I[t]*d_lin[t]**2 - (N_norm[t, 2] - Nv_norm_lin[t, 1])/Nv_norm_lin[t, 1] +
                        2*beta_I[t]*d_lin[t] * (d[t] - d_lin[t])) >= -slack[t, 3]]

        # N_{t+1}^{H,v} = \hat N_{t+1}^{H,v}*exp(-\alpha_H*d_t - \beta_H*d_t^2).
        constr += [(alpha_H[t]*d[t] + beta_H[t]*d[t]**2 - cvx.log(NH_hat_norm[t + 1, 0])) + (
                    cvx.log(Nv_norm_lin[t + 1, 2]) + (N_norm[t + 1, 4] - Nv_norm_lin[t + 1, 2]) / Nv_norm_lin[t + 1, 2]) <= slack[t, 4],
                   (cvx.log(N_norm[t + 1, 4]) + alpha_H[t]*d[t]) +
                        (-cvx.log(NH_hat_norm_lin[t + 1]) + beta_H[t]*d_lin[t]**2 - (NH_hat_norm[t + 1, 0] - NH_hat_norm_lin[t + 1]) / NH_hat_norm_lin[t + 1] +
                        2*beta_H[t]*d_lin[t] * (d[t] - d_lin[t])) >= -slack[t, 5]]

    # Final viable tumor cell constraint.
    # constr += [cvx.sum(N[-1,:])/nt <= 0.01]
    # constr += [(N[-1,0] + N[-1,2] + N[-1,4])/nt <= 1e-6]
    # constr += [(N[-delta_day-1,0] + N[-delta_day-1,2] + N[-delta_day-1,4])/nt <= 1e-4]
    # constr += [N_norm[-delta_day:,0] + N_norm[-delta_day:,2] + N_norm[-delta_day:,4] <= 1e-4*nt/n_scale]
    # constr += [N_norm[11*delta_day,2] <= 1e-6*nt/n_scale]
    if np.isfinite(frac_max_end):
        constr += [N_norm[-1, 0] + N_norm[-1, 2] + N_norm[-1, 4] <= frac_max_end * nt / n_scale]

    # Normal tissue BED constraint.
    if np.isfinite(M_bed):
        constr += [bed_N <= M_bed]

    # Maximum dose (per day) constraint.
    if np.isfinite(d_max_day):
        # constr += [cvx.sum(d[t*delta_day:(t+1)*delta_day]) <= d_max_day for t in range(T_days)]
        constr += [d_per_day <= d_max_day]

    # Enforce schedule of treatment days.
    t_days_all = np.arange(0, T_days, 1)
    t_days_treat_idx = schedule - 1  # Zero-indexing
    t_days_zero_idx = np.setdiff1d(t_days_all, t_days_treat_idx)
    t_days_treat_idx = t_days_treat_idx.astype(int)
    t_days_zero_idx = t_days_zero_idx.astype(int)

    if len(t_days_zero_idx) != 0:
        constr += [d_per_day[t_days_zero_idx] == 0]
    if constant_dose:
        d_cons = Variable(nonneg=True)
        if len(t_days_treat_idx) != 0:
            constr += [d_per_day[t_days_treat_idx] == d_cons]
        else:
            constr += [d_per_day == d_cons]

    prob = Problem(Minimize(obj), constr)
    var_dict = {"d": d, "slack_dyn": slack, "N_norm": N_norm, "NH_hat_norm": NH_hat_norm}
    if constant_dose:
        var_dict.update({"d_cons": d_cons})
    parm_dict = {"d_lin": d_lin, "Nv_norm_lin": Nv_norm_lin, "NH_hat_norm_lin": NH_hat_norm_lin}
    expr_dict = {"N": N, "Nv_lin": Nv_lin, "d_per_day": d_per_day, "normal_bed": bed_N,
                 "NH_hat": NH_hat_norm * n_scale, "NH_hat_lin": NH_hat_norm_lin * n_scale}
    return prob, schedule, var_dict, parm_dict, expr_dict


def construct_problem_sched_rec(nt, T_days, N0, alpha, beta, f_pro_P, T_C, T_loss, delta_t, k_m, ab_ratio_N=3,
                                M_bed=146.67, frac_max_end=1e-8, d_max_day=np.inf, lam_bed=0, lam_l1=0, n_scale=1,
                                R_rec=None, has_slack_dyn=True, has_slack_rec=True, schedule=None, constant_dose=False):
    # Schedule must be an array of positive integers indicating treatment days (rest of days have dose zero).
    if schedule is None:
        schedule = np.arange(0, T_days, 1) + 1  # Default to treating every day.
    if isinstance(schedule, list):
        schedule = np.array(schedule)
    if np.any(schedule <= 0) or np.any(schedule > T_days):
        raise ValueError("schedule must consist of positive integers in [1, {0}]".format(T_days))
    schedule = schedule.astype(int)  # Cast to integer.

    # Problem constants.
    T = int((T_days * 24 * 60) / delta_t)  # Total time steps.
    delta_day = int(24 * 60 / delta_t)  # Number of time steps per day.

    N0_P, N0_I, N0_H = N0
    alpha_P, alpha_I, alpha_H = scalar_to_vec_list(alpha, T)
    beta_P, beta_I, beta_H = scalar_to_vec_list(beta, T)

    c1 = np.exp(f_pro_P * (np.log(2) / T_C) * delta_t)
    c2 = c1 ** (2 * k_m - 1)
    c3 = np.exp(-(np.log(2) / T_loss) * delta_t)

    if R_rec is None:
        R_rec = c1 ** T
    # R_norm = R_rec/n_scale
    R_norm = R_rec

    # Define variables.
    d = Variable(T, nonneg=True)

    # Cell compartment configuration indicators.
    # z_{t,0} = 1{all cells in P compartment}.
    # z_{t,1} = 1{P full, H empty, excess cells in I compartment}.
    # z_{t,2} = 1 - z_{t,0} - z_{t,1} = 1{P and I compartments full}.
    z_ind = Variable((T + 1, 2), boolean=True)
    z_PI = 1 - z_ind[:, 0] - z_ind[:, 1]
    z = cvx.vstack([z_ind[:, 0], z_ind[:, 1], z_PI]).T

    # N = Variable((T+1,6), nonneg = True)             # N_t = (N_t^{P,v}, N_t^{P,d}, N_t^{I,v}, N_t^{I,d}, N_t^{H,v}, N_t^{H,d}).
    N_norm = Variable((T + 1, 6), nonneg=True)  # N_t^{norm} = N_t/n_scale.
    # N_tld = Variable((T+1,6), nonneg = True)         # \tilde N_t = (\tilde N_t^{P,v}, \tilde N_t^{P,d}, \tilde N_t^{I,v}, \tilde N_t^{I,d}, \tilde N_t^{H,v}, \tilde N_t^{H,d}).
    N_tld_norm = Variable((T + 1, 6), nonneg=True)  # \tilde N_t^{norm} = \tilde N_t/n_scale.
    # NH_hat = Variable((T+1,2), nonneg = True)        # \hat N_t^H = (\hat N_t^{H,v}, \hat N_t^{H,d}).
    NH_hat_norm = Variable((T + 1, 2), nonneg=True)  # \hat N_t^{norm,H} = \hat N_t^H/n_scale.

    # Slack for cell dynamics constraints.
    if has_slack_dyn:
        slack_dyn = Variable((T, 6), nonneg=True)
    else:
        slack_dyn = Constant(value=np.zeros((T, 6)))

    # Slack for cell recompartmentalization constraints.
    if has_slack_rec:
        slack_rec = Variable((T, 22), nonneg=True)
    else:
        slack_rec = Constant(value=np.zeros((T, 22)))

    # Define linearization parameters.
    d_lin = Parameter(T, nonneg=True)  # d_t^{(k)} for t = 1,...,T.
    # Nv_lin = Parameter((T+1,3), pos = True)          # N_t^{v,(k)} = (N_t^{P,v,(k)}, N_t^{I,v,(k)}, N_t^{H,v,(k)}).
    Nv_norm_lin = Parameter((T + 1, 3), pos=True)  # N_t^{norm,v,(k)} = N_t^{v,(k)}/n_scale.
    # Nv_tld_lin = Parameter((T+1,3), pos = True)      # \tilde N_t^{v,(k)} = (\tilde N_t^{P,v,(k)}, \tilde N_t^{I,v,(k)}, \tilde N_t^{H,v,(k)}).
    Nv_tld_norm_lin = Parameter((T + 1, 3), pos=True)  # \tilde N_t^{norm,v,(k)} = \tilde N_t^{v,(k)}/n_scale.
    # NH_hat_lin = Parameter(T+1, pos = True)          # \hat N_t^{H,v,(k)}.
    NH_hat_norm_lin = Parameter(T + 1, pos=True)  # \hat N_t^{norm,H,(k)} = \hat N_t^{H,v,(k)}/n_scale.

    N = N_norm * n_scale  # N_t = (N_t^{P,v}, N_t^{P,d}, N_t^{I,v}, N_t^{I,d}, N_t^{H,v}, N_t^{H,d}).
    N_tld = N_tld_norm * n_scale  # \tilde N_t = (\tilde N_t^{P,v}, \tilde N_t^{P,d}, \tilde N_t^{I,v}, \tilde N_t^{I,d}, \tilde N_t^{H,v}, \tilde N_t^{H,d}).
    Nv_lin = Nv_norm_lin * n_scale  # N_t^{v,(k)} = (N_t^{P,v,(k)}, N_t^{I,v,(k)}, N_t^{H,v,(k)}).
    # N_tld_tot = cvx.sum(N_tld, axis = 1)             # \tilde N_t^{tot} = \tilde N_t^{P,v} + \tilde N_t^{P,d} + \tilde N_t^{I,v} + \tilde N_t^{I,d} + \tilde N_t^{H,v} + \tilde N_t^{H,d}.
    N_tld_norm_tot = cvx.sum(N_tld_norm, axis=1)  # \tilde N_t^{norm,tot} = \tilde N_t^{tot}/n_scale.

    # Define expressions.
    d_per_day = cvx.vstack([cvx.sum(d[t * delta_day:(t + 1) * delta_day]) for t in range(T_days)])
    d_per_day = d_per_day[:, 0]  # Flatten into single dimensional vector.
    bed_N = cvx.sum(d_per_day) + cvx.sum_squares(d_per_day) / ab_ratio_N

    # Define objective.
    # obj = cvx.sum(N[1:,:])/nt + cvx.sum(slack_dyn)/T + cvx.sum(slack_rec)/T
    # obj = cvx.sum(N[1:,:])/nt + cvx.sum(slack_dyn)/slack_dyn.size + cvx.sum(slack_rec)/slack_rec.size
    # obj = cvx.sum(N[1:,:])/nt + cvx.sum(slack_dyn)/slack_dyn.size + cvx.sum(slack_rec)/slack_rec.size + lam_bed*cvx.sum_squares(d + 0.5*ab_ratio_N)/T
    obj = fun_objective_rec(nt, T_days, N, delta_t, slack_dyn, slack_rec, d, ab_ratio_N, lam_bed, lam_l1)

    # Define constraints.
    # Initial cell compartment configuration.
    if N0_P > 0 and N0_I == 0 and N0_H == 0:
        constr = [z_ind[0, 0] == 1, z_ind[0, 1] == 0]
    elif N0_P > 0 and N0_I > 0 and N0_H == 0:
        constr = [z_ind[0, 0] == 0, z_ind[0, 1] == 1]
    elif N0_P > 0 and N0_I > 0 and N0_H > 0:
        constr = [z_ind[0, 0] == 0, z_ind[0, 1] == 0]
    elif N0_P == 0 and N0_I == 0 and N0_H == 0:
        raise ValueError("All cell compartments empty. No treatment needed")
    else:
        raise ValueError("Invalid cell compartment configuration")

    constr += [z_ind[:, 0] + z_ind[:, 1] <= 1]  # Cell compartment configurations are mutually exclusive.

    constr += [N_norm[0, 0] == N0_P / n_scale, N_norm[0, 1] == 0,
               N_norm[0, 2] == N0_I / n_scale, N_norm[0, 3] == 0,
               N_norm[0, 4] == N0_H / n_scale, N_norm[0, 5] == 0]

    for t in range(T):
        # Linear cell dynamics.
        # \tilde N_{t+1}^{P,d} = c_2*(N_t^{P,v} + N_t^{P,d}) - (c_2/c_1)*\tilde N_{t+1}^{P,v}.
        # \tilde N_{t+1}^{I,d} = N_t^{I,v} + N_t^{I,d} - \tilde N_{t+1}^{I,v}.
        # \hat N_{t+1}^{H,v} + \hat N_{t+1}^{H,d} = c_3*(N_t^{H,v} + N_t^{H,d}).
        # \tilde N_{t+1}^{H,d} = \hat N_{t+1}^{H,v} + \hat N_{t+1}^{H,d} - \tilde N_{t+1}^{H,v}
        constr += [N_tld_norm[t + 1, 1] == c2 * (N_norm[t, 0] + N_norm[t, 1]) - (c2 / c1) * N_tld_norm[t + 1, 0],
                   N_tld_norm[t + 1, 3] == N_norm[t, 3] + N_norm[t, 2] - N_tld_norm[t + 1, 2],
                   NH_hat_norm[t + 1, 0] + NH_hat_norm[t + 1, 1] == c3 * (N_norm[t, 4] + N_norm[t, 5]),
                   N_tld_norm[t + 1, 5] == NH_hat_norm[t + 1, 0] + NH_hat_norm[t + 1, 1] - N_tld_norm[t + 1, 4]]

        # Nonlinear cell dynamics, with CCP linearization.
        # \tilde N_{t+1}^{P,v} = N_t^{P,v}*c1*exp(-\alpha_P*d_t - \beta_P*d_t^2).
        constr += [(alpha_P[t]*d[t] + beta_P[t]*d[t]**2 - cvx.log(N_norm[t, 0]) - np.log(c1)) + (
                    cvx.log(Nv_tld_norm_lin[t + 1, 0]) + (N_tld_norm[t + 1, 0] - Nv_tld_norm_lin[t + 1, 0]) /
                    Nv_tld_norm_lin[t + 1, 0]) <= slack_dyn[t, 0],
                   (cvx.log(N_tld_norm[t + 1, 0]) + alpha_P[t]*d[t] - np.log(c1)) +
                   (-cvx.log(Nv_norm_lin[t, 0]) + beta_P[t]*d_lin[t]**2 -
                    (N_norm[t, 0] - Nv_norm_lin[t, 0]) / Nv_norm_lin[t, 0] +
                    2*beta_P[t]*d_lin[t]*(d[t] - d_lin[t])) >= -slack_dyn[t, 1]]

        # \tilde N_{t+1}^{I,v} = N_t^{I,v}*exp(-\alpha_I*d_t - \beta_I*d_t^2).
        constr += [(alpha_I[t]*d[t] + beta_I[t]*d[t]**2 - cvx.log(N_norm[t, 2])) + (
                    cvx.log(Nv_tld_norm_lin[t + 1, 1]) + (N_tld_norm[t + 1, 2] - Nv_tld_norm_lin[t + 1, 1]) /
                    Nv_tld_norm_lin[t + 1, 1]) <= slack_dyn[t, 2],
                   (cvx.log(N_tld_norm[t + 1, 2]) + alpha_I[t]*d[t]) +
                   (-cvx.log(Nv_norm_lin[t, 1]) + beta_I[t]*d_lin[t]**2 -
                    (N_norm[t, 2] - Nv_norm_lin[t, 1]) / Nv_norm_lin[t, 1] +
                    2*beta_I[t]*d_lin[t]*(d[t] - d_lin[t])) >= -slack_dyn[t, 3]]

        # \tilde N_{t+1}^{H,v} = \hat N_{t+1}^{H,v}*exp(-\alpha_H*d_t - \beta_H*d_t^2).
        constr += [(alpha_H[t]*d[t] + beta_H[t]*d[t]**2 - cvx.log(NH_hat_norm[t + 1, 0])) + (
                    cvx.log(Nv_tld_norm_lin[t + 1, 2]) + (N_tld_norm[t + 1, 4] - Nv_tld_norm_lin[t + 1, 2]) /
                    Nv_tld_norm_lin[t + 1, 2]) <= slack_dyn[t, 4],
                   (cvx.log(N_tld_norm[t + 1, 4]) + alpha_H[t]*d[t]) +
                   (-cvx.log(NH_hat_norm_lin[t + 1]) + beta_H[t]*d_lin[t]**2 -
                    (NH_hat_norm[t + 1, 0] - NH_hat_norm_lin[t + 1]) / NH_hat_norm_lin[t + 1] +
                    2*beta_H[t]*d_lin[t] * (d[t] - d_lin[t])) >= -slack_dyn[t, 5]]

        # Recompartmentalization.
        # 1. All cells in P compartment, I and H compartments empty.
        # N_t^{P,v} + N_t^{P,d} = \tilde N_t^{tot}, \tilde N_t^{tot} <= N_0^P, N_t^{I,v} + N_t^{I,d} = 0, N_t^{H,v] + N_t^{H,d} = 0.
        constr += [N_norm[t + 1, 0] + N_norm[t + 1, 1] - N_tld_norm_tot - (1 - z[t + 1, 0]) * R_norm <= slack_rec[t, 0],
                   N_norm[t + 1, 0] + N_norm[t + 1, 1] - N_tld_norm_tot + (1 - z[t + 1, 0]) * R_norm >= -slack_rec[t, 1],
                   N_tld_norm_tot[t + 1] - N0_P / n_scale - (1 - z[t + 1, 0]) * R_norm <= slack_rec[t, 2],
                   N_norm[t + 1, 2] + N_norm[t + 1, 3] - (1 - z[t + 1, 0]) * R_norm <= slack_rec[t, 3],
                   # N_norm[t+1,2] + N_norm[t+1,3] + (1 - z[t+1,0])*R_norm >= -slack_rec[t,4] already satisfied by N_norm >= 0,
                   N_norm[t + 1, 4] + N_norm[t + 1, 5] - (1 - z[t + 1, 0]) * R_norm <= slack_rec[t, 5]]
        # N_norm[t+1,4] + N_norm[t+1,5] + (1 - z[t+1,0])*R_norm >= -slack_rec[t,6] already satisfied by N_norm >= 0]

        # 2. P compartment full, H compartment empty, excess cells in I compartment.
        # N_t^{P,v} + N_t^{P,d} = N_0^P, \tilde N_t^{tot} >= N_0^P, \tilde N_t^{tot} <= N0_P + N0_I, N_t^{I,v} + N_t^{I,d} = \tilde N_t^{tot} - N_0^P, N_t^{H,v} + N_t^{H,d} = 0.
        constr += [N_norm[t + 1, 0] + N_norm[t + 1, 1] - N0_P / n_scale - (1 - z[t + 1, 1]) * R_norm <= slack_rec[t, 7],
                   N_norm[t + 1, 0] + N_norm[t + 1, 1] - N0_P / n_scale + (1 - z[t + 1, 1]) * R_norm >= -slack_rec[t, 8],
                   N0_P / n_scale - N_tld_norm_tot[t + 1] - (1 - z[t + 1, 1]) * R_norm <= slack_rec[t, 9],
                   N_tld_norm_tot[t + 1] - N0_P / n_scale - N0_I / n_scale - (1 - z[t + 1, 1]) * R_norm <= slack_rec[t, 10],
                   N_norm[t + 1, 2] + N_norm[t + 1, 3] - N_tld_norm_tot[t + 1] + N0_P / n_scale -
                        (1 - z[t + 1, 1]) * R_norm <= slack_rec[t, 11],
                   N_norm[t + 1, 2] + N_norm[t + 1, 3] - N_tld_norm_tot[t + 1] + N0_P / n_scale +
                        (1 - z[t + 1, 1]) * R_norm >= -slack_rec[t, 12],
                   N_norm[t + 1, 4] + N_norm[t + 1, 5] - (1 - z[t + 1, 1]) * R_norm <= slack_rec[t, 13]]
        # N_norm[t+1,4] + N_norm[t+1,5] - (1 - z[t+1,1])*R_norm >= -slack_rec[t,14] already satisfied by N_norm >= 0]

        # 3. P and I compartments full, excess cells in H compartment.
        # N_t^{P,v} + N_t^{P,d} = N_0^P, N_t^{I,v} + N_t^{I,d} = N_0^I, \tilde N_t^{tot} >= N_0^P + N_0^I, N_t^{H,v} + N_t^{H,d} = \tilde N_t^{tot} - N_0^P - N_0^I.
        constr += [N_norm[t + 1, 0] + N_norm[t + 1, 1] - N0_P / n_scale - (1 - z_PI[t + 1]) * R_norm <= slack_rec[t, 15],
                   N_norm[t + 1, 0] + N_norm[t + 1, 1] - N0_P / n_scale + (1 - z_PI[t + 1]) * R_norm >= -slack_rec[t, 16],
                   N_norm[t + 1, 2] + N_norm[t + 1, 3] - N0_I / n_scale - (1 - z_PI[t + 1]) * R_norm <= slack_rec[t, 17],
                   N_norm[t + 1, 2] + N_norm[t + 1, 3] - N0_I / n_scale + (1 - z_PI[t + 1]) * R_norm >= -slack_rec[t, 18],
                   N0_P / n_scale + N0_I / n_scale - N_tld_norm_tot[t + 1] - (1 - z_PI[t + 1]) * R_norm <= slack_rec[t, 19],
                   N_norm[t + 1, 4] + N_norm[t + 1, 5] - N_tld_norm_tot[t + 1] + N0_P / n_scale + N0_I / n_scale -
                        (1 - z_PI[t + 1]) * R_norm <= slack_rec[t, 20],
                   N_norm[t + 1, 4] + N_norm[t + 1, 5] - N_tld_norm_tot[t + 1] + N0_P / n_scale + N0_I / n_scale +
                        (1 - z_PI[t + 1]) * R_norm >= -slack_rec[t, 21]]

    # Final viable tumor cell constraint.
    if np.isfinite(frac_max_end):
        constr += [N_norm[-1, 0] + N_norm[-1, 2] + N_norm[-1, 4] <= frac_max_end * nt / n_scale]

    # Normal tissue BED constraint.
    if np.isfinite(M_bed):
        constr += [bed_N <= M_bed]

    # Maximum dose (per day) constraint.
    if np.isfinite(d_max_day):
        constr += [cvx.sum(d[t * delta_day:(t + 1) * delta_day]) <= d_max_day for t in range(T_days)]

    # Enforce schedule of treatment days.
    t_days_all = np.arange(0, T_days, 1)
    t_days_treat_idx = schedule - 1  # Zero-indexing
    t_days_zero_idx = np.setdiff1d(t_days_all, t_days_treat_idx)
    t_days_treat_idx = t_days_treat_idx.astype(int)
    t_days_zero_idx = t_days_zero_idx.astype(int)

    if len(t_days_zero_idx) != 0:
        constr += [d_per_day[t_days_zero_idx] == 0]
    if constant_dose:
        d_cons = Variable(nonneg=True)
        if len(t_days_treat_idx) != 0:
            constr += [d_per_day[t_days_treat_idx] == d_cons]
        else:
            constr += [d_per_day == d_cons]

    prob = Problem(Minimize(obj), constr)
    var_dict = {"d": d, "z_ind": z_ind, "slack_dyn": slack_dyn, "slack_rec": slack_rec, "N_norm": N_norm,
                "N_tld_norm": N_tld_norm, "NH_hat_norm": NH_hat_norm}
    if constant_dose:
        var_dict.update({"d_cons": d_cons})
    parm_dict = {"d_lin": d_lin, "Nv_norm_lin": Nv_norm_lin, "Nv_tld_norm_lin": Nv_tld_norm_lin, "NH_hat_norm_lin": NH_hat_norm_lin}
    expr_dict = {"N": N, "N_tld": N_tld, "Nv_lin": Nv_lin, "N_tld_norm_tot": N_tld_norm_tot, "NH_hat": NH_hat_norm * n_scale,
                 "NH_hat_lin": NH_hat_norm_lin * n_scale, "z": z, "d_per_day": d_per_day, "normal_bed": bed_N}
    return prob, schedule, var_dict, parm_dict, expr_dict


def init_parms_sched(parm_dict, T_days, N0, alpha, beta, f_pro_P, T_C, T_loss, delta_t, k_m, d_init=None, ab_ratio_N=3,
                     M_bed=146.67, n_scale=1, schedule=None):
    # Schedule must be an array of positive integers indicating treatment days (rest of days have dose zero).
    if schedule is None:
        schedule = np.arange(0, T_days, 1) + 1  # Default to treating every day.
    if isinstance(schedule, list):
        schedule = np.array(schedule)
    if np.any(schedule <= 0) or np.any(schedule > T_days):
        raise ValueError("schedule must consist of positive integers in [1, {0}]".format(T_days))
    schedule = schedule.astype(int)  # Cast to integer.

    T = int((T_days * 24 * 60) / delta_t)  # Total time steps.
    delta_day = int(24 * 60 / delta_t)     # Number of time steps per day.
    N0_P, N0_I, N0_H = N0
    M_tld_N = ab_ratio_N * M_bed + T * (0.5 * ab_ratio_N) ** 2
    c3 = np.exp(-(np.log(2) / T_loss) * delta_t)

    if d_init is None:
        t_days_treat_idx = (schedule - 1).astype(int)  # Zero-indexing
        T_days_treat = len(t_days_treat_idx)

        d_init = np.zeros(T)
        if T_days_treat != 0:
            # Calculate optimal constant fraction (per time step) with only P compartment, assuming total treatment
            # days is T_days_treat. Rest of (T_days - T_days_treat) days are set to zero dose.
            T_treat = int((T_days_treat * 24 * 60 / delta_t))   # Number of time steps of treatment.
            df = np.max([np.sqrt(M_tld_N / T_treat) - 0.5*ab_ratio_N, 0])
            for t in t_days_treat_idx:
                d_init[t*delta_day:(t+1)*delta_day] = df

    if d_init.shape not in [(T,), (T, 1)]:
        raise ValueError("d_init must be a vector of length {0}".format(T))
    if np.any(d_init < 0):
        raise ValueError("d_init must be non-negative")

    N_init, N_tld_init, N_tld_tot_init, NH_hat_init, z_init = \
        calc_cell_dynamics_three(d_init, N0, alpha, beta, f_pro_P, T_C, T_loss, delta_t, k_m, recomp=True)
    # d_per_day_init = np.array([np.sum(d_init[t*delta_day:(t+1)*delta_day]) for t in range(T_days)])
    # bed_N_init = np.sum(d_per_day_init) + np.sum(d_per_day_init**2)/ab_ratio_N
    # obj_init = np.sum(N_init[1:,:])/nt + lam_bed*bed_N_init/T

    parm_dict["d_lin"].value = d_init
    parm_dict["NH_hat_norm_lin"].value = np.concatenate((0.5 * c3 * np.array([N0_H]), NH_hat_init)) / n_scale
    parm_dict["Nv_norm_lin"].value = np.column_stack((N_init[:, 0], N_init[:, 2], N_init[:, 4])) / n_scale
    if "Nv_tld_norm_lin" in parm_dict:
        parm_dict["Nv_tld_norm_lin"].value = \
            np.row_stack((np.array([N0_P, N0_I, N0_H]), # First row is filler that isn't used in constraints (only for consistent indexing).
                          np.column_stack((N_tld_init[:, 0], N_tld_init[:, 2], N_tld_init[:, 4]))
                         )) / n_scale

    # Add an optional offset to handle numerical error due to near-zero values.
    parm_dict["NH_hat_norm_lin"].value += NH_HAT_NORM_LIN_INIT_OFFSET
    parm_dict["Nv_norm_lin"].value[:, 0] += NV_NORM_LIN_INIT_OFFSET[0]
    parm_dict["Nv_norm_lin"].value[:, 1] += NV_NORM_LIN_INIT_OFFSET[1]
    parm_dict["Nv_norm_lin"].value[:, 2] += NV_NORM_LIN_INIT_OFFSET[2]
    if "Nv_tld_norm_lin" in parm_dict:
        parm_dict["Nv_tld_norm_lin"].value[:, 0] += NV_TLD_NORM_LIN_INIT_OFFSET[0]
        parm_dict["Nv_tld_norm_lin"].value[:, 1] += NV_TLD_NORM_LIN_INIT_OFFSET[1]
        parm_dict["Nv_tld_norm_lin"].value[:, 2] += NV_TLD_NORM_LIN_INIT_OFFSET[2]
    return N_init, d_init, z_init

def solve_ccp_sched(nt, T_days, N0, alpha, beta, f_pro_P, T_C, T_loss, delta_t, k_m, d_init=None, ab_ratio_N=3,
                    M_bed=146.67, frac_max_end=1e-8, d_max_day=np.inf, lam_bed=0, lam_l1=0, n_scale=1, R_rec=None,
                    has_slack_dyn=True, has_slack_rec=True, schedule=None, constant_dose=False, recomp=False,
                    filename=None, *args, **kwargs):
    max_iter = kwargs.pop("max_iter", 1000)
    delta_stop = kwargs.pop("delta_stop", 1e-3)
    verbose = kwargs.get("verbose", False)

    if verbose:
        print("Constructing problem...")
    prob, schedule, var_dict, parm_dict, expr_dict = \
        construct_problem_sched(nt, T_days, N0, alpha, beta, f_pro_P, T_C, T_loss, delta_t, k_m, ab_ratio_N=ab_ratio_N,
                                M_bed=M_bed, frac_max_end=frac_max_end, d_max_day=d_max_day, lam_bed=lam_bed,
                                lam_l1=lam_l1, n_scale=n_scale, R_rec=R_rec, has_slack_dyn=has_slack_dyn,
                                has_slack_rec=has_slack_rec, schedule=schedule, constant_dose=constant_dose,
                                recomp=recomp)

    if verbose:
        print("Initializing parameters...")
    N_init, d_init, z_init = init_parms_sched(parm_dict, T_days, N0, alpha, beta, f_pro_P, T_C, T_loss, delta_t, k_m,
                                              d_init, ab_ratio_N, M_bed, n_scale, schedule)
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
        parm_dict["NH_hat_norm_lin"].value = var_dict["NH_hat_norm"].value[:, 0]
        parm_dict["Nv_norm_lin"].value = np.column_stack((N_norm_val[:, 0], N_norm_val[:, 2], N_norm_val[:, 4]))
        if recomp:
            N_tld_norm_val = var_dict["N_tld_norm"].value
            parm_dict["Nv_tld_norm_lin"].value = np.column_stack(
                (N_tld_norm_val[:, 0], N_tld_norm_val[:, 2], N_tld_norm_val[:, 4]))

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
                pickle.dump(res_hist_list, handle, protocol=pickle.HIGHEST_PROTOCOL)

        finished = (k + 1) >= max_iter or obj_diff <= delta_stop
        k = k + 1
    stop_time = time()
    run_time = stop_time - start_time

    result = {"obj": prob.value, "d": var_dict["d"].value, "N": expr_dict["N"].value, "slack_dyn": var_dict["slack_dyn"].value,
              "d_per_day": expr_dict["d_per_day"].value, "normal_bed": expr_dict["normal_bed"].value, "obj_diff": obj_diff,
              "iterations": k, "run_time": run_time, "schedule": schedule}
    if recomp:
        result["z"] = expr_dict["z"].value
        result["slack_rec"] = var_dict["slack_rec"].value

    # if verbose:
    #     print_result(result)
    return result
