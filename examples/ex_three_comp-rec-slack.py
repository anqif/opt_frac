import numpy as np
import cvxpy as cvx
import cvxpy.settings as cvxpy_s

from time import time
from warnings import warn
from cvxpy import Variable, Parameter, Problem, Minimize

from opt_frac.plot_sim import EQD2_primer_sim_step, plot_dose
from opt_frac.utilities import calc_cell_dynamics_three, plot_plan_results

def main():
    np.random.seed(1)
    fig_path = r'~/Documents/Software/opt_frac/examples/figures/experiments/'
    data_path = r'~/Documents/Software/opt_frac/examples/data/'

    # Problem parameters.
    delta_t = 60                      # Time step in minutes.
    T_days = 14                       # Total days of treatment.
    # T_days = 7
    T = int((T_days*24*60)/delta_t)   # Total time steps.
    delta_day = int(24*60/delta_t)    # Number of time steps per day.
    # T = 20
    
    rhot = 1e6            # Tumor cell density.
    vt = 64               # Volume of a tumorlet.
    nt = rhot*vt          # Total number of cells in a tumorlet.
    clf = 0.92            # Cell loss factor.
    gf = 0.25             # Growth fraction.
    
    f_pro_P = 0.5         # Initial proliferation fraction in P compartment.
    T_C = 2*(24*60)       # Cell cycle time in minutes.
    T_loss = 2*(24*60)    # Cell loss half-time in H compartment in minutes.
    T_lysis = 3*(24*60)   # Lysis half-time in minutes.
    k_m = 0.3
    
    # Photon parameters.
    # alpha_P = 0.382
    # beta_P = 0.0576
    # OER_I = 2.0
    # OER_H = 1.37
    
    # Proton parameters.
    alpha_P = 0.205
    beta_P = alpha_P/2.5
    OER_I = 1.0
    OER_H = 1.05
    
    ab_ratio_N = 3                # Ratio alpha/beta for normal tissue cells.
    M = 146.67                    # Upper bound on BED for normal tissue.
    
    N0_P = (gf/f_pro_P)*nt
    N0_H = clf*gf*(T_loss/T_C)*nt
    N0_I = nt - N0_P - N0_H
    
    # Constants in optimization problem.
    d_max_day = 18                # Maximum total dose per day.
    
    alpha_I = alpha_P/OER_I
    beta_I = beta_P/OER_I**2
    alpha_H = alpha_P/OER_H
    beta_H = beta_P/OER_H**2
    
    ab_ratio_P = alpha_P/beta_P   # Ratio alpha/beta for P compartment.
    ab_ratio_I = alpha_I/beta_I   # Ratio alpha/beta for I compartment.
    ab_ratio_H = alpha_H/beta_H   # Ratio alpha/beta for H compartment.
    
    M_tld_N = ab_ratio_N*M + T*(0.5*ab_ratio_N)**2
    c1 = np.exp(f_pro_P*(np.log(2)/T_C)*delta_t)
    c2 = c1**(2*k_m - 1)
    c3 = np.exp(-(np.log(2)/T_loss)*delta_t)
    
    R = c1**T                     # Make this bigger if problem is infeasible at first.
    n_scale = 0.5*nt              # Normalize initial cell counts N_0 -> N_0/n_scale. Adjust value if solver fails (generally due to precision issues).
    # n_scale = 0.85*nt
    # R_norm = R/n_scale
    R_norm = R
    
    # Optimizer arguments.
    lam_bed = 0
    # lam_bed = 0.025   # Penalty on normal tissue BED term.
    
    solver_name = "MOSEK"
    # verbose = False
    verbose = True
    max_iter = 10
    # max_iter = 30
    # max_iter = 1000
    delta_stop = 1e-3
    
    print("Initial cell count: P compartment = {0}, I compartment = {1}, H compartment = {2}".format(N0_P, N0_I, N0_H))
    print("Constants: c1 = {0}, c2 = {1}, c3 = {2}, n_scale = {3}".format(c1, c2, c3, n_scale))
    
    # Three compartment (P,I,H) problem.
    print("Constructing problem...")
    
    # Define variables.
    d = Variable(T, nonneg = True)
    slack_dyn = Variable((T,6), nonneg = True)
    slack_rec = Variable((T,22), nonneg = True)
    
    # Cell compartment configuration indicators.
    # z_{t,0} = 1{all cells in P compartment}. 
    # z_{t,1} = 1{P full, H empty, excess cells in I compartment}.
    # z_{t,2} = 1 - z_{t,0} - z_{t,1} = 1{P and I compartments full}.
    z = Variable((T+1,2), boolean = True)
    z_PI = 1 - z[:,0] - z[:,1]
    
    # N = Variable((T+1,6), nonneg = True)             # N_t = (N_t^{P,v}, N_t^{P,d}, N_t^{I,v}, N_t^{I,d}, N_t^{H,v}, N_t^{H,d}).
    N_norm = Variable((T+1,6), nonneg = True)          # N_t^{norm} = N_t/n_scale.
    # N_tld = Variable((T+1,6), nonneg = True)         # \tilde N_t = (\tilde N_t^{P,v}, \tilde N_t^{P,d}, \tilde N_t^{I,v}, \tilde N_t^{I,d}, \tilde N_t^{H,v}, \tilde N_t^{H,d}).
    N_tld_norm = Variable((T+1,6), nonneg = True)      # \tilde N_t^{norm} = \tilde N_t/n_scale.
    # NH_hat = Variable((T+1,2), nonneg = True)        # \hat N_t^H = (\hat N_t^{H,v}, \hat N_t^{H,d}).
    NH_hat_norm = Variable((T+1,2), nonneg = True)     # \hat N_t^{norm,H} = \hat N_t^H/n_scale.

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
    bed_N = cvx.sum(d_per_day) + cvx.sum_squares(d_per_day)/ab_ratio_N
    
    # Define objective.
    # obj = cvx.sum(N[1:,:])/nt + cvx.sum(slack_dyn)/T + cvx.sum(slack_rec)/T
    # obj = cvx.sum(N[1:,:])/nt + cvx.sum(slack_dyn)/slack_dyn.size + cvx.sum(slack_rec)/slack_rec.size
    # obj = cvx.sum(N[1:,:])/nt + cvx.sum(slack_dyn)/slack_dyn.size + cvx.sum(slack_rec)/slack_rec.size + lam_bed*cvx.sum_squares(d + 0.5*ab_ratio_N)/T
    slack_reg = cvx.sum(slack_dyn)/slack_dyn.size + cvx.sum(slack_rec)/slack_rec.size
    obj = cvx.sum(N[1:,:]) + slack_reg + lam_bed*bed_N/T
    
    # Define constraints.
    # Initial cell compartment configuration.
    if N0_P > 0 and N0_I == 0 and N0_H == 0:
        constr = [z[0,0] == 1, z[0,1] == 0]
    elif N0_P > 0 and N0_I > 0 and N0_H == 0:
        constr = [z[0,0] == 0, z[0,1] == 1]
    elif N0_P > 0 and N0_I > 0 and N0_H > 0:
        constr = [z[0,0] == 0, z[0,1] == 0]
    elif N0_P == 0 and N0_I == 0 and N0_H == 0:
        print("All cell compartments empty. No treatment needed")
        return
    else:
        raise ValueError("Invalid cell compartment configuration")
    constr += [z[:,0] + z[:,1] <= 1]   # Cell compartment configurations are mutually exclusive.
        
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
        constr += [(alpha_P*d[t] + beta_P*d[t]**2 - cvx.log(N_norm[t,0]) - np.log(c1)) + (cvx.log(Nv_tld_norm_lin[t+1,0]) + (N_tld_norm[t+1,0] - Nv_tld_norm_lin[t+1,0])/Nv_tld_norm_lin[t+1,0]) <= slack_dyn[t,0],
                   (cvx.log(N_tld_norm[t+1,0]) + alpha_P*d[t] - np.log(c1)) + (-cvx.log(Nv_norm_lin[t,0]) + beta_P*d_lin[t]**2 - (N_norm[t,0] - Nv_norm_lin[t,0])/Nv_norm_lin[t,0] + 2*beta_P*d_lin[t]*(d[t] - d_lin[t])) >= -slack_dyn[t,1]]
        
        # \tilde N_{t+1}^{I,v} = N_t^{I,v}*exp(-\alpha_I*d_t - \beta_I*d_t^2).
        constr += [(alpha_I*d[t] + beta_I*d[t]**2 - cvx.log(N_norm[t,2])) + (cvx.log(Nv_tld_norm_lin[t+1,1]) + (N_tld_norm[t+1,2] - Nv_tld_norm_lin[t+1,1])/Nv_tld_norm_lin[t+1,1]) <= slack_dyn[t,2],
                   (cvx.log(N_tld_norm[t+1,2]) + alpha_I*d[t]) + (-cvx.log(Nv_norm_lin[t,1]) + beta_I*d_lin[t]**2 - (N_norm[t,2] - Nv_norm_lin[t,1])/Nv_norm_lin[t,1] + 2*beta_I*d_lin[t]*(d[t] - d_lin[t])) >= -slack_dyn[t,3]]
        
        # \tilde N_{t+1}^{H,v} = \hat N_{t+1}^{H,v}*exp(-\alpha_H*d_t - \beta_H*d_t^2).
        constr += [(alpha_H*d[t] + beta_H*d[t]**2 - cvx.log(NH_hat_norm[t+1,0])) + (cvx.log(Nv_tld_norm_lin[t+1,2]) + (N_tld_norm[t+1,4] - Nv_tld_norm_lin[t+1,2])/Nv_tld_norm_lin[t+1,2]) <= slack_dyn[t,4],
                   (cvx.log(N_tld_norm[t+1,4]) + alpha_H*d[t]) + (-cvx.log(NH_hat_norm_lin[t+1]) + beta_H*d_lin[t]**2 - (NH_hat_norm[t+1,0] - NH_hat_norm_lin[t+1])/NH_hat_norm_lin[t+1] + 2*beta_H*d_lin[t]*(d[t] - d_lin[t])) >= -slack_dyn[t,5]]
        
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
    constr += [N_norm[-1,0] + N_norm[-1,2] + N_norm[-1,4] <= 1e-8*nt/n_scale]
    
    # Normal tissue BED constraint.
    # constr += [cvx.sum_squares(d + 0.5*ab_ratio_N) <= M_tld_N]
    constr += [bed_N <= M]
    
    # Maximum dose (per day) constraint.
    constr += [cvx.sum(d[t*delta_day:(t+1)*delta_day]) <= d_max_day for t in range(T_days)]
    
    # Weekend break assuming we start on Monday.
    # constr += [d[((t+1)*7-2)*delta_day:(t+1)*7*delta_day] == 0 for t in range(T_days // 7)]
    
    # TODO: Monotonic change in recompartmentalization state (restricts MIP search space).
    # constr += [z[T,0] == 1, z[T,1] == 0]     # All cells in P at end of treatment.
    
    prob = Problem(Minimize(obj), constr)
    
    # Initialize parameters.
    print("Initializing parameters...")
    df = max(np.sqrt(M_tld_N/T) - 0.5*ab_ratio_N, 0)   # Optimal (constant) fraction with only P compartment.
    d_init = np.repeat(df, T)
    
    N_init, N_tld_init, N_tld_tot_init, NH_hat_init, z_init = calc_cell_dynamics_three(d_init, [N0_P, N0_I, N0_H], [alpha_P, alpha_I, alpha_H], [beta_P, beta_I, beta_H], f_pro_P, T_C, T_loss, delta_t, k_m, recomp = True)
    # plot_plan_results(d_init, N_init, T, delta_t, [N0_P, N0_I, N0_H], model_name = "(P,I,H) Model, Initial Plan", file_prefix = fig_path + "const_frac")
    # sur_frac_init, eqd2_init, tcp_init, fx_init, schedule_init = EQD2_primer_sim_step(d_init, gf_in = gf, clf_in = clf, delta_t = delta_t, show = True, fileprefix = fig_path + "const_frac")
    d_per_day_init = np.array([np.sum(d_init[t*delta_day:(t+1)*delta_day]) for t in range(T_days)])
    bed_N_init = np.sum(d_per_day_init) + np.sum(d_per_day_init**2)/ab_ratio_N
    
    d_lin.value = d_init
    Nv_norm_lin.value = np.column_stack((N_init[:,0], N_init[:,2], N_init[:,4]))/n_scale
    Nv_tld_norm_lin.value = np.row_stack((np.array([N0_P, N0_I, N0_H]),   # First row is filler that isn't used in constraints (only for consistent indexing).
                                          np.column_stack((N_tld_init[:,0], N_tld_init[:,2], N_tld_init[:,4]))
                                         ))/n_scale
    NH_hat_norm_lin.value = np.concatenate((0.5*c3*np.array([N0_H]), NH_hat_init))/n_scale
    
    print("Starting CCP loop...")
    # obj_prev = np.sum(N_init[1:,:])/nt
    obj_prev = np.sum(N_init[1:,:])/nt + lam_bed*bed_N_init/T
    d_opt = d_init
    N_opt = N_init
    obj_opt = obj_prev
    slack_dyn_opt = np.zeros(slack_dyn.shape)
    slack_rec_opt = np.zeros(slack_rec.shape)
    
    k = 0
    obj_diff = obj_prev
    finished = False
    start_time = time()
    while not finished:
        if k % 10 == 0:
            print("CCP iteration: {0}".format(k))
    
        # Solve linearized problem.
        try:
            prob.solve(solver = solver_name, verbose = verbose)
        except cvx.error.SolverError:
            msg = "Solver {0} failed with status {1}. Terminating prematurely at iteration {2}".format(solver_name, prob.status, k)
            warn(msg)
            break
            
        if prob.status not in cvxpy_s.SOLUTION_PRESENT:
            raise RuntimeError("Solver {0} failed with status {1}".format(solver_name, prob.status))
            # msg = "Solver {0} failed with status {1}. Terminating prematurely at iteration {2}".format(solver_name, prob.status, k)
            # warn(msg)
            # break
        
        # Save optimal values.
        d_opt = d.value
        N_opt = N.value
        obj_opt = prob.value   
        slack_dyn_opt = slack_dyn.value
        slack_rec_opt = slack_rec.value
        
        # Update linearization point.
        d_lin.value = d.value
        Nv_norm_lin.value = np.column_stack((N_norm.value[:,0], N_norm.value[:,2], N_norm.value[:,4]))
        Nv_tld_norm_lin.value = np.column_stack((N_tld_norm.value[:,0], N_tld_norm.value[:,2], N_tld_norm.value[:,4]))
        NH_hat_norm_lin.value = NH_hat_norm.value[:,0]
        
        # Check stopping criterion.
        obj_diff = np.abs(obj_prev - prob.value)
        obj_prev = prob.value
        finished = (k + 1) >= max_iter or obj_diff <= delta_stop
        k = k + 1
    stop_time = time()
    run_time = stop_time - start_time
    
    np.save(data_path + 'three_comp-rec-dose.npy', d_opt)
    # np.save(data_path + "three_comp-rec-nbed_{0}-dose.npy".format(M), d_opt)
    plot_dose(d_opt, gf_in = gf, clf_in = clf, delta_t = delta_t, figsize = (12,8), show = True, fileprefix = fig_path + "three_comp-rec")
    
    print("Optimal objective:", obj_opt)
    # print("Optimal dose vector:", d_opt)
    # print("Optimal slack term:", np.sum(slack.value)/T)
    print("Optimal slack term:")
    print("Cell dynamics = {0}".format(np.sum(slack_dyn_opt)/slack_dyn_opt.size))
    print("Recompartmentalization = {0}".format(np.sum(slack_rec_opt)/slack_rec_opt.size))
    
    print("Optimal cell count:")
    print("P viable = {0}, P doomed = {1}".format(N_opt[-1,0], N_opt[-1,1]))
    print("I viable = {0}, I doomed = {1}".format(N_opt[-1,2], N_opt[-1,3]))
    print("H viable = {0}, H doomed = {1}".format(N_opt[-1,4], N_opt[-1,5]))
    
    print("Absolute change in objective:", obj_diff)
    print("Total iterations:", k)
    print("Elapsed time:", run_time)
    
    print("\nCalculating cell dynamics with optimal dose vector...")
    sur_frac_opt, eqd2_opt, tcp_opt, fx_opt, schedule_opt = EQD2_primer_sim_step(d_opt, gf_in = gf, clf_in = clf, alpha_p_ori = alpha_P, a_over_b = alpha_P/beta_P, oer_i = OER_I, oer_h = OER_H, 
                                                                                 delta_t = delta_t, verbose = False, show = True, filename = fig_path + "three_comp-rec-sf.jpg")
    print("Final survival fraction:", sur_frac_opt[-1,1])
    print("Final survival fraction by compartment: P viable = {0}, I viable = {1}, H viable = {2}".format(sur_frac_opt[-1,2], sur_frac_opt[-1,3], sur_frac_opt[-1,4]))
    print("Final normal tissue BED:", np.sum(fx_opt*(1 + fx_opt/ab_ratio_N)))
    print("Final EQD2: {0}, Final TCP: {1}".format(eqd2_opt, tcp_opt))

if __name__ == "__main__":
    main()
