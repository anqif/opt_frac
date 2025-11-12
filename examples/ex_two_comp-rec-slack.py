import numpy as np
import cvxpy as cvx
import cvxpy.settings as cvxpy_s

from time import time
from cvxpy import Variable, Parameter, Problem, Minimize

from opt_frac.plot_sim import EQD2_primer_sim_step
from opt_frac.utilities import calc_cell_dynamics, plot_plan_results

def main():
    fig_path = r'~/Documents/Software/opt_frac/examples/figures/experiments/'
    data_path = r'~/Documents/Software/opt_frac/examples/data/'

    # Problem parameters.
    delta_t = 15                      # Time step in minutes.
    T_days = 14                        # Total days of treatment.
    T = int((T_days*24*60)/delta_t)   # Total time steps.
    delta_day = int(24*60/delta_t)    # Number of time steps per day.
    # T = 20
    
    rhot = 1e6        # Tumor cell density.
    vt = 64           # Volume of a tumorlet.
    nt = rhot*vt      # Total number of cells in a tumorlet.
    gf = 0.25         # Growth fraction.
    f_pro_P = 0.5     # Initial proliferation fraction in P compartment.
    T_C = 2*(24*60)   # Cell cycle time in minutes.
    k_m = 0.3
    
    alpha_P = 0.382
    beta_P = 0.0576
    OER_I = 2.0
    ab_ratio_N = 3                # Ratio alpha/beta for normal tissue cells.
    M = 146.67                    # Upper bound on BED for normal tissue.
    
    N0_P = (gf/f_pro_P)*nt
    N0_I = nt - N0_P
    
    # Constants in optimization problem.
    d_max_day = 18                # Maximum total dose per day.
    alpha_I = alpha_P/OER_I
    beta_I = beta_P/OER_I**2
    ab_ratio_P = alpha_P/beta_P   # Ratio alpha/beta for P compartment.
    ab_ratio_I = alpha_I/beta_I   # Ratio alpha/beta for I compartment.
    
    M_tld_N = ab_ratio_N*M + T*(0.5*ab_ratio_N)**2
    c1 = np.exp(f_pro_P*(np.log(2)/T_C)*delta_t)
    c2 = c1**(2*k_m - 1)
    R = c1**T                     # Make this bigger if problem is infeasible at first.
    # n_scale = nt                # Normalize initial cell counts N_0 -> N_0/n_scale. Adjust value if solver fails (generally due to precision issues).
    # n_scale = 1e3*nt
    n_scale = 1e-1*nt
    R_norm = R
    # R_norm = R/n_scale
    
    # Optimizer arguments.
    solver_name = "MOSEK"
    # verbose = False
    verbose = True
    max_iter = 30
    # max_iter = 1000
    delta_stop = 1e-3
    
    print("Initial cell count: P compartment = {0}, I compartment = {1}".format(N0_P, N0_I))
    print("Constants: c1 = {0}, c2 = {1}, n_scale = {2}".format(c1, c2, n_scale))
    
    # Two compartment (P,I) problem.
    print("Constructing problem...")
    # Define variables.
    d = Variable(T, nonneg = True)
    z = Variable(T+1, boolean = True)                  # z_t = 0 if all cells in P compartment, z_t = 1 if spillover into I compartment.
    slack_dyn = Variable((T,4), nonneg = True)
    slack_rec = Variable((T,10), nonneg = True)
    
    # N = Variable((T+1,4), nonneg = True)             # N_t = (N_t^{P,v}, N_t^{P,d}, N_t^{I,v}, N_t^{I,d}).
    N_norm = Variable((T+1,4), nonneg = True)          # N_t^{norm} = N_t/n_scale.
    # N_tld = Variable((T+1,4), nonneg = True)         # \tilde N_t = (\tilde N_t^{P,v}, \tilde N_t^{P,d}, \tilde N_t^{I,v}, \tilde N_t^{I,d}).
    N_tld_norm = Variable((T+1,4), nonneg = True)      # \tilde N_t^{norm} = \tilde N_t/n_scale.

    # Define linearization parameters.
    d_lin = Parameter(T, nonneg = True)                # d_t^{(k)} for t = 1,...,T.
    # Nv_lin = Parameter((T+1,2), pos = True)          # N_t^{v,(k)} = (N_t^{P,v,(k)}, N_t^{I,v,(k)}).
    Nv_norm_lin = Parameter((T+1,2), pos = True)       # N_t^{norm,v,(k)} = N_t^{v,(k)}/n_scale.
    # Nv_tld_lin = Parameter((T+1,2), pos = True)      # \tilde N_t^{v,(k)} = (\tilde N_t^{P,v,(k)}, \tilde N_t^{I,v,(k)}).
    Nv_tld_norm_lin = Parameter((T+1,2), pos = True)   # \tilde N_t^{norm,v,(k)} = \tilde N_t^{v,(k)}/n_scale.
    
    N = N_norm*n_scale                                 # N_t = (N_t^{P,v}, N_t^{P,d}, N_t^{I,v}, N_t^{I,d}).
    N_tld = N_tld_norm*n_scale                         # \tilde N_t = (\tilde N_t^{P,v}, \tilde N_t^{P,d}, \tilde N_t^{I,v}, \tilde N_t^{I,d}). 
    Nv_lin = Nv_norm_lin*n_scale                       # N_t^{v,(k)} = (N_t^{P,v,(k)}, N_t^{I,v,(k)}).
    # N_tld_tot = cvx.sum(N_tld, axis = 1)             # \tilde N_t^{tot} = \tilde N_t^{P,v} + \tilde N_t^{P,d} + \tilde N_t^{I,v} + \tilde N_t^{I,d}.
    N_tld_norm_tot = cvx.sum(N_tld_norm, axis = 1)     # \tilde N_t^{norm,tot} = \tilde N_t^{tot}/n_scale.
    
    # Define objective.
    lam = 0.025   # Penalty on normal tissue BED term.
    # obj = cvx.sum(N[1:,:])/nt + cvx.sum(slack)/T
    # obj = cvx.sum(N[1:,:])/nt + cvx.sum(slack_dyn)/(4*T) + cvx.sum(slack_rec)/(10*T)
    obj = cvx.sum(N[1:,:])/nt + cvx.sum(slack_dyn)/(4*T) + cvx.sum(slack_rec)/(10*T) + lam*cvx.sum_squares(d + 0.5*ab_ratio_N)/T
    
    # Define constraints.
    constr = [z[0] == 0] if N0_I == 0 else [z[0] == 1]
    constr += [N_norm[0,0] == N0_P/n_scale, N_norm[0,1] == 0, N_norm[0,2] == N0_I/n_scale, N_norm[0,3] == 0]
    constr += [slack_rec == 0]
    for t in range(T):
        # Linear cell dynamics.
        # \tilde N_{t+1}^{P,d} = c_2*(N_t^{P,v} + N_t^{P,d}) - (c_2/c_1)*\tilde N_{t+1}^{P,v}.
        # \tilde N_{t+1}^{I,d} = N_t^{I,d} + N_t^{I,v} - \tilde N_{t+1}^{I,v}
        constr += [N_tld_norm[t+1,1] == c2*(N_norm[t,0] + N_norm[t,1]) - (c2/c1)*N_tld_norm[t+1,0],
                   N_tld_norm[t+1,3] == N_norm[t,3] + N_norm[t,2] - N_tld_norm[t+1,2]]
        
        # Nonlinear cell dynamics, with CCP linearization.
        # \tilde N_{t+1}^{P,v} = N_t^{P,v}*c1*exp(-\alpha_P*d_t - \beta_P*d_t^2).
        constr += [(alpha_P*d[t] + beta_P*d[t]**2 - cvx.log(N_norm[t,0]) - np.log(c1)) + (cvx.log(Nv_tld_norm_lin[t+1,0]) + (N_tld_norm[t+1,0] - Nv_tld_norm_lin[t+1,0])/Nv_tld_norm_lin[t+1,0]) <= slack_dyn[t,0],
                   (cvx.log(N_tld_norm[t+1,0]) + alpha_P*d[t] - np.log(c1)) + (-cvx.log(Nv_norm_lin[t,0]) + beta_P*d_lin[t]**2 - (N_norm[t,0] - Nv_norm_lin[t,0])/Nv_norm_lin[t,0] + 2*beta_P*d_lin[t]*(d[t] - d_lin[t])) >= -slack_dyn[t,1]]
        
        # \tilde N_{t+1}^{I,v} = N_t^{I,v}*exp(-\alpha_I*d_t - \beta_I*d_t^2).
        constr += [(alpha_I*d[t] + beta_I*d[t]**2 - cvx.log(N_norm[t,2])) + (cvx.log(Nv_tld_norm_lin[t+1,1]) + (N_tld_norm[t+1,2] - Nv_tld_norm_lin[t+1,1])/Nv_tld_norm_lin[t+1,1]) <= slack_dyn[t,2],
                   (cvx.log(N_tld_norm[t+1,2]) + alpha_I*d[t]) + (-cvx.log(Nv_norm_lin[t,1]) + beta_I*d_lin[t]**2 - (N_norm[t,2] - Nv_norm_lin[t,1])/Nv_norm_lin[t,1] + 2*beta_I*d_lin[t]*(d[t] - d_lin[t])) >= -slack_dyn[t,3]]
        
        # Recompartmentalization.
        # State 0: All cells in P compartment, I compartment empty.
        # N_t^{P,v} + N_t^{P,d} = \tilde N_t^{tot}, \tilde N_t^{tot} <= N_0^P, N_t^{I,v} + N_t^{I,d} = 0.
        constr += [N_norm[t+1,0] + N_norm[t+1,1] - N_tld_norm_tot - z[t+1]*R_norm <= slack_rec[t,0], 
                   N_norm[t+1,0] + N_norm[t+1,1] - N_tld_norm_tot + z[t+1]*R_norm >= -slack_rec[t,1],
                   N_tld_norm_tot[t+1] - N0_P/n_scale - z[t+1]*R_norm <= slack_rec[t,2],
                   N_norm[t+1,2] + N_norm[t+1,3] - z[t+1]*R_norm <= slack_rec[t,3]]
                   # N_norm[t+1,2] + N_norm[t+1,3] + z[t+1]*R_norm >= -slack_rec[t,4] already satisfied by N_norm >= 0.
        
        # State 1: P compartment full, excess cells in I compartment.
        # N_t^{P,v} + N_t^{P,d} = N_0^P, \tilde N_t^{tot} >= N_0^P, N_t^{I,v} + N_t^{I,d} = \tilde N_t^{tot} - N_0^P.
        constr += [N_norm[t+1,0] + N_norm[t+1,1] - N0_P/n_scale - (1 - z[t+1])*R_norm <= slack_rec[t,5],
                   N_norm[t+1,0] + N_norm[t+1,1] - N0_P/n_scale + (1 - z[t+1])*R_norm >= -slack_rec[t,6],
                   N0_P/n_scale - N_tld_norm_tot[t+1] - (1 - z[t+1])*R_norm <= slack_rec[t,7],
                   N_norm[t+1,2] + N_norm[t+1,3] - N_tld_norm_tot[t+1] + N0_P/n_scale - (1 - z[t+1])*R_norm <= slack_rec[t,8],
                   N_norm[t+1,2] + N_norm[t+1,3] - N_tld_norm_tot[t+1] + N0_P/n_scale + (1 - z[t+1])*R_norm >= -slack_rec[t,9]]
    
    # Final viable tumor cell constraint.
    # constr += [(N[-1,0] + N[-1,2])/nt <= 0.01]
    constr += [N_norm[-1,0] + N_norm[-1,2] <= 1e-8*nt/n_scale]
    # constr += [N_norm[-delta_day:,0] + N_norm[-delta_day:,2] <= 1e-4*nt/n_scale]
    # constr += [N_norm[11*delta_day,2] <= 1e-6*nt/n_scale]
    
    # Normal tissue BED constraint.
    constr += [cvx.sum_squares(d + 0.5*ab_ratio_N) <= M_tld_N]
    
    # Maximum dose (per day) constraint.
    constr += [cvx.sum(d[t*delta_day:(t+1)*delta_day]) <= d_max_day for t in range(T_days)]
    
    # Weekend break assuming we start on Monday.
    # constr += [d[((t+1)*7-2)*delta_day:(t+1)*7*delta_day] == 0 for t in range(T_days // 7)]
    
    # Single change in recompartmentalization state (restricts MIP search space).
    constr += [z[T] == 0]                    # All cells in P at end of treatment.
    constr += [cvx.sum(z) <= T]              # Recompartmentalize during treatment.
    constr += [cvx.diff(z) <= 0]             # Only move from state 1 to state 0.
    constr += [cvx.sum(cvx.diff(z)) == -1]   # Recompartmentalization only occurs once.
    
    prob = Problem(Minimize(obj), constr)
    
    # Initialize parameters.
    print("Initializing parameters...")
    df = max(np.sqrt(M_tld_N/T) - 0.5*ab_ratio_N, 0)   # Optimal (constant) fraction with only P compartment.
    d_init = np.repeat(df, T)
    
    N_init, N_tld_init, N_tld_tot_init, z_init = calc_cell_dynamics(d_init, N0_P, N0_I, f_pro_P, T_C, delta_t, k_m, alpha_P, beta_P, alpha_I, beta_I, recomp = True)
    # plot_plan_results(d_init, N_init, T, delta_t, N0_P, N0_I, model_name = "(P,I) Model, Initial Plan", file_prefix = fig_path + 'const_frac')
    # sur_frac_init, eqd2_init, tcp_init, fx_init, schedule_init = EQD2_primer_sim_step(d_init, gf_in = gf, clf_in = 0.92, show = True, fileprefix = fig_path + "const_frac")
    
    d_lin.value = d_init
    Nv_norm_lin.value = np.column_stack((N_init[:,0], N_init[:,2]))/n_scale
    Nv_tld_norm_lin.value = np.row_stack((np.array([N0_P, N0_I]),   # First row is filler that isn't used in constraints (only for consistent indexing).
                                          np.column_stack((N_tld_init[:,0], N_tld_init[:,2]))
                                         ))/n_scale
    
    print("Starting CCP loop...")
    k = 0
    obj_prev = np.sum(N_init[1:,:])/nt
    obj_diff = obj_prev
    finished = False
    start_time = time()
    while not finished:
        if k % 10 == 0:
            print("CCP iteration: {0}".format(k))
    
        # Solve linearized problem.
        prob.solve(solver = solver_name, verbose = verbose)
        if prob.status not in cvxpy_s.SOLUTION_PRESENT:
            raise RuntimeError("Solver failed with status {0}".format(prob.status))
        
        # Update linearization point.
        d_lin.value = d.value
        Nv_norm_lin.value = np.column_stack((N_norm.value[:,0], N_norm.value[:,2]))
        Nv_tld_norm_lin.value = np.column_stack((N_tld_norm.value[:,0], N_tld_norm.value[:,2]))
        
        # Check stopping criterion.
        obj_diff = np.abs(obj_prev - prob.value)
        obj_prev = prob.value
        finished = (k + 1) >= max_iter or obj_diff <= delta_stop
        k = k + 1
    stop_time = time()
    run_time = stop_time - start_time
    np.save(data_path + 'two_comp-rec-dose.npy', d.value)
    
    print("Optimal objective:", prob.value)
    # print("Optimal dose vector:", d.value)
    # print("Optimal slack term:", np.sum(slack.value)/T)
    print("Optimal slack term: cell dynamics = {0}, recompartmentalization = {1}".format(np.sum(slack_dyn.value)/(4*T), np.sum(slack_rec.value)/(10*T)))
    print("Optimal cell count: P viable = {0}, P doomed = {1}, I viable = {2}, I doomed = {3}".format(N.value[-1,0], N.value[-1,1], N.value[-1,2], N.value[-1,3]))
    print("Absolute change in objective:", obj_diff)
    print("Total iterations:", k)
    print("Elapsed time:", run_time)
    
    print("Calculating cell dynamics with optimal dose vector...")
    # N_opt, N_tld_opt, N_tld_tot_opt, z_opt = calc_cell_dynamics(d.value, N0_P, N0_I, f_pro_P, T_C, delta_t, k_m, alpha_P, beta_P, alpha_I, beta_I, recomp = True)
    
    # print("Final objective without slack:", np.sum(N_opt[1:,:])/nt)
    # print("Final survival fraction:", (N_opt[-1,0] + N_opt[-1,2])/nt)
    # print("Final survival fraction:", (N_opt[-1,0] + N_opt[-1,2])/np.sum(N_opt[-1,:]))
    # print("Final cell count: P viable = {0}, P doomed = {1}, I viable = {2}, I doomed = {3}".format(N_opt[-1,0], N_opt[-1,1], N_opt[-1,2], N_opt[-1,3]))
    
    # plot_plan_results(d.value, N_opt, T, delta_t, N0_P, N0_I, model_name = "(P,I) Model, Optimal 1-Shot Plan", file_prefix = fig_path + 'two_comp-rec-slack')
    sur_frac_opt, eqd2_opt, tcp_opt, fx_opt, schedule_opt = EQD2_primer_sim_step(d.value, gf_in = gf, clf_in = 0.92, show = True, filename = fig_path + "two_comp-rec-sf.jpg")
    print("Final survival fraction:", sur_frac_opt[-1,1])
    print("Final survival fraction by compartment: P viable = {0}, I viable = {1}, H viable = {2}".format(sur_frac_opt[-1,2], sur_frac_opt[-1,3], sur_frac_opt[-1,4]))
    # print("Final EQD2: {0}, Final TCP: {1}".format(eqd2_opt, tcp_opt))

if __name__ == "__main__":
    main()
