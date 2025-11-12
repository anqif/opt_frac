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
    # delta_t = 60
    T_days = 14                       # Total days of treatment.
    # T_days = 5
    T = int((T_days*24*60)/delta_t)   # Total time steps.
    delta_day = int(24*60/delta_t)    # Number of time steps per day.
    
    rhot = 1e6        # Tumor cell density.
    vt = 64           # Volume of a tumorlet.
    nt = rhot*vt      # Total number of cells in a tumorlet.
    gf = 0.25         # Growth fraction.
    # gf = 0.15
    f_pro_P = 0.5     # Initial proliferation fraction in P compartment.
    T_C = 2*(24*60)   # Cell cycle time in minutes.
    k_m = 0.3
    
    alpha_P = 0.382
    beta_P = 0.0576
    OER_I = 2.0
    ab_ratio_N = 3                # Ratio alpha/beta for normal tissue cells.
    M = 146.67                    # Upper bound on BED for normal tissue.
    # M = 80
    
    N0_P = (gf/f_pro_P)*nt
    N0_I = nt - N0_P
    
    # Constants in optimization problem.
    d_max_day = 18                # Maximum total dose per day.
    # d_max_day = 5
    
    alpha_I = alpha_P/OER_I
    beta_I = beta_P/OER_I**2
    ab_ratio_P = alpha_P/beta_P   # Ratio alpha/beta for P compartment.
    ab_ratio_I = alpha_I/beta_I   # Ratio alpha/beta for I compartment.
    
    M_tld_N = ab_ratio_N*M + T*(0.5*ab_ratio_N)**2
    c1 = np.exp(f_pro_P*(np.log(2)/T_C)*delta_t)
    c2 = c1**(2*k_m - 1)
    # n_scale = nt                # Normalize initial cell counts N_0 -> N_0/n_scale. Adjust value if solver fails (generally due to precision issues).
    # n_scale = 1e3*nt
    n_scale = 0.5*nt   # 1e-1*nt
    
    # Optimizer arguments.
    solver_name = "MOSEK"
    verbose = False
    # max_iter = 30
    max_iter = 1000
    delta_stop = 1e-3
    
    print("Initial cell count: P compartment = {0}, I compartment = {1}".format(N0_P, N0_I))
    print("Constants: c1 = {0}, c2 = {1}, n_scale = {2}".format(c1, c2, n_scale))
    
    # Two compartment (P,I) problem.
    print("Constructing problem...")
    # Define variables.
    d = Variable(T, nonneg = True)
    # N = Variable((T+1,4), nonneg = True)         # N_t = (N_t^{P,v}, N_t^{P,d}, N_t^{I,v}, N_t^{I,d}).
    N_norm = Variable((T+1,4), nonneg = True)      # N_t^{norm} = N_t/n_scale.
    slack = Variable((T,4), nonneg = True)
    
    # Define linearization parameters.
    d_lin = Parameter(T, nonneg = True)            # d_t^{(k)} for t = 1,...,T.
    # Nv_lin = Parameter((T+1,2), pos = True)      # N_t^{v,(k)} = (N_t^{P,v,(k)}, N_t^{I,v,(k)}).
    Nv_norm_lin = Parameter((T+1,2), pos = True)   # N_t^{norm,v,(k)} = N_t^{v,(k)}/n_scale.
    
    N = N_norm*n_scale                             # N_t = (N_t^{P,v}, N_t^{P,d}, N_t^{I,v}, N_t^{I,d}).
    Nv_lin = Nv_norm_lin*n_scale                   # N_t^{v,(k)} = (N_t^{P,v,(k)}, N_t^{I,v,(k)}).
    
    # Define expressions.
    d_per_day = cvx.vstack([cvx.sum(d[t*delta_day:(t+1)*delta_day]) for t in range(T_days)])
    bed_N = cvx.sum(d_per_day) + cvx.sum_squares(d_per_day)/ab_ratio_N
    
    # Define objective.
    # obj = N_norm[-1,0] + N_norm[-1,2]
    # obj = N[-1,0] + N[-1,2]                      # N_T^{P,v} + N_T^{I,v}.
    # obj = cvx.sum(N[1:,0] + N[1:,2])             # \sum_{t=1}^T N_t^{P,v} + N_t^{I,v}.
    # obj = N_norm[-1,0] + N_norm[-1,2] + cvx.sum(slack)/T
    # obj = (N[-1,0] + N[-1,2])/nt + cvx.sum(slack)/T
    
    # obj = cvx.sum(N[1:,:])/nt + cvx.sum(slack)/T
    # obj = cvx.sum(N_norm[1:,:]) + cvx.sum(slack)/(4*T)
    
    lam = 0.025   # Penalty on normal tissue BED term.
    # obj = cvx.sum(N[1:,:])/nt + cvx.sum(slack)/T + cvx.sum(d)/T
    # obj = cvx.sum(N[1:,:])/nt + cvx.sum(slack)/T + lam*cvx.sum_squares(d + 0.5*ab_ratio_N)/T
    obj = cvx.sum(N[1:,:])/nt + cvx.sum(slack)/T + lam*bed_N/T
    
    # Define constraints.
    constr = [N_norm[0,0] == N0_P/n_scale, N_norm[0,1] == 0, N_norm[0,2] == N0_I/n_scale, N_norm[0,3] == 0]
    for t in range(T):
        # Linear cell dynamics.
        constr += [N_norm[t+1,1] == c2*(N_norm[t,0] + N_norm[t,1]) - (c2/c1)*N_norm[t+1,0],
                   N_norm[t+1,3] == N_norm[t,3] + N_norm[t,2] - N_norm[t+1,2]]
        
        # Nonlinear cell dynamics, with CCP linearization.
        # N_{t+1}^{P,v} = N_t^{P,v}*c1*exp(-\alpha_P*d_t - \beta_P*d_t^2).
        constr += [(alpha_P*d[t] + beta_P*d[t]**2 - cvx.log(N_norm[t,0]) - np.log(c1)) + (cvx.log(Nv_norm_lin[t+1,0]) + (N_norm[t+1,0] - Nv_norm_lin[t+1,0])/Nv_norm_lin[t+1,0]) <= slack[t,0],
                   (cvx.log(N_norm[t+1,0]) + alpha_P*d[t] - np.log(c1)) + (-cvx.log(Nv_norm_lin[t,0]) + beta_P*d_lin[t]**2 - (N_norm[t,0] - Nv_norm_lin[t,0])/Nv_norm_lin[t,0] + 2*beta_P*d_lin[t]*(d[t] - d_lin[t])) >= -slack[t,1]]
        
        # N_{t+1}^{I,v} = N_t^{I,v}*exp(-\alpha_I*d_t - \beta_I*d_t^2).
        constr += [(alpha_I*d[t] + beta_I*d[t]**2 - cvx.log(N_norm[t,2])) + (cvx.log(Nv_norm_lin[t+1,1]) + (N_norm[t+1,2] - Nv_norm_lin[t+1,1])/Nv_norm_lin[t+1,1]) <= slack[t,2],
                   (cvx.log(N_norm[t+1,2]) + alpha_I*d[t]) + (-cvx.log(Nv_norm_lin[t,1]) + beta_I*d_lin[t]**2 - (N_norm[t,2] - Nv_norm_lin[t,1])/Nv_norm_lin[t,1] + 2*beta_I*d_lin[t]*(d[t] - d_lin[t])) >= -slack[t,3]]
    
    # Final viable tumor cell constraint.
    # constr += [cvx.sum(N[-1,:])/nt <= 0.01]
    # constr += [(N[-1,0] + N[-1,2])/nt <= 1e-6]
    # constr += [(N[-delta_day-1,0] + N[-delta_day-1,2])/nt <= 1e-4]
    constr += [N_norm[-1,0] + N_norm[-1,2] <= 1e-8*nt/n_scale]
    # constr += [N_norm[-delta_day:,0] + N_norm[-delta_day:,2] <= 1e-4*nt/n_scale]
    # constr += [N_norm[11*delta_day,2] <= 1e-6*nt/n_scale]
    
    # Normal tissue BED constraint.
    # TODO: Should normal tissue BED be calculated by dose per time step or per day?
    # constr += [cvx.sum_squares(d + 0.5*ab_ratio_N) <= M_tld_N]
    # constr += [cvx.sum_squares(d_per_day + 0.5*ab_ratio_N) <= M_tld_N]
    constr += [bed_N <= M]
    
    # Maximum dose (per day) constraint.
    # constr += [cvx.sum(d[t*delta_day:(t+1)*delta_day]) <= d_max_day for t in range(T_days)]
    constr += [d_per_day <= d_max_day]
    
    # Weekend break assuming we start on Monday.
    # constr += [d[((t+1)*7-2)*delta_day:(t+1)*7*delta_day] == 0 for t in range(T_days // 7)]
    
    prob = Problem(Minimize(obj), constr)
    
    # Initialize parameters.
    print("Initializing parameters...")
    df = max(np.sqrt(M_tld_N/T) - 0.5*ab_ratio_N, 0)   # Optimal (constant) fraction with only P compartment.
    d_init = np.repeat(df, T)
    
    # N_init, N_tld_init, N_tld_tot_init, z_init = calc_cell_dynamics(d_init, N0_P, N0_I, f_pro_P, T_C, delta_t, k_m, alpha_P, beta_P, alpha_I, beta_I, recomp = False)
    N_init, N_tld_init, N_tld_tot_init, z_init = calc_cell_dynamics(d_init, N0_P, N0_I, f_pro_P, T_C, delta_t, k_m, alpha_P, beta_P, alpha_I, beta_I, recomp = True)
    # plot_plan_results(d_init, N_init, T, delta_t, N0_P, N0_I, model_name = "(P,I) Model, Initial Plan", file_prefix = fig_path + 'const_frac')
    # sur_frac_init, eqd2_init, tcp_init, fx_init, schedule_init = EQD2_primer_sim_step(d_init, gf_in = gf, clf_in = 0.92, show = True, fileprefix = fig_path + "const_frac")
    d_per_day_init = np.array([np.sum(d_init[t*delta_day:(t+1)*delta_day]) for t in range(T_days)])
    bed_N_init = np.sum(d_per_day_init) + np.sum(d_per_day_init**2)/ab_ratio_N
    
    d_lin.value = d_init
    Nv_norm_lin.value = np.column_stack((N_init[:,0], N_init[:,2]))/n_scale
    
    print("Starting CCP loop...")
    k = 0
    # obj_prev = np.sum(N_init[1:,:])/nt
    obj_prev = np.sum(N_init[1:,:])/nt + lam*bed_N_init/T
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
        
        # Check stopping criterion.
        obj_diff = np.abs(obj_prev - prob.value)
        obj_prev = prob.value
        finished = (k + 1) >= max_iter or obj_diff <= delta_stop
        k = k + 1
    stop_time = time()
    run_time = stop_time - start_time
    
    d_opt = d.value
    obj_opt = prob.value
    slack_avg_opt = np.sum(slack.value)/T
    # N_BED_opt = np.sum(d_opt*(1 + d_opt/ab_ratio_N))

    # Save optimal dose.
    np.save(data_path + 'two_comp-no_rec-dose.npy', d_opt)
    
    print("Optimal objective:", obj_opt)
    # print("Optimal dose vector:", d_opt)
    print("Optimal slack term:", slack_avg_opt)
    # print("Optimal cell count: P viable = {0}, P doomed = {1}, I viable = {2}, I doomed = {3}".format(N.value[-1,0], N.value[-1,1], N.value[-1,2], N.value[-1,3]))
    print("Optimal cell count:")
    print("P viable = {0}, P doomed = {1}".format(N.value[-1,0], N.value[-1,1]))
    print("I viable = {0}, I doomed = {1}".format(N.value[-1,2], N.value[-1,3]))
    # print("Optimal normal tissue BED:", N_BED_opt)
    print("Absolute change in objective:", obj_diff)
    print("Total iterations:", k)
    print("Elapsed time:", run_time)
    
    print("\nCalculating cell dynamics with optimal dose vector...")
    # N_opt, N_tld_opt, N_tld_tot_opt, z_opt = calc_cell_dynamics(d_opt, N0_P, N0_I, f_pro_P, T_C, delta_t, k_m, alpha_P, beta_P, alpha_I, beta_I, recomp = False)
    # N_opt, N_tld_opt, N_tld_tot_opt, z_opt = calc_cell_dynamics(d_opt, N0_P, N0_I, f_pro_P, T_C, delta_t, k_m, alpha_P, beta_P, alpha_I, beta_I, recomp = True)
    
    # print("Final objective without slack:", np.sum(N_opt[1:,:])/nt)
    # print("Final survival fraction:", (N_opt[-1,0] + N_opt[-1,2])/nt)
    # print("Final survival fraction:", (N_opt[-1,0] + N_opt[-1,2])/np.sum(N_opt[-1,:]))
    # print("Final cell count: P viable = {0}, P doomed = {1}, I viable = {2}, I doomed = {3}".format(N_opt[-1,0], N_opt[-1,1], N_opt[-1,2], N_opt[-1,3]))
    
    # plot_plan_results(d.value, N_opt, T, delta_t, N0_P, N0_I, model_name = "(P,I) Model, Optimal 1-Shot Plan", file_prefix = fig_path + 'two_comp-no_rec-slack')
    sur_frac_opt, eqd2_opt, tcp_opt, fx_opt, schedule_opt = EQD2_primer_sim_step(d_opt, gf_in = gf, clf_in = 0.92, delta_t = delta_t, verbose = False, show = True, 
                                                                                 filename = fig_path + "two_comp-no_rec-sf.jpg")
    print("Final survival fraction:", sur_frac_opt[-1,1])
    print("Final survival fraction by compartment: P viable = {0}, I viable = {1}, H viable = {2}".format(sur_frac_opt[-1,2], sur_frac_opt[-1,3], sur_frac_opt[-1,4]))
    print("Final normal tissue BED:", np.sum(fx_opt*(1 + fx_opt/ab_ratio_N)))
    print("Final EQD2: {0}, Final TCP: {1}".format(eqd2_opt, tcp_opt))

if __name__ == "__main__":
    main()
