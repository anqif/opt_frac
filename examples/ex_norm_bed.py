import pickle
import mosek
import cvxpy
import numpy as np
import matplotlib.pyplot as plt
from warnings import warn

from opt_frac.plot_sim import EQD2_primer_sim_step
from opt_frac.optimization import solve_ccp, print_result

def main():
    fig_path = r'~/Documents/Software/opt_frac/examples/figures/experiments/'
    data_path = r'~/Documents/Software/opt_frac/examples/data/'
    show = False

    # Problem parameters.
    delta_t = 60          # Time step (sec) of cell update.
    T_days = 14           # Total days of treatment.
    
    # Cell parameters.
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
    
    N0_P = (gf/f_pro_P)*nt
    N0_H = clf*gf*(T_loss/T_C)*nt
    N0_I = nt - N0_P - N0_H
    N0 = [N0_P, N0_I, N0_H]
    
    # Proton parameters.
    alpha_P = 0.205
    beta_P = alpha_P/2.5
    OER_I = 1.0
    OER_H = 1.05
    
    alpha_I = alpha_P/OER_I
    beta_I = beta_P/OER_I**2
    alpha_H = alpha_P/OER_H
    beta_H = beta_P/OER_H**2

    alpha = [alpha_P, alpha_I, alpha_H]
    beta = [beta_P, beta_I, beta_H]
    
    # Normal tissue parameters. 
    ab_ratio_N = 3                # Ratio alpha/beta for normal tissue cells.
    d_max_day = 18                # Maximum total dose per day.
    # M_bed_list = [80, 120, 140, 160, 210, 216.7, 240, 378]   # Upper bound on BED for normal tissue.
    M_bed_list = [80, 90, 110, 120, 140, 160]
    
    # Algorithm parameters.
    lam_bed = 0
    n_scale = 0.5*nt
    has_slack_dyn = True
    has_slack_rec = True
    weekend_break = False
    recomp = False
    
    max_iter = 1000
    delta_stop = 1e-3
    solver = "MOSEK"
    verbose = False
    
    try:
        print("Importing prior results...")
        with open(data_path + "three_comp-no_rec-nbed.pkl", "rb") as handle:
            result_list = pickle.load(handle)
            bed_norm_max_fitted = [result["bed_normal_max"] for result in result_list]
    except IOError:
        print("No prior results found, starting from scratch...")
        result_list = []
        bed_norm_max_fitted = []

    print("Fitting models with CCP...")
    for M_bed in M_bed_list:
        print("\nMaximum normal tissue BED:", M_bed)
        if M_bed in bed_norm_max_fitted:
            print("Saved results found, moving to next bound in list")
            continue
        
        try:
            result = solve_ccp(nt, T_days, N0, alpha, beta, f_pro_P, T_C, T_loss, delta_t, k_m, ab_ratio_N = ab_ratio_N, M_bed = M_bed, d_max_day = d_max_day, 
                               lam_bed = lam_bed, n_scale = n_scale, has_slack_dyn = has_slack_dyn, has_slack_rec = has_slack_rec, weekend_break = weekend_break, 
                               recomp = recomp, max_iter = max_iter, delta_stop = delta_stop, solver = solver, verbose = verbose)
        except mosek.Error as e:
            msg = "Response code: {0}\nMessage: {1}".format(e.errno, e.msg)
            warn(msg)
            continue
        except cvxpy.error.SolverError:
            msg = "Solver {0} failed. Moving to next weight in list".format(solver)
            warn(msg)
            continue
            
        # print_result(result)
        sur_frac_opt, eqd2_opt, tcp_opt, fx_opt, schedule_opt = EQD2_primer_sim_step(result["d"], gf_in = gf, clf_in = clf, alpha_p_ori = alpha_P, a_over_b = alpha_P/beta_P, 
                                                                                     oer_i = OER_I, oer_h = OER_H, delta_t = delta_t, plot_survival = False)
                                                          
        result["bed_normal_max"] = M_bed
        result["survival"] = sur_frac_opt
        result_list.append(result)
        bed_norm_max_fitted.append(M_bed)
    
        with open(data_path + "three_comp-no_rec-nbed.pkl", "wb") as handle:
            pickle.dump(result_list, handle, protocol = pickle.HIGHEST_PROTOCOL)
    
    # Plot survival fraction vs. normal tissue BED at end of treatment.
    bed_norm_arr = np.array([result["bed_normal"] for result in result_list])
    sur_frac_arr = np.array([result["survival"][-1,1] for result in result_list])
    idx_sort = np.argsort(bed_norm_arr)
    bed_norm_sort = bed_norm_arr[idx_sort]
    sur_frac_sort = sur_frac_arr[idx_sort]
    
    fig = plt.figure(figsize = (12,8))
    plt.semilogy(bed_norm_sort, sur_frac_sort, marker = "o")
    plt.title("Final Survival Fraction vs. Normal Tissue BED\nfor Optimal Fractionation Schedule")
    plt.xlabel("Normal Tissue BED")
    plt.ylabel("Survival Fraction")
    plt.show()
    fig.savefig(fig_path + "three_comp_no_rec-sf_nbed.jpg", bbox_inches = "tight", dpi = 300)
    
if __name__ == "__main__":
    main()
