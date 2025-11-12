import pickle
import numpy as np
import matplotlib.pyplot as plt
from copy import deepcopy

from opt_frac.tests.base_test import BaseTest
from opt_frac.optimization import solve_ccp, print_result
from opt_frac.plot_sim import plot_dose, EQD2_primer_sim_step, EQD2_primer_sim_comp
from opt_frac.utilities import convert_dose_delta

class TestBreak(BaseTest):
    """Unit tests for fixed treatment break"""

    def setUp(self):
        np.random.seed(1)
        super(TestBreak, self).setUp()
        
        # Problem parameters.
        self.delta_t = 60                   # Time step (sec) of cell update.
        self.T_days = 15                    # Total days of treatment.
        # T = int((T_days*24*60)/delta_t)   # Total time steps.
        # delta_day = int(24*60/delta_t)    # Number of time steps per day.
        
        # Normal tissue parameters. 
        # TODO: Run without maximum BED constraint.
        self.M_bed = 146.67                # Upper bound on BED for normal tissue.
        self.d_max_day = 18                # Maximum total dose per day.

        # Treatment pattern.
        # treat_len consecutive dose fractions followed by treat_break consecutive days without any dose.
        self.treat_break = 6             # Consecutive days with zero dose.
        # self.treat_break = 13
        self.treat_len = 1                 # Consecutive days of treatment.
        
        # Algorithm parameters.
        self.lam_bed = 0
        # self.lam_bed = 0.025
        self.has_slack_dyn = True          # Slack on linearized cell dynamics constraints?
        self.has_slack_rec = False         # Slack on recompartmentalization MIP constraints?
        self.solver = "MOSEK"
        self.verbose = False
        self.show = False

    def test_norec(self):
        alpha_P = self.alpha[0]
        beta_P = self.beta[0]
    
        n_scale = 0.01*self.nt     # Treatment days = 15, length = 1, break = 6.
        # n_scale = 0.25*self.nt   # Treatment days = 15, length = 1, break = 13.
        weekend_break = False
        max_iter = 1000
        delta_stop = 1e-3
        fileprefix = "three_comp-no_rec-delta_t_{0}-break_{1}".format(self.delta_t, self.treat_break)
        
        print("Fitting model without recompartmentalization...")
        result = solve_ccp(self.nt, self.T_days, self.N0, self.alpha, self.beta, self.f_pro_P, self.T_C, self.T_loss,
                           self.delta_t, self.k_m, ab_ratio_N = self.ab_ratio_N, M_bed = self.M_bed,
                           d_max_day = self.d_max_day, lam_bed = self.lam_bed, n_scale = n_scale,
                           has_slack_dyn = self.has_slack_dyn, has_slack_rec = self.has_slack_rec,
                           weekend_break = weekend_break, treat_break = self.treat_break, treat_len = self.treat_len,
                           recomp = False, max_iter = max_iter, delta_stop = delta_stop, solver = self.solver,
                           verbose = self.verbose, filename = BaseTest.data_path + fileprefix + "-hist.pkl")
        print_result(result)
        # np.save(BaseTest.data_path + fileprefix + "-dose.npy", result["d"])
        
        plot_dose(result["d"], gf_in = self.gf, clf_in = self.clf, delta_t = self.delta_t, figsize = (12,8), show = self.show, fileprefix = BaseTest.fig_path + fileprefix)
        sur_frac_result = EQD2_primer_sim_step(result["d"], gf_in = self.gf, clf_in = self.clf, alpha_p_ori = alpha_P, a_over_b = alpha_P/beta_P,
                                               oer_i = self.OER_I, oer_h = self.OER_H, delta_t = self.delta_t, plot_survival = True, show = self.show,
                                               filename = BaseTest.fig_path + fileprefix + "-sf.jpg")
        BaseTest.print_sur_frac(sur_frac_result, self.ab_ratio_N)

    def test_rec(self):
        alpha_P = self.alpha[0]
        beta_P = self.beta[0]
    
        n_scale_norec = 0.01*self.nt     # Treatment days = 15, length = 1, break = 6.
        # n_scale_norec = 0.25*self.nt       # Treatment days = 15, length = 1, break = 13.
        n_scale_rec = 0.5*self.nt
        
        weekend_break = False
        max_iter_norec = 1000
        max_iter_rec = 100
        delta_stop = 1e-3
        recalc_norec = True
        
        fileprefix_norec = "three_comp-no_rec-delta_t_{0}-break_{1}".format(self.delta_t, self.treat_break)
        fileprefix_rec = "three_comp-rec-warm_start-delta_t_{0}-break_{1}".format(self.delta_t, self.treat_break)
        
        if recalc_norec:
            print("Fitting model without recompartmentalization...")
            result_norec = solve_ccp(self.nt, self.T_days, self.N0, self.alpha, self.beta, self.f_pro_P, self.T_C, self.T_loss, self.delta_t, self.k_m, ab_ratio_N = self.ab_ratio_N, M_bed = self.M_bed, 
                               d_max_day = self.d_max_day, lam_bed = self.lam_bed, n_scale = n_scale_norec, has_slack_dyn = self.has_slack_dyn, has_slack_rec = self.has_slack_rec, weekend_break = weekend_break, 
                               treat_break = self.treat_break, treat_len = self.treat_len, recomp = False, max_iter = max_iter_norec, delta_stop = delta_stop, solver = self.solver, verbose = self.verbose, 
                               filename = BaseTest.data_path + fileprefix_norec + "-hist.pkl")
            d_norec = result_norec["d"]
        else:
            print("Loading optimal dose without recompartmentalization...")
            # d_norec = np.load(data_path + fileprefix_norec + "-dose.npy")
            with open(BaseTest.data_path + fileprefix_norec + "-hist.pkl", "rb") as handle:
                norec_hist_list = pickle.load(handle)
            d_norec = norec_hist_list[-1]["d"]
        
        print("\nCalculating initial dose for model with recompartmentalization...")
        d_init = convert_dose_delta(d_norec, self.T_days, self.delta_t, self.delta_t)
        
        print("\nFitting model with recompartmentalization using warm start...")
        result_rec = solve_ccp(self.nt, self.T_days, self.N0, self.alpha, self.beta, self.f_pro_P, self.T_C, self.T_loss,
                               self.delta_t, self.k_m, d_init = d_init, ab_ratio_N = self.ab_ratio_N, M_bed = self.M_bed,
                               d_max_day = self.d_max_day, lam_bed = self.lam_bed, n_scale = n_scale_rec,
                               has_slack_dyn = self.has_slack_dyn, has_slack_rec = self.has_slack_rec,
                               weekend_break = weekend_break, treat_break = self.treat_break, treat_len = self.treat_len,
                               recomp = True, max_iter = max_iter_rec, delta_stop = delta_stop, solver = self.solver,
                               verbose = self.verbose, filename = BaseTest.data_path + fileprefix_rec + "-hist.pkl")
        print_result(result_rec)
        d_rec = result_rec["d"]
        # iter_rec = result_rec["iterations"]
        # np.save(data_path + fileprefix_rec + "-dose.npy", result_rec["d"])
        
        plot_dose(d_rec, gf_in = self.gf, clf_in = self.clf, delta_t = self.delta_t, figsize = (12,8), show = self.show,
                  fileprefix = BaseTest.fig_path + fileprefix_rec)
        sur_frac_result = EQD2_primer_sim_step(d_rec, gf_in = self.gf, clf_in = self.clf, alpha_p_ori = alpha_P,
                                               a_over_b = alpha_P/beta_P, oer_i = self.OER_I, oer_h = self.OER_H,
                                               delta_t = self.delta_t, plot_survival = True, show = self.show,
                                               filename = BaseTest.fig_path + fileprefix_rec + "-sf.jpg")
        BaseTest.print_sur_frac(sur_frac_result, self.ab_ratio_N)

        EQD2_primer_sim_comp([d_init, d_rec], gf_list = self.gf, clf_list = self.clf, ab_ratio_N = self.ab_ratio_N,
                             delta_t = self.delta_t, figsize = (12,8), verbose = False, show = self.show,
                             label_list = ["No Recompartmentalization", "Recompartmentalization (Warm Start)"],
                             fileprefix = BaseTest.fig_path + fileprefix_rec + "-comp")
    