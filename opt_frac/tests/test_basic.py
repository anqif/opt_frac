import numpy as np

from opt_frac.tests.base_test import BaseTest
from opt_frac.plot_sim import EQD2_primer_sim_step, plot_dose
from opt_frac.optimization import solve_ccp, print_result

class TestBasic(BaseTest):
    """Unit tests for basic CCP algorithm"""

    def setUp(self):
        np.random.seed(1)
        super(TestBasic, self).setUp()

        # Problem parameters.
        self.delta_t = 60          # Time step (sec) of cell update.
        self.T_days = 14           # Total days of treatment.
        
        # Normal tissue parameters. 
        self.M_bed = 146.67             # Upper bound on BED for normal tissue.
        self.d_max_day = 18             # Maximum total dose per day.

        # Algorithm parameters.
        self.max_iter = 1000
        self.delta_stop = 1e-3
        self.solver = "MOSEK"
        self.verbose = False
        self.show = True

    def test_basic(self):
        alpha_P = self.alpha[0]
        beta_P = self.beta[0]
        
        # Algorithm parameters.
        lam_bed = 0
        n_scale = 0.5*self.nt
        has_slack_dyn = True
        weekend_break = False

        print("Fitting model with CCP...")
        result = solve_ccp(self.nt, self.T_days, self.N0, self.alpha, self.beta, self.f_pro_P, self.T_C, self.T_loss,
                           self.delta_t, self.k_m, ab_ratio_N = self.ab_ratio_N, M_bed = self.M_bed, recomp = False,
                           d_max_day = self.d_max_day, lam_bed = lam_bed, n_scale = n_scale, has_slack_dyn = has_slack_dyn,
                           weekend_break = weekend_break, max_iter = self.max_iter, delta_stop = self.delta_stop,
                           solver = self.solver, verbose = self.verbose)
        print_result(result)
        
        # Save and plot fractionation schedule.
        np.save(BaseTest.data_path + "three_comp-no_rec-dose.npy", result["d"])
        plot_dose(result["d"], gf_in = self.gf, clf_in = self.clf, delta_t = self.delta_t, figsize = (12,8),
                  show = self.show, fileprefix = BaseTest.fig_path + "three_comp-no_rec")
        
        print("\nCalculating cell dynamics with optimal dose vector...")
        sur_frac_result = EQD2_primer_sim_step(result["d"], gf_in = self.gf, clf_in = self.clf, alpha_p_ori = alpha_P,
                                               a_over_b = alpha_P/beta_P, oer_i = self.OER_I, oer_h = self.OER_H,
                                               delta_t = self.delta_t, verbose = self.verbose, show = self.show,
                                               filename = BaseTest.fig_path + "three_comp-no_rec-sf.jpg")
        BaseTest.print_sur_frac(sur_frac_result, self.ab_ratio_N)

    def test_alpha_beta(self):
        T = int((self.T_days * 24 * 60) / self.delta_t)  # Total time steps.

        # Treat with proton therapy for first T//2 time steps, then switch to photon therapy.
        # alpha_vec = [alpha_P, alpha_I, alpha_H]; beta_vec = [beta_P, beta_I, beta_H].
        alpha_vec = [np.concatenate([np.repeat(0.205, T//2), np.repeat(0.305, T - T//2)]),
                     np.concatenate([np.repeat(0.205 / 1.0, T//2), np.repeat(0.305 / 1.7, T - T//2)]),
                     np.concatenate([np.repeat(0.205 / 1.05, T//2), np.repeat(0.305 / 1.37, T - T//2)])]
        beta_vec  = [np.concatenate([np.repeat(0.205 / 2.5, T // 2), np.repeat(0.305 / 2.8, T - T // 2)]),
                     np.concatenate([np.repeat(0.205 / (2.5 * 1.0**2), T // 2), np.repeat(0.305 / (2.8 * 1.7**2), T - T // 2)]),
                     np.concatenate([np.repeat(0.205 / (2.5 * 1.05**2), T // 2), np.repeat(0.305 / (2.8 * 1.37**2), T - T // 2)])]

        # Algorithm parameters.
        lam_bed = 0
        n_scale = 0.5 * self.nt
        has_slack_dyn = True
        weekend_break = False

        print("Fitting model with CCP...")
        result = solve_ccp(self.nt, self.T_days, self.N0, alpha_vec, beta_vec, self.f_pro_P, self.T_C, self.T_loss,
                           self.delta_t, self.k_m, ab_ratio_N=self.ab_ratio_N, M_bed=self.M_bed, recomp=False,
                           d_max_day=self.d_max_day, lam_bed=lam_bed, n_scale=n_scale, has_slack_dyn=has_slack_dyn,
                           weekend_break=weekend_break, max_iter=self.max_iter, delta_stop=self.delta_stop,
                           solver=self.solver, verbose=self.verbose)
        print_result(result)

        plot_dose(result["d"], gf_in = self.gf, clf_in = self.clf, delta_t = self.delta_t, figsize = (12,8), show = self.show)
        # TODO: Figure out how to simulate survival curves when (alpha, beta) parameters may vary over time.
