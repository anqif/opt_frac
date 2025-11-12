import pickle
import numpy as np
import matplotlib.pyplot as plt
from copy import deepcopy

from opt_frac.tests.base_test import BaseTest
from opt_frac.optimization import solve_ccp, print_result
from opt_frac.plot_sim import plot_dose, EQD2_primer_sim_step, EQD2_primer_sim_comp
from opt_frac.utilities import convert_dose_delta

class TestAlgorithm(BaseTest):
    """Unit tests for the 2-step CCP algorithm"""

    def setUp(self):
        np.random.seed(1)
        super(TestAlgorithm, self).setUp()

        # Problem parameters.
        self.delta_t = 60                     # Time step in minutes.
        self.T_days = 15                      # Total days of treatment.
        # T = int((self.T_days*24*60)/self.delta_t)     # Total time steps.
        # delta_day = int(24*60/self.delta_t)      # Number of time steps per day.

        # Normal tissue parameters.
        # self.M_bed = 146.67                   # Upper bound on BED for normal tissue.
        self.M_bed = 378                      # Normal tissue BED for 3 x 18 Gy
        self.d_max_day = 18                   # Maximum total dose per day.

        # Algorithm parameters.
        self.lam_bed = 0
        # self.lam_bed = 0.025
        self.has_slack_dyn = True             # Slack on linearized cell dynamics constraints?
        self.has_slack_rec = False            # Slack on recompartmentalization MIP constraints?
        # self.weekend_break = False
        self.weekend_break = True

        self.refit_norec = True
        self.solver = "MOSEK"
        self.verbose = False
        self.show = False

    def test_norec(self):
        # n_scale = 0.25*self.nt
        n_scale = 0.5*self.nt
        max_iter = 1000
        delta_stop = 1e-3
        fileprefix = "three_comp-no_rec-delta_t_{0}".format(self.delta_t)
        # fileprefix = "three_comp-no_rec-delta_t_{0}-max_bed_{1}".format(self.delta_t, self.M_bed)

        if self.weekend_break:
            fileprefix = "{0}-weekend".format(fileprefix)

        alpha_P = self.alpha[0]
        beta_P = self.beta[0]

        print("Fitting model without recompartmentalization...")
        result = solve_ccp(self.nt, self.T_days, self.N0, self.alpha, self.beta, self.f_pro_P, self.T_C, self.T_loss,
                           self.delta_t, self.k_m, ab_ratio_N=self.ab_ratio_N, M_bed=self.M_bed, d_max_day=self.d_max_day,
                           lam_bed=self.lam_bed, n_scale=n_scale, has_slack_dyn=self.has_slack_dyn, has_slack_rec=self.has_slack_rec,
                           weekend_break=self.weekend_break, recomp=False, max_iter=max_iter, delta_stop=delta_stop,
                           solver=self.solver, verbose=self.verbose, filename=BaseTest.data_path + fileprefix + "-hist.pkl")
        print_result(result)
        # np.save(BaseTest.data_path + fileprefix + "-dose.npy", result["d"])

        plot_dose(result["d"], gf_in=self.gf, clf_in=self.clf, delta_t=self.delta_t, figsize=(12, 8), show=self.show,
                  fileprefix=BaseTest.fig_path + fileprefix)
        sur_frac_result = EQD2_primer_sim_step(result["d"], gf_in=self.gf, clf_in=self.clf, alpha_p_ori=alpha_P,
                                a_over_b=alpha_P / beta_P, oer_i=self.OER_I, oer_h=self.OER_H, delta_t=self.delta_t,
                                plot_survival=True, show=self.show, filename=BaseTest.fig_path + fileprefix + "-sf.jpg")
        BaseTest.print_sur_frac(sur_frac_result, self.ab_ratio_N)
        BaseTest.primer_EQD2_dose_comp(result["d"], self.gf, self.ab_ratio_N, self.delta_t, verbose=True)

    def test_two_step(self):
        # n_scale_norec = 0.25*self.nt
        n_scale_norec = 0.5 * self.nt
        max_iter_norec = 1000
        delta_stop_norec = 1e-3
        # fileprefix_norec = "three_comp-no_rec-delta_t_{0}".format(self.delta_t)
        fileprefix_norec = "three_comp-no_rec-delta_t_{0}-max_bed_{1}".format(self.delta_t, self.M_bed)

        n_scale_rec = 0.5 * self.nt
        max_iter_rec = 100
        delta_stop_rec = 1e-3
        # fileprefix_rec = "three_comp-rec-warm_start-delta_t_{0}".format(self.delta_t)
        fileprefix_rec = "three_comp-rec-warm_start-delta_t_{0}-max_bed_{1}".format(self.delta_t, self.M_bed)

        if self.weekend_break:
            fileprefix_norec = "{0}-weekend".format(fileprefix_norec)
            fileprefix_rec = "{0}-weekend".format(fileprefix_rec)

        alpha_P = self.alpha[0]
        beta_P = self.beta[0]

        if self.refit_norec:
            print("Fitting model without recompartmentalization...")
            result_norec = solve_ccp(self.nt, self.T_days, self.N0, self.alpha, self.beta, self.f_pro_P, self.T_C,
                                     self.T_loss, self.delta_t, self.k_m, ab_ratio_N=self.ab_ratio_N, M_bed=self.M_bed,
                                     d_max_day=self.d_max_day, lam_bed=self.lam_bed, n_scale=n_scale_norec,
                                     has_slack_dyn=self.has_slack_dyn, has_slack_rec=self.has_slack_rec,
                                     weekend_break=self.weekend_break, recomp=False, max_iter=max_iter_norec,
                                     delta_stop=delta_stop_norec, solver=self.solver, verbose=self.verbose,
                                     filename=BaseTest.data_path + fileprefix_norec + "-hist.pkl")
            print_result(result_norec)
            d_norec = result_norec["d"]
            # np.save(data_path + fileprefix_norec + "-dose.npy", result_norec["d"])
        else:
            print("Loading optimal dose without recompartmentalization...")
            # d_norec = np.load(BaseTest.data_path + fileprefix_norec + "-dose.npy")
            with open(BaseTest.data_path + fileprefix_norec + "-hist.pkl", "rb") as handle:
                norec_hist_list = pickle.load(handle)
            d_norec = norec_hist_list[-1]["d"]

        print("\nCalculating initial dose for model with recompartmentalization...")
        d_init = convert_dose_delta(d_norec, self.T_days, self.delta_t, self.delta_t)

        # TODO: Write function that allows us to continue from where we left off in the middle of CCP
        #   (e.g., when it terminates early due to solver scaling issues).
        print("\nFitting model with recompartmentalization using warm start...")
        result_rec = solve_ccp(self.nt, self.T_days, self.N0, self.alpha, self.beta, self.f_pro_P, self.T_C, self.T_loss,
                               self.delta_t, self.k_m, d_init=d_init, ab_ratio_N=self.ab_ratio_N, M_bed=self.M_bed,
                               d_max_day=self.d_max_day, lam_bed=self.lam_bed, n_scale=n_scale_rec,
                               has_slack_dyn=self.has_slack_dyn, has_slack_rec=self.has_slack_rec,
                               weekend_break=self.weekend_break, recomp=True, max_iter=max_iter_rec,
                               delta_stop=delta_stop_rec, solver=self.solver, verbose=self.verbose,
                               filename=BaseTest.data_path + fileprefix_rec + "-hist.pkl")
        print_result(result_rec)
        d_rec = result_rec["d"]
        # np.save(BaseTest.data_path + fileprefix_rec + "-iter_{0}-dose.npy".format(result_rec["iterations"]), result_rec["d"])

        plot_dose(d_rec, gf_in=self.gf, clf_in=self.clf, delta_t=self.delta_t, figsize=(12, 8), show=self.show,
                  fileprefix=BaseTest.fig_path + fileprefix_rec)
        sur_frac_result = EQD2_primer_sim_step(d_rec, gf_in=self.gf, clf_in=self.clf, alpha_p_ori=alpha_P,
                                    a_over_b=alpha_P / beta_P, oer_i=self.OER_I, oer_h=self.OER_H, delta_t=self.delta_t,
                                    plot_survival=True, show=self.show, filename=BaseTest.fig_path + fileprefix_rec + "-sf.jpg")
        BaseTest.print_sur_frac(sur_frac_result, self.ab_ratio_N)

        EQD2_primer_sim_comp([d_init, d_rec], gf_list=self.gf, clf_list=self.clf, ab_ratio_N=self.ab_ratio_N,
                             delta_t=self.delta_t, figsize=(12, 8), verbose=self.verbose, show=self.show,
                             label_list=["No Recompartmentalization", "Recompartmentalization (Warm Start)"],
                             # fileprefix=BaseTest.fig_path + "three_comp-warm_start-iter_{0}-comp".format(max_iter_rec))
                             fileprefix=BaseTest.fig_path + "three_comp-warm_start-iter_{0}-max_bed_{1}-comp".format(max_iter_rec, self.M_bed))
