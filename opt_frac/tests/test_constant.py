import pickle
import numpy as np
import matplotlib.pyplot as plt
from copy import deepcopy

from opt_frac.tests.base_test import BaseTest
from opt_frac.optimization import solve_ccp, print_result
from opt_frac.plot_sim import plot_dose, EQD2_primer_sim_step, EQD2_primer_sim_comp
from opt_frac.schedule import solve_ccp_sched
from opt_frac.utilities import convert_dose_delta

class TestConstant(BaseTest):
    """Unit tests for constant dose schedule"""

    def setUp(self):
        np.random.seed(1)
        super(TestConstant, self).setUp()

        # Problem parameters.
        self.delta_t = 60  # Time step (sec) of cell update.
        self.T_days = 15  # Total days of treatment.

        # Normal tissue parameters.
        self.M_bed = 146.67  # Upper bound on BED for normal tissue.
        self.d_max_day = 18  # Maximum total dose per day.
        # self.d_max_day = 10

        # Algorithm parameters.
        self.lam_bed = 0
        # self.lam_bed = 0.025
        self.has_slack_dyn = True  # Slack on linearized cell dynamics constraints?
        self.has_slack_rec = False  # Slack on recompartmentalization MIP constraints?
        self.weekend_break = False
        self.constant_dose = True

        self.refit_norec = True
        self.solver = "MOSEK"
        self.verbose = False
        self.show = False

    def test_norec(self):
        n_scale = 0.025 * self.nt
        # n_scale = 0.5*self.nt
        max_iter = 1000
        delta_stop = 1e-3
        fileprefix = "three_comp-no_rec-const-delta_t_{0}".format(self.delta_t)

        alpha_P = self.alpha[0]
        beta_P = self.beta[0]

        print("Fitting model without recompartmentalization...")
        result = solve_ccp(self.nt, self.T_days, self.N0, self.alpha, self.beta, self.f_pro_P, self.T_C, self.T_loss,
                           self.delta_t, self.k_m, ab_ratio_N=self.ab_ratio_N, M_bed=self.M_bed, d_max_day=self.d_max_day,
                           lam_bed=self.lam_bed, n_scale=n_scale, has_slack_dyn=self.has_slack_dyn,
                           has_slack_rec=self.has_slack_rec, weekend_break=self.weekend_break, treat_break=0,
                           treat_len=self.T_days, constant_dose=self.constant_dose, recomp=False, max_iter=max_iter,
                           delta_stop=delta_stop, solver=self.solver, verbose=self.verbose,
                           filename=BaseTest.data_path + fileprefix + "-hist.pkl")
        print_result(result)
        # np.save(BaseTest.data_path + fileprefix + "-dose.npy", result["d"])

        plot_dose(result["d"], gf_in=self.gf, clf_in=self.clf, delta_t=self.delta_t, figsize=(12, 8), show=self.show,
                  fileprefix=BaseTest.fig_path + fileprefix)
        sur_frac_result = EQD2_primer_sim_step(result["d"], gf_in=self.gf, clf_in=self.clf, alpha_p_ori=alpha_P,
                                a_over_b=alpha_P / beta_P, oer_i=self.OER_I, oer_h=self.OER_H, delta_t=self.delta_t,
                                plot_survival=True, show=self.show, filename=BaseTest.fig_path + fileprefix + "-sf.jpg")
        BaseTest.print_sur_frac(sur_frac_result, self.ab_ratio_N)

    def test_two_step(self):
        n_scale_norec = 0.025 * self.nt
        # n_scale_norec = 0.5*self.nt
        max_iter_norec = 1000
        delta_stop_norec = 1e-3
        fileprefix_norec = "three_comp-no_rec-const-delta_t_{0}".format(self.delta_t)

        n_scale_rec = 0.5 * self.nt
        max_iter_rec = 2
        delta_stop_rec = 1e-3
        fileprefix_rec = "three_comp-rec-warm_start-const-delta_t_{0}-iter_{1}".format(self.delta_t, max_iter_rec)

        alpha_P = self.alpha[0]
        beta_P = self.beta[0]

        if self.refit_norec:
            print("Fitting model without recompartmentalization...")
            result_norec = solve_ccp(self.nt, self.T_days, self.N0, self.alpha, self.beta, self.f_pro_P, self.T_C,
                                     self.T_loss, self.delta_t, self.k_m, ab_ratio_N=self.ab_ratio_N, M_bed=self.M_bed,
                                     d_max_day=self.d_max_day, lam_bed=self.lam_bed, n_scale=n_scale_norec,
                                     has_slack_dyn=self.has_slack_dyn, has_slack_rec=self.has_slack_rec,
                                     weekend_break=self.weekend_break, treat_break=0, treat_len=self.T_days,
                                     constant_dose=self.constant_dose, recomp=False, max_iter=max_iter_norec,
                                     delta_stop=delta_stop_norec, solver=self.solver, verbose=self.verbose,
                                     filename=BaseTest.data_path + fileprefix_norec + "-hist.pkl")
            print_result(result_norec)
            d_norec = result_norec["d"]
            # np.save(BaseTest.data_path + fileprefix_norec + "-dose.npy", result_norec["d"])
        else:
            print("Loading optimal dose without recompartmentalization...")
            # d_norec = np.load(BaseTest.data_path + fileprefix_norec + "-dose.npy")
            with open(BaseTest.data_path + fileprefix_norec + "-hist.pkl", "rb") as handle:
                norec_hist_list = pickle.load(handle)
            d_norec = norec_hist_list[-1]["d"]

        print("\nCalculating initial dose for model with recompartmentalization...")
        d_init = convert_dose_delta(d_norec, self.T_days, self.delta_t, self.delta_t)

        print("\nFitting model with recompartmentalization using warm start...")
        result_rec = solve_ccp(self.nt, self.T_days, self.N0, self.alpha, self.beta, self.f_pro_P, self.T_C, self.T_loss,
                               self.delta_t, self.k_m, d_init=d_init, ab_ratio_N=self.ab_ratio_N, M_bed=self.M_bed,
                               d_max_day=self.d_max_day, lam_bed=self.lam_bed, n_scale=n_scale_rec,
                               has_slack_dyn=self.has_slack_dyn, has_slack_rec=self.has_slack_rec,
                               weekend_break=self.weekend_break, constant_dose=self.constant_dose, recomp=True,
                               max_iter=max_iter_rec, delta_stop=delta_stop_rec, solver=self.solver, verbose=self.verbose,
                               filename=BaseTest.data_path + fileprefix_rec + "-hist.pkl")
        print_result(result_rec)
        d_rec = result_rec["d"]
        # np.save(data_path + fileprefix_rec + "-dose.npy", result_rec["d"])

        plot_dose(d_rec, gf_in=self.gf, clf_in=self.clf, delta_t=self.delta_t, figsize=(12, 8), show=self.show,
                  fileprefix=BaseTest.fig_path + fileprefix_rec)
        sur_frac_result = EQD2_primer_sim_step(d_rec, gf_in=self.gf, clf_in=self.clf, alpha_p_ori=alpha_P, a_over_b=alpha_P / beta_P,
                                               oer_i=self.OER_I, oer_h=self.OER_H, delta_t=self.delta_t, plot_survival=True,
                                               show=self.show, filename=BaseTest.fig_path + fileprefix_rec + "-sf.jpg")
        BaseTest.print_sur_frac(sur_frac_result, self.ab_ratio_N)

        EQD2_primer_sim_comp([d_init, d_rec], gf_list=self.gf, clf_list=self.clf, ab_ratio_N=self.ab_ratio_N,
                             delta_t=self.delta_t, figsize=(12, 8), verbose=self.verbose, show=self.show,
                             label_list=["No Recompartmentalization", "Recompartmentalization (Warm Start)"],
                             fileprefix=BaseTest.fig_path + "three_comp-warm_start-const-iter_{0}-comp".format(max_iter_rec))

    def test_schedule_norec(self):
        # n_scale = self.nt          # 1-week variable fractionation schedule without weekend break.
        # n_scale = 0.085*self.nt     # 1-week constant fractionation schedule without weekend break.
        n_scale = 0.075*self.nt   # 2-week schedule with weekend break.
        max_iter = 1000
        delta_stop = 1e-3
        constant_dose = False

        # T_days = 7
        # schedule = np.array([1, 2, 3, 4, 5, 6, 7])

        T_days = 12
        schedule = np.array([1, 8, 9, 10, 11])

        # T_days = 14
        # schedule = np.array([1, 2, 3, 4, 5, 8, 9, 10, 11, 12])

        if constant_dose:
            fileprefix = "three_comp-no_rec-const-sched_len_{0}".format(len(schedule))
        else:
            fileprefix = "three_comp-no_rec-sched_len_{0}".format(len(schedule))

        alpha_P = self.alpha[0]
        beta_P = self.beta[0]

        print("Fitting model without recompartmentalization...")
        result = solve_ccp_sched(self.nt, T_days, self.N0, self.alpha, self.beta, self.f_pro_P, self.T_C,
                                 self.T_loss, self.delta_t, self.k_m, ab_ratio_N=self.ab_ratio_N, M_bed=self.M_bed,
                                 d_max_day=self.d_max_day, lam_bed=self.lam_bed, n_scale=n_scale,
                                 has_slack_dyn=self.has_slack_dyn, has_slack_rec=self.has_slack_rec,
                                 schedule=schedule, constant_dose=constant_dose, recomp=False,
                                 max_iter=max_iter, delta_stop=delta_stop, solver=self.solver, verbose=self.verbose,
                                 filename=BaseTest.data_path + fileprefix + "-hist.pkl")
        print_result(result)
        # np.save(BaseTest.data_path + fileprefix + "-dose.npy", result["d"])

        plot_dose(result["d"], gf_in=self.gf, clf_in=self.clf, delta_t=self.delta_t, figsize=(12, 8),
                  show=self.show, fileprefix=BaseTest.fig_path + fileprefix)
        sur_frac_result = EQD2_primer_sim_step(result["d"], gf_in=self.gf, clf_in=self.clf, alpha_p_ori=alpha_P,
                                               a_over_b=alpha_P / beta_P, oer_i=self.OER_I, oer_h=self.OER_H,
                                               delta_t=self.delta_t, plot_survival=True, show=self.show,
                                               filename=BaseTest.fig_path + fileprefix + "-sf.jpg")
        BaseTest.print_sur_frac(sur_frac_result, self.ab_ratio_N)
