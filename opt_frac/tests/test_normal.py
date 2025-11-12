import pickle
import numpy as np
from copy import deepcopy

import matplotlib
matplotlib.use("TkAgg")   # Temporary solution to Qt no plugin error.
import matplotlib.pyplot as plt

from opt_frac.schedule import solve_ccp_sched
from opt_frac.tests.base_test import BaseTest
from opt_frac.optimization import solve_ccp, print_result, InfeasibleError
from opt_frac.plot_sim import EQD2_primer_sim_step, plot_dose_sf_stacked, create_schedule
from opt_frac.utilities import calc_normal_bed, calc_normal_bed_const


class TestNormal(BaseTest):
    """Unit tests for tradeoff between optimal SF and NBED"""

    def setUp(self):
        np.random.seed(1)
        super(TestNormal, self).setUp()
        self.delta_t = 60

        # Normal tissue parameters.
        self.d_max_day = 18          # Maximum total dose per day.
        # self.d_max_day = 10

        # Algorithm parameters.
        self.has_slack_dyn = True    # Slack on linearized cell dynamics constraints?
        self.has_slack_rec = False   # Slack on recompartmentalization MIP constraints?
        # self.weekend_break = False
        self.weekend_break = True

        self.solver = "MOSEK"
        self.verbose = False
        self.show = False
        self.refit = False
        # self.refit = True
        self.filter_results = True

    @staticmethod
    def gen_dict():
        return {"nbed_list": np.array([]), "sf_list": np.array([]), "d_list": [], "sched_list": [], "T_list": []}

    @staticmethod
    def plot_sf_versus_nbed(result, title=None, xlim=None, ylim=None, annotations=None, show=True, filename=None):
        idx_sort = np.argsort(result["nbed_list"])
        nbed_sort = result["nbed_list"][idx_sort]
        sf_sort = result["sf_list"][idx_sort]

        fig = plt.figure(figsize=(12, 8))
        plt.semilogy(nbed_sort, sf_sort, marker="o")
        # plt.xticks(np.arange(np.min(nbed_sort), np.max(nbed_sort) + 1, 1))

        # Annotate each point.
        if annotations is not None:
            ant_sort = [annotations[i] for i in idx_sort]
            dx = 0.02 * (np.max(nbed_sort) - np.min(nbed_sort))
            dy = 0.001 * (np.max(sf_sort) - np.min(sf_sort))
            # dy = 0.05 * (np.max(sf_sort) - np.min(sf_sort))
            for i, txt in enumerate(ant_sort):
                plt.annotate(txt, (nbed_sort[i] + dx, sf_sort[i] + 0.05**i*dy))

        if xlim is not None:
            plt.xlim(xlim)
        if ylim is not None:
            plt.ylim(ylim)
        plt.xlabel("Normal Tissue Biologically Effective Dose (BED)")
        plt.ylabel("Total Survival Fraction")
        if title is not None:
            plt.title(title)
        if show:
            plt.show()
        if filename is not None:
            fig.savefig(BaseTest.fig_exp_path + filename, bbox_inches="tight", dpi=300)

    # TODO: 1) Compare SF vs. NBED curve for a) schedule + constant dose, b) schedule + variable dose, and
    #          c) no schedule + variable dose.
    #       2) Potentially penalize NBED in objective (as well as lower hard NBED constraint to ~80 Gy).
    def test_sched_norec(self):
        # Problem parameters.
        # Proton parameters (variable dose).
        # n_scale = 0.5*self.nt    # [1, 8, 9].
        # n_scale = 0.35*self.nt   # [1, 8, 9, 10] with optional [11].
        # n_scale = 0.2*self.nt    # [1, 8, 9, 10, 11].
        # n_scale = 0.1*self.nt    # [1, 8, 9, 10, 11, 12, 15, 16] with optional [17, 18, 19].
        # n_scale = 0.15*self.nt   # [1, 8, ..., 12, 15, ..., 19, 22, 23, 24].

        # Proton parameters (constant dose).
        n_scale = 0.9625*self.nt
        # n_scale = 0.976*self.nt
        # n_scale = 0.6*self.nt    # [1, 5, 8, 9, 10, 11, 12, 15, 16, 17, 18, 19].
        # n_scale = 0.75*self.nt   # [1, 8, ..., 12, 15, ..., 19, 22, 23, 24].

        # Photon parameters.
        # n_scale = 2*self.nt   # [1, 8, 9].
        # n_scale = 4*self.nt   # [1, 8, 9, 10] with optional [11].
        # n_scale = self.nt     # [1, 8, 9, 10, 11, 12, 15, 16] with optional [17, 18, 19, 22, 23, 24].

        max_iter = 1000
        delta_stop = 1e-3

        # Tuning parameters.
        lam_bed = 0
        constant_dose = False
        # constant_dose = True

        # Range of treatment schedules.
        # parm_dict_list = [{"schedule": [1, 8, 9], "M_bed": calc_normal_bed_const(18, 3, self.ab_ratio_N)}]
        parm_dict_list = [{"schedule": [1, 8, 9], "M_bed": calc_normal_bed_const(18, 3, self.ab_ratio_N)},
                          {"schedule": [1, 8, 9, 10], "M_bed": calc_normal_bed_const(12, 4, self.ab_ratio_N)},
                          {"schedule": [1, 8, 9, 10, 11], "M_bed": calc_normal_bed_const(10, 5, self.ab_ratio_N)},
                          {"schedule": [1, 8, 9, 10, 11, 12, 15, 16], "M_bed": calc_normal_bed_const(7.5, 8, self.ab_ratio_N)},
                          {"schedule": [1, 5, 8, 9, 10, 11, 12, 15, 16, 17, 18, 19], "M_bed": calc_normal_bed_const(5, 12, self.ab_ratio_N)},
                          {"schedule": [1, 5, 8, 9, 10, 11, 12, 15, 16, 17, 18, 19, 22, 23, 24], "M_bed": calc_normal_bed_const(4, 15, self.ab_ratio_N)}
                         ]

        alpha_P = self.alpha[0]
        beta_P = self.beta[0]

        if constant_dose:
            filename = "schedule-no_rec-const-max_iter_{0}.pkl".format(max_iter)
            # fileprefix = "schedule-no_rec-const"
        else:
            filename = "schedule-no_rec-max_iter_{0}.pkl".format(max_iter)
            # fileprefix = "schedule-no_rec"

        result_dict = BaseTest.get_result_dict(filename, TestNormal.gen_dict, self.refit)

        print("Fitting model without recompartmentalization...")
        for parm_dict in parm_dict_list:
            print("\nSchedule = {0}\tMax Normal BED = {1}".format(parm_dict["schedule"], parm_dict["M_bed"]))
            if parm_dict["schedule"] in result_dict["sched_list"]:
                print("Saved results found, moving to next schedule in list")
                continue

            try:
                # Solve for best dose schedule using CCP (no recompartmentalization).
                T_days = np.max(parm_dict["schedule"])
                result = solve_ccp_sched(self.nt, T_days, self.N0, self.alpha, self.beta, self.f_pro_P, self.T_C,
                                         self.T_loss, self.delta_t, self.k_m, ab_ratio_N=self.ab_ratio_N,
                                         M_bed=parm_dict["M_bed"], d_max_day=self.d_max_day, lam_bed=lam_bed,
                                         n_scale=n_scale, has_slack_dyn=self.has_slack_dyn,
                                         has_slack_rec=self.has_slack_rec, schedule=parm_dict["schedule"],
                                         constant_dose=constant_dose, recomp=False, max_iter=max_iter,
                                         delta_stop=delta_stop, solver=self.solver, verbose=self.verbose)
                print_result(result)

                # Simulate cell dynamics using resulting dose schedule.
                sur_frac_result = EQD2_primer_sim_step(result["d"], gf_in=self.gf, clf_in=self.clf, alpha_p_ori=alpha_P,
                                                       a_over_b=alpha_P / beta_P, oer_i=self.OER_I, oer_h=self.OER_H,
                                                       delta_t=self.delta_t, plot_survival=False, show=False)
                sur_frac_tot = sur_frac_result[0][-1,1]
                print("True survival fraction:", sur_frac_tot)

                # Save optimal dose and true survival fraction.
                result_dict["nbed_list"] = np.concatenate([result_dict["nbed_list"], np.array([result["normal_bed"]])])
                result_dict["sf_list"] = np.concatenate([result_dict["sf_list"], np.array([sur_frac_tot])])
                result_dict["d_list"].append(result["d"])
                result_dict["sched_list"].append(parm_dict["schedule"])

                with open(BaseTest.save_path + filename, 'wb') as handle:
                    pickle.dump(result_dict, handle, protocol=pickle.HIGHEST_PROTOCOL)
            except InfeasibleError as error:
                print(error)
                continue

        print("\nPlotting survival fraction vs. normal tissue BED for various schedules...")
        annotations = ["L = {0}".format(len(parm_dict["schedule"])) for parm_dict in parm_dict_list]
        TestNormal.plot_sf_versus_nbed(result_dict, title="Final Survival Fraction vs. Normal Tissue BED (Proton, Variable Dose, Schedule)",
                                       xlim=(0, 300), annotations=annotations, show=self.show,
                                       filename="sf_total_vs_nbed-no_rec-schedule.jpg")

        print("\nPlotting schedule with lowest survival fraction...")
        # idx_min = 3   Photon: [1, 8, 9, 10, 11, 12, 15, 16].
        idx_min = np.argmin(result_dict["sf_list"])
        d_opt = result_dict["d_list"][idx_min]
        sf_opt = result_dict["sf_list"][idx_min]
        nbed_opt = result_dict["nbed_list"][idx_min]
        sched_opt = result_dict["sched_list"][idx_min]

        print("Schedule: {0}".format(sched_opt))
        d_opt_sched = create_schedule(d_opt, delta_t = self.delta_t, only_nonzero = True)[0]
        # d_per_day_opt = create_schedule(d_opt, delta_t = self.delta_t, only_nonzero = False)[0]
        # d_per_day_opt = np.array(d_per_day_opt)
        # d_opt_sched = d_per_day_opt[np.array(sched_opt).astype(int) - 1]
        print("Nonzero Dose: {0}".format(d_opt_sched))
        print("Survival fraction: {0}".format(sf_opt))
        print("Normal tissue BED: {0}".format(nbed_opt))

        fileprefix = "three_comp-no_rec-sched_len_{0}".format(len(sched_opt))
        plot_dose_sf_stacked(d_opt, gf_in=self.gf, clf_in=self.clf, alpha_p_ori=alpha_P, a_over_b=alpha_P / beta_P,
                             oer_i=self.OER_I, oer_h=self.OER_H, delta_t=self.delta_t, figsize=(8, 12),
                             title="Survival Fraction for Final Dose Schedule (Proton, Variable Dose)", show=self.show,
                             filename=BaseTest.fig_exp_path + fileprefix + "-dose_sf.jpg")

    def test_nosched_norec(self):
        # Problem parameters.
        # Proton parameters (variable dose).
        # n_scale = 0.5*self.nt    # [1, 8, 9].
        n_scale = 0.35*self.nt   # [1, 8, 9, 10] with optional [11].
        # n_scale = 0.2*self.nt    # [1, 8, 9, 10, 11].
        # n_scale = 0.1*self.nt    # [1, 8, 9, 10, 11, 12, 15, 16] with optional [17, 18, 19].
        # n_scale = 0.15*self.nt   # [1, 8, ..., 12, 15, ..., 19, 22, 23, 24].

        # Photon parameters.
        # n_scale = 2*self.nt   # [1, 8, 9].
        # n_scale = 4*self.nt   # [1, 8, 9, 10] with optional [11].
        # n_scale = self.nt     # [1, 8, 9, 10, 11, 12, 15, 16] with optional [17, 18, 19, 22, 23, 24].

        max_iter = 1000
        delta_stop = 1e-3

        # Tuning parameters.
        lam_bed = 0
        constant_dose = False
        # constant_dose = True   # With constant_dose, optimal schedule is to just deliver all Gy on the last day.

        # Range of treatment schedules.
        # parm_dict_list = [{"T_days": 9, "M_bed": calc_normal_bed_const(18, 3, self.ab_ratio_N)}]
        parm_dict_list = [{"T_days": 9, "M_bed": calc_normal_bed_const(18, 3, self.ab_ratio_N)},
                          {"T_days": 10, "M_bed": calc_normal_bed_const(12, 4, self.ab_ratio_N)},
                          {"T_days": 11, "M_bed": calc_normal_bed_const(10, 5, self.ab_ratio_N)},
                          {"T_days": 16, "M_bed": calc_normal_bed_const(7.5, 8, self.ab_ratio_N)},
                          {"T_days": 19, "M_bed": calc_normal_bed_const(5, 12, self.ab_ratio_N)},
                          {"T_days": 24, "M_bed": calc_normal_bed_const(4, 15, self.ab_ratio_N)}]

        alpha_P = self.alpha[0]
        beta_P = self.beta[0]

        if constant_dose:
            filename = "no_schedule-no_rec-const-max_iter_{0}.pkl".format(max_iter)
            # fileprefix = "no_schedule-no_rec-const"
        else:
            filename = "no_schedule-no_rec-max_iter_{0}.pkl".format(max_iter)
            # fileprefix = "no_schedule-no_rec"

        result_dict = BaseTest.get_result_dict(filename, TestNormal.gen_dict, self.refit)

        print("Fitting model without recompartmentalization...")
        for parm_dict in parm_dict_list:
            print("\nTreatment Length (Days) = {0}\tMax Normal BED = {1}".format(parm_dict["T_days"], parm_dict["M_bed"]))
            if parm_dict["T_days"] in result_dict["T_list"]:
                print("Saved results found, moving to next treatment length in list")
                continue
            try:
                # Solve for best dose schedule using CCP (no recompartmentalization).
                result = solve_ccp(self.nt, parm_dict["T_days"], self.N0, self.alpha, self.beta, self.f_pro_P, self.T_C,
                                   self.T_loss, self.delta_t, self.k_m, ab_ratio_N=self.ab_ratio_N, M_bed=parm_dict["M_bed"],
                                   d_max_day=self.d_max_day, lam_bed=lam_bed, n_scale=n_scale, has_slack_dyn=self.has_slack_dyn,
                                   has_slack_rec=self.has_slack_rec, weekend_break=False, treat_break=0,
                                   treat_len=parm_dict["T_days"], constant_dose=constant_dose, recomp=False,
                                   max_iter=max_iter, delta_stop=delta_stop, solver=self.solver, verbose=self.verbose)
                print_result(result)

                # Simulate cell dynamics using resulting dose schedule.
                sur_frac_result = EQD2_primer_sim_step(result["d"], gf_in=self.gf, clf_in=self.clf, alpha_p_ori=alpha_P,
                                                       a_over_b=alpha_P / beta_P, oer_i=self.OER_I, oer_h=self.OER_H,
                                                       delta_t=self.delta_t, plot_survival=False, show=False)
                sur_frac_tot = sur_frac_result[0][-1,1]
                print("True survival fraction:", sur_frac_tot)

                # Save optimal dose and true survival fraction.
                result_dict["nbed_list"] = np.concatenate([result_dict["nbed_list"], np.array([result["normal_bed"]])])
                result_dict["sf_list"] = np.concatenate([result_dict["sf_list"], np.array([sur_frac_tot])])
                result_dict["d_list"].append(result["d"])
                result_dict["T_list"].append(parm_dict["T_days"])

                with open(BaseTest.save_path + filename, 'wb') as handle:
                    pickle.dump(result_dict, handle, protocol=pickle.HIGHEST_PROTOCOL)
            except InfeasibleError as error:
                print(error)
                continue

        print("\nPlotting survival fraction vs. treatment time...")
        annotations = ["T = {0}".format(parm_dict["T_days"]) for parm_dict in parm_dict_list]
        TestNormal.plot_sf_versus_nbed(result_dict, title="Final Survival Fraction vs. Normal Tissue BED (Proton, Variable Dose, No Schedule)",
                                       xlim=(0, 300), annotations=annotations, show=self.show,
                                       filename="sf_total_vs_nbed-no_rec-no_schedule.jpg")

        print("\nPlotting schedule with lowest survival fraction...")
        idx_min = np.argmin(result_dict["sf_list"])
        d_opt = result_dict["d_list"][idx_min]
        sf_opt = result_dict["sf_list"][idx_min]
        nbed_opt = result_dict["nbed_list"][idx_min]
        T_opt = result_dict["T_list"][idx_min]

        print("Treatment Length: {0}".format(T_opt))
        d_per_day_opt, sched_opt = create_schedule(d_opt, delta_t = self.delta_t)
        print("Schedule: {0},\tDose: {1}".format(sched_opt, d_per_day_opt))
        print("Survival fraction: {0}".format(sf_opt))
        print("Normal tissue BED: {0}".format(nbed_opt))

        fileprefix = "three_comp-no_rec-no_sched_len_{0}".format(T_opt)
        plot_dose_sf_stacked(d_opt, gf_in=self.gf, clf_in=self.clf, alpha_p_ori=alpha_P, a_over_b=alpha_P / beta_P,
                             oer_i=self.OER_I, oer_h=self.OER_H, delta_t=self.delta_t, figsize=(8, 12),
                             title="Survival Fraction for Final Dose Schedule (Proton, Variable Dose, No Schedule)", show=self.show,
                             filename=BaseTest.fig_exp_path + fileprefix + "-dose_sf.jpg")
