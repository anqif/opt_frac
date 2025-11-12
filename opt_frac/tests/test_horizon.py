import pickle
import numpy as np
import matplotlib.pyplot as plt
from copy import deepcopy

from opt_frac.tests.base_test import BaseTest
from opt_frac.optimization import solve_ccp, print_result, InfeasibleError
from opt_frac.plot_sim import plot_dose, EQD2_primer_sim_step, EQD2_primer_sim_comp, plot_dose_sf_stacked
from opt_frac.utilities import convert_dose_delta, calc_normal_bed

class TestHorizon(BaseTest):
    """Unit tests for the optimal treatment horizon"""

    def setUp(self):
        np.random.seed(1)
        super(TestHorizon, self).setUp()
        self.delta_t = 60

        # Normal tissue parameters.
        self.M_bed = 146.67          # Upper bound on BED for normal tissue.
        self.d_max_day = 18          # Maximum total dose per day.
        # self.d_max_day = 10

        # Algorithm parameters.
        self.lam_bed = 0
        self.has_slack_dyn = True    # Slack on linearized cell dynamics constraints?
        self.has_slack_rec = False   # Slack on recompartmentalization MIP constraints?
        self.weekend_break = False

        self.solver = "MOSEK"
        self.verbose = False
        self.show = True
        self.refit = False
        self.filter_results = True

    @staticmethod
    def gen_dict():
        return {"T_list": np.array([]), "sf_list": np.array([]), "d_list": []}
        # return {"T_list": np.array([]), "sf_list": np.array([]), "nbed_list": np.array([]), "d_list": []}

    @staticmethod
    def filter_by_T(result, T_filt):
        mask = np.in1d(result["T_list"], T_filt)
        d_list = deepcopy(result["d_list"])
        for i in range(len(T_filt)):
            if mask[i]:
                d_list.append(result["d_list"][i])
        result["d_list"] = d_list
        result["T_list"] = result["T_list"][mask]
        result["sf_list"] = result["sf_list"][mask]
        return result

    @staticmethod
    def plot_sf_versus_T(result, show=True, filename=None):
        idx_sort = np.argsort(result["T_list"])
        T_sort = result["T_list"][idx_sort]
        sf_sort = result["sf_list"][idx_sort]

        fig = plt.figure(figsize=(12, 8))
        plt.semilogy(T_sort, sf_sort, marker="o")
        plt.xticks(np.arange(np.min(T_sort), np.max(T_sort) + 1, 1))
        plt.xlabel("Treatment Length (days)")
        plt.ylabel("Total Survival Fraction")
        plt.title("Final Survival Fraction vs. Maximum Treatment Length")
        if show:
            plt.show()
        if filename is not None:
            fig.savefig(BaseTest.fig_exp_path + filename, bbox_inches="tight", dpi=300)

    def test_norec(self):
        # Problem parameters.
        n_scale = 0.5*self.nt
        max_iter = 1000
        delta_stop = 1e-3

        # Range of treatment lengths.
        T_max = 14
        T_min = 5
        T_step = 1

        alpha_P = self.alpha[0]
        beta_P = self.beta[0]
        filename = "treat_len-max_iter_{0}.pkl".format(max_iter)
        result_dict = BaseTest.get_result_dict(filename, TestHorizon.gen_dict, self.refit)

        print("Fitting model without recompartmentalization...")
        T_range = np.arange(T_max, T_min - 1, -T_step)
        for T_days in T_range:
            print("\nT = {0}".format(T_days))
            if T_days in result_dict["T_list"]:
                print("Saved results found, moving to next T in list")
                continue

            try:
                # Solve for best dose schedule using CCP (no recompartmentalization).
                result = solve_ccp(self.nt, T_days, self.N0, self.alpha, self.beta, self.f_pro_P, self.T_C,
                               self.T_loss, self.delta_t, self.k_m, ab_ratio_N=self.ab_ratio_N, M_bed=self.M_bed,
                               d_max_day=self.d_max_day, lam_bed=self.lam_bed, n_scale=n_scale,
                               has_slack_dyn=self.has_slack_dyn, has_slack_rec=self.has_slack_rec,
                               weekend_break=self.weekend_break, recomp=False, max_iter=max_iter,
                               delta_stop=delta_stop, solver=self.solver, verbose=self.verbose)
                print_result(result)

                # Simulate cell dynamics using resulting dose schedule.
                sur_frac_result = EQD2_primer_sim_step(result["d"], gf_in=self.gf, clf_in=self.clf, alpha_p_ori=alpha_P,
                                                       a_over_b=alpha_P / beta_P, oer_i=self.OER_I, oer_h=self.OER_H,
                                                       delta_t=self.delta_t, plot_survival=False, show=False)
                sur_frac_tot = sur_frac_result[0][-1,1]
                print("True survival fraction:", sur_frac_tot)

                # Save optimal dose and true survival fraction.
                result_dict["T_list"] = np.concatenate([result_dict["T_list"], np.array([T_days])])
                result_dict["sf_list"] = np.concatenate([result_dict["sf_list"], np.array([sur_frac_tot])])
                # result_dict["nbed_list"] = np.concatenate([result_dict["nbed_list"], np.array([result["normal_bed"]])])
                result_dict["d_list"].append(result["d"])

                with open(BaseTest.save_path + filename, 'wb') as handle:
                    pickle.dump(result_dict, handle, protocol=pickle.HIGHEST_PROTOCOL)
            except InfeasibleError as error:
                print(error)
                continue
                # print("Problem is infeasible, terminating at T = {0}".format(T_days))
                # break

        if self.filter_results:
            TestHorizon.filter_by_T(result_dict, T_range)

        print("\nPlotting survival fraction vs. treatment time...")
        TestHorizon.plot_sf_versus_T(result_dict, show = self.show, filename = "sf_total_vs_T-no_rec.jpg")

        print("\nPlotting schedule with lowest survival fraction...")
        idx_min = np.argmin(result_dict["sf_list"])
        # idx_min = np.where(result_dict["T_list"] == 4)[0][0]
        d_opt = result_dict["d_list"][idx_min]
        T_opt = result_dict["T_list"][idx_min]
        sf_opt = result_dict["sf_list"][idx_min]
        # nbed_opt = result_dict["nbed_list"][idx_min]
        nbed_opt = calc_normal_bed(d_opt, T_opt, self.delta_t, self.ab_ratio_N)

        print("Treatment length: {0}".format(T_opt))
        print("Survival fraction: {0}".format(sf_opt))
        print("Normal tissue BED: {0}".format(nbed_opt))

        fileprefix = "three_comp-no_rec-T_{0}".format(T_opt)
        # plot_dose(d_opt, gf_in=self.gf, clf_in=self.clf, delta_t=self.delta_t, figsize=(12, 8), show=self.show,
        #           fileprefix=BaseTest.fig_exp_path + fileprefix)
        # sur_frac_result = EQD2_primer_sim_step(d_opt, gf_in=self.gf, clf_in=self.clf, alpha_p_ori=alpha_P,
        #                                        a_over_b=alpha_P / beta_P, oer_i=self.OER_I, oer_h=self.OER_H,
        #                                        delta_t=self.delta_t, plot_survival=True, show=self.show,
        #                                        filename=BaseTest.fig_exp_path + fileprefix + "-sf.jpg")
        # BaseTest.print_sur_frac(sur_frac_result, self.ab_ratio_N)
        plot_dose_sf_stacked(d_opt, gf_in=self.gf, clf_in=self.clf, alpha_p_ori=alpha_P, a_over_b=alpha_P / beta_P,
                             oer_i=self.OER_I, oer_h=self.OER_H, delta_t=self.delta_t, figsize=(8, 12),
                             title="Survival Fraction for Final Dose Schedule", show=self.show,
                             filename=BaseTest.fig_exp_path + fileprefix + "-dose_sf.jpg")

    def test_norec_break(self):
        # Fixed parameters.
        n_scale = 0.5 * self.nt
        max_iter = 1000
        delta_stop = 1e-3

        # Tuned parameters.
        treat_len = 1
        # treat_break = 0

        # Range of treatment lengths.
        # T_max = 14
        # T_min = 7
        # T_step = 1
        # n_break = 1

        T_max = 15
        T_min = 7
        T_step = 2
        n_break = 2

        alpha_P = self.alpha[0]
        beta_P = self.beta[0]
        filename = "treat_len-max_iter_{0}-break_{1}.pkl".format(max_iter, n_break)
        result_dict = BaseTest.get_result_dict(filename, TestHorizon.gen_dict, self.refit)

        print("Fitting model without recompartmentalization...")
        for T_days in range(T_max, T_min - 1, -T_step):
            print("\nT = {0}".format(T_days))
            if T_days in result_dict["T_list"]:
                print("Saved results found, moving to next T in list")
                continue

            if n_break == 1:
                treat_break = T_days - 2   # Treat only at beginning and end.
            elif n_break == 2:
                treat_break = (T_days - 3) // 2   # Treat at beginning, middle, and end.
            else:
                treat_break = 0   # Treat throughout.

            try:
                # Solve for best dose schedule using CCP (no recompartmentalization).
                result = solve_ccp(self.nt, T_days, self.N0, self.alpha, self.beta, self.f_pro_P, self.T_C,
                               self.T_loss, self.delta_t, self.k_m, ab_ratio_N=self.ab_ratio_N, M_bed=self.M_bed,
                               d_max_day=self.d_max_day, lam_bed=self.lam_bed, n_scale=n_scale,
                               has_slack_dyn=self.has_slack_dyn, has_slack_rec=self.has_slack_rec, treat_len=treat_len,
                               treat_break=treat_break, weekend_break=self.weekend_break, recomp=False, max_iter=max_iter,
                               delta_stop=delta_stop, solver=self.solver, verbose=self.verbose)
                print_result(result)

                # Simulate cell dynamics using resulting dose schedule.
                sur_frac_result = EQD2_primer_sim_step(result["d"], gf_in=self.gf, clf_in=self.clf, alpha_p_ori=alpha_P,
                                                       a_over_b=alpha_P / beta_P, oer_i=self.OER_I, oer_h=self.OER_H,
                                                       delta_t=self.delta_t, plot_survival=False, show=False)
                sur_frac_tot = sur_frac_result[0][-1,1]
                print("True survival fraction:", sur_frac_tot)

                # Save optimal dose and true survival fraction.
                result_dict["T_list"] = np.concatenate([result_dict["T_list"], np.array([T_days])])
                result_dict["sf_list"] = np.concatenate([result_dict["sf_list"], np.array([sur_frac_tot])])
                result_dict["d_list"].append(result["d"])

                with open(BaseTest.save_path + filename, 'wb') as handle:
                    pickle.dump(result_dict, handle, protocol=pickle.HIGHEST_PROTOCOL)
            except InfeasibleError as error:
                print(error)
                continue
                # print("Problem is infeasible, terminating at T = {0}".format(T_days))
                # break

        print("\nPlotting survival fraction vs. treatment time...")
        TestHorizon.plot_sf_versus_T(result_dict, show = self.show, filename = "sf_total_vs_T-no_rec-break_{0}.jpg".format(n_break))

        print("\nPlotting schedule with lowest survival fraction...")
        idx_min = np.argmin(result_dict["sf_list"])
        # idx_min = np.where(result_dict["T_list"] == 10)[0][0]
        d_opt = result_dict["d_list"][idx_min]
        T_opt = result_dict["T_list"][idx_min]
        sf_opt = result_dict["sf_list"][idx_min]
        # nbed_opt = result_dict["nbed_list"][idx_min]
        nbed_opt = calc_normal_bed(d_opt, T_opt, self.delta_t, self.ab_ratio_N)

        print("Treatment length: {0}".format(T_opt))
        print("Survival fraction: {0}".format(sf_opt))
        print("Normal tissue BED: {0}".format(nbed_opt))

        fileprefix = "three_comp-no_rec-T_{0}-break_{1}".format(T_opt, n_break)
        # plot_dose(d_opt, gf_in=self.gf, clf_in=self.clf, delta_t=self.delta_t, figsize=(12, 8), show=self.show,
        #           fileprefix=BaseTest.fig_exp_path + fileprefix)
        # sur_frac_result = EQD2_primer_sim_step(d_opt, gf_in=self.gf, clf_in=self.clf, alpha_p_ori=alpha_P,
        #                                        a_over_b=alpha_P / beta_P, oer_i=self.OER_I, oer_h=self.OER_H,
        #                                        delta_t=self.delta_t, plot_survival=True, show=self.show,
        #                                        filename=BaseTest.fig_exp_path + fileprefix + "-sf.jpg")
        # BaseTest.print_sur_frac(sur_frac_result, self.ab_ratio_N)
        plot_dose_sf_stacked(d_opt, gf_in=self.gf, clf_in=self.clf, alpha_p_ori=alpha_P, a_over_b=alpha_P / beta_P,
                             oer_i=self.OER_I, oer_h=self.OER_H, delta_t=self.delta_t, figsize=(8, 12),
                             title="Survival Fraction for Final Dose Schedule", show=self.show,
                             filename=BaseTest.fig_exp_path + fileprefix + "-dose_sf.jpg")
