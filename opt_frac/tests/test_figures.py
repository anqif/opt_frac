import pickle
import numpy as np
import matplotlib.pyplot as plt

from opt_frac.tests.base_test import BaseTest
from opt_frac.plot_sim import *
from opt_frac.utilities import convert_dose_delta, calc_normal_bed_sched


class TestFigures(BaseTest):
    """Unit tests for creating figures for comparison"""

    def setUp(self):
        np.random.seed(1)
        super(TestFigures, self).setUp()

        # Problem parameters.
        self.delta_t = 60              # Time step (sec) of cell update.
        self.T_days = 15               # Total days of treatment.

        # Normal tissue parameters.
        self.M_bed = 146.67             # Upper bound on BED for normal tissue.
        self.d_max_day = 18             # Maximum total dose per day.

        # Algorithm parameters.
        self.max_iter_rec = 2
        self.show = True
        self.weekend_break = False
        self.verbose_print = True

    def calc_result_dict(self, dict_list):
        alpha_P = self.alpha[0]
        beta_P = self.beta[0]
        delta_day = int(24*60 / self.delta_t)  # Number of time steps per day.

        res_dict = {"nbed_list": np.array([]), "sf_list": np.array([]), "d_list": [], "sched_list": []}
        for parm_dict in dict_list:
            sur_frac, s_sbrt, sf_sbrt = primer_simulation(parm_dict["dose"], parm_dict["schedule"], gf_in=self.gf,
                                                          clf_in=self.clf, alpha_p_ori=alpha_P, a_over_b=alpha_P/beta_P,
                                                          oer_i=self.OER_I, oer_h=self.OER_H)
            n_treat_days = len(parm_dict["schedule"])
            normal_bed = n_treat_days * parm_dict["dose"] + n_treat_days * parm_dict["dose"]**2/self.ab_ratio_N
            dose_delta = np.zeros(n_treat_days * delta_day)
            for treat_day in parm_dict["schedule"]:
                dose_delta[((treat_day - 1)*delta_day):(treat_day*delta_day)] = parm_dict["dose"]
            sur_frac_tot = sur_frac[-1, 1]

            res_dict["nbed_list"] = np.concatenate([res_dict["nbed_list"], np.array([normal_bed])])
            res_dict["sf_list"] = np.concatenate([res_dict["sf_list"], np.array([sur_frac_tot])])
            res_dict["d_list"].append(dose_delta)
            res_dict["sched_list"].append(parm_dict["schedule"])
        return res_dict

    @staticmethod
    def plot_sf_versus_nbed_list(result_list, title=None, label_list=None, annotations_list=None, xlim=None, ylim=None, show=True, filename=None):
        if not isinstance(result_list, list):
            result_list = list(result_list)
        if label_list is None:
            label_list = len(result_list)*[None]
        if not isinstance(label_list, list):
            label_list = list(label_list)
        if len(label_list) != len(result_list):
            raise ValueError("label_list must be a list of length {0}".format(len(result_list)))
        if annotations_list is None:
            annotations_list = len(result_list)*[None]
        if not isinstance(annotations_list, list):
            annotations_list = [annotations_list]
        if len(annotations_list) != len(result_list):
            raise ValueError("annotations_list must be a list of length {0}".format(len(result_list)))

        fig = plt.figure(figsize=(12, 8))
        for result, label, annotations in zip(result_list, label_list, annotations_list):
            idx_sort = np.argsort(result["nbed_list"])
            nbed_sort = result["nbed_list"][idx_sort]
            sf_sort = result["sf_list"][idx_sort]
            if label is None:
                plt.semilogy(nbed_sort, sf_sort, marker="o")
            else:
                plt.semilogy(nbed_sort, sf_sort, marker="o", label=label)
            # plt.xticks(np.arange(np.min(nbed_sort), np.max(nbed_sort) + 1, 1))

            # Annotate each point.
            if annotations is not None:
                ant_sort = [annotations[i] for i in idx_sort]
                dx = 0.02 * (np.max(nbed_sort) - np.min(nbed_sort))
                dy = 0.001 * (np.max(sf_sort) - np.min(sf_sort))
                # dy = 0.05 * (np.max(sf_sort) - np.min(sf_sort))
                for i, txt in enumerate(ant_sort):
                    plt.annotate(txt, (nbed_sort[i] + dx, sf_sort[i] + 0.05**i * dy))

        if xlim is not None:
            plt.xlim(xlim)
        if ylim is not None:
            plt.ylim(ylim)

        if title is not None:
            plt.title(title)
        if not all([label == None for label in label_list]):
            plt.legend()
        plt.xlabel("Normal Tissue Biologically Effective Dose (BED)")
        plt.ylabel("Total Survival Fraction")

        if show:
            plt.show()
        if filename is not None:
            fig.savefig(BaseTest.fig_exp_path + filename, bbox_inches="tight", dpi=300)

    def test_norec(self):
        alpha_P = self.alpha[0]
        beta_P = self.beta[0]

        if self.weekend_break:
            filename = "three_comp-no_rec-delta_t_{0}-weekend-hist.pkl".format(self.delta_t)
            figname_sf = "three_comp-no_rec-weekend-sf.jpg"
            figname_dose_sf = "three_comp-no_rec-weekend-dose_sf.jpg"
        else:
            filename = "three_comp-no_rec-delta_t_{0}-hist.pkl".format(self.delta_t)
            figname_sf = "three_comp-no_rec-sf.jpg"
            figname_dose_sf = "three_comp-no_rec-dose_sf.jpg"

        print("Importing dose information...")
        # d_norec = np.load(data_path + "three_comp-no_rec-delta_t_{0}-lam_{1}-dose.npy".format(self.delta_t, self.lam_bed))
        # with open(BaseTest.data_path + "three_comp-no_rec-delta_t_{0}-hist.pkl".format(self.delta_t), "rb") as handle:
        with open(BaseTest.data_path + filename, "rb") as handle:
            norec_hist_list = pickle.load(handle)
        d_norec = norec_hist_list[-1]["d"]

        print("Plotting results without recompartmentalization...")
        plot_dose(d_norec, gf_in=self.gf, clf_in=self.clf, delta_t=self.delta_t, figsize=(12, 8), show=self.show,
                  fileprefix=BaseTest.fig_path + "three_comp-no_rec")
        sur_frac_result = EQD2_primer_sim_step(d_norec, gf_in=self.gf, clf_in=self.clf, alpha_p_ori=alpha_P,
                  a_over_b=alpha_P / beta_P, oer_i=self.OER_I, oer_h=self.OER_H, delta_t=self.delta_t,
                  plot_survival=True, show=self.show, filename=BaseTest.fig_path + figname_sf)
        BaseTest.print_sur_frac(sur_frac_result, self.ab_ratio_N)

        plot_dose_sf_stacked(d_norec, gf_in=self.gf, clf_in=self.clf, alpha_p_ori=alpha_P, a_over_b=alpha_P /beta_P,
                             oer_i=self.OER_I, oer_h=self.OER_H, delta_t=self.delta_t, figsize=(8, 12),
                             title="Survival Fraction for Final Dose Schedule", show=self.show,
                             filename=BaseTest.fig_path + figname_dose_sf)
        BaseTest.primer_EQD2_dose_comp(d_norec, self.gf, self.ab_ratio_N, self.delta_t, verbose = self.verbose_print)

    def test_rec(self):
        alpha_P = self.alpha[0]
        beta_P = self.beta[0]

        print("Importing dose information...")
        # d_rec = np.load(data_path + "three_comp-rec-warm_start-delta_t_{0}-iter_{1}-dose.npy".format(delta_t_rec, max_iter_rec))
        # with open(data_path + "three_comp-rec-delta_t_{0}-iter_{1}-hist.pkl".format(delta_t_rec, max_iter_rec), "rb") as handle:
        with open(BaseTest.data_path + "three_comp-rec-warm_start-delta_t_{0}-hist.pkl".format(self.delta_t), "rb") as handle:
            rec_hist_list = pickle.load(handle)
        d_rec = rec_hist_list[-1]["d"]

        print("Plotting results with recompartmentalization using warm start...")
        plot_dose(d_rec, gf_in=self.gf, clf_in=self.clf, delta_t=self.delta_t, figsize=(12, 8), show=self.show,
                  fileprefix=BaseTest.fig_path + "three_comp-rec-warm_start")
        sur_frac_result = EQD2_primer_sim_step(d_rec, gf_in=self.gf, clf_in=self.clf, alpha_p_ori=alpha_P,
                                               a_over_b=alpha_P / beta_P, oer_i=self.OER_I, oer_h=self.OER_H,
                                               delta_t=self.delta_t, plot_survival=True, show=self.show,
                                               filename=BaseTest.fig_path + "three_comp-rec-warm_start-sf.jpg")
        BaseTest.print_sur_frac(sur_frac_result, self.ab_ratio_N)

        plot_dose_sf_stacked(d_rec, gf_in=self.gf, clf_in=self.clf, alpha_p_ori=alpha_P, a_over_b=alpha_P / beta_P,
                             oer_i=self.OER_I, oer_h=self.OER_H, delta_t=self.delta_t, figsize=(8, 12),
                             title="Survival Fraction for Final Dose Schedule (Warm Start)", show=self.show,
                             filename=BaseTest.fig_path + "three_comp-rec-warm_start-dose_sf.jpg")

    def test_rec_comp(self):
        alpha_P = self.alpha[0]
        beta_P = self.beta[0]
        delta_t_norec = self.delta_t
        delta_t_rec = self.delta_t

        print("Importing dose information...")
        with open(BaseTest.data_path + "three_comp-no_rec-delta_t_{0}-hist.pkl".format(delta_t_norec), "rb") as handle:
            norec_hist_list = pickle.load(handle)
        d_norec = norec_hist_list[-1]["d"]

        with open(BaseTest.data_path + "three_comp-rec-warm_start-delta_t_{0}-hist.pkl".format(delta_t_rec), "rb") as handle:
            rec_hist_list = pickle.load(handle)
        d_rec = rec_hist_list[-1]["d"]

        print("Comparing against baseline...")
        print("\nNo Recompartmentalization")
        BaseTest.primer_EQD2_dose_comp(d_norec, self.gf, self.ab_ratio_N, self.delta_t, verbose = self.verbose_print)

        print("\nRecompartmentalization (Warm Start)")
        BaseTest.primer_EQD2_dose_comp(d_rec, self.gf, self.ab_ratio_N, self.delta_t, verbose = self.verbose_print)

        d_norec_delta_rec = convert_dose_delta(d_norec, self.T_days, delta_t_norec, delta_t_rec)
        EQD2_primer_sim_comp([d_norec_delta_rec, d_rec], gf_list=self.gf, clf_list=self.clf, ab_ratio_N=self.ab_ratio_N,
                             delta_t=delta_t_rec, figsize=(12, 8), show=self.show,
                             label_list=["No Recompartmentalization", "Recompartmentalization (Warm Start)"],
                             fileprefix=BaseTest.fig_path + "three_comp-warm_start-comp")

        # plot_dose_sf_stacked(d_rec, gf_in=self.gf, clf_in=self.clf, alpha_p_ori=alpha_P, a_over_b=alpha_P / beta_P,
        #                      oer_i=self.OER_I, oer_h=self.OER_H, delta_t=delta_t_rec, figsize=(8, 12),
        #                      title="Survival Fraction for Final Dose Schedule (Warm Start)", show=self.show,
        #                      filename=BaseTest.fig_path + "three_comp-rec-warm_start-dose_sf.jpg")

    def test_const_frac(self):
        alpha_P = self.alpha[0]
        beta_P = self.beta[0]

        print("Plotting results for constant fractionation...")
        with open(BaseTest.data_path + "three_comp-rec-warm_start-const-delta_t_{0}-iter_{1}-hist.pkl".format(self.delta_t, self.max_iter_rec), "rb") as handle:
            const_hist_list = pickle.load(handle)
        d_const = const_hist_list[-1]["d"]

        plot_dose(d_const, gf_in=self.gf, clf_in=self.clf, delta_t=self.delta_t, figsize=(12, 8), show=self.show,
                  fileprefix=BaseTest.fig_path + "three_comp-rec-const-delta_t_{0}-iter_{1}".format(self.delta_t, self.max_iter_rec))
        sur_frac_result = EQD2_primer_sim_step(d_const, gf_in=self.gf, clf_in=self.clf, alpha_p_ori=alpha_P,
                                               a_over_b=alpha_P / beta_P, oer_i=self.OER_I, oer_h=self.OER_H,
                                               delta_t=self.delta_t, plot_survival=True, show=self.show,
                                               filename=BaseTest.fig_path + "three_comp-rec-const-delta_t_{0}-iter_{1}-sf.jpg".format(self.delta_t, self.max_iter_rec))
        BaseTest.print_sur_frac(sur_frac_result, self.ab_ratio_N)
        BaseTest.primer_EQD2_dose_comp(d_const, self.gf, self.ab_ratio_N, self.delta_t, verbose = self.verbose_print)

        plot_dose_sf_stacked(d_const, gf_in=self.gf, clf_in=self.clf, alpha_p_ori=alpha_P, a_over_b=alpha_P / beta_P,
                             oer_i=self.OER_I, oer_h=self.OER_H, delta_t=self.delta_t, figsize=(8, 12), show=self.show,
                             title="Survival Fraction for Final Constant Dose Schedule", dose_lim=(0, 16),
                             filename=BaseTest.fig_path + "three_comp-rec-const-delta_t_{0}-iter_{1}-dose_sf.jpg".format(self.delta_t, self.max_iter_rec))

    def test_consec_frac(self):
        alpha_P = self.alpha[0]
        beta_P = self.beta[0]

        fx = [5.5, 5, 5, 5, 5, 5, 5, 15.25]
        schedule = [1, 8, 9, 10, 11, 12, 15, 16]
        n_fracs = len(fx)

        sur_frac, s_sbrt, sf_sbrt = primer_simulation(fx, schedule, gf_in=self.gf)
        eqd2, tcp, nfrac_eqd2 = EQD2_simulation(s_sbrt, gf_in=self.gf)
        nbed_primer = calc_normal_bed_sched(fx, schedule, self.ab_ratio_N)
        print("Variable Fraction Size and Break: {0} Gy delivered on Days {1}".format(fx, schedule))
        print("EQD2: {0},\tTCP: {1}".format(eqd2, tcp))
        print("Normal Tissue BED: {0}".format(nbed_primer))
        plot_schedule_sf_stacked(fx, schedule, gf_in=self.gf, clf_in=self.clf, alpha_p_ori=alpha_P, a_over_b=alpha_P / beta_P,
                                 oer_i=self.OER_I, oer_h=self.OER_H, delta_t=self.delta_t, figsize=(8, 12), show=self.show,
                                 # title="Survival Fraction for Final Constant Dose Schedule",
                                 dose_lim=(0, 16), sf_lim=(1e-20, 1),
                                 filename=BaseTest.fig_path + "three_comp-break-sched_len_{0}-dose_sf.jpg".format(n_fracs))

        sched_consec = [1, 2, 3, 4, 5, 8, 9, 10]
        sur_frac_consec, s_sbrt_consec, sf_sbrt_consec = primer_simulation(fx, sched_consec, gf_in=self.gf)
        eqd2_consec, tcp_consec, nfrac_eqd2_consec = EQD2_simulation(s_sbrt_consec, gf_in=self.gf)
        nbed_primer_consec = calc_normal_bed_sched(fx, sched_consec, self.ab_ratio_N)
        tcp_change_consec = 100 * (tcp - tcp_consec) / tcp_consec
        print("\nVariable Fraction Size Weekday Schedule: {0} Gy delivered on Days {1}".format(fx, sched_consec))
        print("EQD2: {0},\tTCP: {1}".format(eqd2_consec, tcp_consec))
        print("Normal Tissue BED: {0}".format(nbed_primer_consec))
        print("Primer Shot TCP is Higher by {0}%".format(tcp_change_consec))
        plot_schedule_sf_stacked(fx, sched_consec, gf_in=self.gf, clf_in=self.clf, alpha_p_ori=alpha_P, a_over_b=alpha_P / beta_P,
                                 oer_i=self.OER_I, oer_h=self.OER_H, delta_t=self.delta_t, figsize=(8, 12), show=self.show,
                                 # title="Survival Fraction for Final Constant Dose Schedule",
                                 dose_lim=(0, 16), sf_lim=(1e-20, 1),
                                 filename=BaseTest.fig_path + "three_comp-consec-sched_len_{0}-dose_sf.jpg".format(n_fracs))

        fx_const = np.sum(fx) / len(fx)
        sur_frac_concon, s_sbrt_concon, sf_sbrt_concon = primer_simulation(fx_const, sched_consec, gf_in=self.gf)
        eqd2_concon, tcp_concon, nfrac_eqd2_concon = EQD2_simulation(s_sbrt_concon, gf_in=self.gf)
        nbed_primer_concon = calc_normal_bed_sched(fx_const, sched_consec, self.ab_ratio_N)
        tcp_change_concon = 100 * (tcp - tcp_concon) / tcp_concon
        print("\nConstant Fraction Size Weekday Schedule: {0} Gy delivered on Days {1}".format(fx_const, sched_consec))
        print("EQD2: {0},\tTCP: {1}".format(eqd2_concon, tcp_concon))
        print("Normal Tissue BED: {0}".format(nbed_primer_concon))
        print("Primer Shot TCP is Higher by {0}%".format(tcp_change_concon))
        plot_schedule_sf_stacked(fx_const, sched_consec, gf_in=self.gf, clf_in=self.clf, alpha_p_ori=alpha_P, a_over_b=alpha_P / beta_P,
                                 oer_i=self.OER_I, oer_h=self.OER_H, delta_t=self.delta_t, figsize=(8, 12), show=self.show,
                                 # title="Survival Fraction for Final Constant Dose Schedule",
                                 dose_lim=(0, 8), sf_lim=(1e-20, 1),
                                 filename=BaseTest.fig_path + "three_comp-consec_const-sched_len_{0}-dose_sf.jpg".format(n_fracs))

    def test_break(self):
        alpha_P = self.alpha[0]
        beta_P = self.beta[0]
        # delta_t_norec = self.delta_t
        delta_t_rec = self.delta_t

        print("Importing dose information...")
        # with open(BaseTest.data_path + "three_comp-no_rec-delta_t_{0}-hist.pkl".format(delta_t_norec), "rb") as handle:
        #     norec_hist_list = pickle.load(handle)
        # d_norec = norec_hist_list[-1]["d"]
        with open(BaseTest.data_path + "three_comp-rec-warm_start-delta_t_{0}-hist.pkl".format(delta_t_rec), "rb") as handle:
            rec_hist_list = pickle.load(handle)
        d_rec = rec_hist_list[-1]["d"]

        print("Plotting results for different treatment breaks...")
        print("Weekly treatment schedule:")
        d_break_6 = np.load(BaseTest.data_path + "three_comp-rec-warm_start-delta_t_{0}-break_6-dose.npy".format(self.delta_t))
        plot_dose(d_break_6, gf_in = self.gf, clf_in = self.clf, delta_t = self.delta_t, figsize = (12,8), show = self.show,
                  fileprefix = BaseTest.fig_path + "three_comp-rec-warm_start-delta_t_{0}-break_6".format(self.delta_t))
        sur_frac_result_6 = EQD2_primer_sim_step(d_break_6, gf_in = self.gf, clf_in = self.clf, alpha_p_ori = alpha_P,
                                                 a_over_b = alpha_P/beta_P, oer_i = self.OER_I, oer_h = self.OER_H,
                                                 delta_t = self.delta_t, plot_survival = True, show = self.show,
                                                 filename = BaseTest.fig_path + "three_comp-rec-warm_start-delta_t_{0}-break_6-sf.jpg".format(self.delta_t))
        # BaseTest.print_sur_frac(sur_frac_result_6, self.ab_ratio_N)
        BaseTest.primer_EQD2_dose_comp(d_break_6, self.gf, self.ab_ratio_N, self.delta_t, verbose = self.verbose_print)

        plot_dose_sf_stacked(d_break_6, gf_in = self.gf, clf_in = self.clf, alpha_p_ori = alpha_P, a_over_b = alpha_P/beta_P,
                             oer_i = self.OER_I, oer_h = self.OER_H, delta_t = self.delta_t, figsize = (8,12), show = self.show,
                             title = "Survival Fraction for Final Weekly Dose Schedule", dose_lim = (0,16),
                             filename = BaseTest.fig_path + "three_comp-rec-warm_start-break_6-dose_sf.jpg")

        print("\nBiweekly treatment schedule:")
        d_break_13 = np.load(BaseTest.data_path + "three_comp-rec-warm_start-delta_t_{0}-break_13-dose.npy".format(self.delta_t))
        plot_dose(d_break_13, gf_in = self.gf, clf_in = self.clf, delta_t = self.delta_t, figsize = (12,8), show = self.show,
                  fileprefix = BaseTest.fig_path + "three_comp-rec-warm_start-delta_t_{0}-break_13".format(self.delta_t))
        sur_frac_result_13 = EQD2_primer_sim_step(d_break_13, gf_in = self.gf, clf_in = self.clf, alpha_p_ori = alpha_P,
                                                  a_over_b = alpha_P/beta_P, oer_i = self.OER_I, oer_h = self.OER_H,
                                                  delta_t = self.delta_t, plot_survival = True, show = self.show,
                                                  filename = BaseTest.fig_path + "three_comp-rec-warm_start-delta_t_{0}-break_13-sf.jpg".format(self.delta_t))
        # BaseTest.print_sur_frac(sur_frac_result_13, self.ab_ratio_N)
        BaseTest.primer_EQD2_dose_comp(d_break_13, self.gf, self.ab_ratio_N, self.delta_t, verbose=self.verbose_print)

        plot_dose_sf_stacked(d_break_13, gf_in = self.gf, clf_in = self.clf, alpha_p_ori = alpha_P, a_over_b = alpha_P/beta_P,
                             oer_i = self.OER_I, oer_h = self.OER_H, delta_t = self.delta_t, figsize = (8,12), show = self.show,
                             title = "Survival Fraction for Final Biweekly Dose Schedule", dose_lim = (0,16),
                             filename = BaseTest.fig_path + "three_comp-rec-warm_start-break_13-dose_sf.jpg")

        # Compare survival fraction for all treatment break schedules.
        # d_comp_list = [d_norec, d_break_6, d_break_13]
        d_comp_list = [d_rec, d_break_6, d_break_13]
        EQD2_primer_sim_comp(d_comp_list, gf_list = self.gf, clf_list = self.clf, ab_ratio_N = self.ab_ratio_N,
                             delta_t = self.delta_t, figsize = (12,8), show = self.show,
                             label_list = ["No Break", "Weekly Break", "Biweekly Break",],
                             fileprefix = BaseTest.fig_path + "three_comp-warm_start-comp_break")

    def test_break_comp(self):
        alpha_P = self.alpha[0]
        beta_P = self.beta[0]
        # delta_t_norec = self.delta_t
        delta_t_rec = self.delta_t
        cutoff = 1e-3

        print("Plotting results for different treatment breaks...")
        print("Weekly treatment schedule:")
        d_break_6 = np.load(BaseTest.data_path + "three_comp-rec-warm_start-delta_t_{0}-break_6-dose.npy".format(self.delta_t))
        plot_dose(d_break_6, gf_in=self.gf, clf_in=self.clf, delta_t=self.delta_t, figsize=(12, 8), show=self.show,
                  fileprefix=BaseTest.fig_path + "three_comp-rec-warm_start-delta_t_{0}-break_6".format(self.delta_t))
        sur_frac_result_6 = EQD2_primer_sim_step(d_break_6, gf_in=self.gf, clf_in=self.clf, alpha_p_ori=alpha_P,
                                                 a_over_b=alpha_P / beta_P, oer_i=self.OER_I, oer_h=self.OER_H,
                                                 delta_t=self.delta_t, plot_survival=True, show=self.show,
                                                 filename=BaseTest.fig_path + "three_comp-rec-warm_start-delta_t_{0}-break_6-sf.jpg".format(self.delta_t))
        # BaseTest.print_sur_frac(sur_frac_result_6, self.ab_ratio_N)
        BaseTest.primer_EQD2_dose_comp(d_break_6, self.gf, self.ab_ratio_N, self.delta_t, verbose=self.verbose_print)

        plot_dose_sf_stacked(d_break_6, gf_in=self.gf, clf_in=self.clf, alpha_p_ori=alpha_P, a_over_b=alpha_P / beta_P,
                             oer_i=self.OER_I, oer_h=self.OER_H, delta_t=self.delta_t, figsize=(8, 12), show=self.show,
                             title="Survival Fraction for Final Weekly Dose Schedule", dose_lim=(0, 16),
                             filename=BaseTest.fig_path + "three_comp-rec-warm_start-break_6-dose_sf.jpg")

        fx_break_6, sched_break_6 = create_schedule(d_break_6, delta_t=self.delta_t)
        n_frac = len(fx_break_6)
        sched_break_con = np.arange(n_frac) + 1
        plot_schedule(fx_break_6, sched_break_con, gf_in=self.gf, clf_in=self.clf, delta_t=self.delta_t, figsize=(12, 8), show=self.show,
                      fileprefix=BaseTest.fig_path + "three_comp-rec-warm_start-delta_t_{0}-break_6-con".format(self.delta_t))
        sur_frac_con_6 = EQD2_primer_sim_step_sched(fx_break_6, sched_break_con, gf_in=self.gf, clf_in=self.clf, alpha_p_ori=alpha_P,
                                                    a_over_b=alpha_P / beta_P, oer_i=self.OER_I, oer_h=self.OER_H, plot_survival=True, show=self.show,
                                                    filename=BaseTest.fig_path + "three_comp-rec-warm_start-delta_t_{0}-break_6-con-sf.jpg".format(self.delta_t))

        plot_schedule_sf_stacked(fx_break_6, sched_break_con, gf_in=self.gf, clf_in=self.clf, alpha_p_ori=alpha_P,
                                 a_over_b=alpha_P / beta_P, oer_i=self.OER_I, oer_h=self.OER_H, delta_t=self.delta_t,
                                 figsize=(8, 12), show=self.show, title="Survival Fraction for Final Weekly Dose Schedule\nDelivered in Consecutive Fractions",
                                 dose_lim=(0, 16), filename=BaseTest.fig_path + "three_comp-rec-warm_start-break_6-con-dose_sf.jpg")

    def test_sf_nbed(self):
        max_iter = 1000
        alpha_P = self.alpha[0]
        beta_P = self.beta[0]
        delta_day = int(24 * 60 / self.delta_t)  # Number of time steps per day.

        # Survival fraction vs. normal tissue BED for baseline fractionation schedules.
        base_dict_list = [{"dose": 18,  "schedule": [1, 2, 3]},
                          {"dose": 12,  "schedule": [1, 2, 3, 4]},
                          {"dose": 10,  "schedule": [1, 2, 3, 4, 5]},
                          {"dose": 7.5, "schedule": [1, 2, 3, 4, 5, 8, 9, 10]},
                          {"dose": 5,   "schedule": [1, 2, 3, 4, 5, 8, 9, 10, 11, 12, 15, 16]},
                          {"dose": 4,   "schedule": [1, 2, 3, 4, 5, 8, 9, 10, 11, 12, 15, 16, 17, 18, 19]}
                         ]
        base_res_dict = self.calc_result_dict(base_dict_list)

        # Survival fraction vs. normal tissue BED for baseline primer schedules (Jeho & Zeno's paper).
        base_primer_dict_list = [{"dose": 18,  "schedule": [1, 8, 9]},
                                 {"dose": 12,  "schedule": [1, 8, 9, 10]},
                                 {"dose": 10,  "schedule": [1, 8, 9, 10, 11]},
                                 {"dose": 7.5, "schedule": [1, 8, 9, 10, 11, 12, 15, 16]},
                                 {"dose": 5,   "schedule": [1, 5, 8, 9, 10, 11, 12, 15, 16, 17, 18, 19]},
                                 {"dose": 4,   "schedule": [1, 5, 8, 9, 10, 11, 12, 15, 16, 17, 18, 19, 22, 23, 24]}
                                ]
        base_primer_res_dict = self.calc_result_dict(base_primer_dict_list)

        print("Importing optimal variable fraction fixed schedule...")
        with open(BaseTest.save_path + "schedule-no_rec-max_iter_{0}.pkl".format(max_iter), "rb") as handle:
            fixed_res_dict = pickle.load(handle)

        def calc_nbed_const(dose, nfrac):
            return nfrac * dose * (1 + dose / self.ab_ratio_N)

        fixed_dict_list = [{"schedule": [1, 8, 9], "M_bed": calc_nbed_const(18, 3)},
                           {"schedule": [1, 8, 9, 10], "M_bed": calc_nbed_const(12, 4)},
                           {"schedule": [1, 8, 9, 10, 11], "M_bed": calc_nbed_const(10, 5)},
                           {"schedule": [1, 8, 9, 10, 11, 12, 15, 16], "M_bed": calc_nbed_const(7.5, 8)},
                           {"schedule": [1, 5, 8, 9, 10, 11, 12, 15, 16, 17, 18, 19], "M_bed": calc_nbed_const(5, 12)},
                           {"schedule": [1, 5, 8, 9, 10, 11, 12, 15, 16, 17, 18, 19, 22, 23, 24], "M_bed": calc_nbed_const(4, 15)}
                          ]

        print("Importing optimal variable fraction variable schedule...")
        with open(BaseTest.save_path + "no_schedule-no_rec-max_iter_{0}.pkl".format(max_iter), "rb") as handle:
            var_res_dict = pickle.load(handle)

        var_dict_list = [{"T_days": 9, "M_bed": calc_nbed_const(18, 3)},
                         {"T_days": 10, "M_bed": calc_nbed_const(12, 4)},
                         {"T_days": 11, "M_bed": calc_nbed_const(10, 5)},
                         {"T_days": 16, "M_bed": calc_nbed_const(7.5, 8)},
                         {"T_days": 19, "M_bed": calc_nbed_const(5, 12)},
                         {"T_days": 24, "M_bed": calc_nbed_const(4, 15)}]

        print("Plotting survival fraction vs. normal tissue BED...")
        annotations_list = [["L = {0}".format(len(parm_dict["schedule"])) for parm_dict in base_dict_list],
                            ["L = {0}".format(len(parm_dict["schedule"])) for parm_dict in base_primer_dict_list],
                            ["L = {0}".format(len(parm_dict["schedule"])) for parm_dict in fixed_dict_list],
                            ["L = {0}".format(parm_dict["T_days"])        for parm_dict in var_dict_list]]
        TestFigures.plot_sf_versus_nbed_list([base_res_dict, base_primer_res_dict, fixed_res_dict, var_res_dict],
                                             title="Final Survival Fraction vs. Normal Tissue BED (Proton Therapy)",
                                             label_list=["Constant Dose, Daily Schedule", "Constant Dose, Fixed Schedule",
                                                         "Variable Dose, Fixed Schedule", "Variable Dose, Variable Schedule"],
                                             xlim=(0, 450), annotations_list=annotations_list, show=self.show,
                                             filename="sf_total_vs_nbed-comp-schedule.jpg")
