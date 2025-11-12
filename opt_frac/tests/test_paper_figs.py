import pickle
import numpy as np
import matplotlib.pyplot as plt

from opt_frac.tests.base_test import BaseTest
from opt_frac.plot_sim import *
from opt_frac.utilities import calc_normal_bed_sched, calc_normal_bed_const

class TestPaperFigs(BaseTest):
    """Unit tests for creating manuscript figures"""

    def setUp(self):
        np.random.seed(1)
        super(TestPaperFigs, self).setUp()

        # Problem parameters.
        self.delta_t = 60              # Time step (sec) of cell update.
        self.T_days = 15               # Total days of treatment.

        # Normal tissue parameters.
        self.M_bed = 146.67             # Upper bound on BED for normal tissue.
        self.d_max_day = 18             # Maximum total dose per day.

        # Algorithm parameters.
        self.max_iter_rec = 2
        self.weekend_break = True
        self.verbose_print = True

        # Plotting parameters.
        SMALL_SIZE = 18
        MEDIUM_SIZE = 20
        LARGE_SIZE = 22

        SMALL_WIDTH = 1.5
        MEDIUM_WIDTH = 3
        LARGE_WIDTH = 4

        color_cyc = plt.rcParams["axes.prop_cycle"].by_key()["color"]
        self.color_dict = {"Total": "black", "P": color_cyc[3], "I": color_cyc[0], "H": color_cyc[2]}
        # self.color_dict = {"Total": color_cyc[0], "P": color_cyc[1], "I": color_cyc[2], "H": color_cyc[3]}
        self.file_ext = "pdf"
        self.show = False

        # plt.rc('font', size=SMALL_SIZE)              # controls default text sizes
        # plt.rc('axes', titlesize=LARGE_SIZE)         # fontsize of the axes title
        plt.rc('axes', labelsize=LARGE_SIZE)     # fontsize of the axis x and y labels
        plt.rc('xtick', labelsize=SMALL_SIZE)    # fontsize of the tick labels
        plt.rc('ytick', labelsize=SMALL_SIZE)    # fontsize of the tick labels
        plt.rc('legend', fontsize=SMALL_SIZE)    # legend fontsize
        plt.rc('legend', frameon=False)
        # plt.rc('figure', titlesize=LARGE_SIZE)         # fontsize of the figure title
        plt.rc('figure', labelsize=LARGE_SIZE)   # fontsize of the figure x and y labels.
        plt.rc('lines', linewidth=MEDIUM_WIDTH)  # width of the plot lines

    @staticmethod
    def make_consec_sched(schedule, weekend_break = False):
        day = 1
        sched_consec = []
        for i in range(len(schedule)):
            if weekend_break:
                if day % 6 == 0:
                    day += 2
                elif day % 7 == 0:
                    day += 1
            sched_consec.append(day)
            day += 1
        return sched_consec

    def test_constant_comp(self):
        alpha_P = self.alpha[0]
        beta_P = self.beta[0]

        schedule = np.array([1, 2, 3, 4, 5, 8, 9, 10, 11, 12, 15, 16])  # With weekend break.
        n_fracs = len(schedule)
        M_bed = calc_normal_bed_const(7.5, 8, self.ab_ratio_N)  # NBED from 7.5 Gy x 8 fractions.
        M_bed = int(M_bed)

        # Import results for optimal consecutive weekday schedule with variable dose size.
        filename = "schedule-no_rec-nfracs_{0}-Mbed_{1}-convar.pkl".format(n_fracs, M_bed)
        with open(BaseTest.save_path + filename, "rb") as handle:
            results_dict = pickle.load(handle)
        fx_var = results_dict["fx"]
        schedule_var = results_dict["schedule"]

        sur_frac_var, s_sbrt_var, sf_sbrt_var = primer_simulation(fx_var, schedule_var, gf_in=self.gf)
        eqd2_var, tcp_var, nfrac_eqd2_var = EQD2_simulation(s_sbrt_var, gf_in=self.gf)
        nbed_var = calc_normal_bed_sched(fx_var, schedule_var, self.ab_ratio_N)
        print("Variable Fraction Size and Break: {0} Gy delivered on Days {1}".format(fx_var, schedule_var))
        print("EQD2: {0},\tTCP: {1}".format(eqd2_var, tcp_var))
        print("Normal Tissue BED: {0}".format(nbed_var))
        plot_schedule_sf_stacked(fx_var, schedule_var, gf_in=self.gf, clf_in=self.clf, alpha_p_ori=alpha_P, a_over_b=alpha_P / beta_P,
                                 oer_i=self.OER_I, oer_h=self.OER_H, delta_t=self.delta_t, figsize=(8, 12), show=self.show,
                                 # title = "Survival Fraction for Optimal Weekday Schedule with Variable Fraction Size",
                                 dose_lim=(0, 16), sf_lim=(1e-20, 1), color_dict=self.color_dict, show_subtitle=False,
                                 show_legend=True,
                                 filename=BaseTest.fig_paper_path + "dose_sf-nfracs_{0}-weekday_sched-opt_var_dose.{1}".format(n_fracs, self.file_ext))

        # Import results for optimal consecutive weekday schedule with constant dose size.
        filename = "schedule-no_rec-nfracs_{0}-Mbed_{1}-concon.pkl".format(n_fracs, M_bed)
        with open(BaseTest.save_path + filename, "rb") as handle:
            results_dict = pickle.load(handle)
        fx_con = results_dict["fx"]
        schedule_con = results_dict["schedule"]

        sur_frac_con, s_sbrt_con, sf_sbrt_con = primer_simulation(fx_con, schedule_con, gf_in=self.gf)
        eqd2_con, tcp_con, nfrac_eqd2_con = EQD2_simulation(s_sbrt_con, gf_in=self.gf)
        nbed_con = calc_normal_bed_sched(fx_con, schedule_con, self.ab_ratio_N)
        print("Constant Fraction Size, Variable Break: {0} Gy delivered on Days {1}".format(fx_con, schedule_con))
        print("EQD2: {0},\tTCP: {1}".format(eqd2_con, tcp_con))
        print("Normal Tissue BED: {0}".format(nbed_con))
        plot_schedule_sf_stacked(fx_con, schedule_con, gf_in=self.gf, clf_in=self.clf, alpha_p_ori=alpha_P, a_over_b=alpha_P / beta_P,
                                 oer_i=self.OER_I, oer_h=self.OER_H, delta_t=self.delta_t, figsize=(8, 12), show=self.show,
                                 # title = "Survival Fraction for Optimal Weekday Schedule with Constant Fraction Size",
                                 dose_lim=(0, 16), sf_lim=(1e-20, 1), color_dict=self.color_dict, show_subtitle=False,
                                 show_legend=True, show_ylabel=False,
                                 filename=BaseTest.fig_paper_path + "dose_sf-nfracs_{0}-weekday_sched-opt_const_dose.{1}".format(n_fracs, self.file_ext))

    def test_break_comp(self):
        alpha_P = self.alpha[0]
        beta_P = self.beta[0]

        n_fracs = 8
        M_bed = calc_normal_bed_const(7.5, 8, self.ab_ratio_N)   # NBED from 7.5 Gy x 8 fractions

        # Import results.
        filename = "schedule-no_rec-nfracs_{0}-Mbed_{1}.pkl".format(n_fracs, int(M_bed))
        with open(BaseTest.save_path + filename, "rb") as handle:
            results_dict = pickle.load(handle)
        fx = results_dict["fx"]
        schedule = results_dict["schedule"]
        sched_consec = self.make_consec_sched(schedule, self.weekend_break)
        fx_const = np.sum(fx) / n_fracs

        # fx = [5.5, 5, 5, 5, 5, 5, 5, 15.25]
        # fx = [5.75157316, 5.21019916, 5.21162042, 5.20307717, 5.20420604, 5.20044611, 5.17524184, 15.44008791]
        # schedule = [1, 8, 9, 10, 11, 12, 15, 16]
        # sched_consec = [1, 2, 3, 4, 5, 8, 9, 10]
        # n_fracs = len(fx)
        # fx_const = np.sum(fx) / n_fracs

        sur_frac_prmvar, s_sbrt_prmvar, sf_sbrt_prmvar = primer_simulation(fx, schedule, gf_in=self.gf)
        eqd2_prmvar, tcp_prmvar, nfrac_eqd2_prmvar = EQD2_simulation(s_sbrt_prmvar, gf_in=self.gf)
        nbed_prmvar = calc_normal_bed_sched(fx, schedule, self.ab_ratio_N)
        print("Variable Fraction Size and Break: {0} Gy delivered on Days {1}".format(fx, schedule))
        print("EQD2: {0},\tTCP: {1}".format(eqd2_prmvar, tcp_prmvar))
        print("Normal Tissue BED: {0}".format(nbed_prmvar))
        plot_schedule_sf_stacked(fx, schedule, gf_in=self.gf, clf_in=self.clf, alpha_p_ori=alpha_P, a_over_b=alpha_P / beta_P,
                                 oer_i=self.OER_I, oer_h=self.OER_H, delta_t=self.delta_t, figsize=(8, 12), show=self.show,
                                 # title = "Survival Fraction for Optimal Primer Shot Schedule",
                                 dose_lim=(0, 16), sf_lim=(1e-20, 1), color_dict=self.color_dict, show_subtitle=False,
                                 show_legend=False, show_xlabel=True, show_ylabel=True,
                                 filename=BaseTest.fig_paper_path + "dose_sf-nfracs_{0}-primer_sched-var_dose.{1}".format(n_fracs, self.file_ext))

        sur_frac_wkdvar, s_sbrt_wkdvar, sf_sbrt_wkdvar = primer_simulation(fx, sched_consec, gf_in=self.gf)
        eqd2_wkdvar, tcp_wkdvar, nfrac_eqd2_wkdvar = EQD2_simulation(s_sbrt_wkdvar, gf_in=self.gf)
        nbed_wkdvar = calc_normal_bed_sched(fx, sched_consec, self.ab_ratio_N)
        tcp_change_wkdvar = 100 * (tcp_prmvar - tcp_wkdvar) / tcp_wkdvar
        print("\nWeekday Schedule with Variable Fraction Size: {0} Gy delivered on Days {1}".format(fx, sched_consec))
        print("EQD2: {0},\tTCP: {1}".format(eqd2_wkdvar, tcp_wkdvar))
        print("Normal Tissue BED: {0}".format(nbed_wkdvar))
        print("Optimal Primer Shot TCP is Higher by {0}%".format(tcp_change_wkdvar))
        plot_schedule_sf_stacked(fx, sched_consec, gf_in=self.gf, clf_in=self.clf, alpha_p_ori=alpha_P, a_over_b=alpha_P / beta_P,
                                 oer_i=self.OER_I, oer_h=self.OER_H, delta_t=self.delta_t, figsize=(8, 12), show=self.show,
                                 # title = "Survival Fraction for Weekday Schedule with Variable Fraction Size",
                                 dose_lim=(0, 16), sf_lim=(1e-20, 1), color_dict=self.color_dict, show_subtitle=False,
                                 show_legend=False, show_xlabel=True, show_ylabel=True,
                                 filename=BaseTest.fig_paper_path + "dose_sf-nfracs_{0}-weekday_sched-var_dose.{1}".format(n_fracs, self.file_ext))

        sur_frac_prmcon, s_sbrt_prmcon, sf_sbrt_prmcon = primer_simulation(fx_const, schedule, gf_in=self.gf)
        eqd2_prmcon, tcp_prmcon, nfrac_eqd2_prmcon = EQD2_simulation(s_sbrt_prmcon, gf_in=self.gf)
        nbed_prmcon = calc_normal_bed_sched(fx_const, schedule, self.ab_ratio_N)
        tcp_change_prmcon = 100 * (tcp_prmvar - tcp_prmcon) / tcp_prmcon
        print("\nPrimer Shot Schedule with Constant Fraction Size: {0} Gy delivered on Days {1}".format(fx_const, schedule))
        print("EQD2: {0},\tTCP: {1}".format(eqd2_prmcon, tcp_prmcon))
        print("Normal Tissue BED: {0}".format(nbed_prmcon))
        print("Optimal Primer Shot TCP is Higher by {0}%".format(tcp_change_prmcon))
        plot_schedule_sf_stacked(fx_const, schedule, gf_in=self.gf, clf_in=self.clf, alpha_p_ori=alpha_P, a_over_b=alpha_P / beta_P,
                                 oer_i=self.OER_I, oer_h=self.OER_H, delta_t=self.delta_t, figsize=(8, 12), show=self.show,
                                 # title = "Survival Fraction for Primer Shot Schedule with Constant Fraction Size",
                                 dose_lim=(0, 16), sf_lim=(1e-20, 1), color_dict=self.color_dict, show_subtitle=False,
                                 show_legend=False, show_xlabel=True, show_ylabel=True,
                                 filename=BaseTest.fig_paper_path + "dose_sf-nfracs_{0}-primer_sched-const_dose.{1}".format(n_fracs, self.file_ext))

        sur_frac_wkdcon, s_sbrt_wkdcon, sf_sbrt_wkdcon = primer_simulation(fx_const, sched_consec, gf_in=self.gf)
        eqd2_wkdcon, tcp_wkdcon, nfrac_eqd2_wkdcon = EQD2_simulation(s_sbrt_wkdcon, gf_in=self.gf)
        nbed_wkdcon = calc_normal_bed_sched(fx_const, sched_consec, self.ab_ratio_N)
        tcp_change_wkdcon = 100 * (tcp_prmvar - tcp_wkdcon) / tcp_wkdcon
        print("\nWeekday Schedule with Constant Fraction Size: {0} Gy delivered on Days {1}".format(fx_const, sched_consec))
        print("EQD2: {0},\tTCP: {1}".format(eqd2_wkdcon, tcp_wkdcon))
        print("Normal Tissue BED: {0}".format(nbed_wkdcon))
        print("Optimal Primer Shot TCP is Higher by {0}%".format(tcp_change_wkdcon))
        plot_schedule_sf_stacked(fx_const, sched_consec, gf_in=self.gf, clf_in=self.clf, alpha_p_ori=alpha_P, a_over_b=alpha_P / beta_P,
                                 oer_i=self.OER_I, oer_h=self.OER_H, delta_t=self.delta_t, figsize=(8, 12), show=self.show,
                                 # title = "Survival Fraction for Weekday Schedule with Constant Fraction Size",
                                 dose_lim=(0, 16), sf_lim=(1e-20, 1), color_dict=self.color_dict, show_subtitle=False,
                                 show_legend=True, show_xlabel=True, show_ylabel=True,
                                 filename=BaseTest.fig_paper_path + "dose_sf-nfracs_{0}-weekday_sched-const_dose.{1}".format(n_fracs, self.file_ext))

    def test_break_opt_comp(self):
        alpha_P = self.alpha[0]
        beta_P = self.beta[0]

        n_fracs = 8
        M_bed = calc_normal_bed_const(7.5, 8, self.ab_ratio_N)  # NBED from 7.5 Gy x 8 fractions
        M_bed = int(M_bed)

        # Import results for consecutive weekday schedule with constant dose.
        filename = "schedule-no_rec-nfracs_{0}-Mbed_{1}-concon.pkl".format(n_fracs, M_bed)
        with open(BaseTest.save_path + filename, "rb") as handle:
            results_cc = pickle.load(handle)
        fx = results_cc["fx"]
        schedule = results_cc["schedule"]

        sur_frac_cc, s_sbrt_cc, sf_sbrt_cc = primer_simulation(fx, schedule, gf_in=self.gf)
        eqd2_cc, tcp_cc, nfrac_eqd2_cc = EQD2_simulation(s_sbrt_cc, gf_in=self.gf)
        nbed_cc = calc_normal_bed_sched(fx, schedule, self.ab_ratio_N)
        print("Consecutive Weekday Schedule with Constant Fraction Size: {0} Gy delivered on Days {1}".format(fx, schedule))
        print("EQD2: {0},\tTCP: {1}".format(eqd2_cc, tcp_cc))
        print("Normal Tissue BED: {0}".format(nbed_cc))
        plot_schedule_sf_stacked(fx, schedule, gf_in=self.gf, clf_in=self.clf, alpha_p_ori=alpha_P, a_over_b=alpha_P / beta_P,
                                 oer_i=self.OER_I, oer_h=self.OER_H, delta_t=self.delta_t, figsize=(8, 12), show=self.show,
                                 # title = "Survival Fraction for Consecutive Weekday Schedule with Constant Fraction Size",
                                 dose_lim=(0, 19), sf_lim=(1e-20, 1), color_dict=self.color_dict, show_subtitle=False, days_lim=(0, 17),
                                 show_xlabel=True, xtick_max=16, show_ylabel=True, ytick_step=2, ytick_max=18, show_legend=True, leg_loc=(0.47, 0.67),
                                 filename=BaseTest.fig_paper_path + "dose_sf-nfracs_{0}-Mbed_{1}-concon.{2}".format(n_fracs, M_bed, self.file_ext))

        # Import results for consecutive weekday schedule with variable dose.
        filename = "schedule-no_rec-nfracs_{0}-Mbed_{1}-convar.pkl".format(n_fracs, M_bed)
        with open(BaseTest.save_path + filename, "rb") as handle:
            results_cv = pickle.load(handle)
        fx = results_cv["fx"]
        schedule = results_cv["schedule"]

        sur_frac_cv, s_sbrt_cv, sf_sbrt_cv = primer_simulation(fx, schedule, gf_in=self.gf)
        eqd2_cv, tcp_cv, nfrac_eqd2_cv = EQD2_simulation(s_sbrt_cv, gf_in=self.gf)
        nbed_cv = calc_normal_bed_sched(fx, schedule, self.ab_ratio_N)
        print("\nConsecutive Weekday Schedule with Variable Fraction Size: {0} Gy delivered on Days {1}".format(fx, schedule))
        print("Avg Fraction Size: {0},\tMax Fraction Size: {1}".format(np.sum(fx) / n_fracs, np.max(fx)))
        print("EQD2: {0},\tTCP: {1}".format(eqd2_cv, tcp_cv))
        print("Normal Tissue BED: {0}".format(nbed_cv))
        plot_schedule_sf_stacked(fx, schedule, gf_in=self.gf, clf_in=self.clf, alpha_p_ori=alpha_P, a_over_b=alpha_P / beta_P,
                                 oer_i=self.OER_I, oer_h=self.OER_H, delta_t=self.delta_t, figsize=(8, 12), show=self.show,
                                 # title = "Survival Fraction for Consecutive Weekday Schedule with Variable Fraction Size",
                                 dose_lim=(0, 19), sf_lim=(1e-20, 1), color_dict=self.color_dict, show_subtitle=False, days_lim=(0, 17),
                                 show_xlabel=True, xtick_max=16, show_ylabel=True, ytick_step=2, ytick_max=18, show_legend=False,
                                 filename=BaseTest.fig_paper_path + "dose_sf-nfracs_{0}-Mbed_{1}-convar.{2}".format(n_fracs, M_bed, self.file_ext))

        # Import results for primer shot schedule with variable dose.
        filename = "schedule-no_rec-nfracs_{0}-Mbed_{1}-primvar.pkl".format(n_fracs, M_bed)
        with open(BaseTest.save_path + filename, "rb") as handle:
            results_pv = pickle.load(handle)
        fx = results_pv["fx"]
        schedule = results_pv["schedule"]

        sur_frac_pv, s_sbrt_pv, sf_sbrt_pv = primer_simulation(fx, schedule, gf_in=self.gf)
        eqd2_pv, tcp_pv, nfrac_eqd2_pv = EQD2_simulation(s_sbrt_pv, gf_in=self.gf)
        nbed_pv = calc_normal_bed_sched(fx, schedule, self.ab_ratio_N)
        print("\nPrimer Shot Schedule with Variable Fraction Size: {0} Gy delivered on Days {1}".format(fx, schedule))
        print("Avg Fraction Size: {0},\tMax Fraction Size: {1}".format(np.sum(fx) / n_fracs, np.max(fx)))
        print("EQD2: {0},\tTCP: {1}".format(eqd2_pv, tcp_pv))
        print("Normal Tissue BED: {0}".format(nbed_pv))
        plot_schedule_sf_stacked(fx, schedule, gf_in=self.gf, clf_in=self.clf, alpha_p_ori=alpha_P, a_over_b=alpha_P / beta_P,
                                 oer_i=self.OER_I, oer_h=self.OER_H, delta_t=self.delta_t, figsize=(8, 12), show=self.show,
                                 # title = "Survival Fraction for Primer Shot Schedule with Variable Fraction Size",
                                 dose_lim=(0, 19), sf_lim=(1e-20, 1), color_dict=self.color_dict, show_subtitle=False, days_lim=(0, 17),
                                 show_xlabel=True, xtick_max=16, show_ylabel=True, ytick_step=2, ytick_max=18, show_legend=False,
                                 filename=BaseTest.fig_paper_path + "dose_sf-nfracs_{0}-Mbed_{1}-primvar.{2}".format(n_fracs, M_bed, self.file_ext))

    def test_nfracs_comp(self):
        alpha_P = self.alpha[0]
        beta_P = self.beta[0]
        M_bed = calc_normal_bed_const(7.5, 8, self.ab_ratio_N)  # NBED from 7.5 Gy x 8 fractions
        M_bed = int(M_bed)

        # Import results with 4 fractions.
        n_fracs = 4
        filename = "schedule-no_rec-nfracs_{0}-Mbed_{1}.pkl".format(n_fracs, M_bed)
        with open(BaseTest.save_path + filename, "rb") as handle:
            results_nf4 = pickle.load(handle)
        fx = results_nf4["fx"]
        schedule = results_nf4["schedule"]

        # fx = [ 8.22912438, 8.12094888, 8.11511922, 17.03862243]
        # schedule = [1, 8, 9, 10]
        # n_fracs = len(fx)

        sur_frac_nf4, s_sbrt_nf4, sf_sbrt_nf4 = primer_simulation(fx, schedule, gf_in=self.gf)
        eqd2_nf4, tcp_nf4, nfrac_eqd2_nf4 = EQD2_simulation(s_sbrt_nf4, gf_in=self.gf)
        nbed_nf4 = calc_normal_bed_sched(fx, schedule, self.ab_ratio_N)
        print("Primer Shot with {0} Fractions: {1} Gy delivered on Days {2}".format(n_fracs, fx, schedule))
        print("Avg Fraction Size: {0},\tMax Fraction Size: {1}".format(np.sum(fx) / n_fracs, np.max(fx)))
        print("EQD2: {0},\tTCP: {1}".format(eqd2_nf4, tcp_nf4))
        print("Normal Tissue BED: {0}".format(nbed_nf4))
        plot_schedule_sf_stacked(fx, schedule, gf_in=self.gf, clf_in=self.clf, alpha_p_ori=alpha_P, a_over_b=alpha_P / beta_P,
                                 oer_i=self.OER_I, oer_h=self.OER_H, delta_t=self.delta_t, figsize=(8, 12), show=self.show,
                                 # title = "Survival Fraction for Primer Shot Schedule with {0} Fractions".format(n_fracs),
                                 dose_lim=(0, 19), sf_lim=(1e-20, 1), color_dict=self.color_dict, show_subtitle=False, days_lim=(0, 20), xtick_step=2,
                                 show_legend=True, show_xlabel=True, show_ylabel=True, ytick_step=2, ytick_max=18, leg_loc=(0.49, 0.7), # leg_loc=(0.49, 0), # leg_loc=(0.395, 0.025),
                                 filename=BaseTest.fig_paper_path + "dose_sf-nfracs_{0}-Mbed_{1}.{2}".format(n_fracs, M_bed, self.file_ext))

        # Import results with 8 fractions.
        n_fracs = 8
        filename = "schedule-no_rec-nfracs_{0}-Mbed_{1}.pkl".format(n_fracs, M_bed)
        with open(BaseTest.save_path + filename, "rb") as handle:
            results_nf8 = pickle.load(handle)
        fx = results_nf8["fx"]
        schedule = results_nf8["schedule"]

        # fx = [ 8.22912438, 8.12094888, 8.11511922, 17.03862243]
        # schedule = [1, 8, 9, 10]
        # n_fracs = len(fx)

        sur_frac_nf8, s_sbrt_nf8, sf_sbrt_nf8 = primer_simulation(fx, schedule, gf_in=self.gf)
        eqd2_nf8, tcp_nf8, nfrac_eqd2_nf8 = EQD2_simulation(s_sbrt_nf8, gf_in=self.gf)
        nbed_nf8 = calc_normal_bed_sched(fx, schedule, self.ab_ratio_N)
        print("\nPrimer Shot with {0} Fractions: {1} Gy delivered on Days {2}".format(n_fracs, fx, schedule))
        print("Avg Fraction Size: {0},\tMax Fraction Size: {1}".format(np.sum(fx)/n_fracs, np.max(fx)))
        print("EQD2: {0},\tTCP: {1}".format(eqd2_nf8, tcp_nf8))
        print("Normal Tissue BED: {0}".format(nbed_nf8))
        plot_schedule_sf_stacked(fx, schedule, gf_in=self.gf, clf_in=self.clf, alpha_p_ori=alpha_P, a_over_b=alpha_P / beta_P,
                                 oer_i=self.OER_I, oer_h=self.OER_H, delta_t=self.delta_t, figsize=(8, 12), show=self.show,
                                 # title = "Survival Fraction for Primer Shot Schedule with {0} Fractions".format(n_fracs),
                                 dose_lim=(0, 19), sf_lim=(1e-20, 1), color_dict=self.color_dict, show_subtitle=False, days_lim=(0, 20), xtick_step=2,
                                 show_legend=False, show_xlabel=True, show_ylabel=True, ytick_step=2, ytick_max=18, # leg_loc=(0.25, 0.025),
                                 filename=BaseTest.fig_paper_path + "dose_sf-nfracs_{0}-Mbed_{1}.{2}".format(n_fracs, M_bed, self.file_ext))

        # Import results with 11 fractions.
        n_fracs = 11
        filename = "schedule-no_rec-nfracs_{0}-Mbed_{1}.pkl".format(n_fracs, M_bed)
        with open(BaseTest.save_path + filename, "rb") as handle:
            results_nf11 = pickle.load(handle)
        fx = results_nf11["fx"]
        schedule = results_nf11["schedule"]

        # fx = [4.6750122, 4.5606846, 4.56110347, 4.55654972, 4.55911114, 4.55466291, 4.55648779, 4.55292182,
        #       4.55447484, 4.55120587, 4.54634139, 13.77054063]
        # schedule = [1, 5, 8, 9, 10, 11, 12, 15, 16, 17, 18, 19]
        # n_fracs = len(fx)

        sur_frac_nf11, s_sbrt_nf11, sf_sbrt_nf11 = primer_simulation(fx, schedule, gf_in=self.gf)
        eqd2_nf11, tcp_nf11, nfrac_eqd2_nf11 = EQD2_simulation(s_sbrt_nf11, gf_in=self.gf)
        nbed_nf11 = calc_normal_bed_sched(fx, schedule, self.ab_ratio_N)
        print("\nPrimer Shot with {0} Fractions: {1} Gy delivered on Days {2}".format(n_fracs, fx, schedule))
        print("Avg Fraction Size: {0},\tMax Fraction Size: {1}".format(np.sum(fx) / n_fracs, np.max(fx)))
        print("EQD2: {0},\tTCP: {1}".format(eqd2_nf11, tcp_nf11))
        print("Normal Tissue BED: {0}".format(nbed_nf11))
        plot_schedule_sf_stacked(fx, schedule, gf_in=self.gf, clf_in=self.clf, alpha_p_ori=alpha_P, a_over_b=alpha_P / beta_P,
                                 oer_i=self.OER_I, oer_h=self.OER_H, delta_t=self.delta_t, figsize=(8, 12), show=self.show,
                                 # title = "Survival Fraction for Primer Shot Schedule with {0} Fractions".format(n_fracs),
                                 dose_lim=(0, 19), sf_lim=(1e-20, 1), color_dict=self.color_dict, show_subtitle=False, days_lim=(0, 20),
                                 show_legend=False, show_xlabel=True, show_ylabel=True, xtick_step=2, ytick_step=2, ytick_max=18,
                                 filename=BaseTest.fig_paper_path + "dose_sf-nfracs_{0}-Mbed_{1}.{2}".format(n_fracs, M_bed, self.file_ext))

    def test_nbed_comp(self):
        alpha_P = self.alpha[0]
        beta_P = self.beta[0]
        n_fracs = 8    # Schedule: [1, 8, 9, 10, 11, 12, 15, 16].

        # Import results with max NBED = 160 Gy.
        M_bed = calc_normal_bed_const(5, 12, self.ab_ratio_N)  # Max NBED from 5 Gy x 12 fractions = 160.
        M_bed = int(M_bed)
        filename = "schedule-no_rec-nfracs_{0}-Mbed_{1}.pkl".format(n_fracs, M_bed)
        with open(BaseTest.save_path + filename, "rb") as handle:
            results_M160 = pickle.load(handle)
        fx = results_M160["fx"]
        schedule = results_M160["schedule"]

        sur_frac_M160, s_sbrt_M160, sf_sbrt_M160 = primer_simulation(fx, schedule, gf_in=self.gf)
        eqd2_M160, tcp_M160, nfrac_eqd2_M160 = EQD2_simulation(s_sbrt_M160, gf_in=self.gf)
        nbed_M160 = calc_normal_bed_sched(fx, schedule, self.ab_ratio_N)
        print("Primer Shot with {0} Fractions: {1} Gy delivered on Days {2}".format(n_fracs, fx, schedule))
        print("Avg Fraction Size: {0},\tMax Fraction Size: {1}".format(np.sum(fx) / n_fracs, np.max(fx)))
        print("EQD2: {0},\tTCP: {1}".format(eqd2_M160, tcp_M160))
        print("Normal Tissue BED: {0},\tUpper Bound on Normal BED: {1}".format(nbed_M160, M_bed))
        plot_schedule_sf_stacked(fx, schedule, gf_in=self.gf, clf_in=self.clf, alpha_p_ori=alpha_P, a_over_b=alpha_P / beta_P,
                                 oer_i=self.OER_I, oer_h=self.OER_H, delta_t=self.delta_t, figsize=(8, 12), show=self.show,
                                 # title = "Survival Fraction for Primer Shot Schedule with NBED Bound = {0}".format(M_bed),
                                 dose_lim=(0, 19), sf_lim=(1e-20, 1), color_dict=self.color_dict, show_subtitle=False,
                                 show_legend=True, show_xlabel=True, show_ylabel=True, ytick_step=2, ytick_max=18, leg_loc=(0.31, 0.025),
                                 filename=BaseTest.fig_paper_path + "dose_sf-nfracs_{0}-Mbed_{1}.{2}".format(n_fracs, M_bed, self.file_ext))

        # Import results with max NBED = 216 Gy.
        M_bed = calc_normal_bed_const(10, 5, self.ab_ratio_N)  # Max NBED from 10 Gy x 5 fractions = 216.
        M_bed = int(M_bed)
        filename = "schedule-no_rec-nfracs_{0}-Mbed_{1}.pkl".format(n_fracs, M_bed)
        with open(BaseTest.save_path + filename, "rb") as handle:
            results_M216 = pickle.load(handle)
        fx = results_M216["fx"]
        schedule = results_M216["schedule"]

        sur_frac_M216, s_sbrt_M216, sf_sbrt_M216 = primer_simulation(fx, schedule, gf_in=self.gf)
        eqd2_M216, tcp_M216, nfrac_eqd2_M216 = EQD2_simulation(s_sbrt_M216, gf_in=self.gf)
        nbed_M216 = calc_normal_bed_sched(fx, schedule, self.ab_ratio_N)
        print("\nPrimer Shot with {0} Fractions: {1} Gy delivered on Days {2}".format(n_fracs, fx, schedule))
        print("Avg Fraction Size: {0},\tMax Fraction Size: {1}".format(np.sum(fx)/n_fracs, np.max(fx)))
        print("EQD2: {0},\tTCP: {1}".format(eqd2_M216, tcp_M216))
        print("Normal Tissue BED: {0},\tUpper Bound on Normal BED: {1}".format(nbed_M216, M_bed))
        plot_schedule_sf_stacked(fx, schedule, gf_in=self.gf, clf_in=self.clf, alpha_p_ori=alpha_P, a_over_b=alpha_P / beta_P,
                                 oer_i=self.OER_I, oer_h=self.OER_H, delta_t=self.delta_t, figsize=(8, 12), show=self.show,
                                 # title = "Survival Fraction for Primer Shot Schedule with NBED Bound = {0}".format(M_bed),
                                 dose_lim=(0, 19), sf_lim=(1e-20, 1), color_dict=self.color_dict, show_subtitle=False,
                                 show_legend=False, show_xlabel=True, show_ylabel=True, ytick_step=2, ytick_max=18,
                                 filename=BaseTest.fig_paper_path + "dose_sf-nfracs_{0}-Mbed_{1}.{2}".format(n_fracs, M_bed, self.file_ext))

        # Import results with max NBED = 378 Gy.
        M_bed = calc_normal_bed_const(18, 3, self.ab_ratio_N)  # Max NBED from 18 Gy x 3 fractions = 378.
        M_bed = int(M_bed)
        filename = "schedule-no_rec-nfracs_{0}-Mbed_{1}.pkl".format(n_fracs, M_bed)
        with open(BaseTest.save_path + filename, "rb") as handle:
            results_M378 = pickle.load(handle)
        fx = results_M378["fx"]
        schedule = results_M378["schedule"]

        sur_frac_M378, s_sbrt_M378, sf_sbrt_M378 = primer_simulation(fx, schedule, gf_in=self.gf)
        eqd2_M378, tcp_M378, nfrac_eqd2_M378 = EQD2_simulation(s_sbrt_M378, gf_in=self.gf)
        nbed_M378 = calc_normal_bed_sched(fx, schedule, self.ab_ratio_N)
        print("\nPrimer Shot with {0} Fractions: {1} Gy delivered on Days {2}".format(n_fracs, fx, schedule))
        print("Avg Fraction Size: {0},\tMax Fraction Size: {1}".format(np.sum(fx) / n_fracs, np.max(fx)))
        print("EQD2: {0},\tTCP: {1}".format(eqd2_M378, tcp_M378))
        print("Normal Tissue BED: {0},\tUpper Bound on Normal BED: {1}".format(nbed_M378, M_bed))
        plot_schedule_sf_stacked(fx, schedule, gf_in=self.gf, clf_in=self.clf, alpha_p_ori=alpha_P, a_over_b=alpha_P / beta_P,
                                 oer_i=self.OER_I, oer_h=self.OER_H, delta_t=self.delta_t, figsize=(8, 12), show=self.show,
                                 # title = "Survival Fraction for Primer Shot Schedule with NBED Bound = {0}".format(M_bed),
                                 dose_lim=(0, 19), sf_lim=(1e-20, 1), color_dict=self.color_dict, show_subtitle=False,
                                 show_legend=False, show_xlabel=True, show_ylabel=True, ytick_step=2, ytick_max=18,
                                 filename=BaseTest.fig_paper_path + "dose_sf-nfracs_{0}-Mbed_{1}.{2}".format(n_fracs, M_bed, self.file_ext))
