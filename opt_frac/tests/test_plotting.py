import numpy as np
import matplotlib.pyplot as plt

from opt_frac.tests.base_test import BaseTest
from opt_frac.plot_sim import *
from opt_frac.utilities import calc_cell_dynamics, plot_plan_results

class TestPlotting(BaseTest):
    """Unit tests for basic plotting functions"""

    def setUp(self):
        super(TestPlotting, self).setUp()

        # Problem parameters.
        self.delta_t = 60
        self.T_days = 14                                       # Total days of treatment.
        self.T = int((self.T_days * 24 * 60) / self.delta_t)   # Total time steps.
        self.delta_day = int(24 * 60 / self.delta_t)           # Number of time steps per day.
        self.M_bed = 146.67                                    # Upper bound on BED for normal tissue.

        self.show = False
        self.verbose = False

        # Plot display parameters.
        SMALL_SIZE = 16
        MEDIUM_SIZE = 18
        LARGE_SIZE = 20

        # plt.rc('font', size=SMALL_SIZE)          # controls default text sizes
        plt.rc('axes', titlesize=LARGE_SIZE)     # fontsize of the axes title
        plt.rc('axes', labelsize=MEDIUM_SIZE)    # fontsize of the x and y labels
        plt.rc('xtick', labelsize=SMALL_SIZE)    # fontsize of the tick labels
        plt.rc('ytick', labelsize=SMALL_SIZE)    # fontsize of the tick labels
        # plt.rc('legend', fontsize=SMALL_SIZE)    # legend fontsize
        # plt.rc('figure', titlesize=LARGE_SIZE)  # fontsize of the figure title

        SMALL_WIDTH = 1.5
        LARGE_WIDTH = 4
        plt.rc('lines', linewidth=LARGE_WIDTH)

    def test_plot_dose(self):
        M_tld_N = self.ab_ratio_N * self.M_bed + self.T * (0.5 * self.ab_ratio_N) ** 2

        # Define dose vector in time steps.
        df = max(np.sqrt(M_tld_N / self.T) - 0.5 * self.ab_ratio_N,
                 0)  # Optimal (constant) fraction with only P compartment.
        dose = np.repeat(df, self.T)

        plot_dose(dose, gf_in = self.gf, clf_in = self.clf, delta_t = self.delta_t, figsize = (12, 8), line = False)
        plot_dose(dose, gf_in = self.gf, clf_in = self.clf, delta_t = self.delta_t, figsize = (12, 8), line = True)

    def test_primer_sim(self):
        M_tld_N = self.ab_ratio_N * self.M_bed + self.T * (0.5 * self.ab_ratio_N) ** 2

        # Define dose vector in time steps.
        df = max(np.sqrt(M_tld_N/self.T) - 0.5*self.ab_ratio_N, 0)   # Optimal (constant) fraction with only P compartment.
        dose_1 = np.repeat(df, self.T)

        dose_2 = np.zeros(self.T)
        dose_2[:5*self.delta_day] = 10

        dose_3 = np.zeros(self.T)
        dose_3[:self.delta_day] = 10

        # dose_4 = np.zeros(self.T)
        # dose_4[14*self.delta_day:18*self.delta_day] = 10

        # dose_5 = np.zeros(self.T)
        # for day in [1, 15, 16, 17, 18]:
        #     dose_5[day*self.delta_day] = 10

        # dose_6 = np.load(BaseTest.data_path + 'three_comp-no_rec-dose-14_days-gf_0.25-lam_0.025-delta_60.npy')

        # Plot resulting survival fraction with dose schedule.
        dose_list = [dose_1, dose_2, dose_3]
        # dose_list = [dose_1, dose_2, dose_3, dose_4, dose_5]
        # dose_list = [dose_1, dose_2, dose_3, dose_4, dose_5, dose_6]
        for dose in dose_list:
            print("\nDose Schedule: {0}".format(dose))
            sur_frac_result = EQD2_primer_sim_step(dose, gf_in=self.gf, clf_in=self.clf, delta_t=self.delta_t, show=self.show)
            BaseTest.print_sur_frac(sur_frac_result, self.ab_ratio_N)

    def test_primer_comp(self):
        # Plot comparison of doses with different growth fractions.
        dose_1 = np.load(BaseTest.data_path + 'three_comp-no_rec-dose-14_days-gf_0.15-lam_0.025-delta_60.npy')
        dose_2 = np.load(BaseTest.data_path + 'three_comp-no_rec-dose-14_days-gf_0.25-lam_0.025-delta_60.npy')
        EQD2_primer_sim_comp([dose_1, dose_2], gf_list = [0.15, 0.25], clf_list = self.clf, delta_t = self.delta_t,
                             show = self.show, fileprefix = BaseTest.fig_path + 'three_comp-no_rec-comp_gf')

    def test_primer_nbed_comp(self):
        # Plot comparison of doses with different normal tissue BED constraints.
        nbed_plot = [100, 120, 140, 210]
        nbed_max_vec = np.load(BaseTest.data_path + 'three_comp-no_rec-gf_0.25-nbed_vector.npy')
        dose_mat = np.load(BaseTest.data_path + 'three_comp-no_rec-gf_0.25-dose_mat.npy')

        dose_list = []
        for nbed_max in nbed_plot:
            idx, = np.where(nbed_max_vec == nbed_max)
            if len(idx) == 0:
                continue
            dose_list.append(dose_mat[:,idx])
        EQD2_primer_sim_comp(dose_list, gf_list = self.gf, clf_list = self.clf, delta_t = self.delta_t, show = self.show,
                             fileprefix = BaseTest.fig_path + 'three_comp-no_rec-comp_bed')

    def test_pbt(self):
        # Proton parameters.
        alpha_P = self.alpha[0]
        beta_P = self.beta[0]

        # Plot comparison of PBT (proton) dose schedules for different treatment lengths.
        # dose = np.load(BaseTest.data_path + 'three_comp-no_rec-dose-pbt-5_days-gf_0.25-lam_0.025-delta_t_{0}.npy'.format(self.delta_t))
        # plot_dose(dose, gf_in=self.gf, clf_in=self.clf, delta_t=self.delta_t, figsize=(12,8), show=self.show,
        #           fileprefix = BaseTest.fig_path + "three_comp-no_rec")

        dose = np.load(BaseTest.data_path + 'three_comp-no_rec-delta_t_{0}-T_{1}-dose.npy'.format(self.delta_t, self.T_days))
        plot_dose_bar(dose, gf_in=self.gf, clf_in=self.clf, delta_t=self.delta_t, show=self.show)

        dose_list = []
        # T_list = [5, 7, 14]
        T_list = [14]
        fig = plt.figure(figsize = (12,8))
        plt.title("Final Dose Schedule for GF = {0}, CLF = {1}".format(self.gf, self.clf))
        plt.xlabel("Day")
        plt.ylabel("Dose (Gy)")
        for T in T_list:
            # dose = np.load(BaseTest.data_path + 'three_comp-no_rec-dose-pbt-{0}_days-gf_0.25-lam_0.025-delta_{1}.npy'.format(T, self.delta_t))
            dose = np.load(BaseTest.data_path + 'three_comp-no_rec-delta_t_{0}-T_{1}-dose.npy'.format(self.delta_t, T))
            # plot_dose(dose, gf_in=self.gf, clf_in=self.clf, delta_t=self.delta_t, show=self.show)
            plot_dose_line(dose, gf_in=self.gf, clf_in=self.clf, delta_t=self.delta_t, label = "T = {0} days".format(T),
                           newfig=False, show=self.show)
            dose_list.append(dose)
        plt.ylim(bottom = 0)
        plt.legend()
        if self.show:
            plt.show()
        fig.savefig(BaseTest.fig_path + 'three_comp-no_rec-comp_days-dose.jpg')

        # Plot PBT survival fraction for each dose schedule.
        for k in range(len(T_list)):
            print("T = {0} days".format(T_list[k]))
            dose = dose_list[k]
            sur_frac_result = EQD2_primer_sim_step(dose, gf_in=self.gf, clf_in=self.clf, alpha_p_ori=alpha_P,
                                a_over_b=alpha_P/beta_P, oer_i=self.OER_I, oer_h=self.OER_H, delta_t=self.delta_t, show=self.show,
                                filename = BaseTest.fig_path + "three_comp-no_rec-delta_t_{0}-T_{1}-sf.jpg".format(self.delta_t, T_list[k]))
            BaseTest.print_sur_frac(sur_frac_result, self.ab_ratio_N)
