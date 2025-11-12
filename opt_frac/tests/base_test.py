import pickle
import numpy as np
import scipy.stats as stats

from opt_frac.plot_sim import create_schedule
from opt_frac.simulation import primer_simulation, EQD2_simulation
from opt_frac.utilities import calc_normal_bed_sched

import matplotlib
matplotlib.use("TkAgg")   # Temporary solution to Qt no plugin error.
import matplotlib.pyplot as plt
import os

# Base class for unit tests.
from unittest import TestCase
from warnings import warn

class BaseTest(TestCase):
    base_path = os.path.expanduser(os.sep.join(["~", "Documents", "Software", "opt_frac", "examples"]))
    data_path = os.sep.join([base_path, "data", ""])
    fig_path = os.sep.join([base_path, "figures", ""])
    save_path = os.sep.join([base_path, "results", ""])
    # fig_exp_path = os.sep.join([fig_path, "experiments", ""])
    fig_exp_path = os.sep.join([fig_path, "experiments", "impt_proton", ""])
    # fig_exp_path = os.sep.join([fig_path, "experiments", "imrt_proton", ""])
    fig_paper_path = os.sep.join([fig_path, "paper", ""])

    color_cyc = plt.rcParams["axes.prop_cycle"].by_key()["color"]
    color_dict = {"Total": color_cyc[0], "P": color_cyc[1], "I": color_cyc[2], "H": color_cyc[3]}

    def setUp(self):
        # Cell parameters.
        rhot = 1e6                              # Tumor cell density.
        vt = 64                                 # Volume of a tumorlet.
        self.nt = rhot * vt                     # Total number of cells in a tumorlet.
        self.clf = 0.92                         # Cell loss factor.
        self.gf = 0.25                          # Growth fraction.

        self.f_pro_P = 0.5                      # Initial proliferation fraction in P compartment.
        self.T_C = 2 * (24 * 60)                # Cell cycle time in minutes.
        self.T_loss = 2 * (24 * 60)             # Cell loss half-time in H compartment in minutes.
        self.T_lysis = 3 * (24 * 60)            # Lysis half-time in minutes.
        self.k_m = 0.3
        self.ab_ratio_N = 3                     # Ratio alpha/beta for normal tissue cells.

        N0_P = (self.gf / self.f_pro_P) * self.nt
        N0_H = self.clf * self.gf * (self.T_loss / self.T_C) * self.nt
        N0_I = self.nt - N0_P - N0_H
        self.N0 = [N0_P, N0_I, N0_H]

        # Proton parameters.
        alpha_P_proton = 0.205
        beta_P_proton = alpha_P_proton / 2.5
        OER_I_proton = 1.0
        OER_H_proton = 1.05

        # Photon parameters
        alpha_P_photon = 0.305
        beta_P_photon = alpha_P_photon / 2.8
        OER_I_photon = 1.7
        OER_H_photon = 1.37

        # Set this to parameters corresponding to modality (proton/photon).
        # alpha_P = alpha_P_proton
        # beta_P = beta_P_proton
        # self.OER_I = OER_I_proton
        # self.OER_H = OER_H_proton

        alpha_P = alpha_P_photon
        beta_P = beta_P_photon
        self.OER_I = OER_I_photon
        self.OER_H = OER_H_photon

        alpha_I = alpha_P / self.OER_I
        beta_I = beta_P / self.OER_I**2
        alpha_H = alpha_P / self.OER_H
        beta_H = beta_P / self.OER_H**2
        self.alpha = [alpha_P, alpha_I, alpha_H]
        self.beta = [beta_P, beta_I, beta_H]

    # AssertAlmostEqual for lists.
    def assertItemsAlmostEqual(self, a, b, places: int = 5) -> None:
        if np.isscalar(a):
            a = [a]
        else:
            a = self.mat_to_list(a)
        if np.isscalar(b):
            b = [b]
        else:
            b = self.mat_to_list(b)
        for i in range(len(a)):
            self.assertAlmostEqual(a[i], b[i], places)

    # Overridden method to assume lower accuracy.
    def assertAlmostEqual(self, a, b, places: int = 5, msg = None, delta = None) -> None:
        super(BaseTest, self).assertAlmostEqual(a, b, places = places, msg = msg, delta = delta)

    def mat_to_list(self, mat):
        """Convert a numpy matrix to a list.
        """
        if isinstance(mat, (np.matrix, np.ndarray)):
            return np.asarray(mat).flatten('F').tolist()
        else:
            return mat

    @staticmethod
    def get_result_dict(filename, gen_dict, recalc, *args, **kwargs):
        if recalc:
            print("\nRecalculating dict...")
            result_dict = gen_dict(*args, **kwargs)
            with open(BaseTest.save_path + filename, 'wb') as handle:
                pickle.dump(result_dict, handle, protocol=pickle.HIGHEST_PROTOCOL)
        else:
            try:
                print("\nImporting dict...")
                with open(BaseTest.save_path + filename, 'rb') as handle:
                    result_dict = pickle.load(handle)
            except IOError:
                print("No prior data found, recalculating dict...")
                result_dict = gen_dict(*args, **kwargs)
                with open(BaseTest.save_path + filename, 'wb') as handle:
                    pickle.dump(result_dict, handle, protocol=pickle.HIGHEST_PROTOCOL)
        return result_dict

    @staticmethod
    def print_sur_frac(result, ab_ratio_N):
        sur_frac_opt, eqd2_opt, tcp_opt, fx_opt, schedule_opt = result
        print("True survival fraction:", sur_frac_opt[-1, 1])
        print("True survival fraction by compartment: P viable = {0}, I viable = {1}, H viable = {2}".format(
            sur_frac_opt[-1, 2], sur_frac_opt[-1, 3], sur_frac_opt[-1, 4]))
        print("True normal tissue BED:", np.sum(fx_opt * (1 + fx_opt / ab_ratio_N)))
        print("True EQD2: {0}, True TCP: {1}".format(eqd2_opt, tcp_opt))

    @staticmethod
    def primer_EQD2_dose_comp(dose, gf=0.25, ab_ratio_N=3, delta_t=15, verbose=False):
        fx, schedule = create_schedule(dose, delta_t=delta_t)
        BaseTest.primer_EQD2_comp(fx, schedule, gf=gf, ab_ratio_N=ab_ratio_N, verbose=verbose)

    @staticmethod
    def primer_EQD2_comp(fx, schedule, gf=0.25, ab_ratio_N=3, verbose=False):
        sur_frac, s_sbrt, sf_sbrt = primer_simulation(fx, schedule, gf_in=gf)
        eqd2, tcp, nfrac_eqd2 = EQD2_simulation(s_sbrt, gf_in=gf)

        nfrac_primer = len(schedule)
        dur_primer = np.max(schedule)
        dur_eqd2 = nfrac_eqd2 + 2 * (nfrac_eqd2 // 5)  # Count 2 days for weekend break every 5 consecutive fractions.
        # nbed_primer = nfrac_primer * fx * (1 + fx / ab_ratio_N)
        nbed_primer = calc_normal_bed_sched(fx, schedule, ab_ratio_N)
        nbed_eqd2 = nfrac_eqd2 * 2 * (1 + 2 / ab_ratio_N)

        print("\n===============================================================")
        print("Primer Shot: {0} Gy delivered on Days {1}".format(fx, schedule))
        print("===============================================================")
        print("EQD2: {0},\tTCP: {1}".format(eqd2, tcp))
        print("Normal Tissue BED: {0}".format(nbed_primer))
        if verbose:
            print("Surviving Cells: {0}".format(s_sbrt))
            print("Survival Fraction: {0}".format(sf_sbrt))
            print("Number of Fractions: {0}".format(nfrac_primer))
            print("Duration of Treatment (Days): {0}".format(dur_primer))

        # Same TCP delivered using 2 Gy/day, 5 fractions/week with weekend break.
        print("\nEQD2 Schedule (2 Gy/day, 5 fractions/week)")
        # print("EQD2: {0},\tTCP: {1}".format(eqd2, tcp))
        print("Normal Tissue BED: {0}".format(nbed_eqd2))
        if verbose:
            print("Number of Fractions: {0}".format(nfrac_eqd2))
            print("Duration of Treatment (Days): {0}".format(dur_eqd2))

        nbed_change = 100 * (nbed_eqd2 - nbed_primer) / nbed_primer
        print("Relative % Difference in Normal Tissue BED from Primer Shot: {0}".format(nbed_change))

        # Same fraction size(s) delivered consecutively using 5 fractions/week with weekend break.
        sur_frac_con, s_sbrt_con, sf_sbrt_con = primer_simulation(fx, [i + 1 for i in range(nfrac_primer)], gf_in=gf)
        eqd2_con, tcp_con, nfrac_eqd2_con = EQD2_simulation(s_sbrt_con, gf_in=gf)

        print("\nConsecutive Fractions (5 fractions/week)")
        print("EQD2: {0},\tTCP: {1}".format(eqd2_con, tcp_con))
        # if verbose:
        #     print("Number of Fractions: {0}".format(nfrac_primer))
        #     print("Duration of Treatment (Days): {0}".format(nfrac_primer))

        tcp_change = 100 * (tcp_con - tcp) / tcp
        print("Relative % Difference in TCP from Primer Shot: {0}".format(tcp_change))

        # Constant fractions delivered consecutively using 5 fractions/week with weekend break, where the fraction size
        # is chosen such that normal tissue BED is equal to that from primer shot schedule.
        if not np.isscalar(fx):
            # Solve for constant fraction that achieves same normal tissue BED as primer shot schedule, i.e.,
            # Find d >= 0 such that n_frac*d(1 + d/(alpha/beta)) = nbed.
            fx_equiv = np.max(np.roots([nfrac_primer / ab_ratio_N, nfrac_primer, -nbed_primer]))  # Take positive root.
            # fx_equiv = (-nfrac_primer + np.sqrt(nfrac_primer**2 + 4*nfrac_primer*nbed_primer/ab_ratio_N))/(2*nfrac_primer/ab_ratio_N)

            sur_frac_equiv, s_sbrt_equiv, sf_sbrt_equiv = primer_simulation(fx_equiv,
                                                                            [i + 1 for i in range(nfrac_primer)],
                                                                            gf_in=gf)
            eqd2_equiv, tcp_equiv, nfrac_eqd2_equiv = EQD2_simulation(s_sbrt_equiv, gf_in=gf)

            print("\nConsecutive Constant Fractions (5 fractions/week)")
            print("EQD2: {0},\tTCP: {1}".format(eqd2_equiv, tcp_equiv))
            # if verbose:
            #     print("Number of Fractions: {0}".format(nfrac_primer))
            #     print("Duration of Treatment (Days): {0}".format(nfrac_primer))

            tcp_change = 100 * (tcp_equiv - tcp) / tcp
            print("Relative % Difference in TCP from Primer Shot: {0}".format(tcp_change))

        # TODO: Same treatment length delivered using 2 Gy/day, 5 fractions/week with weekend break.