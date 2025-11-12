import pickle
import numpy as np
from copy import deepcopy

import matplotlib
matplotlib.use("TkAgg")   # Temporary solution to Qt no plugin error.
import matplotlib.pyplot as plt

from opt_frac.schedule import solve_ccp_sched
from opt_frac.tests.base_test import BaseTest
from opt_frac.optimization import solve_ccp, print_result, InfeasibleError
from opt_frac.plot_sim import EQD2_primer_sim_step, plot_dose_sf_stacked, create_schedule, plot_schedule_sf_stacked
from opt_frac.simulation import primer_simulation, EQD2_simulation
from opt_frac.utilities import calc_normal_bed, calc_normal_bed_sched, calc_normal_bed_const

class TestPaperSims(BaseTest):
    """Unit tests for simulations involving different numbers of fractions and normal tissue BED bounds"""

    def setUp(self):
        np.random.seed(1)
        super(TestPaperSims, self).setUp()
        self.delta_t = 60

        # Normal tissue parameters.
        self.d_max_day = 18          # Maximum total dose per day.
        # self.d_max_day = 10

        # Algorithm parameters.
        self.has_slack_dyn = True    # Slack on linearized cell dynamics constraints?
        self.has_slack_rec = False   # Slack on recompartmentalization MIP constraints?
        self.weekend_break = False

        self.solver = "MOSEK"
        self.verbose = False
        self.show = True
        # self.refit = False
        self.refit = True

    @staticmethod
    def gen_dict():
        return {"schedule": np.array([]), "M_bed": None, "d": np.array([]), "fx": np.array([]), "nbed": None,
                "primer_sim": {"sur_frac": None, "s_sbrt": None, "sf_sbrt": None},
                "eqd2_sim": {"eqd2": None, "tcp": None, "nfrac_eqd2": None}}

    def run_test_frac_gen(self, schedule, M_bed, n_scale = None, constant_dose = False, name_suffix = None):
        if n_scale is None:
            n_scale = self.nt

        # Problem parameters
        alpha_P = self.alpha[0]
        beta_P = self.beta[0]
        max_iter = 1000
        delta_stop = 1e-3

        T_days = np.max(schedule)
        n_fracs = len(schedule)

        if name_suffix is None:
            filename = "schedule-no_rec-nfracs_{0}-Mbed_{1}.pkl".format(n_fracs, M_bed)
        else:
            filename = "schedule-no_rec-nfracs_{0}-Mbed_{1}-{2}.pkl".format(n_fracs, M_bed, name_suffix)
        result_dict = BaseTest.get_result_dict(filename, TestPaperSims.gen_dict, self.refit)

        # Solve for best dose schedule using CCP (no recompartmentalization).
        print("Planning with Schedule = {0}, Max Normal Tissue BED = {1}".format(schedule, M_bed))
        if len(result_dict["schedule"]) == 0 or self.refit:
            result = solve_ccp_sched(self.nt, T_days, self.N0, self.alpha, self.beta, self.f_pro_P, self.T_C,
                                     self.T_loss, self.delta_t, self.k_m, ab_ratio_N=self.ab_ratio_N,
                                     M_bed=M_bed, d_max_day=self.d_max_day, lam_bed=0, n_scale=n_scale,
                                     has_slack_dyn=self.has_slack_dyn, has_slack_rec=self.has_slack_rec,
                                     schedule=schedule, constant_dose=constant_dose, recomp=False, max_iter=max_iter,
                                     delta_stop=delta_stop, solver=self.solver, verbose=self.verbose)
            # print_result(result)
            d_opt = result["d"]
            fx_opt, schedule_opt = create_schedule(d_opt, delta_t=self.delta_t)
            assert np.all(np.array(schedule_opt) == schedule)

            # Calculate true TCP using EQD2 simulation to achieve primer shot's simulated survival rate.
            sur_frac, s_sbrt, sf_sbrt = primer_simulation(fx_opt, schedule, gf_in=self.gf)
            eqd2, tcp, nfrac_eqd2 = EQD2_simulation(s_sbrt, gf_in=self.gf)
            nbed = calc_normal_bed(d_opt, T_days, self.delta_t, ab_ratio_N=self.ab_ratio_N)

            # Save results.
            result_dict = {"schedule": schedule_opt, "M_bed": M_bed, "d": d_opt, "fx": fx_opt, "nbed": nbed,
                           "primer_sim": {"sur_frac": sur_frac, "s_sbrt": s_sbrt, "sf_sbrt": sf_sbrt},
                           "eqd2_sim": {"eqd2": eqd2, "tcp": tcp, "nfrac_eqd2": nfrac_eqd2}}

            with open(BaseTest.save_path + filename, "wb") as handle:
                pickle.dump(result_dict, handle, protocol=pickle.HIGHEST_PROTOCOL)

        print("{0} Gy delivered on Days {1}".format(result_dict["fx"], result_dict["schedule"]))
        print("EQD2: {0},\tTCP: {1}".format(result_dict["eqd2_sim"]["eqd2"], result_dict["eqd2_sim"]["tcp"]))
        print("Normal Tissue BED: {0}".format(result_dict["nbed"]))

        if name_suffix is None:
            figname = "dose_sf-nfracs_{0}-Mbed_{1}.jpg".format(n_fracs, M_bed)
        else:
            figname = "dose_sf-nfracs_{0}-Mbed_{1}-{2}.jpg".format(n_fracs, M_bed, name_suffix)

        # plot_dose_sf_stacked(result_dict["d"], gf_in=self.gf, clf_in=self.clf, alpha_p_ori=alpha_P, a_over_b=alpha_P / beta_P,
        #                      oer_i=self.OER_I, oer_h=self.OER_H, delta_t=self.delta_t, figsize=(8, 12), show=self.show,
        #                      dose_lim=(0, 18), sf_lim=(1e-20, 1), color_dict=self.color_dict, show_legend=True,
        #                      # title="Survival Fraction for Optimal Primer Shot Schedule",
        #                      filename=BaseTest.fig_exp_path + figname)

        plot_schedule_sf_stacked(result_dict["fx"], result_dict["schedule"], gf_in=self.gf, clf_in=self.clf,
                                 alpha_p_ori=alpha_P, a_over_b=alpha_P / beta_P, oer_i=self.OER_I, oer_h=self.OER_H,
                                 delta_t=self.delta_t, figsize=(8, 12), show=self.show, dose_lim=(0, 18),
                                 sf_lim=(1e-20, 1), color_dict=self.color_dict, show_legend=True,
                                 # title="Survival Fraction for Optimal Primer Shot Schedule",
                                 filename=BaseTest.fig_exp_path + figname)

    def test_concon_8(self):
        schedule = np.array([1, 2, 3, 4, 5, 8, 9, 10])
        M_bed = calc_normal_bed_const(7.5, 8, self.ab_ratio_N)  # NBED from 7.5 Gy x 8 fractions = 210 Gy.
        M_bed = int(M_bed)
        self.run_test_frac_gen(schedule, M_bed, n_scale = 0.2*self.nt, constant_dose = True, name_suffix = "concon")   # Photon.

    def test_convar_8(self):
        schedule = np.array([1, 2, 3, 4, 5, 8, 9, 10])
        M_bed = calc_normal_bed_const(7.5, 8, self.ab_ratio_N)  # NBED from 7.5 Gy x 8 fractions = 210 Gy.
        M_bed = int(M_bed)
        self.run_test_frac_gen(schedule, M_bed, n_scale = 2.15*self.nt, name_suffix = "convar")   # Photon.

    def test_primvar_8(self):
        schedule = np.array([1, 8, 9, 10, 11, 12, 15, 16])
        M_bed = calc_normal_bed_const(7.5, 8, self.ab_ratio_N)  # NBED from 7.5 Gy x 8 fractions = 210 Gy.
        M_bed = int(M_bed)
        self.run_test_frac_gen(schedule, M_bed, n_scale = 0.5*self.nt, name_suffix = "primvar")   # Photon.

    def test_concon_16(self):
        schedule = np.array([1, 2, 3, 4, 5, 8, 9, 10, 11, 12, 15, 16])  # With weekend break.
        M_bed = calc_normal_bed_const(5, 12, self.ab_ratio_N)   # NBED from 5 Gy x 12 fractions.
        # M_bed = calc_normal_bed_const(7.5, 8, self.ab_ratio_N)  # NBED from 7.5 Gy x 8 fractions.
        M_bed = int(M_bed)
        self.run_test_frac_gen(schedule, M_bed, n_scale = 0.25*self.nt, constant_dose = True, name_suffix = "concon")   # 5 Gy x 12.
        # self.run_test_frac_gen(schedule, M_bed, n_scale = 1.15*self.nt, constant_dose = True, name_suffix = "concon")   # 7.5 Gy x 8.

    def test_convar_16(self):
        schedule = np.array([1, 2, 3, 4, 5, 8, 9, 10, 11, 12, 15, 16])   # With weekend break.
        # M_bed = calc_normal_bed_const(5, 12, self.ab_ratio_N)   # NBED from 5 Gy x 12 fractions.
        M_bed = calc_normal_bed_const(7.5, 8, self.ab_ratio_N)  # NBED from 7.5 Gy x 8 fractions.
        M_bed = int(M_bed)
        self.run_test_frac_gen(schedule, M_bed, n_scale = 0.25*self.nt, name_suffix = "convar")

    def test_nfracs_4(self):
        schedule = np.array([1, 8, 9, 10])
        # M_bed = calc_normal_bed_const(12, 4, self.ab_ratio_N)   # NBED from 12 Gy x 4 fractions.
        M_bed = calc_normal_bed_const(7.5, 8, self.ab_ratio_N)  # NBED from 7.5 Gy x 8 fractions.
        M_bed = int(M_bed)
        self.run_test_frac_gen(schedule, M_bed, n_scale = 4.5*self.nt)    # Photon.
        # self.run_test_frac_gen(schedule, M_bed, n_scale = 0.35*self.nt)   # Proton.

    def test_nfracs_8(self):
        schedule = np.array([1, 8, 9, 10, 11, 12, 15, 16])
        # M_bed = calc_normal_bed_const(5, 12, self.ab_ratio_N)   # NBED from 5 Gy x 12 fractions.
        M_bed = calc_normal_bed_const(7.5, 8, self.ab_ratio_N)   # NBED from 7.5 Gy x 8 fractions.
        M_bed = int(M_bed)

        # Photon.
        self.run_test_frac_gen(schedule, M_bed, n_scale=0.5*self.nt)  # 7.5 Gy x 8.

        # Proton.
        # self.run_test_frac_gen(schedule, M_bed, n_scale = 0.1*self.nt)   # 7.5 Gy x 8.
        # self.run_test_frac_gen(schedule, M_bed, n_scale = 0.2*self.nt, constant_dose = True, name_suffix = "primcon")   # 7.5 Gy x 8.
        # self.run_test_frac_gen(schedule, M_bed, n_scale=0.3*self.nt, constant_dose=True, name_suffix="primcon")   # 5 Gy by 12.

    def test_nfracs_11(self):
        schedule = np.array([1, 8, 9, 10, 11, 12, 15, 16, 17, 18, 19])
        M_bed = calc_normal_bed_const(7.5, 8, self.ab_ratio_N)  # NBED from 7.5 Gy x 8 fractions.
        M_bed = int(M_bed)
        self.run_test_frac_gen(schedule, M_bed, n_scale = 0.5*self.nt)     # Photon.
        # self.run_test_frac_gen(schedule, M_bed, n_scale = 0.075*self.nt)   # Proton.

    def test_nfracs_12(self):
        schedule = np.array([1, 5, 8, 9, 10, 11, 12, 15, 16, 17, 18, 19])
        # M_bed = calc_normal_bed_const(5, 12, self.ab_ratio_N)   # NBED from 5 Gy x 12 fractions.
        M_bed = calc_normal_bed_const(7.5, 8, self.ab_ratio_N)  # NBED from 7.5 Gy x 8 fractions.
        M_bed = int(M_bed)
        self.run_test_frac_gen(schedule, M_bed, n_scale = self.nt)         # Photon.
        # self.run_test_frac_gen(schedule, M_bed, n_scale = 0.075*self.nt)   # Proton. Note: n_scale = 0.05*self.nt also works.

    def test_nbed_low(self):
        schedule = np.array([1, 8, 9, 10, 11, 12, 15, 16])
        M_bed = calc_normal_bed_const(5, 12, self.ab_ratio_N)  # Max NBED from 5 Gy x 12 fractions = 160.
        M_bed = int(M_bed)
        self.run_test_frac_gen(schedule, M_bed, n_scale=0.75*self.nt)  # Photon.
        # self.run_test_frac_gen(schedule, M_bed, n_scale=0.2*self.nt)   # Proton.

    def test_nbed_mid(self):
        schedule = np.array([1, 8, 9, 10, 11, 12, 15, 16])
        # M_bed = calc_normal_bed_const(7.5, 8, self.ab_ratio_N)  # Max NBED from 7.5 Gy x 8 fractions = 210.
        M_bed = calc_normal_bed_const(10, 5, self.ab_ratio_N)   # Max NBED from 10 Gy x 5 fractions = 216.
        M_bed = int(M_bed)
        self.run_test_frac_gen(schedule, M_bed, n_scale=0.75*self.nt)  # Photon.
        # self.run_test_frac_gen(schedule, M_bed, n_scale=0.1*self.nt)   # Proton.

    def test_nbed_high(self):
        schedule = np.array([1, 8, 9, 10, 11, 12, 15, 16])
        # M_bed = calc_normal_bed_const(12, 4, self.ab_ratio_N)  # Max NBED from 12 Gy x 4 fractions = 240.
        M_bed = calc_normal_bed_const(18, 3, self.ab_ratio_N)   # Max NBED from 18 Gy x 3 fractions = 378.
        M_bed = int(M_bed)

        # Note: For photon parameters, we have convergence with M_bed = 300 & n_scale = 1.0, M_bed = 320 & n_scale = 1.45,
        #   M_bed = 327 & n_scale = 1.8.
        self.run_test_frac_gen(schedule, M_bed, n_scale=2.05*self.nt)  # Photon. Note: Remember to change constants at top of schedule.py.
        # self.run_test_frac_gen(schedule, M_bed, n_scale=0.1*self.nt)   # Proton.
