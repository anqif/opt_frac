import numpy as np
import matplotlib.pyplot as plt

from opt_frac.tests.base_test import BaseTest
from opt_frac.plot_sim import *
from opt_frac.utilities import calc_normal_bed_sched, calc_normal_bed_const


class TestSimulation(BaseTest):
    """Unit tests for cell simulation functions"""

    def setUp(self):
        np.random.seed(1)
        super(TestSimulation, self).setUp()

        self.gf = 0.25
        self.ab_ratio_N = 3
        self.log_comp = True
        self.verbose = False

    def primer_sim_verify(self, fx, schedule, gf = 0.25, ab_ratio_N = 3, sf_true = None, places = 5, log = False, verbose = False):
        sur_frac, s_sbrt, sf_sbrt = primer_simulation(fx, schedule, gf_in=gf)
        # nbed = len(schedule) * fx * (1 + fx / ab_ratio_N)
        nbed = calc_normal_bed_sched(fx, schedule, ab_ratio_N)

        if verbose:
            print("\n{0} Gy delivered on Days {1}".format(fx, schedule))
            print("Survival Fraction: {0},\tNormal Tissue BED: {1}".format(sf_sbrt, nbed))

        if sf_true is not None:
            if sf_true < 0:
                raise ValueError("sf_true must be a nonnegative scalar")
            elif sf_true == 0 or sf_sbrt == 0:
                self.assertAlmostEqual(sf_sbrt, sf_true, places = places)
            else:
                if log:
                    self.assertAlmostEqual(np.log(sf_sbrt), np.log(sf_true), places = places)
                else:
                    self.assertAlmostEqual(sf_sbrt, sf_true, places = places)
        return sur_frac, s_sbrt, sf_sbrt, nbed

    def EQD2_sim_verify(self, s_sbrt, gf = 0.25, ab_ratio_N = 3, eqd2_true = None, tcp_true = None, places = 5, log = False, verbose = False):
        eqd2, tcp, num_fracs = EQD2_simulation(s_sbrt, gf_in = gf)
        fx_std = 2
        # nbed = num_fracs * fx_std * (1 + fx_std / ab_ratio_N)
        nbed = calc_normal_bed_const(fx_std, num_fracs, ab_ratio_N)

        if verbose:
            print("\n{0} Gy delivered in {1} Days with Weekend Break".format(fx_std, num_fracs))
            print("EQD2: {0},\tTCP: {1},\tNormal Tissue BED: {2}".format(eqd2, tcp, nbed))

        if eqd2_true is not None:
            self.assertAlmostEqual(eqd2, eqd2_true, places = places)
        if tcp_true is not None:
            if tcp_true < 0:
                raise ValueError("tcp_true must be a nonnegative scalar")
            elif tcp == 0 or tcp_true == 0:
                self.assertAlmostEqual(tcp, tcp_true, places = places)
            else:
                if log:
                    self.assertAlmostEqual(np.log(tcp), np.log(tcp_true), places = places)
                else:
                    self.assertAlmostEqual(tcp, tcp_true, places = places)
            self.assertAlmostEqual(tcp, tcp_true, places = places)
        return eqd2, tcp, nbed

    def test_basic(self):
        sur_frac_1, s_sbrt_1, sf_sbrt_1 = primer_simulation(10, [1, 15, 16, 17, 18], gf_in=self.gf)
        sur_frac_2, s_sbrt_2, sf_sbrt_2 = primer_simulation([14, 9, 9, 9, 9], [1, 15, 16, 17, 18], gf_in=self.gf)

        eqd2_1, tcp_1, nfrac_1 = EQD2_simulation(s_sbrt_1, gf_in=self.gf)
        eqd2_2, tcp_2, nfrac_2 = EQD2_simulation(s_sbrt_2, gf_in=self.gf)

    def test_benchmark(self):
        self.primer_sim_verify(5, [1, 2, 3, 4, 5], gf=self.gf, ab_ratio_N=self.ab_ratio_N, sf_true=None)
        self.primer_sim_verify(10, [1, 5, 11], gf=self.gf, ab_ratio_N=self.ab_ratio_N, sf_true=None)

        self.EQD2_sim_verify(100, gf=self.gf, ab_ratio_N=self.ab_ratio_N, eqd2_true=None, tcp_true=None)
        self.EQD2_sim_verify(1e6, gf=self.gf, ab_ratio_N=self.ab_ratio_N, eqd2_true=None, tcp_true=None)

    def test_constant_primer(self):
        def primer_sim_test(fx, schedule, sf_true, places = 6):
            return self.primer_sim_verify(fx, schedule, gf=self.gf, ab_ratio_N=self.ab_ratio_N, sf_true=sf_true,
                                          places=places, log=self.log_comp, verbose=self.verbose)

        primer_sim_test(18, [1, 15, 16], sf_true = 1.80055451574456e-27)
        primer_sim_test(12, [1, 15, 16, 17], sf_true = 4.01970310783555e-19)
        primer_sim_test(10, [1, 15, 16, 17, 18], sf_true = 3.14852631583764e-18)
        primer_sim_test(7.5, [1, 15, 16, 17, 18, 19, 22, 23], sf_true = 2.53998908040949e-19)
        primer_sim_test(5, [1, 9, 15, 16, 17, 18, 19, 22, 23, 24, 25, 26], sf_true = 1.18881622717972e-16)
        primer_sim_test(4, [1, 5, 15, 16, 17, 18, 19, 22, 23, 24, 25, 26, 29, 30, 31], sf_true = 6.49605876267178e-15)

    def test_constant_EQD2(self):
        def EQD2_sim_test(s_sbrt, eqd2_true, tcp_true, places = 6):
            return self.EQD2_sim_verify(s_sbrt, gf=self.gf, ab_ratio_N=self.ab_ratio_N, eqd2_true=eqd2_true,
                                        tcp_true=tcp_true, places=places, log=self.log_comp, verbose=self.verbose)

        EQD2_sim_test(5.401663548992980e-17, eqd2_true = 2.078406374282604e+02, tcp_true = 0.949324568163257)
        EQD2_sim_test(1.205910932652372e-08, eqd2_true = 1.393951274066048e+02, tcp_true = 0.942631047077028)
        EQD2_sim_test(9.445578949875306e-08, eqd2_true = 1.334601511399029e+02, tcp_true = 0.940454956179937)
        EQD2_sim_test(7.619967243353366e-09, eqd2_true = 1.429222685964274e+02, tcp_true = 0.943650182802347)
        EQD2_sim_test(3.566448682419092e-06, eqd2_true = 1.192462016678157e+02, tcp_true = 0.931420749797299)
        EQD2_sim_test(1.948817629293370e-04, eqd2_true = 1.055811296331876e+02, tcp_true = 0.912230906833012)

    def test_constant_comp(self):
        BaseTest.primer_EQD2_comp(18, [1, 15, 16], gf=self.gf, ab_ratio_N=self.ab_ratio_N, verbose=self.verbose)
        BaseTest.primer_EQD2_comp(12, [1, 15, 16, 17], gf=self.gf, ab_ratio_N=self.ab_ratio_N, verbose=self.verbose)
        BaseTest.primer_EQD2_comp(10, [1, 15, 16, 17, 18], gf=self.gf, ab_ratio_N=self.ab_ratio_N, verbose=self.verbose)
        # BaseTest.primer_EQD2_comp(7.5, [1, 2, 3, 4, 5, 8, 9, 10], gf=self.gf, ab_ratio_N=self.ab_ratio_N, verbose=self.verbose)
        # BaseTest.primer_EQD2_comp(7.5, [1, 8, 9, 10, 11, 12, 15, 16], gf=self.gf, ab_ratio_N=self.ab_ratio_N, verbose=self.verbose)
        BaseTest.primer_EQD2_comp(7.5, [1, 15, 16, 17, 18, 19, 22, 23], gf=self.gf, ab_ratio_N=self.ab_ratio_N, verbose=self.verbose)
        # BaseTest.primer_EQD2_comp(5, [1, 2, 3, 4, 5, 8, 9, 10, 11, 12, 15, 16], gf=self.gf, ab_ratio_N=self.ab_ratio_N, verbose=self.verbose)
        # BaseTest.primer_EQD2_comp(5, [1, 2, 8, 9, 10, 11, 12, 15, 16, 17, 18, 19], gf=self.gf, ab_ratio_N=self.ab_ratio_N, verbose=self.verbose)
        BaseTest.primer_EQD2_comp(5, [1, 9, 15, 16, 17, 18, 19, 22, 23, 24, 25, 26], gf=self.gf, ab_ratio_N=self.ab_ratio_N, verbose=self.verbose)
        BaseTest.primer_EQD2_comp(4, [1, 5, 15, 16, 17, 18, 19, 22, 23, 24, 25, 26, 29, 30, 31], gf=self.gf, ab_ratio_N=self.ab_ratio_N, verbose=self.verbose)

    def test_variable_comp(self):
        BaseTest.primer_EQD2_comp([10, 6, 17], [1, 8, 9], gf=self.gf, ab_ratio_N=self.ab_ratio_N, verbose=self.verbose)
        BaseTest.primer_EQD2_comp(7*[6] + [16], [1, 8, 9, 10, 11, 12, 15, 16], gf=self.gf, ab_ratio_N=self.ab_ratio_N, verbose=self.verbose)
        BaseTest.primer_EQD2_comp([4.5] + 9*[2] + [15], [1, 8, 9, 10, 11, 12, 15, 16, 17, 18, 19], gf=self.gf, ab_ratio_N=self.ab_ratio_N, verbose=self.verbose)

        # Compare two fractions (primer and final with break in between) with continuous fractions (primer, low dose, final without any break).
        # BaseTest.primer_EQD2_comp([9.5, 9.5], [1, 14], gf=self.gf, ab_ratio_N=self.ab_ratio_N, verbose=self.verbose)
        BaseTest.primer_EQD2_comp([25, 29], [1, 14], gf=self.gf, ab_ratio_N=self.ab_ratio_N, verbose = self.verbose)
        BaseTest.primer_EQD2_comp([6] + 12*[3] + [10], np.arange(14) + 1, gf=self.gf, ab_ratio_N=self.ab_ratio_N, verbose=self.verbose)

    def test_variable_sched(self):
        BaseTest.primer_EQD2_comp([11, 8, 8, 17.5], [1, 8, 9, 10], gf=self.gf, ab_ratio_N=self.ab_ratio_N, verbose=self.verbose)
        BaseTest.primer_EQD2_comp([8, 8, 8, 8, 16], [1, 8, 9, 10, 11], gf=self.gf, ab_ratio_N=self.ab_ratio_N, verbose=self.verbose)
        BaseTest.primer_EQD2_comp([5.5, 5, 5, 5, 5, 5, 5, 15.25], [1, 8, 9, 10, 11, 12, 15, 16], gf=self.gf, ab_ratio_N=self.ab_ratio_N, verbose=self.verbose)
