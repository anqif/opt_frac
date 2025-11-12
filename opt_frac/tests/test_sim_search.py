import pickle
import numpy as np
import itertools

from opt_frac.tests.base_test import BaseTest
from opt_frac.simulation import primer_simulation, EQD2_simulation
from opt_frac.utilities import calc_frac_const_sched, calc_normal_bed_sched

NBED_ROUND_DIGITS = 6

class TestSimSearch(BaseTest):
    def setUp(self):
        np.random.seed(1)
        super(TestSimSearch, self).setUp()

        self.d_max_day = 18             # Maximum total dose per day.
        self.weekend_break = True

        self.recalc = True
        self.save_results = True

    def gen_potential_days(self, T_max, weekend_break = True):
        num_week_full = T_max // 7
        num_days_extra = T_max % 7

        t = 1
        days_list = []
        for w in range(num_week_full):
            days_list += [np.arange(t, t + 5)]
            if not weekend_break:
                days_list += [np.array([t + 5, t + 6])]
            t = t + 7

        if num_days_extra > 5 and weekend_break:
            days_list += [np.arange(t, t + 5)]
        else:
            days_list += [np.arange(t, t + num_days_extra)]
        days_all = np.concatenate(days_list)
        return days_all

    def gen_potential_scheds(self, T_max, weekend_break = True):
        days_vec = self.gen_potential_days(T_max, weekend_break = weekend_break)

        sched_combos = []
        for t in range(T_max):
            sched_combos += list(itertools.combinations(days_vec, t + 1))
        return sched_combos

    def search_frac_const(self, T_max, nbed, weekend_break = True):
        sched_list = self.gen_potential_scheds(T_max, weekend_break = weekend_break)

        fx_best = 0
        sf_best = np.inf
        s_best = None
        sur_frac_best = None
        sched_best = None

        for schedule in sched_list:
            fx_const = calc_frac_const_sched(schedule, nbed, ab_ratio_N=self.ab_ratio_N)
            if fx_const == 0:
                continue

            sur_frac, s_sbrt, sf_sbrt = primer_simulation(fx_const, schedule, gf_in=self.gf)
            if sf_sbrt < sf_best:
                fx_best = fx_const
                sf_best = sf_sbrt
                s_best = s_sbrt
                sur_frac_best = sur_frac
                sched_best = list(schedule)

        return fx_best, sched_best, sur_frac_best, s_best, sf_best

    def print_result(self, result_dict):
        print("\nBest Constant Fraction Size and Schedule {0} Weekend Break:".format("with" if result_dict["weekend"] else "without"))
        print("\t{0} Gy delivered on Days {1}".format(result_dict["fx"], result_dict["schedule"]))
        # print("Maximum Allowed Treatment Length: {0} Days".format(result_dict["Tmax"]))
        print("Normal Tissue BED (Actual): {0} Gy".format(np.round(result_dict["nbed"], NBED_ROUND_DIGITS)))
        print("EQD2: {0}".format(result_dict["eqd2_sim"]["eqd2"]))
        print("TCP: {0}".format(result_dict["eqd2_sim"]["tcp"]))

    def test_search(self):
        T_max = 19
        nbed_tar = 208.14

        if self.weekend_break:
            filename = "brute_search-Tmax_{0}-nbed_{1}.pkl".format(T_max, nbed_tar)
        else:
            filename = "brute_search-no_wkd-Tmax_{0}-nbed_{1}.pkl".format(T_max, nbed_tar)

        import_succ = True
        if not self.recalc:
            try:
                print("\nImporting prior results...")
                with open(BaseTest.save_path + filename, 'rb') as handle:
                    result_dict = pickle.load(handle)
                self.print_result(result_dict)
            except IOError:
                print("No prior results found, re-running brute search...")
                import_succ = False

        if self.recalc or not import_succ:
            print("Starting brute search for constant fractionation schedule...")
            print("Normal Tissue BED (Target): {0} Gy".format(nbed_tar))
            print("Maximum Allowed Treatment Length: {0} Days".format(T_max))
            fx_const, schedule_best, sur_frac_best, s_best, sf_best = self.search_frac_const(T_max, nbed = nbed_tar, weekend_break = self.weekend_break)
            eqd2_best, tcp_best, nfrac_eqd2_best = EQD2_simulation(s_best, gf_in=self.gf)
            nbed_act = calc_normal_bed_sched(fx_const, schedule_best, self.ab_ratio_N)

            result_dict = {"fx": fx_const, "schedule": schedule_best, "nbed": nbed_act, "Tmax": T_max,
                           "weekend": self.weekend_break,
                           "primer_sim": {"sur_frac": sur_frac_best, "s_sbrt": s_best, "sf_sbrt": sf_best},
                           "eqd2_sim": {"eqd2": eqd2_best, "tcp": tcp_best, "nfrac_eqd2": nfrac_eqd2_best}}
            self.print_result(result_dict)

            # Save results.
            if self.save_results:
                with open(BaseTest.save_path + filename, "wb") as handle:
                    pickle.dump(result_dict, handle, protocol=pickle.HIGHEST_PROTOCOL)
