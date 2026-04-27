# Introduction
This is a Python package for designing optimal dose schedules for cancer radiotherapy using sequential mixed-integer convex programming, a 
variant of the convex-concave procedure (CCP). Our dose scheduling algorithm is built upon a mechanistic tumor-dose response model, which 
captures the impact of hypoxia and reoxygenation on cellular radiosensitivity, and allows fraction sizes and treatment breaks to be adjusted 
dynamically to account for changes in the tumor microenvironment. For details on the model and implementation, please see the below references.

# Installation
The easiest way to install the package is using the [pip package manager](https://pypi.org/project/pip), which is included in most standard installations of Python.
For installation via HTTP, enter the following on command line:

```bash
pip install git+https://github.com/anqif/opt_frac.git
```

If you have SSH access set up on GitHub, enter the command

```bash
pip install git+ssh://git@github.com:anqif/opt_frac.git
```

# Example
Below is a simple example of our algorithm. More examples can be found in the `opt_frac\tests` subfolder.

```python
import opt_frac
from opt_frac.optimization import solve_ccp, print_result
from opt_frac.plot_sim import plot_dose

# Cell parameters.
rhot = 1e6                         # Tumor cell density.
vt = 64                            # Volume of a tumorlet.
nt = rhot * vt                     # Total number of cells in a tumorlet.
clf = 0.92                         # Cell loss factor.
gf = 0.25                          # Growth fraction.

f_pro_P = 0.5                      # Initial proliferation fraction in P compartment.
T_C = 2 * (24 * 60)                # Cell cycle time in minutes.
T_loss = 2 * (24 * 60)             # Cell loss half-time in H compartment in minutes.
k_m = 0.3                          # Survival probability of progeny after mitosis.
ab_ratio_N = 3                     # Ratio alpha/beta for normal tissue cells.

N0_P = (gf / f_pro_P) * nt
N0_H = clf * gf * (T_loss / T_C) * nt
N0_I = nt - N0_P - N0_H
N0 = [N0_P, N0_I, N0_H]            # Initial cell count in each compartment.

alpha_P = 0.305
beta_P = alpha_P / 2.8
OER_I = 1.7                        # Oxygen enhancement ratio
OER_H = 1.37

alpha_I = alpha_P / OER_I
beta_I = beta_P / OER_I**2
alpha_H = alpha_P / OER_H
beta_H = beta_P / OER_H**2
alpha = [alpha_P, alpha_I, alpha_H]
beta = [beta_P, beta_I, beta_H]

# Problem parameters.
delta_t = 60                      # Time step (sec) of cell update.
T_days = 14                       # Total days of treatment.

# Normal tissue parameters. 
M_bed = 146.67                    # Upper bound on BED3 for normal tissue.
d_max_day = 18                    # Maximum total dose per day.

# Algorithm parameters.
n_scale = 0.5*nt                  # Scaling factor on cell counts (adjust if needed to improve convergence rate).
has_slack_dyn = True              # Include slack variable in cell dynamic constraints?
weekend_break = True              # Enforce a weekend break?

# Fit model using sequential mixed-integer convex programming algorithm.
result = solve_ccp(nt, T_days, N0, alpha, beta, f_pro_P, T_C, T_loss, delta_t, k_m, ab_ratio_N = ab_ratio_N, M_bed = M_bed, recomp = False,
				   d_max_day = d_max_day, n_scale = n_scale, has_slack_dyn = has_slack_dyn, weekend_break = weekend_break, max_iter = 1000, 
				   delta_stop = 1e-3, solver = "MOSEK", verbose = True)

print_result(result)

# Plot optimal dose schedule.
plot_dose(result["d"], gf_in = gf, clf_in = clf, delta_t = delta_t)
```

# References
[1] Fu A, Gouw ZAR, Jeong J, Deasy JO. Optimal Radiotherapy Dose Scheduling with Variable Fraction Sizes and Breaks via Sequential Mixed-Integer Convex Programming. Phys Imaging Radiat Oncol. Under revision, April 2026.

[2] Gouw ZAR, Jeong J, Rimner A, Lee NY, Jackson A, Fu A, Sonke J-J, Deasy JO. "Primer Shot" Fractionation with an Early Treatment Break is Theoretically Superior to Consecutive Weekday Fractionation Schemes for Early-Stage Non-Small Cell Lung Cancer. Radiother Oncol. 2024; 190(1): 110006. https://doi.org/10.1016/j.radonc.2023.110006.

[3] Jeong J, Shoghi KI, Deasy JO. Modelling the Interplay Between Hypoxia and Proliferation in Radiotherapy Tumour Response. Phys Med Biol. 2013; 58(14): 4897. https://doi.org/10.1088/0031-9155/58/14/4897.

[4] Jeong J, Oh JH, Sonke JJ, Belderbos J, Bradley JD, Fontanella AN, Rao SS, Deasy JO. Modeling the Cellular Response of Lung Cancer to Radiation Therapy for a Broad Range of Fractionation Schedules. Clin Cancer Res. 2017; 23(18): 5469—79. https://doi.org/10.1158/1078-0432.ccr-16-3277.