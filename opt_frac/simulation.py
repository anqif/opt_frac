import numpy as np

from scipy.optimize import root_scalar
from warnings import warn

###########################################################################
# <<<<<<<<<<<<<<<<< RT fractional dose for EQD2 estimation >>>>>>>>>>>>>>>>
###########################################################################

def EQD2_simulation(s_sbrt, gf_in = 0.25, clf_in = 0.92, alpha_p_ori = 0.305, a_over_b = 2.8, oer_i = 1.7, oer_h = 1.37):
    #######################################################################
    ##                            INITIALIZATION                         ##
    #######################################################################
    # EQD2 calculation constants.
    rho_t = 1e6
    v_t_ref = 3e4
    f_s = 0.01
    t_c = 2
    f_p_pro_in = 0.5
    ht_loss = 2
    k_m = 0.3
    ht_lys = 3
    
    F_p_cyc = [0.56, 0.24, 0.2]
    Alpha_ratio_p_cyc = [2, 3]
    tumor_size = '-'
    tcp_goal = '-'
    vcp_goal = '-'
    d_t = 15

    # a_over_b = 2.8
    # oer_i = 1.7
    # oer_h = 1.37
    # alpha_p_ori = 0.305  
    beta_p_ori = alpha_p_ori/a_over_b 

    v_t = 3e4

    VCP = vcp_goal
    n_t = rho_t*v_t
    n_t_ref = rho_t*v_t_ref
    total_clono_cell = n_t*f_s
    delta_t = d_t/(60*24)
    t_start = 0
    
    comp_size = np.zeros(3)
    comp_size_ref = np.zeros(3)
    clf = clf_in
    gf = gf_in
    
    # Initial cell distribution for a specific CLF and GF value.
    f_p_pro = f_p_pro_in

    comp_size[0] = gf/f_p_pro*n_t
    comp_size[1] = (1 - gf*(1/f_p_pro_in + clf*ht_loss/t_c))*n_t
    comp_size[2] = clf*gf*ht_loss/t_c*n_t

    comp_size_ref[0] = gf/f_p_pro*n_t_ref
    comp_size_ref[1] = (1 - gf*(1/f_p_pro_in + clf*ht_loss/t_c))*n_t_ref
    comp_size_ref[2] = clf*gf*ht_loss/t_c*n_t_ref
    
    # Save the number of cells into a matrix.
    f_p = comp_size[0]/np.sum(comp_size)
    f_i = comp_size[1]/np.sum(comp_size)
    f_h = comp_size[2]/np.sum(comp_size)

    # Set EQD2 specific parameters.
    d = 2
    alpha_p = alpha_p_ori
    alpha_i = alpha_p_ori/oer_i
    alpha_h = alpha_p_ori/oer_h
    beta_p = beta_p_ori
    beta_i = beta_p_ori/(oer_i**2)
    beta_h = beta_p_ori/(oer_h**2)
    s_eqd2 = 0
    sf_eqd2 = 0
    eqd2 = 0

    #######################################################################
    ##                            SIMULATION                             ##
    #######################################################################
    # Assign proliferating fraction to the initial value.
    f_p_pro = f_p_pro_in

    # Cell distribution in each compartment 
    # [0: P viable, 1: P doomed, 2: I viable, 3: I doomed, 4: H viable, 5: H doomed, 6: Lysis].
    # Initially, all compartments are fully filled with viable cells.
    # Note: "comp_size" is the size of each compartment [0: P, 1: I, 2: H].
    cell_dist = np.zeros(7)
    cell_dist[0] = comp_size_ref[0]
    cell_dist[2] = comp_size_ref[1]
    cell_dist[4] = comp_size_ref[2]
    
    # Variables: t: time (day), j: # of fractions, add_time: additional time for weekend break, 
    #            cum_cell_dist: cumulative cell distribution for each time increment.
    t = 0       
    j = 0
    add_time = 0
    cum_cell_dist = []
    s_eqd2_pre = s_eqd2
    eqd2_pre = eqd2
    
    # Treat until the SF becomes equivalent to that of the SBRT regime.
    while (cell_dist[0] + cell_dist[2] + cell_dist[4]) > s_sbrt:
        # Change in f_p_pro (k_p) as blood supply improves.
        f_p_pro = 1 - 0.5*(cell_dist[0] + cell_dist[1])/comp_size[0]
        
        # RT fraction.
        if t > (t_start + j + add_time - delta_t/2) and t < (t_start + j + add_time + delta_t/2):
            cell_dist[1] = cell_dist[1] + cell_dist[0]*(1 - np.exp(-alpha_p*d - beta_p*d**2))
            cell_dist[0] = cell_dist[0]*np.exp(-alpha_p*d - beta_p*d**2)
            cell_dist[3] = cell_dist[3] + cell_dist[2]*(1 - np.exp(-alpha_i*d - beta_i*d**2))
            cell_dist[2] = cell_dist[2]*np.exp(-alpha_i*d - beta_i*d**2)
            cell_dist[5] = cell_dist[5] + cell_dist[4]*(1 - np.exp(-alpha_h*d - beta_h*d**2))
            cell_dist[4] = cell_dist[4]*np.exp(-alpha_h*d - beta_h*d**2)
            j = j + 1

            # Weekend break.
            if j % 5 == 0:
                add_time = add_time + 2
        
        # Cell proliferation and death.
        cell_dist[0] = cell_dist[0]*(2)**(f_p_pro*delta_t/t_c)
        h_pre = cell_dist[4] + cell_dist[5]
        cell_dist[4] = cell_dist[4]*(0.5)**(delta_t/ht_loss)
        cell_dist[5] = cell_dist[5]*(0.5)**(delta_t/ht_loss)
        p_d_pre = cell_dist[1]
        cell_dist[1] = cell_dist[1]*(2)**(f_p_pro*(2*k_m-1)*delta_t/t_c)
        
        # Mitotically dead cell in 1 time step.
        md = p_d_pre - cell_dist[1] + (h_pre - cell_dist[4] - cell_dist[5])
        cell_dist[6] = cell_dist[6] + md
        cell_dist[6] = cell_dist[6]*(0.5)**(delta_t/ht_lys)
        
        # Recompartmentalization of the cell.
        if (cell_dist[0] + cell_dist[1]) >= comp_size[0]:
            p_ex = (cell_dist[0] + cell_dist[1]) - comp_size[0]
            p_ratio = cell_dist[0]/(cell_dist[0] + cell_dist[1])
            cell_dist[0] = comp_size[0]*p_ratio
            cell_dist[1] = comp_size[0]*(1 - p_ratio)
            cell_dist[2] = cell_dist[2] + p_ex*p_ratio
            cell_dist[3] = cell_dist[3] + p_ex*(1 - p_ratio)
        elif (cell_dist[2] + cell_dist[3]) > 0:
            if (cell_dist[2] + cell_dist[3]) > (comp_size[0] - (cell_dist[0] + cell_dist[1])):
                p_def = comp_size[0] - (cell_dist[0] + cell_dist[1])
                i_ratio = cell_dist[2]/(cell_dist[2] + cell_dist[3])
                cell_dist[0] = cell_dist[0] + p_def*i_ratio
                cell_dist[1] = cell_dist[1] + p_def*(1 - i_ratio)
                cell_dist[2] = cell_dist[2] - p_def*i_ratio
                cell_dist[3] = cell_dist[3] - p_def*(1 - i_ratio)
            else:
                cell_dist[0] = cell_dist[0] + cell_dist[2]
                cell_dist[1] = cell_dist[1] + cell_dist[3]
                cell_dist[2] = 0
                cell_dist[3] = 0
                if (cell_dist[4] + cell_dist[5]) > 0:
                    if (cell_dist[4] + cell_dist[5]) > (comp_size[0] - (cell_dist[0] + cell_dist[1])):
                        p_def = comp_size[0] - (cell_dist[0] + cell_dist[1])
                        h_ratio = cell_dist[4]/(cell_dist[4] + cell_dist[5])
                        cell_dist[0] = cell_dist[0] + p_def*h_ratio
                        cell_dist[1] = cell_dist[1] + p_def*(1 - h_ratio)
                        cell_dist[4] = cell_dist[4] - p_def*h_ratio
                        cell_dist[5] = cell_dist[5] - p_def*(1 - h_ratio)
                    else:
                        cell_dist[0] = cell_dist[0] + cell_dist[4]
                        cell_dist[1] = cell_dist[1] + cell_dist[5]
                        cell_dist[4] = 0
                        cell_dist[5] = 0
        
        if (cell_dist[2] + cell_dist[3]) >= comp_size[1]:
            i_ex = (cell_dist[2] + cell_dist[3]) - comp_size[1]
            i_ratio = cell_dist[2]/(cell_dist[2] + cell_dist[3])
            cell_dist[2] = comp_size[1]*i_ratio
            cell_dist[3] = comp_size[1]*(1 - i_ratio)
            cell_dist[4] = cell_dist[4] + i_ex*i_ratio
            cell_dist[5] = cell_dist[5] + i_ex*(1 - i_ratio)
        elif (cell_dist[4] + cell_dist[5]) > 0:
            if (cell_dist[4] + cell_dist[5]) > (comp_size[1] - (cell_dist[2] + cell_dist[3])):
                i_def = comp_size[1] - (cell_dist[2] + cell_dist[3])
                h_ratio = cell_dist[4]/(cell_dist[4] + cell_dist[5])
                cell_dist[2] = cell_dist[2] + i_def*h_ratio
                cell_dist[3] = cell_dist[3] + i_def*(1 - h_ratio)
                cell_dist[4] = cell_dist[4] - i_def*h_ratio
                cell_dist[5] = cell_dist[5] - i_def*(1 - h_ratio)
            else:
                cell_dist[2] = cell_dist[2] + cell_dist[4]
                cell_dist[3] = cell_dist[3] + cell_dist[5]
                cell_dist[4] = 0
                cell_dist[5] = 0
        
        # Increase time step and store the number of cells in each compartment.
        t = t + delta_t
        cum_cell_dist = cum_cell_dist + [np.copy(cell_dist)]

        s_eqd2_pre = s_eqd2
        sf_eqd2_pre = sf_eqd2
        eqd2_pre = eqd2

        s_eqd2 = cell_dist[0] + cell_dist[2] + cell_dist[4]
        sf_eqd2 = s_eqd2/np.sum(comp_size)
        eqd2 = j*d

    if t == 0:
        eqd2 = 0
        tcp = 0
        warn("Initial cell distribution below target survival rate from SBRT")
    else:
        eqd2 = eqd2_pre + ((eqd2 - eqd2_pre)/(s_eqd2_pre - s_eqd2))*(s_eqd2_pre - s_sbrt)
        tcp = 0.95/(1 + (62.1/eqd2)**6)   # TD_50 = 62.1 Gy, gamma_50 = 1.5.
    n_frac = j

    # Note: Normal Tissue BED = n_treat*dose*(1 + dose/(alpha/beta)) = eqd2*(1 + 2/3) for normal alpha/beta ratio = 3.
    # Number of treatments = n_treat = eqd2/dose = eqd2/2, Duration of treatment = n_treat + add_time.

    # return {"eqd2": eqd2, "tcp": tcp, "n_frac": n_frac}
    return eqd2, tcp, n_frac

#########################################################################
# <<<<<<<<<<<<<<<<< RT fractional dose for SBRT schedule >>>>>>>>>>>>>>>>
#########################################################################

def primer_simulation(fx_in, schedule_in, gf_in = 0.25, clf_in = 0.92, alpha_p_ori = 0.305, a_over_b = 2.8, oer_i = 1.7, oer_h = 1.37):
    #######################################################################
    ##                            INITIALIZATION                         ##
    #######################################################################
    # Primer shot simulation constants.
    rho_t = 1e6
    v_t_ref = 3e4
    f_s = 0.01
    t_c = 2
    f_p_pro_in = 0.5
    ht_loss = 2
    k_m = 0.3
    ht_lys = 3
    
    F_p_cyc = [0.56, 0.24, 0.2]
    Alpha_ratio_p_cyc = [2, 3]
    tumor_size = '-'
    tcp_goal = '-'
    vcp_goal = '-'
    d_t = 15                     # Time step used in simulation (minutes).

    # a_over_b = 2.8
    # oer_i = 1.7
    # oer_h = 1.37
    # alpha_p_ori = 0.305  
    beta_p_ori = alpha_p_ori/a_over_b

    v_t = 3e4
    
    alpha_p = alpha_p_ori
    beta_p = beta_p_ori

    VCP = vcp_goal
    n_t = rho_t*v_t
    n_t_ref = rho_t*v_t_ref
    total_clono_cell = n_t*f_s
    delta_t = d_t/(60*24)        # Time step used in simulation (days).
    t_start = 0
    
    IC = []
    GF = []
    TCP = []
    TD50 = []
    BED = []
    Reox_time = []
    Reox_time2 = []
    Treat_duration = []
    vec_leng = [] 
    p_pre = []
    i_pre = []
    h_pre = []
    T_end = []
    
    comp_size = np.zeros(3)
    comp_size_ref = np.zeros(3)
    clf = clf_in
    gf = gf_in
    
    # Initial cell distribution for a specific CLF and GF value.
    f_p_pro = f_p_pro_in

    comp_size[0] = gf/f_p_pro*n_t
    comp_size[1] = (1 - gf*(1/f_p_pro_in + clf*ht_loss/t_c))*n_t
    comp_size[2] = clf*gf*ht_loss/t_c*n_t

    comp_size_ref[0] = gf/f_p_pro*n_t_ref
    comp_size_ref[1] = (1 - gf*(1/f_p_pro_in + clf*ht_loss/t_c))*n_t_ref
    comp_size_ref[2] = clf*gf*ht_loss/t_c*n_t_ref
    
    # Save the number of cells into a matrix.
    f_p = comp_size[0]/np.sum(comp_size)
    f_i = comp_size[1]/np.sum(comp_size)
    f_h = comp_size[2]/np.sum(comp_size)
    
    # Find effective alpha/beta ratio.
    Treat_day = schedule_in
    n_frac_sbrt = len(Treat_day)
    duration_sbrt = np.max(Treat_day)
    # duration_sbrt = 0 if n_frac_sbrt == 0 else np.max(Treat_day)
    if np.isscalar(fx_in):
        if fx_in < 0:
            raise ValueError("fx_in must be nonnegative")
        d = np.repeat(fx_in, n_frac_sbrt)
    elif len(fx_in) == n_frac_sbrt:
        fx_in = np.array(fx_in)
        if np.any(fx_in < 0):
            raise ValueError("fx_in must only contain nonnegative values")
        d = fx_in
    else:
        raise ValueError("fx_in must be a nonnegative scalar or vector of length {0}".format(n_frac_sbrt))
    
    Alpha_p_cyc = np.zeros(3)
    f = lambda alpha_s: F_p_cyc[0]*np.exp(-Alpha_ratio_p_cyc[0]*alpha_s*2 - Alpha_ratio_p_cyc[0]*(alpha_s/a_over_b)*4) + \
                        F_p_cyc[1]*np.exp(-alpha_s*2 - (alpha_s/a_over_b)*4) + \
                        F_p_cyc[2]*np.exp(-Alpha_ratio_p_cyc[1]*alpha_s*2 - Alpha_ratio_p_cyc[1]*(alpha_s/a_over_b)*4) - np.exp(-alpha_p*2 - (alpha_p/a_over_b)*4)
    f_root = root_scalar(f, bracket = [0.1, 0.5], method = "toms748")   # Solution should be around 0.3.
    Alpha_p_cyc[1] = f_root.root
    Alpha_p_cyc[0] = Alpha_p_cyc[1]*Alpha_ratio_p_cyc[0]
    Alpha_p_cyc[2] = Alpha_p_cyc[1]*Alpha_ratio_p_cyc[1]

    alpha_p_eff = np.zeros(n_frac_sbrt)
    beta_p_eff = np.zeros(n_frac_sbrt)
    for j in range(n_frac_sbrt):
        Su_p = F_p_cyc[0]*np.exp(-Alpha_p_cyc[0]*d[j] - (Alpha_p_cyc[0]/a_over_b)*d[j]**2) + \
               F_p_cyc[1]*np.exp(-Alpha_p_cyc[1]*d[j] - (Alpha_p_cyc[1]/a_over_b)*d[j]**2) + \
               F_p_cyc[2]*np.exp(-Alpha_p_cyc[2]*d[j] - (Alpha_p_cyc[2]/a_over_b)*d[j]**2)
        alpha_p_eff[j] = -np.log(Su_p)/(d[j]*(1 + (d[j]/a_over_b)))
        beta_p_eff[j] = (alpha_p_eff[j]/a_over_b)

    Su_i_2gy = np.exp(-alpha_p/oer_i*2 - (alpha_p/a_over_b)/(oer_i**2)*2**2)
    oer_i_g1 = (-(Alpha_p_cyc[0]*2) - np.sqrt((Alpha_p_cyc[0]*2)**2 - 4*np.log(Su_i_2gy)*(Alpha_p_cyc[0]/a_over_b)*2**2))/(2*np.log(Su_i_2gy))
    Su_h_2gy = np.exp(-alpha_p/oer_h*2 - (alpha_p/a_over_b)/(oer_h**2)*2**2)
    oer_h_g1 = (-(Alpha_p_cyc[0]*2) - np.sqrt((Alpha_p_cyc[0]*2)**2 - 4*np.log(Su_h_2gy)*(Alpha_p_cyc[0]/a_over_b)*2**2))/(2*np.log(Su_h_2gy))
    alpha_i = Alpha_p_cyc[0]/oer_i_g1
    beta_i = (Alpha_p_cyc[0]/a_over_b)/(oer_i_g1**2)
    alpha_h = Alpha_p_cyc[0]/oer_h_g1
    beta_h = (Alpha_p_cyc[0]/a_over_b)/(oer_h_g1**2)
    
    alpha_p = alpha_p_eff
    beta_p = beta_p_eff

    #######################################################################
    ##                            SIMULATION                             ##
    #######################################################################
    # Assign proliferating fraction to the initial value.
    f_p_pro = f_p_pro_in

    # Cell distribution in each compartment 
    # [0: P viable, 1: P doomed, 2: I viable, 3: I doomed, 4: H viable, 5: H doomed, 6: Lysis].
    # Initially, all compartments are fully filled with viable cells.
    # Note: "comp_size" is the size of each compartment [0: P, 1: I, 2: H].
    cell_dist = np.zeros(7)
    cell_dist[0] = comp_size[0]
    cell_dist[2] = comp_size[1]
    cell_dist[4] = comp_size[2]
    
    # Variables: t: time (day), j: # of fractions, add_time: additional time for weekend break, 
    #            cum_cell_dist: cumulative cell distribution for each time increment.
    t = 0       
    j = 0
    cum_cell_dist_sbrt = []
    
    # Treat for specific SBRT schedule.
    while t < (t_start + (np.max(Treat_day) - 1) + delta_t/2):
    # while t < (t_start + (duration_sbrt - 1) + delta_t/2):
        # Change in f_p_pro (k_p) as blood supply improves.
        f_p_pro = 1 - 0.5*(cell_dist[0] + cell_dist[1])/comp_size[0]

        # RT fraction.
        if t > (t_start + (Treat_day[j] - 1) - delta_t/2) and t < (t_start + (Treat_day[j] - 1) + delta_t/2):
            cell_dist[1] = cell_dist[1] + cell_dist[0]*(1 - np.exp(-alpha_p[j]*d[j] - beta_p[j]*d[j]**2))
            cell_dist[0] = cell_dist[0]*np.exp(-alpha_p[j]*d[j] - beta_p[j]*d[j]**2)
            cell_dist[3] = cell_dist[3] + cell_dist[2]*(1 - np.exp(-alpha_i*d[j] - beta_i*d[j]**2))
            cell_dist[2] = cell_dist[2]*np.exp(-alpha_i*d[j] - beta_i*d[j]**2)
            cell_dist[5] = cell_dist[5] + cell_dist[4]*(1 - np.exp(-alpha_h*d[j] - beta_h*d[j]**2))
            cell_dist[4] = cell_dist[4]*np.exp(-alpha_h*d[j] - beta_h*d[j]**2)
            j = j + 1
        
        # Cell proliferation and death.
        cell_dist[0] = cell_dist[0]*(2)**(f_p_pro*delta_t/t_c)
        h_pre = cell_dist[4] + cell_dist[5]
        cell_dist[4] = cell_dist[4]*(0.5)**(delta_t/ht_loss)
        cell_dist[5] = cell_dist[5]*(0.5)**(delta_t/ht_loss)
        p_d_pre = cell_dist[1]
        cell_dist[1] = cell_dist[1]*(2)**(f_p_pro*(2*k_m - 1)*delta_t/t_c)
        
        # Mitotically dead cell in 1 time step.
        md = p_d_pre - cell_dist[1] + (h_pre - cell_dist[4] - cell_dist[5])
        cell_dist[6] = cell_dist[6] + md
        cell_dist[6] = cell_dist[6]*(0.5)**(delta_t/ht_lys)
        
        # Recompartmentalization of the cell.
        if (cell_dist[0] + cell_dist[1]) >= comp_size[0]:   # P compartment full.                                         
            p_ex = (cell_dist[0] + cell_dist[1]) - comp_size[0]                                 
            p_ratio = cell_dist[0]/(cell_dist[0] + cell_dist[1])                                
            cell_dist[0] = comp_size[0]*p_ratio
            cell_dist[1] = comp_size[0]*(1 - p_ratio)
            cell_dist[2] = cell_dist[2] + p_ex*p_ratio
            cell_dist[3] = cell_dist[3] + p_ex*(1 - p_ratio)
        elif (cell_dist[2] + cell_dist[3]) > 0:             # I compartment contains cells.
            if (cell_dist[2] + cell_dist[3]) > (comp_size[0] - (cell_dist[0] + cell_dist[1])):
                p_def = comp_size[0] - (cell_dist[0] + cell_dist[1])
                i_ratio = cell_dist[2]/(cell_dist[2] + cell_dist[3])
                cell_dist[0] = cell_dist[0] + p_def*i_ratio
                cell_dist[1] = cell_dist[1] + p_def*(1 - i_ratio)
                cell_dist[2] = cell_dist[2] - p_def*i_ratio
                cell_dist[3] = cell_dist[3] - p_def*(1 - i_ratio)
            else:
                cell_dist[0] = cell_dist[0] + cell_dist[2]
                cell_dist[1] = cell_dist[1] + cell_dist[3]
                cell_dist[2] = 0
                cell_dist[3] = 0
                if (cell_dist[4] + cell_dist[5]) > 0:        # H compartment contains cells.
                    if (cell_dist[4] + cell_dist[5]) > (comp_size[0] - (cell_dist[0] + cell_dist[1])):
                        p_def = comp_size[0] - (cell_dist[0] + cell_dist[1])
                        h_ratio = cell_dist[4]/(cell_dist[4] + cell_dist[5])
                        cell_dist[0] = cell_dist[0] + p_def*h_ratio
                        cell_dist[1] = cell_dist[1] + p_def*(1 - h_ratio)
                        cell_dist[4] = cell_dist[4] - p_def*h_ratio
                        cell_dist[5] = cell_dist[5] - p_def*(1 - h_ratio)
                    else:
                        cell_dist[0] = cell_dist[0] + cell_dist[4]
                        cell_dist[1] = cell_dist[1] + cell_dist[5]
                        cell_dist[4] = 0
                        cell_dist[5] = 0
                        
        if (cell_dist[2] + cell_dist[3]) >= comp_size[1]:   # I compartment full.
            i_ex = (cell_dist[2] + cell_dist[3]) - comp_size[1]
            i_ratio = cell_dist[2]/(cell_dist[2] + cell_dist[3])
            cell_dist[2] = comp_size[1]*i_ratio
            cell_dist[3] = comp_size[1]*(1 - i_ratio)
            cell_dist[4] = cell_dist[4] + i_ex*i_ratio
            cell_dist[5] = cell_dist[5] + i_ex*(1 - i_ratio)
        elif (cell_dist[4] + cell_dist[5]) > 0:             # H compartment contains cells.
            if (cell_dist[4] + cell_dist[5]) > (comp_size[1] - (cell_dist[2] + cell_dist[3])):
                i_def = comp_size[1] - (cell_dist[2] + cell_dist[3])
                h_ratio = cell_dist[4]/(cell_dist[4] + cell_dist[5])
                cell_dist[2] = cell_dist[2] + i_def*h_ratio
                cell_dist[3] = cell_dist[3] + i_def*(1 - h_ratio)
                cell_dist[4] = cell_dist[4] - i_def*h_ratio
                cell_dist[5] = cell_dist[5] - i_def*(1 - h_ratio)
            else:
                cell_dist[2] = cell_dist[2] + cell_dist[4]
                cell_dist[3] = cell_dist[3] + cell_dist[5]
                cell_dist[4] = 0
                cell_dist[5] = 0
        
        # Increase time step and store the number of cells in each compartment.
        t = t + delta_t
        cum_cell_dist_sbrt = cum_cell_dist_sbrt + [np.copy(cell_dist)]

    #######################################################################
    ##                              RESULTS                              ##
    #######################################################################
    # Collect results.
    cum_cell_dist_sbrt = np.row_stack(cum_cell_dist_sbrt)   # Row = time, column = cell distribution.
    s_sbrt = cell_dist[0] + cell_dist[2] + cell_dist[4]     # Total viable cells.
    sf_sbrt = s_sbrt/np.sum(comp_size)
    n_frac_sbrt = len(Treat_day)
    duration_sbrt = np.max(Treat_day)
    ntd2 = n_frac_sbrt*d*(1 + (d/a_over_b))/(1 + (2/a_over_b))
    d_sbrt = d
    t_sbrt = t
    
    # Calculate SF of each compartment over time for SBRT schedule.
    # sur_frac = np.zeros((cum_cell_dist_sbrt.shape[0] + 2, 5))   # Time Step (days), Total SF, P comp SF, I comp SF, H comp SF.
    # sur_frac[0,:] = np.array([0,    1, comp_size[0]/np.sum(comp_size), comp_size[1]/np.sum(comp_size), comp_size[2]/np.sum(comp_size)])
    # sur_frac[1,:] = np.array([0.99, 1, comp_size[0]/np.sum(comp_size), comp_size[1]/np.sum(comp_size), comp_size[2]/np.sum(comp_size)])
    
    # for t in range(cum_cell_dist_sbrt.shape[0]):
    #     tot = cum_cell_dist_sbrt[t,0] + cum_cell_dist_sbrt[t,2] + cum_cell_dist_sbrt[t,4]   # Total viable cells.
    #     p = cum_cell_dist_sbrt[t,0]
    #     i = cum_cell_dist_sbrt[t,2]
    #     h = cum_cell_dist_sbrt[t,4]
    #     full = np.sum(comp_size)
    #     sur_frac[t+2,:] = np.array([95/96 + (t+1)/96, tot/full, p/full, i/full, h/full])

    # t_frac = np.arange(0, t_sbrt + delta_t, delta_t) + 1   # Time step (days).
    t_frac = np.arange(cum_cell_dist_sbrt.shape[0] + 1)*delta_t + 1
    sur_frac = np.zeros((cum_cell_dist_sbrt.shape[0] + 1, 4))   # Total SF, P comp SF, I comp SF, H comp SF.
    full_size = np.sum(comp_size)

    sur_frac[0,:] = np.array([1, comp_size[0]/full_size, comp_size[1]/full_size, comp_size[2]/full_size])
    for t in range(cum_cell_dist_sbrt.shape[0]):
        p = cum_cell_dist_sbrt[t,0]   # P viable.
        i = cum_cell_dist_sbrt[t,2]   # I viable.
        h = cum_cell_dist_sbrt[t,4]   # H viable.
        tot = p + i + h
        sur_frac[t+1,:] = np.array([tot/full_size, p/full_size, i/full_size, h/full_size])
    sur_frac = np.column_stack([t_frac, sur_frac])   # Concatenate time step (days) as first column.

    # return {"tot_viable": s_sbrt, "sur_frac": sf_sbrt, "sur_frac_hist": sur_frac, "num_frac": n_frac_sbrt,
    #         "duration": duration_sbrt, "ntd2": ntd2, "d": d_sbrt, "t": t_sbrt}
    return sur_frac, s_sbrt, sf_sbrt
