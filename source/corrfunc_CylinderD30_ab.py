from matplotlib import rcParams
import matplotlib
import matplotlib.pyplot as plt 
import numpy as np
import h5py as h5
from Corrfunc.mocks.DDrppi_mocks import DDrppi_mocks
from Corrfunc.utils import convert_rp_pi_counts_to_wp
from corr_func import *
from Corrfunc.theory.DD import DD
from Corrfunc.utils import convert_3d_counts_to_cf
import collections
import matplotlib as mpl
##Specify rcParams for Matplotlib##
mpl.rcParams['axes.labelsize'] = 35
mpl.rcParams['legend.fontsize'] = 35
mpl.rcParams['axes.titlesize'] = 35
mpl.rcParams['xtick.major.size'] = 15
mpl.rcParams['xtick.major.width'] = 2
mpl.rcParams['xtick.minor.size'] = 8
mpl.rcParams['xtick.minor.width'] = 2
mpl.rcParams['ytick.major.size'] = 15
mpl.rcParams['ytick.major.width'] = 2
mpl.rcParams['ytick.minor.size'] = 8
mpl.rcParams['ytick.minor.width'] = 2
mpl.rcParams['xtick.labelsize'] = 35
mpl.rcParams['ytick.labelsize'] = 35
font = {'family' : 'serif',
        'weight' : 'normal',
        'size'   : 20}
from matplotlib import gridspec

# Define the dictionary storing all correlation and projected correlations
xi_dict = collections.defaultdict(list)
wp_dict = collections.defaultdict(list)

# Save the outputs, do not need to rerun everytime
def wp_xi(phase):
    halo_path = '/users/PAS0023/osu0218/Project/For_Chenxiao/New_Catalogs/'
    fname_particle = halo_path + 'particle_subsample-fid'+str(phase)+'cosmo2-z0p3.hdf5'
    fname_halo = halo_path + 'NHOD_0.10_11.7_11.7_12.9_1.00_0.0_0.0_1.0_1.0_0.0_'+str(phase)+'_z0p300_richness_d30.00.h5'
    fname_galaxy = halo_path + 'NHOD_0.60_12.7_11.0_13.8_1.50_0.0_0.0_1.0_0.6_0.0_'+str(phase)+'_z0p3.hdf5'
    f_particle = h5.File(fname_particle, 'r')
    f_halo = h5.File(fname_halo, 'r')
    f_galaxy = h5.File(fname_galaxy, 'r')

    data_halos = f_halo['halos']
    data_gals = f_galaxy['particles']
    data_part = f_particle['particles']
    print(data_halos.dtype)
    # unit mpc/h

    data_halos_mass_gt_2_14 = prop_cut(data_halos, 'mass', 2e14, 1e16) #1e16 refers to infinity
    X_halos = data_halos_mass_gt_2_14['x'][:].astype(np.float32) # Figure 2
    Y_halos = data_halos_mass_gt_2_14['y'][:].astype(np.float32)
    Z_halos = data_halos_mass_gt_2_14['z'][:].astype(np.float32)

    # make lambda cut and pick the top 7747 halos in abundance, then rerun the calculation of correlations
    data_halos_lambda = data_halos['lambda']
    Sort = np.argsort(-data_halos_lambda)
    sort = Sort[:len(X_halos)]
    X_halos_match = data_halos['x'][sort].astype(np.float32) # X-cluster, Figure 3
    Y_halos_match = data_halos['y'][sort].astype(np.float32)
    Z_halos_match = data_halos['z'][sort].astype(np.float32)

    X_gals = data_gals['x'][:].astype(np.float32) 
    Y_gals = data_gals['y'][:].astype(np.float32)
    Z_gals = data_gals['z'][:].astype(np.float32)

    X_part = data_part['x'][:].astype(np.float32)
    Y_part = data_part['y'][:].astype(np.float32)
    Z_part = data_part['z'][:].astype(np.float32)

    print(len(X_halos), len(Y_halos), len(Z_halos))
    print(len(X_halos_match), len(Y_halos_match), len(Z_halos_match))
    print(len(X_gals), len(Y_gals), len(Z_gals))
    print(len(X_part), len(Y_part), len(Z_part))
    
    # calculate xi
    nrbins = 30
    bins = np.logspace(np.log10(0.1), np.log10(100.0), nrbins + 1)
    r = (bins[:-1] + bins[1:]) / 2.0
    boxsize = 1100
    xi_hg = xi_pairs_cross(X_halos,Y_halos,Z_halos,X_gals,Y_gals,Z_gals,bins,boxsize)
    xi_hh = xi_pairs_cross(X_halos,Y_halos,Z_halos,X_halos,Y_halos,Z_halos,bins,boxsize)
    xi_hm = xi_pairs_cross(X_halos,Y_halos,Z_halos,X_part,Y_part,Z_part,bins,boxsize)
    xi_gg = xi_pairs_cross(X_gals,Y_gals,Z_gals,X_gals,Y_gals,Z_gals,bins,boxsize)
    xi_mm = xi_pairs_cross(X_part,Y_part,Z_part,X_part,Y_part,Z_part,bins,boxsize)
    np.savetxt("abacus_phases/xi_abacus_phase"+str(phase)+".txt",np.transpose(np.array([r, xi_hg, xi_hh, xi_hm, xi_gg, xi_mm])))
    
    # calculate wp
    pimax = 100
    nrbins = 30
    bins = np.logspace(np.log10(0.1), np.log10(100.0), nrbins + 1)
    rp = (bins[0:-1] + bins[1:]) / 2.0
    wp_hg = wp_pairs_cross(X_halos,Y_halos,Z_halos,X_gals,Y_gals,Z_gals,pimax,bins,boxsize)
    wp_hh = wp_pairs_cross(X_halos,Y_halos,Z_halos,X_halos,Y_halos,Z_halos,pimax,bins,boxsize)
    wp_hm = wp_pairs_cross(X_halos,Y_halos,Z_halos,X_part,Y_part,Z_part,pimax,bins,boxsize)
    wp_gg = wp_pairs_cross(X_gals,Y_gals,Z_gals,X_gals,Y_gals,Z_gals,pimax,bins,boxsize)
    wp_mm = wp_pairs_cross(X_part,Y_part,Z_part,X_part,Y_part,Z_part,pimax,bins,boxsize)
    np.savetxt("abacus_phases/wp_abacus_phase"+str(phase)+".txt",np.transpose(np.array([rp, wp_hg, wp_hh, wp_hm, wp_gg, wp_mm])))

    # Cell to do switch halos to halos after abundance matching, converting from h to c##
    X_halos = X_halos_match # X_clusters
    Y_halos = Y_halos_match
    Z_halos = Z_halos_match
    
    nrbins = 30
    bins = np.logspace(np.log10(0.1), np.log10(100.0), nrbins + 1)
    r = (bins[:-1] + bins[1:]) / 2.0
    boxsize = 1100
    xi_cg = xi_pairs_cross(X_halos,Y_halos,Z_halos,X_gals,Y_gals,Z_gals,bins,boxsize)
    xi_cc = xi_pairs_cross(X_halos,Y_halos,Z_halos,X_halos,Y_halos,Z_halos,bins,boxsize)
    xi_cm = xi_pairs_cross(X_halos,Y_halos,Z_halos,X_part,Y_part,Z_part,bins,boxsize)
    xi_gg = xi_pairs_cross(X_gals,Y_gals,Z_gals,X_gals,Y_gals,Z_gals,bins,boxsize)
    xi_mm = xi_pairs_cross(X_part,Y_part,Z_part,X_part,Y_part,Z_part,bins,boxsize)
    np.savetxt("abacus_phases/xi_ab_abacus_phase"+str(phase)+".txt",np.transpose(np.array([r, xi_cg, xi_cc, xi_cm, xi_gg, xi_mm])))

    pimax = 100
    nrbins = 30
    bins = np.logspace(np.log10(0.1), np.log10(100.0), nrbins + 1)
    rp = (bins[0:-1] + bins[1:]) / 2.0
    wp_cg = wp_pairs_cross(X_halos,Y_halos,Z_halos,X_gals,Y_gals,Z_gals,pimax,bins,boxsize)
    wp_cc = wp_pairs_cross(X_halos,Y_halos,Z_halos,X_halos,Y_halos,Z_halos,pimax,bins,boxsize)
    wp_cm = wp_pairs_cross(X_halos,Y_halos,Z_halos,X_part,Y_part,Z_part,pimax,bins,boxsize)
    wp_gg = wp_pairs_cross(X_gals,Y_gals,Z_gals,X_gals,Y_gals,Z_gals,pimax,bins,boxsize)
    wp_mm = wp_pairs_cross(X_part,Y_part,Z_part,X_part,Y_part,Z_part,pimax,bins,boxsize)
    np.savetxt("abacus_phases/wp_ab_abacus_phase"+str(phase)+".txt",np.transpose(np.array([rp, wp_cg, wp_cc, wp_cm, wp_gg, wp_mm])))

# run wp_xi
def run_wp_xi():
    for i in range(20):
        wp_xi(i)

# averaging the result of 20 phases
def ave_phase():
    nrbins = 30
    xi_list = ["r", "xi_hg", "xi_hh", "xi_hm", "xi_gg", "xi_mm"]
    xi_ab_list = ["r", "xi_ab_hg", "xi_ab_hh", "xi_ab_hm", "xi_ab_gg", "xi_ab_mm"]
    for xi in xi_list:
        xi_dict[xi] = [0] * nrbins
    for xi in xi_ab_list:
        xi_dict[xi] = [0] * nrbins

    nrpbins = 30
    wp_list = ["rp", "wp_hg", "wp_hh", "wp_hm", "wp_gg", "wp_mm"]
    wp_ab_list = ["rp", "wp_ab_hg", "wp_ab_hh", "wp_ab_hm", "wp_ab_gg", "wp_ab_mm"]
    for wp in wp_list:
        wp_dict[wp] = [0] * nrpbins
    for wp in wp_ab_list:
        wp_dict[wp] = [0] * nrpbins

    for phase in range(20):
        xi_file = np.loadtxt("abacus_phases/xi_abacus_phase"+str(phase)+".txt")
        xi_ab_file = np.loadtxt("abacus_phases/xi_ab_abacus_phase"+str(phase)+".txt")
        for (index, xi) in enumerate(xi_list):
            xi_dict[xi] += xi_file[:,index] / 20.
        for (index, xi) in enumerate(xi_ab_list):
            if index == 0:
                continue
            xi_dict[xi] += xi_ab_file[:,index] / 20.

        wp_file = np.loadtxt("abacus_phases/wp_abacus_phase"+str(phase)+".txt")
        wp_ab_file = np.loadtxt("abacus_phases/wp_ab_abacus_phase"+str(phase)+".txt")
        for (index, wp) in enumerate(wp_list):
            wp_dict[wp] += wp_file[:,index] / 20.
        for (index, wp) in enumerate(wp_ab_list):
            if index == 0:
                continue
            wp_dict[wp] += wp_ab_file[:,index] / 20.
            
# $b_g$ and $b_h$ in four different ways using $\xi$
def bias_xi(ab = False):  
    r = xi_dict["r"]
    xi_hg = xi_dict["xi_ab_hg"] if ab else xi_dict["xi_hg"]
    xi_hh = xi_dict["xi_ab_hh"] if ab else xi_dict["xi_hh"]
    xi_hm = xi_dict["xi_ab_hm"] if ab else xi_dict["xi_hm"]
    xi_gg = xi_dict["xi_ab_gg"] if ab else xi_dict["xi_gg"]
    xi_mm = xi_dict["xi_ab_mm"] if ab else xi_dict["xi_mm"]
    # Plot xicg vs rp
    fig = plt.figure(figsize=(12, 18))
    # set height ratios for sublots
    gs = gridspec.GridSpec(2, 1, height_ratios=[2, 1]) 
    gs.update(wspace=0.025, hspace=0) 

    r_min = 11   #8 for h and 10 for c
    # the fisrt subplot
    ax0 = plt.subplot(gs[0])
    # log scale for axis Y of the first subplot
    ax0.set_yscale("log")
    ax0.set_xscale("log")
    ax0.set_ylabel("$\\xi$")
#     ax0.set_yticks(np.logspace(-1, 4, num=50))
    ax0.minorticks_on()
    ax0.set_yticks([1.e-3, 1.e-2, 1.e-1, 1, 1.e+1, 1.e+2, 1.e+3, 1.e+4], minor=True)
    #ax0.set_xlabel("$r$")
    line_hg, = ax0.plot(r, xi_hg, color='orange', linestyle='-', linewidth=3)
    line_hh, = ax0.plot(r[r_min:30], xi_hh[r_min:30], color='red', linestyle='-', linewidth=3)
    line_hm, = ax0.plot(r, xi_hm, color='blue', linestyle='-', linewidth=3)
    line_gg, = ax0.plot(r, xi_gg, color='green', linestyle='-', linewidth=3)
    line_mm, = ax0.plot(r, xi_mm, color='black', linestyle='-', linewidth=3)

    #the second subplot
    # shared axis X
    ax1 = plt.subplot(gs[1], sharex = ax0)
    line_hg_ratio, = ax1.plot(r, [k/np.sqrt(m*n) for k, m, n in zip(xi_hg, xi_gg, xi_mm)], color='orange', linestyle='solid', linewidth=3)
    line_hh_ratio, = ax1.plot(r[r_min:30], [np.sqrt(m/n) for m, n in zip(xi_hh, xi_mm)][r_min:30], color='red', linestyle='solid', linewidth=3)
    line_hm_ratio, = ax1.plot(r, [m/n for m, n in zip(xi_hm, xi_mm)], color='blue', linestyle='solid', linewidth=3)
    line_gg_ratio, = ax1.plot(r, [np.sqrt(m/n) for m, n in zip(xi_gg, xi_mm)], color='green', linestyle='solid', linewidth=3)
    line_bg_hg_hm, = ax1.plot(r, [m/n for m, n in zip(xi_hg, xi_hm)], color='green', linestyle=':', linewidth=3)
    
    # print the average halo bias at large scale using the average of two observables
    bh_lg_scale = (np.array([m/n for m, n in zip(xi_hm, xi_mm)]) + np.array([k/np.sqrt(m*n) for k, m, n in zip(xi_hg, xi_gg, xi_mm)])) / 2
    bg_lg_scale = [np.sqrt(m/n) for m, n in zip(xi_gg, xi_mm)]
    print(ab, "xi")
    print("bh at rp > 10:", bh_lg_scale[20:])
    print("ave_bh at rp > 10:", sum(bh_lg_scale[20:])/len(bh_lg_scale[20:]))
    print("bg at rp > 10:", bg_lg_scale[20:])
    print("ave_bg at rp > 10:", sum(bg_lg_scale[20:])/len(bg_lg_scale[20:]))

    plt.setp(ax0.get_xticklabels(), visible=False)
    ax1.set_xlabel("$r$ [$h^{-1}$Mpc]")
    ax1.set_ylabel("$b(r)$")
    ax1.set_ylim(1, 6)
#     yticks = ax1.yaxis.get_major_ticks()
#     yticks[-1].label1.set_visible(False)
    # remove vertical gap between subplots
    plt.subplots_adjust(hspace=.0)
    
    if ab:
        ax0.legend((line_hg, line_hh, line_hm, line_gg, line_mm), ('$\\xi_{cg}$', '$\\xi_{cc}$', '$\\xi_{cm}$', '$\\xi_{gg}$', '$\\xi_{mm}$'), loc='upper right')
        ax1.legend((line_hg_ratio, line_hh_ratio, line_hm_ratio, line_gg_ratio, line_bg_hg_hm), ('$b_c \simeq \\xi_{cg} \\left(\\xi_{gg} \\xi_{mm}\\right)^{-1/2}$', '$b_c \simeq \\left(\\xi_{cc} / \\xi_{mm}\\right)^{1/2}$', '$b_c \simeq \\left(\\xi_{cm} / \\xi_{mm}\\right)$', '$b_g \simeq \\left(\\xi_{gg} / \\xi_{mm}\\right)^{1/2}$', '$b_g \simeq \\xi_{cg} / \\xi_{cm}$'), loc='upper left', fontsize=30)
         # remove vertical gap between subplots
        plt.savefig("xi_abacus_ab_match_bias.pdf",dpi=300,bbox_inches='tight')
    else:
        ax0.legend((line_hg, line_hh, line_hm, line_gg, line_mm), 
                   ('$\\xi_{hg}$', '$\\xi_{hh}$', '$\\xi_{hm}$', '$\\xi_{gg}$', '$\\xi_{mm}$'), loc='upper right')
        ax1.legend((line_hg_ratio, line_hh_ratio, line_hm_ratio, line_gg_ratio, line_bg_hg_hm), 
                   ('$b_h \simeq \\xi_{hg} \\left(\\xi_{gg} \\xi_{mm}\\right)^{-1/2}$', '$b_h \simeq \\left(\\xi_{hh} / \\xi_{mm}\\right)^{1/2}$', '$b_h \simeq \\left(\\xi_{hm} / \\xi_{mm}\\right)$', '$b_g \simeq \\left(\\xi_{gg} / \\xi_{mm}\\right)^{1/2}$', '$b_g \simeq \\xi_{hg} / \\xi_{hm}$'), loc='upper left', fontsize=30)
        plt.savefig("xi_abacus_bias.pdf",dpi=300,bbox_inches='tight')

        
# $b_g$ and $b_h$ in four different ways using $wp$, wp_ab_hg is wpcg
def bias_wp(ab = False):
    r = wp_dict["rp"]
    wp_hg = wp_dict["wp_ab_hg"] if ab else wp_dict["wp_hg"]
    wp_hh = wp_dict["wp_ab_hh"] if ab else wp_dict["wp_hh"]
    wp_hm = wp_dict["wp_ab_hm"] if ab else wp_dict["wp_hm"]
    wp_gg = wp_dict["wp_ab_gg"] if ab else wp_dict["wp_gg"]
    wp_mm = wp_dict["wp_ab_mm"] if ab else wp_dict["wp_mm"]
    np.savetxt("output_txt/wp_ave_phase_ab.txt",np.transpose(np.array([r, wp_hg, wp_hh, wp_hm, wp_gg, wp_mm])))
    # Plot xicg vs rp
    fig = plt.figure(figsize=(12, 18))
    # set height ratios for sublots
    gs = gridspec.GridSpec(2, 1, height_ratios=[2, 1]) 
    gs.update(wspace=0.025, hspace=0) 

    r_min = 11 #6 for h and 4 for c
    r_max = 28 #
    # the fisrt subplot
    ax0 = plt.subplot(gs[0])
    # log scale for axis Y of the first subplot
    ax0.set_yscale("log")
    ax0.set_xscale("log")
    ax0.set_ylabel("$w_p$")
    ax0.minorticks_on()
    ax0.set_yticks([1.e-1, 1, 1.e+1, 1.e+2, 1.e+3, 1.e+4], minor=True)
    #ax0.set_xlabel("$r$")
    line_hg, = ax0.plot(r, wp_hg, color='orange', linestyle='-', linewidth=3)
    line_hh, = ax0.plot(r[r_min:], wp_hh[r_min:], color='red', linestyle='-', linewidth=3)
    line_hm, = ax0.plot(r, wp_hm, color='blue', linestyle='-', linewidth=3)
    line_gg, = ax0.plot(r, wp_gg, color='green', linestyle='-', linewidth=3)
    line_mm, = ax0.plot(r, wp_mm, color='black', linestyle='-', linewidth=3)

    # the second subplot
    # shared axis X
    ax1 = plt.subplot(gs[1], sharex = ax0)
    line_hg_ratio, = ax1.plot(r, [k/np.sqrt(m*n) for k, m, n in zip(wp_hg, wp_gg, wp_mm)], color='orange', linestyle='solid', linewidth=3)
    line_hh_ratio, = ax1.plot(r[r_min:], [np.sqrt(m/n) for m, n in zip(wp_hh, wp_mm)][r_min:], color='red', linestyle='solid', linewidth=3)
    line_hm_ratio, = ax1.plot(r, [m/n for m, n in zip(wp_hm, wp_mm)], color='blue', linestyle='solid', linewidth=3)
    line_gg_ratio, = ax1.plot(r, [np.sqrt(m/n) for m, n in zip(wp_gg, wp_mm)], color='green', linestyle='solid', linewidth=3)
    line_bg_hg_hm, = ax1.plot(r, [m/n for m, n in zip(wp_hg, wp_hm)], color='green', linestyle=':', linewidth=3)

    # print the average halo bias at large scale using the average of two observables
    bh_lg_scale = (np.array([m/n for m, n in zip(wp_hm, wp_mm)]) + np.array([k/np.sqrt(m*n) for k, m, n in zip(wp_hg, wp_gg, wp_mm)])) / 2
    bg_lg_scale = [np.sqrt(m/n) for m, n in zip(wp_gg, wp_mm)]
    print(ab, "wp")
    print("bh at rp > 10:", bh_lg_scale[20:])
    print("ave_bh at rp > 10:", sum(bh_lg_scale[20:])/len(bh_lg_scale[20:]))
    print("bg at rp > 10:", bg_lg_scale[20:])
    print("ave_bg at rp > 10:", sum(bg_lg_scale[20:])/len(bg_lg_scale[20:]))

    plt.setp(ax0.get_xticklabels(), visible=False)
    ax1.set_xlabel("$r_p$ [$h^{-1}$Mpc]")
    ax1.set_ylabel("$b(r_p)$")
    ax1.set_ylim(1, 6)
#     yticks = ax1.yaxis.get_major_ticks()
#     yticks[-1].label1.set_visible(False)
    # remove vertical gap between subplots
    plt.subplots_adjust(hspace=.0)
    
    if ab:
        ax0.legend((line_hg, line_hh, line_hm, line_gg, line_mm), ('$w_{p,cg}$', '$w_{p,cc}$', '$w_{p,cm}$', '$w_{p,gg}$', '$w_{p,mm}$'), loc='upper right')
        ax1.legend((line_hg_ratio, line_hh_ratio, line_hm_ratio, line_gg_ratio, line_bg_hg_hm), ('$b_c \simeq w_{p,cg} \\left(w_{p,gg} w_{p,mm}\\right)^{-1/2}$', '$b_c \simeq \\left(w_{p,cc} / w_{p,mm}\\right)^{1/2}$', '$b_c \simeq \\left(w_{p,cm} / w_{p,mm}\\right)$', '$b_g \simeq \\left(w_{p,gg} / w_{p,mm}\\right)^{1/2}$',  '$b_g \simeq w_{p,hg} / w_{p,hm}$'), loc='upper left', fontsize=30)
        plt.savefig("wp_abacus_ab_match_bias.pdf",dpi=300,bbox_inches='tight')
    else :
        ax0.legend((line_hg, line_hh, line_hm, line_gg, line_mm), ('$w_{p,hg}$', '$w_{p,hh}$', '$w_{p,hm}$', '$w_{p,gg}$', '$w_{p,mm}$'), loc='upper right')
        ax1.legend((line_hg_ratio, line_hh_ratio, line_hm_ratio, line_gg_ratio, line_bg_hg_hm), 
                   ('$b_h \simeq w_{p,hg} \\left(w_{p,gg} w_{p,mm}\\right)^{-1/2}$', '$b_h \simeq \\left(w_{p,hh} / w_{p,mm}\\right)^{1/2}$', '$b_h \simeq \\left(w_{p,hm} / w_{p,mm}\\right)$', '$b_g \simeq \\left(w_{p,gg} / w_{p,mm}\\right)^{1/2}$', '$b_g \simeq w_{p,hg} / w_{p,hm}$'), loc='upper left', fontsize=30)
        plt.savefig("wp_abacus_bias.pdf",dpi=300,bbox_inches='tight')
        
        
# Compare $\xi_{mm}$ with (xihm^2 xigg) / xihg^2
def xi_compare():
    r = xi_dict["r"]
    xi_hg = xi_dict["xi_hg"]
    xi_hm = xi_dict["xi_hm"]
    xi_gg = xi_dict["xi_gg"]
    xi_mm = xi_dict["xi_mm"]
    xi_cg = xi_dict["xi_ab_hg"]
    xi_cm = xi_dict["xi_ab_hm"]
    # print(r)
    # ratio of the average
    xi_mm_compare = [a**2 * b / c**2 for a, b, c in zip(xi_hm, xi_gg, xi_hg)]
    xi_mm_compare_c = [a**2 * b / c**2 for a, b, c in zip(xi_cm, xi_gg, xi_cg)]

    # ratio of the average
    # xi_mm_compare = xi_dict["xi_mm_compare"]
    # xi_mm_compare_c = xi_dict["xi_mm_compare_c"]

    # Plot xicg vs rp
    fig = plt.figure(figsize=(12, 18))
    # set height ratios for sublots
    gs = gridspec.GridSpec(2, 1, height_ratios=[2, 1]) 
    gs.update(wspace=0.025, hspace=0) 

    r_min = 12   #8 for h and 10 for c
    # the fisrt subplot
    ax0 = plt.subplot(gs[0])
    # log scale for axis Y of the first subplot
    ax0.set_yscale("log")
    ax0.set_xscale("log")
    ax0.set_ylabel("$\\xi$")
    line_mm, = ax0.plot(r, xi_mm, color='black', linestyle='-', linewidth=3)
    line_mm_compare, = ax0.plot(r, xi_mm_compare, color='black', linestyle='-.', linewidth=3)
    line_mm_compare_c, = ax0.plot(r, xi_mm_compare_c, color='black', linestyle='dotted', linewidth=3)

    #the second subplot
    # shared axis X
    ax1 = plt.subplot(gs[1], sharex = ax0)
    line_xi_mm_ratio = ax1.plot(r, [n/m for n, m in zip(xi_mm_compare, xi_mm)], color='black', linestyle='-.', linewidth = 3)
    line_xi_mm_ratio_c = ax1.plot(r, [n/m for n, m in zip(xi_mm_compare_c, xi_mm)], color='black', linestyle='dotted', linewidth = 3)

    plt.setp(ax0.get_xticklabels(), visible=False)
    ax1.set_xlabel("$r$ [$h^{-1}$Mpc]")
    ax1.set_ylabel("$\mathrm{ratio\ to}\ \\xi_{mm}$")
    ax1.set_ylim(0.9, 1.1)
    ax1.axhline(1, xmin=0, xmax=1000, color='black', linestyle=':', linewidth=3)
    yticks = ax1.yaxis.get_major_ticks()
    yticks[-1].label1.set_visible(False)

    ax0.legend((line_mm, line_mm_compare, line_mm_compare_c), 
               ('$\\xi_{mm}$','$\\xi_{mm}^h\equiv\\frac{\\left(\\xi_{hm}\\right)^2\\xi_{gg}}{\\left(\\xi_{hg}\\right)^2}$', '$\\xi_{mm}^c\equiv\\frac{\\left(\\xi_{cm}\\right)^2\\xi_{gg}}{\\left(\\xi_{cg}\\right)^2}$'), loc='lower left')
    # ax1.legend((line_xi_mm_ratio), 
    #            ('xi_mm'), loc='upper left', fontsize=30)
    # remove vertical gap between subplots
    plt.subplots_adjust(hspace=.0)
    plt.savefig("xi_abacus_compare_ximm.pdf",dpi=300,bbox_inches='tight')

    
# Compare $wp_{mm}$ with (wphm^2 wpgg) / wphg^2
def wp_compare():
    r = wp_dict["rp"]
    wp_hg = wp_dict["wp_hg"]
    wp_hm = wp_dict["wp_hm"]
    wp_gg = wp_dict["wp_gg"]
    wp_mm = wp_dict["wp_mm"]
    wp_cg = wp_dict["wp_ab_hg"]
    wp_cm = wp_dict["wp_ab_hm"]
    # print(r, wp_hg)
    wp_mm_compare = [a**2 * b / c**2 for a, b, c in zip(wp_hm, wp_gg, wp_hg)]
    wp_mm_compare_c = [a**2 * b / c**2 for a, b, c in zip(wp_cm, wp_gg, wp_cg)]

    # Plot xicg vs rp
    fig = plt.figure(figsize=(12, 18))
    # set height ratios for sublots
    gs = gridspec.GridSpec(2, 1, height_ratios=[2, 1]) 
    gs.update(wspace=0.025, hspace=0) 

    # the first subplot
    ax0 = plt.subplot(gs[0])
    # log scale for axis Y of the first subplot
    ax0.set_yscale("log")
    ax0.set_xscale("log")
    ax0.set_ylabel("$w_p$")
    line_mm, = ax0.plot(r, wp_mm, color='black', linestyle='-', linewidth=3)
    line_mm_compare, = ax0.plot(r, wp_mm_compare, color='black', linestyle='-.', linewidth=3)
    line_mm_compare_c, = ax0.plot(r, wp_mm_compare_c, color='black', linestyle='dotted', linewidth=3)

    #the second subplot
    # shared axis X
    ax1 = plt.subplot(gs[1], sharex = ax0)
    line_wp_mm_ratio = ax1.plot(r, [n/m for n, m in zip(wp_mm_compare, wp_mm)], color='black', linestyle='-.', linewidth = 3)
    line_wp_mm_ratio_c = ax1.plot(r, [n/m for n, m in zip(wp_mm_compare_c, wp_mm)], color='black', linestyle='dotted', linewidth = 3)

    plt.setp(ax0.get_xticklabels(), visible=False)
    ax1.set_xlabel("$r_p$ [$h^{-1}$Mpc]")
    ax1.set_ylabel("$\mathrm{ratio\ to}\ w_{p,mm}$")
    ax1.set_ylim(0.9, 1.1)
    ax1.axhline(1, xmin=0, xmax=1000, color='black', linestyle=':', linewidth=3)
    yticks = ax1.yaxis.get_major_ticks()
    yticks[-1].label1.set_visible(False)

    ax0.legend((line_mm, line_mm_compare, line_mm_compare_c), 
               ('$w_{p,mm}$','$w_{p,mm}^h\equiv\\frac{\\left(w_{p,hm}\\right)^2w_{p,gg}}{\\left(w_{p,hg}\\right)^2}$', '$w_{p,mm}^c\equiv\\frac{\\left(w_{p,cm}\\right)^2w_{p,gg}}{\\left(w_{p,cg}\\right)^2}$'), loc='lower left')
    plt.subplots_adjust(hspace=.0)
    plt.savefig("wp_abacus_compare_ximm.pdf",dpi=300,bbox_inches='tight')

    
# compare Camb non-linear spectrum with corrfunc linear spectrum
def cal_nonLinear_xi():
    print("importing camb")
    import camb
    import math
    import scipy.integrate as integrate
    import scipy.interpolate as interpolate
    from camb import model, initialpower
    #Now get matter power spectra and sigma8 at redshift 0 and 0.8
    pars = camb.CAMBparams()
    pars.set_cosmology(H0=67.26, ombh2=0.022, omch2=0.1199) #fiducial cosmology in Abacus
    pars.InitPower.set_params(ns=0.9652, As=2e-09*(0.83/0.79)**2)
    #Note non-linear corrections couples to smaller scales than you want
    pars.set_matter_power(redshifts=[0., 0.3], kmax=100.0)
#     pars.set_matter_power(redshifts=[0.3], kmax=100.0)
    #Non-Linear spectra (Halofit)
    pars.NonLinear = model.NonLinear_both
    results = camb.get_results(pars)
    results.calc_power_spectra(pars)
    kh_nonlin, z_nonlin, pk_nonlin = results.get_matter_power_spectrum(minkh=1e-6, maxkh=100, npoints = 800)
#     print("sigma8=", results.get_sigma8())
#     print("sigma8_today=", results.get_sigma8_0())
#     print(type(kh_nonlin))
#     print(type(pk_nonlin))
#     print(np.shape(kh_nonlin))
#     print(np.shape(pk_nonlin[1:,].reshape(800,)))
    np.savetxt("abacus_phases/pk_non_linear_camb.txt",np.transpose(np.array([kh_nonlin, pk_nonlin[1:,].reshape(800,)])))
    
    pk_nonLin = pk_nonlin[1,:]
    print(type(kh_nonlin), type(pk_nonlin))
    pk_model = interpolate.interp1d(kh_nonlin, pk_nonLin)
    rmin = 0.1
    rmax = 100
    nbins = 30
    bins = np.logspace(np.log10(rmin), np.log10(rmax), nbins +1)
    r = (bins[:-1] + bins[1:]) / 2.0
    xi = []
    for r_ in r:
        f = lambda k: pk_model(k) * math.sin(k*r_) * k
        xi_ = 1 / (2*math.pi**2*r_) * integrate.quad(f, 1e-6, 90.,  limit=1000)[0]
        xi.append(xi_)
    np.savetxt("abacus_phases/xi_non_linear_camb.txt",np.transpose(np.array([r, xi])))
    
    
def nonLinear_compare():   
    cal_nonLinear_xi()
    r_xi_mm = xi_dict["r"]
    xi_mm = xi_dict["xi_mm"]
    np.savetxt("abacus_phases/xi_mm_compare_abacus.txt",np.transpose(np.array([r_xi_mm, xi_mm])))
    camb_file = np.loadtxt("abacus_phases/xi_non_linear_camb.txt")
    r_non_linear = camb_file[:,0]
    xi_non_linear = camb_file[:,1]
    print("xi_mm:", xi_mm)
    print("xi_non_linear:", xi_non_linear)
    print(xi_mm/xi_non_linear)
    fig = plt.figure(figsize=(12, 18))
    # set height ratios for sublots
    gs = gridspec.GridSpec(2, 1, height_ratios=[2, 1]) 
    gs.update(wspace=0.025, hspace=0) 
    # the fisrt subplot
    ax0 = plt.subplot(gs[0])
    # log scale for axis Y of the first subplot
    ax0.set_yscale("log")
    ax0.set_xscale("log")
    ax0.set_ylabel("$\\xi$")
    line_xi_mm, = ax0.plot(r_xi_mm, xi_mm, color='black', linestyle='-', linewidth=3)
    line_xi_non_linear, = ax0.plot(r_non_linear, xi_non_linear, color='black', linestyle='-.', linewidth=3)
    plt.subplots_adjust(hspace=.0)
    plt.savefig("xi_lin_nonLin_compare.pdf",dpi=300,bbox_inches='tight')

    
if __name__ == "__main__":
#     run_wp_xi() 
    ave_phase()
    bias_xi()
    bias_wp()
    bias_xi(True)
    bias_wp(True)
    xi_compare()
    wp_compare()
    #nonLinear_compare()
