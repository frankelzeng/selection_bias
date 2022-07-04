import numpy as np
import emcee
import h5py as h5
import corner
import pandas as pd
import seaborn as sns

### Define matplotlib parameters
import matplotlib as mpl
import matplotlib.pyplot as plt 
from matplotlib.ticker import FormatStrFormatter
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

def read():
    mcmc = h5.File("test_1000_wp.h5", 'r')['mcmc']
    chain = mcmc['chain']
    log_prob = mcmc['log_prob']
    print("log_prob is: ", log_prob[0])
    print("chain is: ", chain[0])
    print("finish log_prob check")
    reader = emcee.backends.HDFBackend("test_1000_wp.h5", read_only=True) ## write my own h5 reader
    flatchain = reader.get_chain(flat=True)
    #print(np.shape(flatchain[10000:]))
    print(len(flatchain))
    #ndim, nsamples = 2, 10000
    #np.random.seed(42)
    #samples = np.random.randn(ndim * nsamples).reshape([nsamples, ndim])
    #55prin(t"samples_shape: ", np.shape(samples))
    figure = corner.corner(flatchain[5000:], plot_datapoints=False, plot_density=False, levels=(0.39,0.86), show_titles=True, title_fmt=".4f",  title_kwargs={"fontsize": 15}, labels=[r"$\Omega_m$", r"$\sigma_8$"], label_kwargs={"fontsize": 15}, range=[(0.25,0.3), (0.8,0.86)], truths = [0.265036359274889, 0.83]) #check this
    #corner.corner(flat_chain,bins=20,labels=labels,ranges=ranges,verbose=True,show_titles=True,plot_datapoints=False,plot_density=False,levels=levels,color='black',smooth=0.6,hist_kwargs=dict(linewidth=3),label_kwargs=dict(fontsize=20),contour_kwargs=dict(linewidths=3))
    plt.savefig('corner_wp.png')
    return
    
def read_likelihood():
    dummy = h5.File("test_1000_wp.h5",'r')['mcmc/log_prob']
    #print(dummy[0])
    return

def plot_cov(filename):
    datContent = [i.strip().split() for i in open("cov/"+filename).readlines()] #replace by genfromtxt
    print(np.shape(datContent))
    datContent = np.array(datContent)
    datContent = datContent.astype(float)
    ax = sns.heatmap(datContent, vmin=-1, vmax=1)
    plt.savefig('cov.pdf')
    
    wp_comp_model = np.loadtxt("wp_comp_model.txt")
    print(wp_comp_model)
    diag = np.diagonal(datContent)
    np.around(diag, decimals=2)
    nbins = 10
    rpmin = 10
    rpmax = 100
    pimax = 100
    bins = np.logspace(np.log10(rpmin), np.log10(rpmax), nbins +1)
    rp = (bins[:-1] + bins[1:]) / 2.0
    np.around(rp, decimals=-1)
#     print(len(rp))
#     print(len(diag[10:]))
    fig = plt.figure(figsize=(12, 18))
    # set height ratios for sublots
    gs = gridspec.GridSpec(2, 1, height_ratios=[2, 1]) 
    gs.update(wspace=0.025, hspace=0) 
    # the first subplot
    ax0 = plt.subplot(gs[0])
    ax0.yaxis.set_major_formatter(FormatStrFormatter('%.2f'))
    ax0.xaxis.set_major_formatter(FormatStrFormatter('%.2f'))
    ax0.set_yscale("linear")
    ax0.set_xscale("log")
    ax0.set_xticks(np.arange(10,100,10))
    ax0.set_xticklabels(np.arange(10,90,10))
    ax0.set_ylabel("frac")
    ax0.set_xlabel("rp")
    line, = ax0.plot(rp, diag[10:]/wp_comp_model, color = 'black', linestyle='-', linewidth=3)
    plt.savefig("diag_cov.pdf",dpi=300, bbox_inches='tight')
###
    return

# plot_cov("corr_analytic_1e14_z0p3.txt")
plot_cov("cov_analytic_1e14_z0p3.txt")



