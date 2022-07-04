import emcee
import h5py as h5
import multiprocessing
from multiprocessing import Pool
import numpy as np
import argparse
import camb
import math
import time
import scipy.integrate as integrate
import scipy.interpolate as interpolate
import scipy.linalg as linalg
from camb import model, initialpower

parser = argparse.ArgumentParser()
parser.add_argument('backend') #the name of the hdf5 file you output the chain to
parser.add_argument('--previous_chain') #if you need to continue an old chain use this argument
args = parser.parse_args()
## examples of using arg parser
# backend = args.backend
# pimax = float(args.pimax)
##

##### Some quantities to calfulate log-likelihood #####
rmin = 1
rmax = 200
nbins = 20
bins = np.logspace(np.log10(rmin), np.log10(rmax), nbins +1)
r = (bins[:-1] + bins[1:]) / 2.0 # r bins to calculate xi from power spectrum
# obs = input xi_mm average_20

def comp_chisq(obs, model, icov):
    print("begin comp_chisq")
    ###assert obs, model and icov are in the same dimension###
    diff = obs - model
#     dummy = diff.T @ icov @ diff
    chisq = np.dot(np.transpose(diff), np.dot(icov, diff))
    print("end comp_chisq")
    return chisq
# import wphm, wpgg and wphg (strckly ordered as obtained from model) from abacus, make sure the covariance and datavectors
# in the same order
def read_wp_obs():
    wp_file = np.loadtxt("output_txt/wp_ave_phase.txt") #[r, wp_hg, wp_hh, wp_hm, wp_gg, wp_mm], should replaced by wp_c*** 0701
    wp_hm = wp_file[20:,3]
    wp_gg = wp_file[20:,4]
    wp_hg = wp_file[20:,1]
    wp = np.concatenate((wp_hm, wp_gg, wp_hg))
    print("obs is", wp)
    return np.array(wp)
read_wp_obs()
# read the covariances for three observables
def read_cov():
    # create a dummy covariances in a more
#     wp_file = np.loadtxt("cov/wp_ave_phase.txt") #[r, wp_hg, wp_hh, wp_hm, wp_gg, wp_mm]
#     wp_hm = wp_file[20:,3]
#     wp_gg = wp_file[20:,4]
#     wp_hg = wp_file[20:,1]
#     wp = np.concatenate((wp_hm, wp_gg, wp_hg))
#     cov = np.diag((np.array(wp)*0.01)**2)

    cov_hg_gg = np.loadtxt("cov/cov_analytic_1e14_z0p3.txt") # 0-9 wphg, 10-20 wpgg, split and reorder the matrix 0701
    rho_m = 2.77e+11 * 0.314  # omega_m = 0.314
    cov_hm = np.loadtxt("cov/DS_cov.dat")/(rho_m)**2 # 0-9 wphm
    cov = linalg.block_diag(cov_hg_gg, cov_hm)
    return cov
#compute the model
def comp_model(H0, ombh2, omch2, sigma8, bh, bg, r=r):
    print("begin comp_model")
    pars = camb.CAMBparams()
    pars.set_cosmology(H0, ombh2, omch2) #fiducial cosmology in Abacus
    pars.InitPower.set_params(ns=0.9652, As=2e-09*(sigma8/0.79)**2)
    pars.set_matter_power(redshifts=[0., 0.3], kmax=100.0) ####only leave one redshift####
    pars.NonLinear = model.NonLinear_both
    ## New block to calculate result
    results = camb.results.CAMBdata()
    results.calc_power_spectra(pars)
    ## New block to calculate result
    #### This part of code should be replaced
    #results = camb.get_results(pars)
    #results.calc_power_spectra(pars)
    #### This part of code should be replaced
    kh_nonlin, z_nonlin, pk_nonlin = results.get_matter_power_spectrum(minkh=1e-6, maxkh=100, npoints = 800) # npoints could decrease 04/27
    pk_nonLin = pk_nonlin[1,:]
    pk_model = interpolate.interp1d(kh_nonlin, pk_nonLin)
    xi = []
    for r_ in r:
        f = lambda k: pk_model(k) * math.sin(k*r_) * k
        xi_ = 1 / (2*math.pi**2*r_) * integrate.quad(f, 1e-6, 90.,  limit = 1000)[0] # limit could decrease, change to log 04/27
        xi.append(xi_)
    xi = np.array(xi)
    # integrate along xi_model to obtain wp
    xi_model = interpolate.interp1d(r, xi)
    wp = []
    nbins = 10
    rpmin = 10
    rpmax = 100
    pimax = 100
    bins = np.logspace(np.log10(rpmin), np.log10(rpmax), nbins +1)
    rp = (bins[:-1] + bins[1:]) / 2.0
#     step = 0.001 # This is possible to be adjusted. Time profiling of the function, compare wp with different step size 04/27
    for rp_ in rp:
        f = lambda pi_: 2*xi_model(np.sqrt(pi_**2 + rp_**2))
        wp_ = integrate.quad(f, 0, pimax, limit=1000)[0]
#         print(wp_)
        wp.append(wp_)
    wp = np.array(wp)
    print("model is ", np.concatenate((wp*bh, wp*bg**2, wp*bh*bg)))
    # np.savetxt("wp_comp_model.txt",np.transpose(wp))
    # print("end comp_model, wp is:", wp)
    return np.concatenate((wp*bh, wp*bg**2, wp*bh*bg)) ##wp_hm, wp_gg, wp_hg
obs = read_wp_obs() ## reading in abacus result
cov = read_cov()
icov = np.linalg.inv(cov)
#####

#################################################################################################################################
############################################### Defining log-likelihood functions ###############################################
#################################################################################################################################

def lnprob(x):
# 0 - omegaM, 1 - sigma8, 2 - bh, 3 - bg
  if (x[0]<0 or x[0]> 1) or (x[1]<0 or x[1]>1.5) or (x[2]<1 or x[2]>10) or (x[3]<0.5 or x[3]>5): #enforce your priors here, x is an array/list of parameters
    return -np.inf
  else:
    try:
        #Write your likelihood here, compute chisq. You can use if statements to switch between different datavectors, in this case ximm or wpmm.
        #omegaM, sigma8, bh, bg = x
        print("x0=", x[0], "x1=", x[1], "x2=", x[2], "x3=", x[3])
        print("comp_model")
        model = comp_model(H0=67.26, ombh2=0.02222, omch2=x[0]*0.6726**2, sigma8=x[1], bh=x[2], bg=x[3], r=r)
#     make sure obs is array
        print("comp_chisq")
        chisq = comp_chisq(obs, model, icov)
        print("chisq=", chisq)
        return (-chisq / 2.0)
    except:
      return - np.inf

# ##check lnprob##
# print("line 86")
# print(lnprob([0.3, 0.8]))
#################################################################################################################################
#################################################### Initializing Walkers #######################################################
#################################################################################################################################

#X params, 1000 walkers, paper says you want lots of walkers. The variable p0 is an initial point in HOD param space.
ndim, nwalkers = 4, 20

OM_init = np.linspace(0.25, 0.35, nwalkers)
sigma8_init = np.linspace(0.7, 0.9, nwalkers)
bh_init = np.linspace(3.5, 4.5, nwalkers)
bg_init = np.linspace(1, 2, nwalkers)
np.random.shuffle(OM_init)
np.random.shuffle(sigma8_init)
np.random.shuffle(bh_init)
np.random.shuffle(bg_init)
p0 = np.transpose(np.array([OM_init, sigma8_init, bh_init, bg_init]))
i = 0
while(i < len(p0)):
    print("lnprob is ",i, lnprob(p0[i]))
    i += 1
#################################################################################################################################
######################################################## Running emcee ##########################################################
#################################################################################################################################

#### could delete this section
if args.previous_chain:
  old_chain = []
  infile = h5.File(args.previous_chain, 'r')
  dummy = infile['mcmc/chain']
  dummy_chain = np.ndarray.flatten(np.array(dummy))
  dummy_chain = dummy_chain[0::ndim]
  dummy_chain = dummy_chain[np.where(dummy_chain > 0)]
  previous_length = int(len(dummy_chain)/nwalkers)
  print("Previous Length: "+str(previous_length))
  infile.close()
####
print("line 121 is running")

with Pool() as pool: #this setups the multiprocessing, if you request  a pitzer node this will be 20 processes
  if args.previous_chain:
    new_backend = emcee.backends.HDFBackend(args.previous_chain)
    new_sampler = emcee.EnsembleSampler(nwalkers, ndim, lnprob, pool=pool, backend=new_backend)
    new_sampler.run_mcmc(None, 5-previous_length)
  else:
    filename = str(args.backend)
    backend = emcee.backends.HDFBackend(filename, name = 'mcmc')
    backend.reset(nwalkers, ndim)
    sampler = emcee.EnsembleSampler(nwalkers, ndim, lnprob, pool=pool, backend=backend)
    pos, prob, state = sampler.run_mcmc(p0, 1)
    sampler.reset()
    sampler.run_mcmc(pos, nsteps=5, rstate0 = state, log_prob0 = prob)
