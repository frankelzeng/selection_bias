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

def comp_chisq(obs, model_camb, icov):
    ###assert obs, model and icov are in the same dimension###
    diff = obs - model_camb
    chisq = np.dot(np.transpose(diff), np.dot(icov, diff))
    print("chisq from comp_chisq func:", chisq)
    return chisq

# import wphm, wphg and wpgg (strckly ordered as obtained from model) from abacus, make sure the covariance and datavectors
# in the same order
def read_wp_obs():
    wp_file = np.loadtxt("output_txt/wp_ave_phase_ab.txt") #[r, wp_hg, wp_hh, wp_hm, wp_gg, wp_mm], should replaced by wp_c*** 0701
    wp_hm = wp_file[20:,3]
    wp_gg = wp_file[20:,4]
    wp_hg = wp_file[20:,1]
    wp = np.concatenate((wp_hm, wp_hg, wp_gg))
    print("obs is", wp)
    return np.array(wp)

# read the covariances for three observables
def read_cov():
    cov_hg_gg = np.loadtxt("cov/cov_analytic_1e14_z0p3_v2.txt") # 0-9 wphg, 10-20 wpgg, split and reorder the matrix 0701
#     rho_m = 2.77e+11 * 0.314  # omega_m = 0.314
    cov_hm = np.loadtxt("cov/wp_hm_cov.dat") # 0-9 wphm
    cov = linalg.block_diag(cov_hm, cov_hg_gg)
    print(np.shape(cov))
    cov_diag = np.diag(np.diag(cov))
    return cov

#compute the model
def comp_model(H0, ombh2, omch2, sigma8, bh, bg, r=r):
    print("begin comp_model")
    pars = camb.CAMBparams()
    pars.set_cosmology(H0, ombh2, omch2)
    pars.InitPower.set_params(ns=0.9652, As=2.07e-09*(sigma8/0.83)**2)
    pars.set_matter_power(redshifts=[0.29999999, 0.3], kmax=200.0)
    pars.NonLinear = model.NonLinear_both
    results = camb.results.CAMBdata()
    results.calc_power_spectra(pars)
    kh_nonlin, z_nonlin, pk_nonlin = results.get_matter_power_spectrum(minkh=1e-6, maxkh=100, npoints = 800)
    pk_nonLin = pk_nonlin[1,:]
    pk_model = interpolate.interp1d(kh_nonlin, pk_nonLin)
    xi = []
    for r_ in r:
        f = lambda k: pk_model(k) * math.sin(k*r_) * k
        xi_ = 1 / (2*math.pi**2*r_) * integrate.quad(f, 1e-6, 90.,  limit = 1000)[0]
        xi.append(xi_)
    xi = np.array(xi)
    xi_model = interpolate.interp1d(r, xi)
    wp = []
    nbins = 10
    rpmin = 10
    rpmax = 100
    pimax = 100
    bins = np.logspace(np.log10(rpmin), np.log10(rpmax), nbins +1)
    rp = (bins[:-1] + bins[1:]) / 2.0
    for rp_ in rp:
        f = lambda pi_: 2*xi_model(np.sqrt(pi_**2 + rp_**2))
        wp_ = integrate.quad(f, 0, pimax, limit=1000)[0]
        wp.append(wp_)
    wp = np.array(wp)
    print("wp_mm=", ",".join(map(str, wp)))
    print("model is ", ",".join(map(str, np.concatenate((wp*bh, wp*bh*bg, wp*bg**2)))))
    # np.savetxt("wp_comp_model.txt",np.transpose(wp))
    return np.concatenate((wp*bh, wp*bh*bg, wp*bg**2)) ##wp_hm, wp_hg, wp_gg, 

obs = read_wp_obs() ## reading in abacus result
cov = read_cov()
icov = np.linalg.inv(cov)
# print("cov is", cov)
# print("icov is", icov)
#####

#################################################################################################################################
############################################### Defining log-likelihood functions ###############################################
#################################################################################################################################
#omegaM = 0.314, sigma8 = 0.8, bh = 4.0, bg = 1.75 for a test on lnprob(x)
def lnprob(x):
# 0 - omegaM, 1 - sigma8, 2 - bh, 3 - bg
  if (x[0]<0.2 or x[0]>0.4) or (x[1]<0.5 or x[1]>1.) or (x[2]<3. or x[2]>6.) or (x[3]<1. or x[3]>3.): #enforce your priors here, x is an array/list of parameters
    return -np.inf
  else:
    try:
        #Write your likelihood here, compute chisq. You can use if statements to switch between different datavectors, in this case ximm or wpmm.
        #omegaM, sigma8, bh, bg = x
        print("x0 omc=", x[0], "x1 sigma8=", x[1], "x2 bh=", x[2], "x3 bg=", x[3])
        model_camb = comp_model(H0=67.26, ombh2=0.02222, omch2=x[0]*0.6726**2 - 0.02222, sigma8=x[1], bh=x[2], bg=x[3], r=r)
        # make sure obs is array
        chisq = comp_chisq(obs, model_camb, icov)
        print("chisq=", chisq, ", model[0]=", model_camb[0])
        print("lnprob=", -chisq/2.0)
        print("---------------------------------------------")
        return (-chisq / 2.0)
    except:
        return - np.inf
print("test_lnprob :")
lnprob([0.314, 0.83, 4.097, 1.639]) #fiducial
lnprob([0.32423147170450756, 0.8339006133533368, 4.226672331803058, 1.6302754437658702]) #best-fit
lnprob([0.3221, 0.8356, 4.1768, 1.6196]) #center-distribution
       
# model_camb_sktime = [23.34398274, 17.98101815, 13.39205323, 9.60027973, 6.59655283, 4.27085896, 2.60148009, 1.51298191, 0.86094068, 0.64037695, 36.61355905, 28.20208861, 21.00458766, 15.05743097, 10.3462755, 6.69857189, 4.08025682, 2.37301634, 1.35033095, 1.00439071, 17.59321201, 13.55140928, 10.09293205, 7.2352588, 4.97149753, 3.2187364, 1.96060763, 1.14026007, 0.64884866, 0.48262062] #positive chisq 98
# model_camb_local = [23.04090818, 17.90216449, 13.4582044, 9.72079147, 6.70220711, 4.33991548, 2.63787503, 1.51482679, 0.83943462, 0.52852096, 36.13820577, 28.07841163, 21.10834157, 15.246446, 10.51198754, 6.80688268, 4.13733998, 2.37590992, 1.31660006, 0.82895166, 17.3647996, 13.49198115, 10.14278694, 7.32608257, 5.05112396, 3.27078091, 1.98803671, 1.14165047, 0.6326406, 0.39832026] #negative chisq -6
# comp_chisq(obs, model_camb_sktime, icov)
# comp_chisq(obs, model_camb_local, icov)
exit()

# ##check lnprob##
#################################################################################################################################
#################################################### Initializing Walkers #######################################################
#################################################################################################################################

#X params, 1000 walkers, paper says you want lots of walkers. The variable p0 is an initial point in HOD param space.
ndim, nwalkers = 4, 500 

OM_init = np.linspace(0.3, 0.32, nwalkers)
sigma8_init = np.linspace(0.8, 0.85, nwalkers)
bh_init = np.linspace(3.8, 4.2, nwalkers)
bg_init = np.linspace(1.5, 1.8, nwalkers)
np.random.shuffle(OM_init)
np.random.shuffle(sigma8_init)
np.random.shuffle(bh_init)
np.random.shuffle(bg_init)
p0 = np.transpose(np.array([OM_init, sigma8_init, bh_init, bg_init]))

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

with Pool() as pool: #this setups the multiprocessing, if you request  a pitzer node this will be 20 processes
  if args.previous_chain:
    new_backend = emcee.backends.HDFBackend(args.previous_chain)
    new_sampler = emcee.EnsembleSampler(nwalkers, ndim, lnprob, pool=pool, backend=new_backend)
    new_sampler.run_mcmc(None, 5000-previous_length)
  else:
    filename = str(args.backend)
    backend = emcee.backends.HDFBackend(filename, name = 'mcmc')
    backend.reset(nwalkers, ndim)
    sampler = emcee.EnsembleSampler(nwalkers, ndim, lnprob, pool=pool, backend=backend)
    pos, prob, state = sampler.run_mcmc(p0, 1)
    sampler.reset()
    sampler.run_mcmc(pos, nsteps=5000, rstate0 = state, log_prob0 = prob)
