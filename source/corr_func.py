import numpy as np
from Corrfunc.mocks.DDrppi_mocks import DDrppi_mocks
from Corrfunc.theory.DDrppi import DDrppi
import astropy.units as u
from astropy.cosmology import Planck15
from astropy.coordinates import SkyCoord
import scipy.interpolate
from Corrfunc.theory.DD import DD
from Corrfunc.utils import convert_3d_counts_to_cf
from Corrfunc.utils import convert_rp_pi_counts_to_wp


# Define a galaxy class to get zspec+noise from zspec
class Gal:
    """ Galaxy class represents a data point in redmagic"""
    
    def __init__(self, ra=0, dec=0, z=0, sig=0, err=0, z_nor=0):
        """ Create a new galaxy data point with ra,dec,cz values"""
        self.ra = ra
        self.dec = dec
        self.z = z
        self.sig = sig
        self.err = err
        self.z_nor = z_nor

# Select observables in a redshift range
def z_cut(d, z_type, z_low, z_high):
    d_cut = d[d[z_type] > z_low]
    d_cut = d_cut[d_cut[z_type] < z_high]
    return d_cut

# Select observables in a redshift range
def mass_cut(d, mass_type, mass_low, mass_high):
    d_cut = d[d[mass_type] > mass_low]
    d_cut = d_cut[d_cut[mass_type] < mass_high]
    return d_cut

# Select observables in some property range
def prop_cut(d, prop_type, prop_low, prop_high):
    d_cut = d[d[prop_type] > prop_low]
    d_cut = d_cut[d_cut[prop_type] < prop_high]
    return d_cut

# calculate xigg from rp pi with RA, DEC, CZ using DDrppi_mocks
def xi_gg_DDrppiMocks(RA, RA_rand, DEC, DEC_rand, CZ, CZ_rand, nbins, pimax, cosmology, nthreads, rp_min, rp_max):
    bins = np.logspace(np.log10(rp_min), np.log10(rp_max), nbins + 1)
    rp = (bins[0:-1] + bins[1:]) / 2.0
    autocorr = 1
    DD = DDrppi_mocks(autocorr, cosmology, nthreads, pimax, bins, RA.astype(np.float32), DEC.astype(np.float32),
                      CZ.astype(np.float32), verbose=True)
    autocorr = 0
    DR = DDrppi_mocks(autocorr, cosmology, nthreads, pimax, bins, RA.astype(np.float32), DEC.astype(np.float32),
                      CZ.astype(np.float32), RA2=RA_rand.astype(np.float32), DEC2=DEC_rand.astype(np.float32),
                      CZ2=CZ_rand.astype(np.float32), verbose=True)
    autocorr = 1
    RR = DDrppi_mocks(autocorr, cosmology, nthreads, pimax, bins, RA_rand.astype(np.float32), 
                      DEC_rand.astype(np.float32), CZ_rand.astype(np.float32), verbose=True)
    ND1 = float(len(RA))
    NR1 = float(len(RA_rand))
    np.savetxt("xi_gg_DD.txt", DD)
    np.savetxt("xi_gg_DR.txt", DR)
    np.savetxt("xi_gg_RR.txt", RR)
    return 0

# calculate wp with RA,DEC,CZ using DDrppi_mocks
def lz_est(RA, RA_rand, DEC, DEC_rand, CZ, CZ_rand, nbins, pimax, cosmology, nthreads, rp_min, rp_max):
    bins = np.logspace(np.log10(rp_min), np.log10(rp_max), nbins + 1)
    rp = (bins[0:-1] + bins[1:]) / 2.0
    autocorr = 1
    DD = DDrppi_mocks(autocorr, cosmology, nthreads, pimax, bins, RA.astype(np.float32), DEC.astype(np.float32),
                      CZ.astype(np.float32), verbose=True)

    autocorr = 0
    DR = DDrppi_mocks(autocorr, cosmology, nthreads, pimax, bins, RA.astype(np.float32), DEC.astype(np.float32),
                      CZ.astype(np.float32), RA2=RA_rand.astype(np.float32), DEC2=DEC_rand.astype(np.float32),
                      CZ2=CZ_rand.astype(np.float32), verbose=True)

    autocorr = 1
    RR = DDrppi_mocks(autocorr, cosmology, nthreads, pimax, bins, RA_rand.astype(np.float32), 
                      DEC_rand.astype(np.float32), CZ_rand.astype(np.float32), verbose=True)
    
    ND1 = float(len(RA))
    NR1 = float(len(RA_rand))
    wp = convert_rp_pi_counts_to_wp(ND1, ND1, NR1, NR1, DD, DR, DR, RR, nbins, pimax)
#     fac = ND1/NR1
#     wp = []
#     for n in range(0, int(len(RR)/int(pimax))):
#         wp_ = 0
#         for m in range(0, int(pimax)):
#             index = n*int(pimax) + m
#             #print(fac)
#             RR_ = RR[index][4] * (fac ** 2.0)
#             DD_ = DD[index][4]
#             DR_ = DR[index][4] * fac
#             wp_ += 2.0 * ( (DD_ - 2.0 * DR_ + RR_) / RR_ )
#             wp_ += 2.0 * ( DD_ / RR_ - 1)
#         wp.append(wp_)
    return rp, wp;

# calculate wp with XYZ using DDrppi
def wp_corrfunc(car_D1, car_R1, nbins, pimax, cosmology, nthreads):
    bins = np.logspace(np.log10(0.3), np.log10(30.0), nbins + 1)
    rp = (bins[0:-1] + bins[1:]) / 2.0
     # Translate cartesian coordinates so that all values are positive
    x_min = np.min([np.min(car_D1[:,0]), np.min(car_R1[:,0])])
    y_min = np.min([np.min(car_D1[:,1]), np.min(car_R1[:,1])])
    z_min = np.min([np.min(car_D1[:,2]), np.min(car_R1[:,2])])  
    for car in (car_D1, car_R1):
        car[:, 0] -= x_min - 10 # be away from the edge of the box
        car[:, 1] -= y_min - 10
        car[:, 2] -= z_min - 10
    ND1 = len(car_D1[:,0])
    NR1 = len(car_R1[:,0])
    
    # Made a permutation of coordinates here
    X = car_D1[:,0]
    Y = car_D1[:,1]
    Z = car_D1[:,2]
    X_rand = car_R1[:,0]
    Y_rand = car_R1[:,1]
    Z_rand = car_R1[:,2]
    
    autocorr = 1
    DD = DDrppi(autocorr, nthreads, pimax, bins, X, Y, Z, verbose=True, periodic=False)
    
    autocorr = 0
    DR = DDrppi(autocorr, nthreads, pimax, bins, X, Y, Z, X2=X_rand, Y2=Y_rand, Z2=Z_rand, verbose=True, periodic=False)
    
    autocorr = 1
    RR = DDrppi(autocorr, nthreads, pimax, bins, X_rand, Y_rand, Z_rand, verbose=True, periodic=False)
    
    wp = convert_rp_pi_counts_to_wp(ND1, ND1, NR1, NR1, DD, DR, DR, RR, nbins, pimax)
    
    return rp, wp

# Convert RA, DEC, CZ to cartesian coordinates
def convert_car(RA, DEC, CZ):
    zGrid = np.linspace(0, 1.0, 3000)
    disGrid =  [Planck15.comoving_distance(z_).value * 0.68 for z_ in zGrid]
    disInterp = scipy.interpolate.interp1d(zGrid, disGrid)
    R = disInterp(CZ/299792.)
    fac = np.pi /180.
    X = R * np.cos(fac*RA) * np.cos(fac*DEC)
    Y = R * np.sin(fac*RA) * np.cos(fac*DEC)
    Z = R * np.sin(fac*DEC)
    np.savetxt("cartesian.txt",np.transpose(np.array([X, Y, Z])))
    return np.transpose(np.array([X, Y, Z]))

def convert_car_astropy(RA, DEC, CZ):
    zGrid = np.linspace(0, 1.0, 3000)
    disGrid =  [Planck15.comoving_distance(z_).value * 0.68 for z_ in zGrid]
    disInterp = scipy.interpolate.interp1d(zGrid, disGrid)
    R = disInterp(CZ/299792.)
    c = SkyCoord(ra=RA*u.degree, dec=DEC*u.degree, distance=R*u.mpc)
    X = c.cartesian.x
    Y = c.cartesian.y 
    Z = c.cartesian.z
    np.savetxt("cartesian_" + str("1") + ".txt", np.transpose(np.array([X, Y, Z])))
    return np.transpose(np.array([X, Y, Z]))

def xi_pairs_cross(X1, Y1, Z1, X2, Y2, Z2, bins, boxsize):
    xi = []
    r = (bins[:-1] + bins[1:]) / 2.0
    autocorr=0
    nthreads=8
    N1 = len(X1)
    N2 = len(X2)
    DD_counts = DD(autocorr, nthreads, bins, X1, Y1, Z1,
               X2=X2, Y2=Y2, Z2=Z2, periodic=True, verbose=True)
    DD_pairs = []
    rmin = []
    rmax = []
    for DD_ in DD_counts:
        DD_pairs.append(DD_[3])
        rmin.append(DD_[0])
        rmax.append(DD_[1])
    rmin = np.array(rmin)
    rmax = np.array(rmax)
    DD_pairs = np.array(DD_pairs)
    RR = N1*N2/(boxsize**3.) * 4./3 * np.pi * (rmax**3. - rmin**3.)
    xi = DD_pairs/RR - 1
    return xi

def wp_pairs_cross(X1, Y1, Z1, X2, Y2, Z2, pimax, bins, boxsize):
    wp = []
    autocorr=0
    nthreads=8
    N1 = len(X1)
    N2 = len(X2)
    DD_counts = DDrppi(autocorr, nthreads, pimax, bins, X1, Y1, Z1,
               X2=X2, Y2=Y2, Z2=Z2, periodic=True, verbose=True)
    for n in range(0, int(len(DD_counts)/int(pimax))):
        wp_ = 0
        for m in range(0, int(pimax)):
            index = n*int(pimax) + m
            DD_ = DD_counts[index][4]
            RR_ = N1*N2/(boxsize**3.) * 2. * np.pi * (DD_counts[index][1]**2. - DD_counts[index][0]**2.)
            wp_ += 2.0 * ( DD_ / RR_ - 1)
        wp.append(wp_)
    return wp
    
# Integrate xi(r) to get wp
def wp_integrate_xi(xi, r, nbins, pimax, rpmin, rpmax):
    xi_model = scipy.interpolate.interp1d(r, xi)
    wp = []
    bins = np.logspace(np.log10(rpmin), np.log10(rpmax), nbins +1)
    rp = (bins[:-1] + bins[1:]) / 2.0
    step = 0.001
    for i in range(0, len(rp)):
        rp_ = rp[i]
        wp_ = 0
        for pi_ in np.arange(0, pimax, step):
            wp_ += 2*xi_model(np.sqrt(pi_**2 + rp_**2))*step
        wp.append(wp_)
#         print(wp_)
    return wp
    
# Calculate bin and cf values for xi of r interpolation
def xi_r_model(car_D1, car_D2, car_R1, car_R2):
    
    # Translate cartesian coordinates so that all values are positive
    x_min = np.min([np.min(car_D1[:,0]), np.min(car_D2[:,0]), 
                   np.min(car_R1[:,0]), np.min(car_R2[:,0])])
    
    y_min = np.min([np.min(car_D1[:,1]), np.min(car_D2[:,1]), 
                   np.min(car_R1[:,1]), np.min(car_R2[:,1])])
    
    z_min = np.min([np.min(car_D1[:,2]), np.min(car_D2[:,2]), 
                   np.min(car_R1[:,2]), np.min(car_R2[:,2])])
    
    for car in (car_D1, car_D2, car_R1, car_R2):
        car[:, 0] -= x_min - 10
        car[:, 1] -= y_min - 10
        car[:, 2] -= z_min - 10
    ND1 = len(car_D1[:,0])
    ND2 = len(car_D2[:,0])
    NR1 = len(car_R1[:,0])
    NR2 = len(car_R2[:,0])
    
    # Calculate the 3d correlation function using the modified coordinates
    #pimax = 100
    rmin = 0.3
    rmax = np.sqrt(30. ** 2 + (135*2.0) ** 2)
    nthreads = 2
    nbins = 50
    bins = np.logspace(np.log10(rmin), np.log10(rmax), nbins+1)
    #np.savetxt("andres_xi.txt", np.transpose(np.array([bins[1:], bins[:-1], bins[1:]])))

    D1D2 = DD(0, nthreads, bins, car_D1[:,0], car_D1[:,1], car_D1[:,2], 
              X2=car_D2[:,0], Y2=car_D2[:,1],Z2=car_D2[:,2], periodic=False)
    
    D1R2 = DD(0, nthreads, bins, car_D1[:,0], car_D1[:,1], car_D1[:,2], 
              X2=car_R2[:,0], Y2=car_R2[:,1],Z2=car_R2[:,2], periodic=False)
    
    D2R1 = DD(0, nthreads, bins, car_D2[:,0], car_D2[:,1], car_D2[:,2], 
              X2=car_R1[:,0], Y2=car_R1[:,1],Z2=car_R1[:,2], periodic=False)
    
    R1R2 = DD(0, nthreads, bins, car_R1[:,0], car_R1[:,1], car_R1[:,2], 
              X2=car_R2[:,0], Y2=car_R2[:,1],Z2=car_R2[:,2], periodic=False)
    cf = convert_3d_counts_to_cf(ND1, ND2, NR1, NR2, D1D2, D1R2, D2R1, R1R2)
    cf = np.nan_to_num(cf)
    bin_fit = (bins[:-1] + bins[1:]) / 2.0
    np.savetxt("xi_r.txt",np.transpose(np.array([bins[1:], bins[:-1], np.zeros(len(cf)), cf])))
    return

# Interpolate xi_r from bin and cf
def xi_r_interpolate(bin_fit, cf):
    xi_model = scipy.interpolate.interp1d(bin_fit, cf)
    return xi_model

# Abel Transformation to get wp
def wp_abel(xi_model, nbins, pimax, rpmin, rpmax):
    wp = []
    bins = np.logspace(np.log10(rpmin), np.log10(rpmax), nbins +1)
    rp = (bins[:-1] + bins[1:]) / 2.0
    step = 0.001
    for i in range(0, len(rp)):
        rp_ = rp[i]
        wp_ = 0
        for r_ in np.arange(rp_*1.001, pimax, step):
            wp_ += 2*xi_model(r_)*r_/np.sqrt(r_**2 - rp_**2)*step
        wp.append(wp_)
        print(wp_)
    return rp, wp

# nbins = 80 to calculate xi for inv_abel_trans
# nbins = 30 for plotting
def xi_inv_abel(rp, wp, rmin, rmax, nbins):
    xi = []  
    bins = np.logspace(np.log10(rmin), np.log10(rmax), nbins + 1)
    r = (bins[:-1] + bins[1:]) / 2.0
    wp_model = scipy.interpolate.interp1d(rp, wp)
    step = 0.001
    for i in range(0, len(r)):
        r_ = r[i]
        xi_ = 0
        # Use np.trapz to replace the inner for loop
        for rp_ in np.arange(r_*1.0005, 230, step):
            slope = (wp_model(rp_ + step) - wp_model(rp_ )) / step
            #print(slope)
            xi_ += -1/3.1415926 * slope * 1/np.sqrt(rp_**2 - r_**2) * step
        xi.append(xi_)
    return r, xi

# Wang et.al 2019, gaussian function of delta_pi for wpgg
def f_dpi(delta_pi, sigma_z):
    return 1/(np.sqrt(2*3.1415926*2) *sigma_z)  * np.exp(- delta_pi**2/(4*sigma_z**2))

# Wang et.al 2019, gaussian function of delta_pi for wpcg
def f_dpi_wpcg(delta_pi, sigma_z):
    return 1/(np.sqrt(2*3.1415926) *sigma_z)  * np.exp(- delta_pi**2/(2*sigma_z**2))

# Wang et.al 2019, returns xi as a function (matrix) of rp and pi, and xi as a function of rp
def xi_matrix_model(model, nbins, pimax, int_max):
    rp = [] #len(rp) = nbins
    pi = [] #len(pi) = pimax
    wp = [] #len(wp) = nbins
    xi = np.zeros([pimax, nbins])
    f_dpi_list = []
    for i in range(pimax):
        # Define pi bins
        pi.append(i + 1) 
    bins = np.logspace(np.log10(0.3), np.log10(30.0), nbins + 1)
    # Define rp bins
    rp = (bins[:-1] + bins[1:]) / 2.0
    # Convert sigma_z to Mpc/h, disInterp(0.03) = 131.8 Mpc/h
    zGrid = np.linspace(0, 1.0, 3000)
    disGrid =  [Planck15.comoving_distance(z_).value * 0.68 for z_ in zGrid]
    disInterp = scipy.interpolate.interp1d(zGrid, disGrid)
    sigma_z = disInterp(0.025) 
    #sigma_z = 89
    print("sigma_z=", sigma_z)
    for rp_ in range(0, nbins):
        wp_ = 0
        for pi_ in range(0, pimax):
             # Integrate to some multiples of sigma, sqrt(2) comes from the conversion between sigma_z and sigma_pair
            for delta_pi in np.arange(-sigma_z*int_max*np.sqrt(2), sigma_z*int_max*np.sqrt(2), 0.1):
                r = np.sqrt(rp[rp_]**2 + (pi[pi_] - delta_pi) ** 2)
                increment =  model(r) * f_dpi(delta_pi, sigma_z) * 0.1
                #increment =  model(r) * f_dpi_wpcg(delta_pi, sigma_z) * 0.1
                xi[pi_][rp_] += increment
            wp_ += 2 * xi[pi_][rp_]
        print('wp_ = ', wp_)
        wp.append(wp_)
    return rp, pi, xi, wp, f_dpi_list

def lz_cross(RA, RA_rand, DEC, DEC_rand, CZ, CZ_rand, nbins, pimax, cosmology, nthreads, rp_min, rp_max):
    bins = np.logspace(np.log10(rp_min), np.log10(rp_max), nbins + 1)
    
    autocorr = 0
    DR = DDrppi_mocks(autocorr, cosmology, nthreads, pimax, bins, RA.astype(np.float32), DEC.astype(np.float32),
                      CZ.astype(np.float32), RA2=RA_rand.astype(np.float32), DEC2=DEC_rand.astype(np.float32),
                      CZ2=CZ_rand.astype(np.float32), verbose=True)
    return DR