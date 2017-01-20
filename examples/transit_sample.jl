#!/usr/bin/env python
# -*- coding: utf-8 -*-

#from __future__ import division, print_function

using Optim
include("transit_model_sampled.jl")
include("transit_orb.jl")
include("occultquad.jl")
include("kepler.jl")
#import os
#import sys
#import corner
#import emcee3
#import pickle
#import fitsio
using FITSIO
#import numpy as np
#import matplotlib.pyplot as plt
#from scipy.optimize import minimize

using celerite
using PyPlot
#from plot_setup import setup, get_figsize
#from transit_model import RotationTerm, TransitModel

#setup()
#np.random.seed(42)

fname= "data/kplr001430163-2013011073258_llc.fits"
fits_data = FITS(fname);
t_tmp = float(read(fits_data[2],"TIME"))
sapq = read(fits_data[2],"SAP_QUALITY")
y_tmp = float(read(fits_data[2],"PDCSAP_FLUX"))
yerr_tmp = float(read(fits_data[2],"PDCSAP_FLUX_ERR"))
nt = size(t_tmp)[1]
# See how many "good" data points there are:
good = falses(nt)
good_count = 0
for i=1:nt
  if isfinite(t_tmp[i]) && isfinite(y_tmp[i]) && sapq[i] == 0
    good[i] = true
    good_count += 1
  end
end
# Now, make arrays to store the good data:
N =  1000
t = zeros(N)
y = zeros(N)
yerr = zeros(N)
count = 1
for i=1:nt
  if isfinite(t_tmp[i]) && isfinite(y_tmp[i]) && sapq[i] == 0 && count <= N
    t[count]=t_tmp[i]
    y[count]=y_tmp[i]
    yerr[count]=yerr_tmp[i]
    count +=1
  end
end
dtmax = 0.0
for i=2:N
  if t[i]-t[i-1] > dtmax
    dtmax = t[i]-t[i-1]
  end
end
# The final number of data points we have is count:
# Shift the zero-point of time to be the middle of the dataset:
t -= 0.5*(minimum(t)+maximum(t))

hdr = read_header(fits_data[2]);

#data, hdr = fitsio.read("data/kplr001430163-2013011073258_llc.fits",
#                        header=True)

texp = float(hdr["INT_TIME"] * hdr["NUM_FRM"]) / 60. / 60. / 24.

#N = 1000
#m = data["SAP_QUALITY"] == 0
#m &= np.isfinite(data["TIME"])
#m &= np.isfinite(data["PDCSAP_FLUX"])
#t = np.ascontiguousarray(data["TIME"][m], dtype=np.float64)[:N]
#y = np.ascontiguousarray(data["PDCSAP_FLUX"][m], dtype=np.float64)[:N]
#yerr = np.ascontiguousarray(data["PDCSAP_FLUX_ERR"][m], dtype=np.float64)[:N]
#t -= 0.5 * (t.min() + t.max())

true_params = [
    0.0,         # mean flux
    log(8.0),    # period
    log(0.015),  # Rp / Rs
    log(0.5),    # duration
    0.0,            # t_0
    0.5,            # impact
    0.5,            # q_1
    0.5,            # q_2
]

println("Calling transit model")
true_model = transit_model_sampled(t,true_params)
#u1 = 2.*sqrt(true_params[8])*true_params[9]
#u2 = sqrt(true_params[8])*(1.-2.*true_params[9])
#x = 1.-sqrt(1.-true_params[7]^2)
#depth = exp(2.*true_params[4])*(1.-u1*x-u2*x^2)/(1.-u1/3.-u2/6.)
#print("Depth [ppt]: ",depth*1e3)
plot(t,true_model)
println("Called transit model")
read(STDIN,Char)
# Build the true model
#true_model = TransitModel(
#    texp,
#    0.0,
#    np.log(8.0),    # period
#    np.log(0.015),  # Rp / Rs
#    np.log(0.5),    # duration
#    0.0,            # t_0
#    0.5,            # impact
#    0.5,            # q_1
#    0.5,            # q_2
#)
#true_params = np.array(true_model.get_parameter_vector())

# Inject the transit into the data
true_transit = 1e-3*true_model + 1.0
y = y.*true_transit

# Normalize the data
med = median(y)
y = (y./ med - 1.0).* 1e3
yerr *= 1e3 / med

# Set up the GP model
#mean = TransitModel(
#    texp,
#    0.0,
#    np.log(8.0),
#    np.log(0.015),
#    np.log(0.5),
#    0.0,
#    0.5,
#    0.5,
#    0.5,
#    bounds=[
#        (-0.5, 0.5),
#        np.log([7.9, 8.1]),
#        (np.log(0.005), np.log(0.1)),
#        (np.log(0.4), np.log(0.6)),
#        (-0.1, 0.1),
#        (0, 1.0), (1e-5, 1-1e-5), (1e-5, 1-1e-5)
#    ]
#)
var = std(y)^2
pname = ["mean flux","log(period [d])","log(R_p/R_*)","log(T [d])","t_0 [d]","b","q1","q2","log(sig^2)","log(amplitude)","log(tau)","log(Pstar)","log(f)"]
gpp = [2.*log(var*0.5),log(var),log(0.5*maximum(t)),log(4.5),0.0]
#minp = [-0.5,log(7.9),log(.005),log(0.4),-.1,0,  1e-5,  1e-5, log(var*0.01), log(var*0.01), log(dtmax)                ,  -8.0   ]
#maxp = [ 0.5, log(8.1), log(.100), log(0.6), .1, 1.0, 1-1e-5, 1-1e-5, log(var*100.), log(var*100.), log(maximum(t)-minimum(t)), log(5.0)]
lower = [-0.5, log(7.9), log(.005), log(0.4),-.1, 0.0,   1e-5,   1e-5, log(var*0.01), log(var*0.01), log(dtmax)                ,               3.0*log(texp),   -8.0  ]
upper = [ 0.5, log(8.1), log(.100), log(0.6), .1, 1.0, 1-1e-5, 1-1e-5, log(var*100.), log(var*100.), log(maximum(t)-minimum(t)), 0.5*(maximum(t)-minimum(t)), log(5.0)]

# To do:
# 1).  Add in bounds on parameters. [x]
# 2).  Write a function that calls transit calculation & calls likelihood computation. [ ]
# 3).  Compute the mean GP "prediction". [ ]
# 4).  Run MCMC. [ ]
# 5).  Make plots of lightcurve & results. [ ]

# Likelihood function

# Define width, nex.
p0 = 1
p = 2
m1 = 2(p-p0)+p0+2
width = 2m1+1
nex = (4(p-p0)+2p0+1)*(N-1)+1
aex = zeros(width,nex)
al_small= zeros(m1,nex)
indx = collect(1:nex)
bex = zeros(nex)

param = [true_params;gpp]

function neg_log_like(param)
# Transiting planet parameters:
ptrans = param[1:8]
# Compute the transit model:
trans_model = transit_model_sampled(t,ptrans)
# Gaussian process rotation kernel parameters:
gpp = param[9:13]
alpha_real = zeros(p)
alpha_imag = zeros(p)
beta_real = zeros(p)
beta_imag = zeros(p)
# Convert from Rotation kernel parameters to celerite kernel parameters:
f = exp(gpp[5])
# Define alpha_real, alpha_imag, beta_real, beta_imag
alpha_real[1] = exp(gpp[2])/(2.+f)
alpha_real[1] = exp(gpp[2])*(1.+f)/(2.+f)
beta_real[1] = exp(-gpp[3])
beta_real[2] = exp(-gpp[3])
beta_imag[2] = 2pi*exp(-gpp[4])
# White noise component:
w0 = exp(gpp[1])
# Compute the banded extended matrix:
logdeta = compile_matrix_symm(alpha_real,alpha_imag,beta_real,beta_imag,w0,t,nex,aex,al_small,indx)
# Subtract the mean before computing the likelihood:
resid = y-trans_model
# Use banded LU decomposition to compute the log likelihood:
log_like = compute_likelihood(p,p0,resid,aex,al_small,indx,logdeta,bex)
println(param)
println("-log(like): ",-log_like)
return -log_like
end

# Optimize the likelihood:
tic()
println("Optimizing log likelihood")
xdiff = param
#result = optimize(DifferentiableFunction(neg_log_like), xdiff, lower, upper, Fminbox(), optimizer=LBFGS, optimizer_o= Optim.Options(autodiff = true))
result = optimize(DifferentiableFunction(neg_log_like), xdiff, lower, upper, Fminbox())
#result = optimize(neg_log_like, xdiff)


toq()

 
#kernel = RotationTerm(
#    np.log(np.var(y)), np.log(0.5*t.max()), np.log(4.5), 0.0,
#    bounds=[
#        np.log(np.var(y) * np.array([0.01, 100])),
#        np.log([np.max(np.diff(t)), (t.max() - t.min())]),
#        np.log([3*np.median(np.diff(t)), 0.5*(t.max() - t.min())]),
#        [-8.0, np.log(5.0)],
#    ]
#)

#gp = celerite.GP(kernel, mean=mean, fit_mean=True,
#                 log_white_noise=2*np.log(0.5*yerr.min()),
#                 fit_white_noise=True)
#gp.compute(t, yerr)

param = Optim.minimizer(result)

#print("Initial log-likelihood: ",Optim.minimum(result))
print("Initial log-likelihood: ",neg_log_like(param))
#println(Optim.minimizer(result))

read(STDIN,Char)

# Define the model
def neg_log_like(params, y, gp):
    gp.set_parameter_vector(params)
    return -gp.log_likelihood(y)

# Optimize with random restarts
p0 = gp.get_parameter_vector()
bounds = gp.get_parameter_bounds()
r = minimize(neg_log_like, p0, method="L-BFGS-B", bounds=bounds, args=(y, gp))
gp.set_parameter_vector(r.x)
ml_params = np.array(r.x)
print("Maximum log-likelihood: {0}".format(gp.log_likelihood(y)))

# Compute the maximum likelihood predictions
x = np.linspace(t.min(), t.max(), 5000)
trend = gp.predict(y, t, return_cov=False)
trend -= gp.mean.get_value(t) - gp.mean.mean_flux
mu, var = gp.predict(y, x, return_var=True)
std = np.sqrt(var)
mean_mu = gp.mean.get_value(x)
mu -= mean_mu
wn = np.exp(gp.log_white_noise.value)
ml_yerr = np.sqrt(yerr**2 + wn)

# Plot the maximum likelihood predictions
fig, (ax1, ax2) = plt.subplots(2, 1, sharex=True, figsize=get_figsize(1, 2))
ax1.errorbar(t - t.min(), y, yerr=ml_yerr, fmt=".k", capsize=0)
ax1.plot(x - t.min(), mu)
ax1.set_ylim(-0.72, 0.72)
ax1.yaxis.set_major_locator(plt.MaxNLocator(5))
ax1.set_ylabel("raw [ppt]")

ax2.errorbar(t - t.min(), y-trend, yerr=ml_yerr, fmt=".k", capsize=0)
ax2.plot(x - t.min(), mean_mu - gp.mean.mean_flux)
ax2.set_xlim(0, t.max()-t.min())
ax2.set_ylim(-0.41, 0.1)
ax2.yaxis.set_major_locator(plt.MaxNLocator(5))
ax2.set_ylabel("de-trended [ppt]")
ax2.set_xlabel("time [days]")
fig.savefig("transit-ml.pdf")

# Save the current state of the GP and data
with open("transit.pkl", "wb") as f:
    pickle.dump((gp, y), f, -1)

if os.path.exists("transit.h5"):
    result = input("MCMC save file exists. Overwrite? (type 'yes'): ")
    if result.lower() != "yes":
        sys.exit(0)

# Do the MCMC
def log_prob(params):
    gp.set_parameter_vector(params)
    lp = gp.log_prior()
    if not np.isfinite(lp):
        return -np.inf
    return gp.log_likelihood(y) + lp

# Initialize
print("Running MCMC sampling...")
ndim = len(ml_params)
nwalkers = 32
pos = ml_params + 1e-5 * np.random.randn(nwalkers, ndim)
lp = np.array(list(map(log_prob, pos)))
m = ~np.isfinite(lp)
while np.any(m):
    pos[m] = ml_params + 1e-5 * np.random.randn(m.sum(), ndim)
    lp[m] = np.array(list(map(log_prob, pos[m])))
    m = ~np.isfinite(lp)

# Sample
sampler = emcee3.Sampler(backend=emcee3.backends.HDFBackend("transit.h5"))
with emcee3.pools.InterruptiblePool() as pool:
    ensemble = emcee3.Ensemble(emcee3.SimpleModel(log_prob), pos, pool=pool)
    sampler.run(ensemble, 15000, progress=True)

# Plot the parameter constraints
samples = np.array(sampler.get_coords(discard=5000, flat=True, thin=13))
samples = samples[:, 1:5]
samples[:, :3] = np.exp(samples[:, :3])
truths = np.array(true_params[1:5])
truths[:3] = np.exp(truths[:3])
fig = corner.corner(samples, truths=truths,
                    labels=[r"period", r"$R_\mathrm{P}/R_\star$", r"duration",
                            r"$t_0$"])
fig.savefig("transit-corner.pdf")
