# Tests the LAPACK implementation made by Dan.

import Optim
import PyPlot
import celerite


# Try out the quasi-periodic inference with LAPACK:
rand(42)
#nt = [64,256,1024,4096,16384,65536,262144]
nt = 1024
nj = [1,2,4,8,16,32,64]
njj = length(nj)
timing_vanilla  = Array(Float64,njj)
timing_lapack = Array(Float64,njj)
# The input coordinates must be sorted (time series has a gap)
#t = sort(cat(1, 3.8 * rand(57), 5.5 + 4.5 * rand(68)))
t = collect(linspace(0,10,nt))

# Randomly vary the size of error bars to simulate heteroscedastic noise
yerr = 0.08 + (0.22-0.08)*rand(length(t))

# Noisy model
y = 0.2*(t-5.0) + sin(3.0*t + 0.1*(t-5.0).^2) + yerr .* randn(length(t))

# Linear spacing for true model
true_t = linspace(0, 10, 5000)
true_y = 0.2*(true_t-5) + sin(3*true_t + 0.1*(true_t-5).^2)

for ij = 1:njj
# A 'granulation' component
  Q = 1.0 / sqrt(2.0)
  w0 = 3.0
  S0 = var(y) / (w0 * Q)
  kernel = celerite.SHOTerm(log(S0), log(Q), log(w0))
  if nj[ij] > 1
    for j=1:nj[ij]
# NJ periodic component
      logQ = rand()
      logw0 = rand()
      logS0 = rand()
      kernel = kernel + celerite.SHOTerm(logS0, logQ, logw0)
    end
  end
# Initialize values for computation:
#celerite.TermSum((celerite.SHOTerm(-0.7432145976901582,-0.34657359027997275,1.0986122886681096),celerite.SHOTerm(-1.089788187970131,0.0,1.0986122886681096)))

# First, run without lapack:
  tic()
  gp = celerite.Celerite(kernel, use_lapack = false)
  celerite.compute!(gp, t, yerr)
  log_like_vanilla = celerite.log_likelihood(gp,y)
  timing_vanilla[ij]=toc()
# Now, run with lapack:
  tic()
  gp = celerite.Celerite(kernel, use_lapack = true)
  celerite.compute!(gp, t, yerr)
  celerite.log_likelihood(gp,y)
  log_like_lapack = celerite.log_likelihood(gp,y)
  timing_lapack[ij] = toc()
  println("Without lapack: ",log_like_vanilla," with lapack: ",log_like_lapack)
end

PyPlot.loglog(nj,timing_vanilla)
PyPlot.loglog(nj,timing_lapack,linestyle="dashed")
read(STDIN,Char)


vector = celerite.get_parameter_vector(gp.kernel)
mask = ones(Bool, length(vector))
mask[2] = false  # Don't fit for the first Q
function nll(params)
    vector[mask] = params
    celerite.set_parameter_vector!(gp.kernel, vector)
    celerite.compute!(gp, t, yerr)
    return -celerite.log_likelihood(gp, y)
end

result = Optim.optimize(nll, vector[mask], Optim.LBFGS())
result

vector[mask] = Optim.minimizer(result)
celerite.set_parameter_vector!(gp.kernel, vector)

mu, variance = celerite.predict(gp, y, true_t, return_var=true)
sigma = sqrt(variance)

PyPlot.plot(true_t, true_y, "k", lw=1.5, alpha=0.3)
PyPlot.errorbar(t, y, yerr=yerr, fmt=".k", capsize=0)
PyPlot.plot(true_t, mu, "g")
PyPlot.fill_between(true_t, mu+sigma, mu-sigma, color="g", alpha=0.3)
PyPlot.xlabel("x")
PyPlot.ylabel("y")
PyPlot.xlim(0, 10)
PyPlot.ylim(-2.5, 2.5);

read(STDIN,Char)
PyPlot.clf()

omega = exp(linspace(log(0.1), log(20), 5000))
psd = celerite.get_psd(gp.kernel, omega)

for term in gp.kernel.terms
    PyPlot.plot(omega, celerite.get_psd(term, omega), "--")
end
PyPlot.plot(omega, psd)

PyPlot.yscale("log")
PyPlot.xscale("log")
PyPlot.xlim(omega[1], omega[end])
PyPlot.xlabel("omega")
PyPlot.ylabel("S(omega)");
