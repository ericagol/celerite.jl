import Optim
import PyPlot
#import celerite
include("../src/celerite.jl")

srand(42)

# The input coordinates must be sorted
t = sort(cat(1, 3.8 * rand(57), 5.5 + 4.5 * rand(68)))
yerr = 0.08 + (0.22-0.08)*rand(length(t))
y = 0.2*(t-5.0) + sin.(3.0*t + 0.1*(t-5.0).^2) + yerr .* randn(length(t))

true_t = linspace(0, 10, 5000)
true_y = 0.2*(true_t-5) + sin.(3*true_t + 0.1*(true_t-5).^2)

PyPlot.plot(true_t, true_y, "k", lw=1.5, alpha=0.3)
PyPlot.errorbar(t, y, yerr=yerr, fmt=".k", capsize=0)
PyPlot.xlabel("x")
PyPlot.ylabel("y")
PyPlot.xlim(0, 10)
PyPlot.ylim(-2.5, 2.5);

Q = 1.0 / sqrt(2.0)
w0 = 3.0
S0 = var(y) / (w0 * Q)
kernel = celerite.SHOTerm(log(S0), log(Q), log(w0))

# A periodic component
Q = 1.0
w0 = 3.0
S0 = var(y) / (w0 * Q)
kernel = kernel + celerite.SHOTerm(log(S0), log(Q), log(w0))

celerite.TermSum((celerite.SHOTerm(-0.7432145976901582,-0.34657359027997275,1.0986122886681096),celerite.SHOTerm(-1.089788187970131,0.0,1.0986122886681096)))

gp = celerite.Celerite(kernel)
celerite.compute!(gp, t, yerr)
celerite.log_likelihood(gp, y)

mu, variance = celerite.predict_full(gp, y, true_t, return_var=true)
sigma = sqrt(variance)

PyPlot.plot(true_t, true_y, "k", lw=1.5, alpha=0.3)
PyPlot.errorbar(t, y, yerr=yerr, fmt=".k", capsize=0)
PyPlot.plot(true_t, mu, "g")
PyPlot.fill_between(true_t, mu+sigma, mu-sigma, color="g", alpha=0.3)
PyPlot.xlabel("x")
PyPlot.ylabel("y")
PyPlot.xlim(0, 10)
PyPlot.ylim(-2.5, 2.5);
