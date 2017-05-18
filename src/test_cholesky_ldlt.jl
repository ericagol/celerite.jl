# Translating DFM's python version:
using PyPlot
#using celerite
include("gp.jl")
include("sturms_theorem.jl")

#function cholesky_ssm_complex(J0,J) # returns N_test,time_complex
#N = 1000
# Number of real, exponential celerite kernel terms:
J0 = 0
# Total number of complex terms:
J = 16

# Iterate until we have a positive definite kernel (defined by Sturm's theorem):
num_pos_root = 1
aj = rand(J)
bj = [zeros(Float64,J0);rand(J-J0)]
cj = rand(Float64,J)
dj = [zeros(Float64,J0);rand(J-J0)]
while num_pos_root > 0
  x = zeros(Float64,0)
  for j=1:J
    push!(x,aj[j],bj[j],cj[j],dj[j])
    num_pos_root = sturms_theorem(x)
  end
  if num_pos_root > 0
    aj = rand(J)
    bj = [zeros(Float64,J0);rand(J-J0)]
    cj = rand(J)
    dj = [zeros(Float64,J0);rand(J-J0)]
  end
end


# Run a timing test:
#N_test = [64,128,256,512,1024,2048,4096,8192,16384,32768,65536,131072,262144,524288]
N_test = [512]
#N_test = [64,256,1024,4096,16384]
#N_test = [N]
ntest = length(N_test)
time_complex = zeros(Float64,ntest)
itest = 1
time_prior = 0.0
yerr = 0.1
#while itest <= ntest && time_prior < 3.0
ntrial = 2
#while itest <= ntest
  N = N_test[itest]
# Generate some random time data:
  t = sort(rand(N)).*100
  y0 = sin.(t)
  kernel = RealTerm(log(aj[1]),log(cj[1]))
  for i=2:J0
    kernel = kernel + RealTerm(log(aj[i]),log(cj[i]))
  end
  for i=J0+1:J
    kernel = kernel + ComplexTerm(log(aj[i]),log(bj[i]),log(cj[i]),log(dj[i]))
  end
  gp = Celerite(kernel)
# Cholesky method
# Compute Cholesky factorization:
  if itest == 9
    logdet_test = compute_ldlt!(gp, t, yerr)
    @profile logdet_test = compute_ldlt!(gp, t, yerr)
    Profile.print(format=:flat)
#  else
  end
  logdet_test = 0.0
  logdet_test = compute_ldlt!(gp, t, yerr)
  time_zero = tic()
  for itrial = 1:ntrial
    logdet_test = compute_ldlt!(gp, t, yerr)
  end
#  end
  time_complex[itest]=toc()/ntrial
# Now do full solve (if N_test isn't too big):
  if N < 2000
    logdetK,K = full_solve(t,y0,aj,bj,cj,dj,yerr)
    println("Determinant: ",logdetK," ",logdet_test)
    println("Vector: ", maximum(abs.((K \ y0) .- apply_inverse_ldlt(gp, y0))))
  end
  println(N_test[itest]," ",time_complex[itest])
  time_prior = time_complex[itest]
  M = N*4
  tpred = sort(rand(M)).*200
  ypred = predict_ldlt!(gp, t, y0, tpred)
  ypred_full = predict_full_ldlt(gp, y0, tpred; return_cov = false)
  itest +=1
#end
scatter(t,y0)
plot(tpred,ypred)
plot(tpred,ypred_full)
println("Prediction error: ", maximum(abs.(ypred .- ypred_full)))

#loglog(N_test,time_complex)
data = readdlm("c_speed.txt",',')
N_c = vec(data[:,4])
t_c = vec(data[:,5])
#loglog(N_test,(N_test./256).*2e-3)
#loglog(N_c,t_c)
#println(time_complex./((N_test./256).*2e-3))
println(time_complex./t_c)

## Convert from X' back to X:
## (Not sure if I need to do this)
#for n=1:N
#  for j=1:J
#    X[n,2*j-1] *= exp(cj[j]*t[n])
#    X[n,2*j  ] *= exp(cj[j]*t[n])
#  end
#end

# Check factorization:
#L = tril(*(u, X'), -1)
#for i=1:N
#  L[i,i]=D[i]
#end
  
#println("Cholesky error: ",maximum(abs(*(L, L') - K)))

