using Base.Test
using celerite

omega = 2pi/12.203317
alpha_r = [1.0428542, 0.38361831, 0.30345984/2, 0.30345984/2]
alpha_i = [0. ,               0. , 0.1, 0.1]
beta = [complex(0.1229159,0.0),complex(0.48922908,0.0),complex(0.09086397,omega),complex(0.09086397,-omega)]
nt = 512
t = collect(linspace(0,nt-1,nt))
acf = zeros(nt)
p = length(alpha_r)

# First solve in traditional manner:
w = 0.03027 * ones(nt)
A = zeros(Float64,nt,nt)
for i=1:nt
  A[i,i] += w[i]
  for j=1:nt
    for k=1:p
      A[i,j] += alpha_r[k] * exp(-real(beta[k])*abs(t[j]-t[i]))*cos(imag(beta[k])*(t[j]-t[i]))
      A[i,j] += alpha_i[k] * exp(-real(beta[k])*abs(t[j]-t[i]))*sin(abs(imag(beta[k])*(t[j]-t[i])))
    end
  end
end

# Compute a realization of correlated noise with this covariance matrix.
y=randn(nt)
# First, do Cholesky decomposition:
sqrta = chol(A)
# Now make a realization of the correlated noise:
corrnoise=*(transpose(sqrta),y);

# Now, solve for A*y = x to get inverse of Kernel (A) times the correlated noise:
x2 = \(A,corrnoise)

# Take dot product with correlated noise, and compare with original noise realization:
@test_approx_eq dot(y,y) dot(corrnoise,x2)
println("Dot product of white noise:                          ",dot(y,y))
println("Dot product of correlated noise with inverse kernel: ",dot(corrnoise,x2))

# Now use Ambikarasan O(N) method:
y = corrnoise
n = nt
alpha_final = [1.0428542, 0.38361831, 0.30345984]
alpha_imag =  [0. ,  0. ,  0.2]
beta_real_final = [0.1229159,0.48922908,0.09086397]
beta_imag_final = [0.0,0.0,omega]
w0 = 0.03027
p_final = 3
p0_final = 2
nex_final = (4(p_final-p0_final)+2p0_final+1)*(n-1)+1
m1_final = 2(p_final-p0_final)+p0_final+2
if p0_final == p_final
  m1_final = p0_final + 1
end
width_final = 2m1_final+1
aex_final = zeros(Float64,width_final,nex_final)
al_small_final = zeros(Float64,m1_final,nex_final)
indx_final= collect(1:nex_final)
tic()
logdeta_final = compile_matrix_symm(alpha_final,alpha_imag,beta_real_final,beta_imag_final,w0,t,nex_final,aex_final,al_small_final,indx_final)
tic()
bex = zeros(Float64,nex_final)
log_like_final = compute_likelihood(p_final,p0_final,y,aex_final,al_small_final,indx_final,logdeta_final,bex)

# Now select solution:
#x1 = zeros(n)
x3 = zeros(n)
log_like3 = 0.0
for i=1:n
  log_like3 += x2[i]*y[i]
end
log_like3 = -0.5 * log_like3
log_like3 += -0.5*logdet(A) - 0.5*n*log(2*pi)

@test_approx_eq logdeta_final logdet(A)
@test_approx_eq log_like_final log_like3
println("Log Determinant: ",logdeta_final," ",logdet(A))
println("Log Likelihood:  ",log_like_final," ",log_like3)
