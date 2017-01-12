using PyPlot

include("simulate_exp_gp2.jl")
nt = 10
t = collect(linspace(0,10,nt))
beta = complex(0.05,1.0)
#beta = 0.3+im*0.0
alpha = 1.0+im*0.0
#println(alpha)
#println(beta)
#beta = abs(randn())+im*randn()
#alpha = randn()+im*randn()
c1 = real(alpha)*real(beta)+imag(alpha)*imag(beta)
c2 = real(alpha)*real(beta)-imag(alpha)*imag(beta)
while c1 < 0 || c2 < 0
  beta = 0.01*abs(randn())+im*randn()
  alpha = randn()+im*randn()
  c1 = real(alpha)*real(beta)+imag(alpha)*imag(beta)
  c2 = real(alpha)*real(beta)-imag(alpha)*imag(beta)
end
println(c1," ",c2)
clf()
#for i=1:1
ndev = randn(nt)
data = simulate_exp_gp2(t,alpha,beta,ndev)
#datac = simulate_exp_gp2(t,conj(alpha),conj(beta),ndev)
#datac = simulate_exp_gp2(t,alpha,conj(beta),ndev)
# Now build matrix & Cholesky:
mat = zeros(nt,nt)
for i=1:nt
  for j=1:nt
    dt = abs(t[i]-t[j])
    mat[i,j]=exp(-real(beta)*dt)*(real(alpha)*cos(imag(beta)*dt)+imag(alpha)*sin(imag(beta)*dt))
  end
end
chmat=chol(mat)
sim=*(transpose(chmat),ndev)
# Now compute cholesky decomposition analytically:
chol_anal = zeros(Complex{Float64},nt,nt)
# Loop over columns (see 10/26/16 notes):
gam = 0.0+im*0.0
for i=1:nt
  if i > 1
    gam = sqrt(1.-exp(-2.0*beta*(t[i]-t[i-1])))
  else
    gam = 1.0
  end
# Loop over rows:
  for j=i:nt
    chol_anal[i,j]=sqrt(alpha)*gam*exp(-beta*(t[j]-t[i]))
  end
end
plot(t,0.5*real(data),color="red")
plot(t,sim,color="blue")
plot(t,sim-0.5*real(data),color="green")
#println(abs(alpha)," ",std(real(data)))
#end
