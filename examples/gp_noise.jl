using PyPlot

nt = 2048
t = collect(linspace(0,10,nt))
s0 = 1.0
period = 0.3
om0 = 2pi/period
q = 1000.0
alpha = s0*om0*(1 + im/sqrt(4.*q^2-1.))
beta = om0/2/q*(1+im*sqrt(4.*q^2-1.))

# Now build matrix & Cholesky:
mat = zeros(nt,nt)
for i=1:nt
  for j=1:nt
    dt = abs.(t[i]-t[j])
    mat[i,j]=exp(-real(beta)*dt)*(real(alpha)*cos(imag(beta)*dt)+imag(alpha)*sin(imag(beta)*dt))
  end
end
chmat=transpose(chol(mat))

nfft = nt
ndev = randn(nt)
sim=*(chmat,ndev)
#plot(t,sim,color="blue")

numuniquepts = ceil(Int64,(nfft+1)/2)

spec = fft(sim)
spec = spec[1:numuniquepts]
spec = abs.(spec).^2/nfft
if rem(nfft,2) == 1
  spec[2:end]=spec[2:end]*2
else
  spec[2:end-1]=spec[2:end-1]*2
end

f = (0:numuniquepts-1)/(t[2]-t[1])/nfft

loglog(f,spec)

# Now compute noisy spectrum:
noise = 50.0
simn = sim + noise*randn(nt)

specn = fft(simn)
specn = specn[1:numuniquepts]
specn = abs.(specn).^2/nfft
if rem(nfft,2) == 1
  specn[2:end]=specn[2:end]*2
else
  specn[2:end-1]=specn[2:end-1]*2
end


loglog(f,specn)
om = 2pi*f
spect = s0*om0^4./((om.^2-om0^2).^2+om0^2.*om.^2./q^2)
loglog(f,spect)
