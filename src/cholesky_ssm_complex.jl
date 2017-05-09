# Translating DFM's python version:
include("terms.jl")
include("cholesky_dfm.jl")

type Celerite
    kernel::Term
    computed::Bool
    D::Vector{Float64}
    Xp::Array{Float64}
    up::Array{Float64}
    phi::Array{Float64}
    x::Vector{Float64}
    logdet::Float64
    n::Int64
    J::Int64

#    Celerite(kernel) = new(kernel, false, [], [], [], [], [])
    Celerite(kernel) = new(kernel, false, zeros(Float64,0),zeros(Float64,0,0), zeros(Float64,0,0), zeros(Float64,0,0), zeros(Float64,0))
end

include("predict.jl")
include("predict_full.jl")

function cholesky!(a_real::Vector{Float64}, c_real::Vector{Float64},
                   a_comp::Vector{Float64}, b_comp::Vector{Float64}, 
                   c_comp::Vector{Float64}, d_comp::Vector{Float64},
                   t::Vector{Float64}, var::Vector{Float64}, Xp::Array{Float64,2}, 
                   phi::Array{Float64,2}, up::Array{Float64,2}, D::Vector{Float64})

  # Compute all the dimensions
  J_real = length(a_real)
  J_comp = length(a_comp)
  N = length(t)
  J = J_real + 2 * J_comp
  # Check that dimensions match:
  assert(length(a_real)==length(c_real))
  assert(length(a_comp)==length(b_comp))
  assert(length(a_comp)==length(c_comp))
  assert(length(a_comp)==length(d_comp))
  assert(length(t)==length(var))
  suma = 0.0
  for j=1:J_real
    suma += a_real[j]
  end
  for j=1:J_comp
    suma += a_comp[j]
  end
  # Set up the matrices
  # Allocate storage for diagonal and for Cholesky Xp vector:
  if size(Xp) != (J, N) || size(phi) != (J, N-1) || size(D) != N || size(up) != (J,N)
# The following is Xp=exp(-c*t)*X:
    Xp   = zeros(Float64, J, N)
    phi = zeros(Float64, J, N-1)
    up   = zeros(Float64, J, N)
    D   = zeros(Float64, N)
  else
# Zero-out the arrays & vectors:
    fill!(Xp, 0.0)
    fill!(phi, 0.0)
    fill!(up, 0.0)
    fill!(D, 0.0)
  end
# Allocate temporary storage matrices:
  S = zeros(Float64,J,J)
  St = zeros(Float64,J,J)
# Then the rest
  vi = zeros(Float64,J)
# Compute a vector that is used in computing S, Xp:
  for i=1:N-1
# Compute the time difference:
    dt=t[i+1]-t[i]
    for j=1:J_real
      phi[j,i]=exp(-c_real[j]*dt)
    end
    for j=1:J_comp
      phi[J_real+2*j-1,i]=exp(-c_comp[j]*dt)
      phi[J_real+2*j  ,i]=phi[J_real+2*j-1,i]
    end
  end

# Explicit first step
  n=1
  D[n] = sqrt(var[n]+suma)
  for j=1:J_real
#  X' = V'/D
    Xp[j,n] = 1.0 / D[n]
    up[j,n] = a_real[j]
  end
  for j=1:J_comp
#  X' = V'/D
    cd = cos(d_comp[j]*t[n])
    sd = sin(d_comp[j]*t[n])
    Xp[J_real+2*j-1,n] = cd / D[n]
    Xp[J_real+2*j  ,n] = sd / D[n]
    up[J_real+2*j-1,n] = a_comp[j]*cd+b_comp[j]*sd
    up[J_real+2*j  ,n] =-b_comp[j]*cd+a_comp[j]*sd
  end
  S = *(Xp,Xp')
# loop over the remaining time steps
  for n=2:N
    for j=1:J_real
      up[j,n]= a_real[j]
      vi[j]= 1.0
    end
    for j=1:J_comp
      cd = cos(d_comp[j]*t[n])
      sd = sin(d_comp[j]*t[n])
      up[J_real+2*j-1,n]= a_comp[j]*cd+b_comp[j]*sd
      up[J_real+2*j  ,n]=-b_comp[j]*cd+a_comp[j]*sd
      vi[J_real+2*j-1]= cd
      vi[J_real+2*j  ]= sd
    end
    St = *(phi[1:J,n-1],phi[1:J,n-1]').*S
    sumas=sum(*(up[1:J,n],up[1:J,n]').*St)
    D[n] = sqrt(var[n]+suma - sumas)
  # Next, compute the new values of X':
    Xp[1:J,n]=(vi-vec(sum(broadcast(*,up[1:J,n],St),1)))./D[n]
# Final step to updating S matrix:
    S = St + *(Xp[1:J,n],Xp[1:J,n]')
  end
return D,Xp,up,phi
end


function compute!(gp::Celerite, x, yerr = 0.0)
  coeffs = get_all_coefficients(gp.kernel)
  var = yerr.^2 + zeros(Float64, length(x))
# Call the choleksy function to decompose & update
# the components of gp with X,D,V,U,etc. 
  gp.n = length(x)
#  println(size(x)," ",size(var)," ",size(gp.Xp)," ",size(gp.phi)," ",size(gp.up)," ",size(gp.D))
# Something is wrong with the following line, which I need to debug:  [ ]
#  gp.D,gp.Xp,gp.up,gp.phi = cholesky!(coeffs..., convert(Vector{Float64},x), var, gp.Xp, gp.phi, gp.up, gp.D)
  @time gp.D,gp.Xp,gp.up,gp.phi = cholesky_dfm!(coeffs..., x, var, gp.Xp, gp.phi, gp.up, gp.D)
  gp.J = size(gp.Xp)[1]
# Compute the log determinant (square the determinant of the Cholesky factor):
  gp.logdet = 2.0*sum(log(gp.D))
#  gp.logdet = sum(log(gp.D))
  gp.x = x
  gp.computed = true
  return gp.logdet
end

function invert_lower(gp::Celerite,y)
# Applies just the lower inverse:  L^{-1}.y:
  @assert(gp.computed)
  N = gp.n
  @assert(length(y)==N)
  z = zeros(Float64,N)
# The following lines solve L.z = y for z:
  z[1] = y[1]/gp.D[1]
  f = zeros(Float64,gp.J)
  for n =2:N # in range(1, N):
    f = gp.phi[:,n-1] .* (f + gp.Xp[:,n-1] .* z[n-1])
    z[n] = (y[n] - sum(gp.up[:,n].*f))/gp.D[n]
  end
  return z
end

function apply_inverse(gp::Celerite, y)
  @assert(gp.computed)
  N = gp.n
  @assert(length(y)==N)
  z = zeros(Float64,N)
# The following lines solve L.z = y for z:
  z[1] = y[1]/gp.D[1]
  f = zeros(Float64,gp.J)
  for n =2:N # in range(1, N):
    f = gp.phi[:,n-1] .* (f + gp.Xp[:,n-1] .* z[n-1])
    z[n] = (y[n] - sum(gp.up[:,n].*f))/gp.D[n]
  end
# The following solves L^T.z = y for z:
  y = copy(z)
  z = zeros(Float64,N)
  z[N] = y[N] / gp.D[N]
  f = zeros(Float64,gp.J)
  for n=N-1:-1:1 #in range(N-2, -1, -1):
    f = gp.phi[:,n] .* (f +  gp.up[:,n+1].*z[n+1])
    z[n] = (y[n] - sum(gp.Xp[:,n].*f)) / gp.D[n]
  end
# The result is the solution of L.L^T.z = y for z,
# or z = {L.L^T}^{-1}.y = L^{T,-1}.L^{-1}.y
  return z
end

function simulate_gp(gp::Celerite, y)
# Multiplies Cholesky factor times random Gaussian vector (y is N(1,0) ) to simulate
# a Gaussian process.
# If iid is zeros, then draw from random normal deviates:
# Check that Cholesky factor has been computed
# Carry out multiplication
# Return simulated correlated noise vector
N=gp.n
@assert(length(y)==N)
z = zeros(Float64,N)
z[1] = gp.D[1]*y[1]
f = zeros(Float64,gp.J)
for n =2:N # in range(1, N):
    f = gp.phi[:,n-1] .* (f + gp.Xp[:,n-1] .* y[n-1])
    z[n] = gp.D[n]*y[n] + sum(gp.up[:,n].*f)
end
# Returns z=L.y
return z
end

function log_likelihood(gp::Celerite, y)
    @assert(gp.computed)
    if size(y, 2) != 1
        error("y must be 1-D")
    end
    alpha = apply_inverse(gp, y)
    nll = gp.logdet + gp.n * log(2*pi)
    for i in 1:gp.n
        nll = nll + alpha[i] * y[i]
    end
    return -0.5 * nll
end

function full_solve(t::Vector,y0::Vector,aj::Vector,bj::Vector,cj::Vector,dj::Vector,yerr::Float64)
  N = length(t)
  @assert(length(y0)==length(t))
  u = zeros(Float64,N,J*2)
  v = zeros(Float64,N,J*2)
# Compute the full U/V matrices:
  for i=1:N
    for j=1:J
      expcjt = exp(-cj[j]*t[i])
      cosdjt = cos(dj[j]*t[i])
      sindjt = sin(dj[j]*t[i])
      u[i,j*2-1]=aj[j]*expcjt*cosdjt+bj[j]*expcjt*sindjt
      u[i,j*2  ]=aj[j]*expcjt*sindjt-bj[j]*expcjt*cosdjt
      v[i,j*2-1]=cosdjt/expcjt
      v[i,j*2  ]=sindjt/expcjt
    end
  end
# Diagonal components:
  diag = fill(yerr^2 + sum(aj),N)

# Compute the kernel:
  K = zeros(Float64,N,N)
  for i=1:N
    for j=1:N
      K[i,j] = sum(aj.*exp(-cj.*abs(t[i]-t[j])).*cos(dj.*abs(t[i]-t[j]))+bj.*exp(-cj.*abs(t[i]-t[j])).*sin(dj.*abs(t[i]-t[j])))
    end
    K[i,i]=diag[i]
  end

# Check that equation (1) holds:
  K0 = tril(*(u, v'), -1) + triu(*(v, u'), 1)
  for i=1:N
    K0[i,i] = diag[i]
  end
  println("Semiseparable error: ",maximum(abs(K - K0)))
  return logdet(K),K
end
