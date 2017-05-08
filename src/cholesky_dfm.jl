function _reshape!(A::Array{Float64}, dims...)
    if size(A) != dims
        A = Array{Float64}(dims...)
    end
    return A
end

function cholesky_dfm!(a_real::Vector{Float64}, c_real::Vector{Float64},
                       a_comp::Vector{Float64}, b_comp::Vector{Float64}, 
                       c_comp::Vector{Float64}, d_comp::Vector{Float64},
                       t::Vector{Float64}, diag::Vector{Float64}, X::Array{Float64,2}, 
                       phi::Array{Float64,2}, u::Array{Float64,2}, D::Vector{Float64})
# Compute the dimensions of the problem:
    N = length(t)
# Number of real components:
    J_real = length(a_real)
# Number of complex components:
    J_comp = length(a_comp)
# Rank of semi-separable components:
    J = J_real + 2*J_comp
# phi is used to stably compute exponentials between time steps:
    phi = _reshape!(phi, J, N-1)
# u, X & D are low-rank matrices and diagonal component:
    u = _reshape!(u, J, N)
    X = _reshape!(X, J, N)
    D = _reshape!(D, N)

# Sum over the diagonal kernel amplitudes:    
    a_sum = sum(a_real) + sum(a_comp)
# Compute the first element:
    D[1] = sqrt(diag[1] + a_sum)
    value = 1.0 / D[1]
    for j in 1:J_real
        u[j, 1] = a_real[j]
        X[j, 1] = value
    end
# We are going to compute cosine & sine recursively - allocate arrays for each complex
# component:
    cd::Vector{Float64} = zeros(J_comp)
    sd::Vector{Float64} = zeros(J_comp)
# Initialize the computation of X:
    for j in 1:J_comp
        cd[j] = cos(d_comp[j]*t[1])
        sd[j] = sin(d_comp[j]*t[1])
        u[J_real+2*j-1, 1] = a_comp[j]*cd[j] + b_comp[j]*sd[j]
        u[J_real+2*j  , 1] = a_comp[j]*sd[j] - b_comp[j]*cd[j]
        X[J_real+2*j-1, 1] = cd[j]*value
        X[J_real+2*j, 1]   = sd[j]*value
    end
# Allocate array for recursive computation of low-rank matrices:   
    S::Array{Float64, 2} = zeros(J, J)
    for j in 1:J
      for k in 1:j
        S[k,j] = X[k,1]*X[j,1]
      end
    end
# Allocate temporary variables:
    phij = 0.0 ; dx = 0.0 ; dcd = 0.0 ; dsd = 0.0 ; cdtmp= 0.0 ; uj = 0.0 ;
    Xj = 0.0 ; Dn = 0.0 ; Sk = 0.0 ; uk = 0.0 ; tmp = 0.0 ; tn = 0.0 ;
# Loop over remaining indices:
    @inbounds for n in 2:N
        # Update phi
        tn = t[n]
# Using time differences stabilizes the exponential component and speeds
# up cosine/sine computation:
        dx = tn - t[n-1]
# Compute real components of the low-rank matrices:
        for j in 1:J_real
            phi[j, n-1] = exp(-c_real[j]*dx)
            u[j, n] = a_real[j]
            X[j, n] = 1.0
        end
# Compute complex components:
        for j in 1:J_comp
            value = exp(-c_comp[j]*dx)
            phi[J_real+2*j-1, n-1] = value
            phi[J_real+2*j, n-1]   = value
            cdtmp = cd[j]
            dcd = cos(d_comp[j]*dx)
            dsd = sin(d_comp[j]*dx)
            cd[j] = cdtmp*dcd-sd[j]*dsd
            sd[j] = sd[j]*dcd+cdtmp*dsd
        # Update u and initialize X
            u[J_real+2*j-1, n] = a_comp[j]*cd[j] + b_comp[j]*sd[j]
            u[J_real+2*j  , n] = a_comp[j]*sd[j] - b_comp[j]*cd[j]
            X[J_real+2*j-1, n  ] = cd[j]
            X[J_real+2*j  , n  ] = sd[j]
        end
        
        # Update S
        for j in 1:J
            phij = phi[j,n-1]
            for k in 1:j
                S[k, j] = phij*phi[k, n-1]*S[k, j]
            end
        end
        
        # Update D and X
        Dn = 0.0
        for j in 1:J
            uj = u[j,n]
            Xj = X[j,n]
            for k in 1:j-1
                Sk = S[k, j]
                tmp = uj * Sk
                uk = u[k,n]
                Dn += uk * tmp
                Xj -= uk*Sk
                X[k, n] -= tmp
            end
            tmp = uj * S[j, j]
            Dn += .5*uj * tmp
            X[j, n] = Xj - tmp
        end
# Finalize computation of D:
        Dn = sqrt(diag[n]+a_sum-2.0*Dn)
        D[n] = Dn
# Finalize computation of X:
        for j in 1:J
            X[j, n] /= Dn
        end
        # Update S
        Xj = 0.0
        for j in 1:J
            Xj = X[j,n]
            for k in 1:j
                S[k, j] += Xj*X[k, n]
            end
        end
    end
# Finished looping over n.  Now return components to the calling routine
# so that these may be used in arithmetic:
    return D,X,u,phi
end
