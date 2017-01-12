function simulate_exp_gp2(t,alpha,beta,ndev)
#
# Function for simulating a GP based upon an exponential correlation
# matrix using an analytic form for the Cholesky decomposition and
# a computation in O(N) operations.  The autocorrelation function is:
#
#    K_ij = alpha * exp(-beta*(|t_i-t_j|))
#
# Requirements:
#  - alpha should be a real positive number.
#  - The times t *must* be sorted in order from least to greatest.
#  - beta may be complex
#  - ndev: normal deviates drawn from N(0,1) with the same length as time vector t.
# Output:
#  - data is a GP drawn from this correlation function with length nt
#
nt = length(t)
data = zeros(eltype(beta),nt)
data[1] = sqrt(alpha/2.)*ndev[1]
gamma = zero(eltype(beta))
for i=2:nt
  gamma = exp(-beta*(t[i]-t[i-1]))
#  if i == 1
#     println(i," ",abs(sqrt(1.0-gamma^2))," ",abs(gamma))
#  end
  data[i] = sqrt(1.0-gamma^2)*sqrt(alpha/2.)*ndev[i]+gamma*data[i-1]
end
return data
end
