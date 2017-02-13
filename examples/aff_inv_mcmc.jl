nvary = npararm
ivary = collect(1:nvary)
nwalkers = nvary * 2
nsteps = 1000
# Set up arrays to hold the results:
par_mcmc = zeros(nwalkers,nsteps,nparam)
log_like_mcmc = zeros(nwalkers,nsteps)
prior_mcmc = zeros(nwalkers,nsteps)
# Initialize walkers:
par_trial = copy(param)
log_like_best = copy(log_like)

function compute_prior(param)
# Check that parameters are inbounds:
inbounds = false
for i=1:ivary
  if param[ivary] =< lower[ivary] || param[ivary] >= upper[ivary]
    inbounds = false
  end
end
if inbounds
  log_prior = 1.0
  ilog = [2,3,4,9,10,11,12,13]
  for i=1:8
    log_prior *= exp(param[ilog[i]])
  end
else
  log_prior = -Inf
end
#pname = ["mean flux","log(period [d])","log(R_p/R_*)","log(T [d])","t_0 [d]","b","q1","q2","log(sig^2)","log(amplitude)","log(tau)","log(Pstar)","log(f)"]
return log_prior
end

for j=1:nwalkers
# Select from within uncertainties:
  log_like_trial = 1e100
# Only initiate models with reasonable chi-square values:
  while log_like_trial > (log_like_best + 1e2)
    outbounds = true
    while outbounds
      par_trial[ivary] = param[ivary] + sigma[ivary].*randn(nvary)
      prior_trial = compute_prior(par_trial)
      if isfinite(prior_trial)
        outbounds = true
      end
    end
    log_like_trial = -neg_log_like(par_trial)
    println("log_like_trial: ",log_like_trial," ",par_trial)
  end
  log_like_mcmc[j,1]=log_like_trial
  prior_mcmc[j,1]=prior_trial
  par_mcmc[j,1,:]=par_trial
  println("Success: ",par_trial,log_like_trial)
end
# Initialize scale length & acceptance counter:
ascale = 2.0
accept = 0
param_best = copy(param)
tic()
###  Go through the markov chain steps & make sure they are computed properly. ###
# Next, loop over steps in markov chain:
for i=2:nsteps
  for j=1:nwalkers
    ipartner = j
# Choose another walker to 'breed' a trial step with:
    while ipartner == j
      ipartner = ceil(Int,rand()*nwalkers)
    end
# Now, choose a new set of parameters for the trial step:
    z=(rand()*(sqrt(ascale)-1.0/sqrt(ascale))+1.0/sqrt(ascale))^2
    par_trial = copy(param_plane)
    par_trial[ivary]=vec(z*par_mcmc[j,i-1,ivary]+(1.0-z)*par_mcmc[ipartner,i-1,ivary])
# Compute model & chi-square:
    chi_trial,cov_trial,logdeta_trial=compute_chi(par_trial)
    prior_trial=compute_prior(par_trial,cov_trial)
    if chi_trial < chi_best
      param_best = par_trial
      chi_best = chi_trial
      println("New best chi-square: ",chi_best)
    end
# Next, determine whether to accept this trial step:
#    alp = z^(nparam-1)*exp((-0.5*(chi_trial - chi_mcmc[j,i-1])+prior_trial-prior_mcmc[j,i-1])*temperature)
    alp = z^(nvary-1)*exp((-0.5*(chi_trial - chi_mcmc[j,i-1])+prior_trial-prior_mcmc[j,i-1])*temperature)
    if rand() < 0.0001
      println("Step: ",i," Walker: ",j," Chi-square: ",chi_trial," Prob: ",alp," Frac: ",accept/(mod(i-1,1000)*nwalkers+j))
    end
    if alp >= rand()
# If step is accepted, add it to the chains!
      par_mcmc[j,i,:] = par_trial
      chi_mcmc[j,i] = chi_trial
      prior_mcmc[j,i] = prior_trial
      logdeta_mcmc[j,i]= logdeta_trial
      for ip=1:nplanet
        sigmass_mcmc[j,i,ip]=sqrt(cov_trial[ip*3,ip*3])
      end
      accept = accept + 1
    else
# If step is rejected, then copy last step:
      par_mcmc[j,i,:] = par_mcmc[j,i-1,:]
      chi_mcmc[j,i] = chi_mcmc[j,i-1]
      logdeta_mcmc[j,i] = logdeta_mcmc[j,i-1]
      sigmass_mcmc[j,i,:] = sigmass_mcmc[j,i-1,:]
      prior_mcmc[j,i] = prior_mcmc[j,i-1]
    end
  end
  if mod(i,1000) == 0
    elapsed = toc()
    acc_rate = accept/(1000*nwalkers)
    ascale = 1+(ascale-1)*acc_rate/0.25
    println("Number of steps: ",i," acceptance rate: ",acc_rate," remaining time: ",(nsteps-i)/1000*elapsed/3600," hr; new ascale: ",ascale)
    accept = 0
    tic()
  end
end
toc()


