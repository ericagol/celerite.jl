# Translating DFM's python version:
@testset "Cholesky LDLT" for (J0, J) in ((1, 1), (0, 16))
    # J0 = Number of real, exponential celerite kernel terms
    # J  = Total number of terms

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
            num_pos_root = celerite.sturms_theorem(x)
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
#    N_test = [512]
    N_test = [32]
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
#    y0 = sin.(t)
    i=1
    if J0 > 0
        kernel = celerite.RealTerm(log(aj[i]),log(cj[i]))
        while i < J0
            i +=1
            kernel = kernel + celerite.RealTerm(log(aj[i]),log(cj[i]))
        end
    end
    if i < J
        if J0 == 0
            kernel = celerite.ComplexTerm(log(aj[i]),log(bj[i]),log(cj[i]),log(dj[i]))
        end
        while i < J
            i +=1
            kernel = kernel + celerite.ComplexTerm(log(aj[i]),log(bj[i]),log(cj[i]),log(dj[i]))
        end
    end
    gp = celerite.Celerite(kernel)
    # Cholesky method
    # Compute Cholesky factorization:
    if itest == 9
        logdet_test = celerite.compute_ldlt!(gp, t, yerr)
        @profile logdet_test = celerite.compute_ldlt!(gp, t, yerr)
        Profile.print(format=:flat)
        #  else
    end
    logdet_test = celerite.compute_ldlt!(gp, t, yerr)
    # Generate random noise, and a realization of the GP:
    noise = randn(N)
    y0 = celerite.simulate_gp_ldlt(gp,noise)
    # Check that this inverse works:
    noise_test = celerite.invert_lower_ldlt(gp,y0)./sqrt(gp.D)
    println("noise recovered? ",maximum(abs(noise-noise_test))," ",maximum(noise)," ",maximum(noise_test))
    time_zero = tic()
    for itrial = 1:ntrial
        logdet_test = celerite.compute_ldlt!(gp, t, yerr)
    end
    #  end
    time_complex[itest]=toc()/ntrial
    # Now do full solve (if N_test isn't too big):
    if N < 2000
        logdetK,K = celerite.full_solve(t,y0,aj,bj,cj,dj,yerr)
# Check the log determinant:
        println("Log det: ",logdetK,logdet_test)
        @test isapprox(logdetK,logdet_test)
# Check that simulation works:
        K_lower = chol(K)'  # chol(K) returns upper triangular cholesky matrix, so we need to transpose.
        y0_full = *(K_lower,noise)
# Check factorization by reconstructing cholesky factor from low-rank decomposition:
        Umat = copy(gp.up)
        Wmat = copy(gp.W)
        for n=1:N
          if J0 > 0
            for j=1:J0
              Umat[j,n] *= exp(-cj[j]*t[n])
              Wmat[j,n] *= exp( cj[j]*t[n])
            end
          end
          if (J-J0) > 0
            for j=1:J-J0
              Umat[J0+2j-1,n] *= exp(-cj[j]*t[n])
              Umat[J0+2j  ,n] *= exp(-cj[j]*t[n])
              Wmat[J0+2j-1,n] *= exp( cj[j]*t[n])
              Wmat[J0+2j  ,n] *= exp( cj[j]*t[n])
            end
          end
        end
        L = tril(*(Umat', Wmat), -1)
        # Need to add in identity and then multipy by D^{1/2}:
        for i=1:N
          L[i,i] = 1.0
          for j=1:N
            L[i,j] *=sqrt(gp.D[j])
          end
        end
        println("Cholesky error: ",maximum(abs(L - K_lower)))
        println("Cholesky error: ",maximum(abs(*(L, L') - K)))
        println("y0: ",y0)
        println("y0_full: ",y0_full)
        println("D: ",gp.D)
        println("y0: ",maximum(abs(y0-y0_full)))
# Check that the solver works:
        z = celerite.apply_inverse_ldlt(gp, y0)
        z_full = K \ y0
        println("K \\ y0: ",maximum(abs(z_full-z)))
        @test isapprox(z_full,z)
# Check that the "chi-square" gives the correct value:
        println("N: ",N," dot(y0,z): ",dot(y0,z)," dot(noise,noise): ",dot(noise,noise))
#        @test isapprox(dot(y0,z),dot(noise,noise))
# Check that the log likelihood is computed correctly:
        nll = celerite.log_likelihood_ldlt(gp, y0)
        nll_full = -.5*(logdetK+N*log(2pi)+dot(z_full,z_full))
        println("Log likelihoods: ",nll," ",nll_full)
        @test isapprox(nll,nll_full)
    end
    println(N_test[itest]," ",time_complex[itest])
    time_prior = time_complex[itest]
    M = N*4
    tpred = sort!(rand(M)) .* 200
    ypred = celerite.predict_ldlt!(gp, t, y0, tpred)
    ypred_full = celerite.predict_full_ldlt(gp, y0, tpred; return_cov = false)
    itest +=1
    #end
    @test isapprox(ypred,ypred_full)

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

end
