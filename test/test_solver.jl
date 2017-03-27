using Base.Test
import celerite

function test_solver(use_lapack)
    srand(42)
    N = 100
    x = sort(10*rand(N))
    y = sin(x)
    yerr = 0.01 + 0.01*rand(N)

    kernel = celerite.RealTerm(0.5, 1.0) + celerite.SHOTerm(0.1, 2.0, -0.5)
    gp = celerite.Celerite(kernel, use_lapack=use_lapack)

    K = celerite.get_matrix(gp, x)
    for n in 1:N
        K[n, n] = K[n, n] + yerr[n]^2
    end

    # Compute using celerite
    celerite.compute!(gp, x, yerr)
    ll = celerite.log_likelihood(gp, y)

    # Compute directly
    ll0 = -0.5*(sum(y .* (K \ y)) + logdet(K) + N*log(2.0*pi))

    @test isapprox(ll, ll0)
end

test_solver(false)
test_solver(true)
