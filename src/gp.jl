include("terms.jl")
include("bandec_trans.jl")
include("banbks_trans.jl")

type Celerite
    kernel::Term
    computed::Bool
    a::Array{Float64}
    al::Array{Float64}
    ipiv::Array{Int64}
    logdet::Float64
    n::Int64
    width::Int64
    dim_ext::Int64
    block_size::Int64

    Celerite(kernel) = new(kernel, false, [], [], [])
end

function build_extended_system(alpha_real::Vector, beta_real::Vector,
                               alpha_complex_real::Vector, alpha_complex_imag::Vector,
                               beta_complex_real::Vector, beta_complex_imag::Vector,
                               x::Vector, A::Array, offset_factor)
    # Compute all the dimensions
    p_real = length(alpha_real)
    p_complex = length(alpha_complex_real)
    n = length(x)
    block_size = 2 * p_real + 4 * p_complex + 1
    dim_ext = block_size * (n - 1) + 1
    width = p_real + 2 * p_complex + 2
    if (p_complex == 0)
        width = p_real + 1
    end

    # This will be needed if we decide to use LAPACK too
    offset = offset_factor * width

    # Set up the extended matrix.
    if size(A) != (1+width+offset, dim_ext)
        A = zeros(Float64, 1+width+offset, dim_ext)
    else
        fill!(A, 0.0)
    end

    # Compute the diagonal element
    sum_alpha = sum(alpha_real) + sum(alpha_complex_real)

    # Pre-compute the phis and psis
    phi_real = Array(Float64, p_real)
    phi_complex = Array(Float64, p_complex)
    psi_complex = Array(Float64, p_complex)
    dt = x[2] - x[1]
    for j in 1:p_real
        phi_real[j] = exp(-beta_real[j] * dt)
    end
    for j in 1:p_complex
        amp = exp(-beta_complex_real[j] * dt)
        arg = beta_complex_imag[j] * dt
        phi_complex[j] = amp * cos(arg)
        psi_complex[j] = -amp * sin(arg)
    end

    for k in 1:n-1
        # First column
        col = block_size * (k - 1) + 1
        row = offset + 1
        A[row, col] = sum_alpha
        row = row + 1
        for j in 1:p_real
            A[row, col] = phi_real[j]
            row = row+1
        end
        for j in 1:p_complex
            A[row, col] = phi_complex[j]
            A[row+1, col] = psi_complex[j]
            row = row + 2
        end

        # Block 1
        col = col + 1
        row2 = row - 1
        row = offset
        if k > 1
            row3 = offset - p_real - 2*p_complex
            for j in 1:p_real
                A[row3, col] = phi_real[j]
                A[row, col] = phi_real[j]
                A[row2, col] = -1.0
                col = col + 1
                row = row - 1
            end
            for j in 1:p_complex
                A[row3, col] = phi_complex[j]
                A[row3+1, col] = psi_complex[j]
                A[row, col] = phi_complex[j]
                A[row2, col] = -1.0

                A[row3-1, col+1] = psi_complex[j]
                A[row3, col+1] = -phi_complex[j]
                A[row-1, col+1] = psi_complex[j]
                A[row2, col+1] = 1.0
                col = col + 2
                row = row - 2
            end
        else
            for j in 1:p_real
                A[row, col] = phi_real[j]
                A[row2, col] = -1.0
                col = col+1
                row = row-1
            end
            for j in 1:p_complex
                A[row, col] = phi_complex[j]
                A[row2, col] = -1.0
                A[row-1, col+1] = psi_complex[j]
                A[row2, col+1] = 1.0
                col = col+2
                row = row-2
            end
        end

        # Block 3
        row = offset - p_real - 2*p_complex + 1;
        row3 = row2 - p_real;
        if k-1 < n-2
            # Update the phis and psis
            dt = x[k+2] - x[k+1]
            for j in 1:p_real
                phi_real[j] = exp(-beta_real[j] * dt)
            end
            for j in 1:p_complex
                amp = exp(-beta_complex_real[j] * dt)
                arg = beta_complex_imag[j] * dt;
                phi_complex[j] = amp * cos(arg)
                psi_complex[j] = -amp * sin(arg)
            end

            for j in 1:p_real
                A[row, col] = -1.0
                A[row2 - j + 1, col] = alpha_real[j]
                A[row2 + 1, col] = phi_real[j]
                col = col+1
            end
            for j in 1:p_complex
                A[row, col] = -1.0
                A[row3 - 2*(j-1), col] = alpha_complex_real[j]
                A[row2 + 1, col] = phi_complex[j]
                A[row2 + 2, col] = psi_complex[j]

                A[row, col+1] = 1.0
                A[row3 - 2*(j-1) - 1, col+1] = alpha_complex_imag[j]
                A[row2, col+1] = psi_complex[j]
                A[row2 + 1, col+1] = -phi_complex[j]
                col = col + 2
            end
        else
            for j in 1:p_real
                A[row, col] = -1.0
                A[row2 - j + 1, col] = alpha_real[j]
                col = col+1
            end
            for j in 1:p_complex
                A[row, col] = -1.0
                A[row3 - 2*(j-1), col] = alpha_complex_real[j]
                A[row, col+1] = 1.0
                A[row3 - 2*(j-1) - 1, col+1] = alpha_complex_imag[j]
                col = col + 2
            end
        end

        for j in 1:p_real
            A[row, col] = alpha_real[j]
            row = row+1
        end
        for j in 1:p_complex
            A[row, col] = alpha_complex_real[j]
            A[row+1, col] = alpha_complex_imag[j]
            row = row+2
        end
    end

    A[offset+1, end] = sum_alpha
    return width, dim_ext, block_size, A
end

function compute(gp::Celerite, x, yerr=0.0)
    gp.n = length(x)
    offset_factor = 1
    coeffs = get_all_coefficients(gp.kernel)
    gp.width, gp.dim_ext, gp.block_size, gp.a = build_extended_system(coeffs..., convert(Vector{Float64}, x), gp.a, offset_factor)

    # Add the yerr
    var = yerr^2 + zeros(Float64, length(x))
    offset = offset_factor * gp.width + 1
    for k in 1:length(var)
        j = (k-1)*gp.block_size+1
        gp.a[offset, j] = gp.a[offset, j] + var[k]
    end

    # Resize the work arrays if needed
    if size(gp.al) != (gp.width, gp.dim_ext)
        gp.al = Array(Float64, gp.width, gp.dim_ext)
    end
    if length(gp.ipiv) != gp.dim_ext
        gp.ipiv = Array(Int64, gp.dim_ext)
    end

    bandec_trans(gp.a, gp.dim_ext, gp.width, gp.width, gp.al, gp.ipiv)
    gp.logdet = 0.0
    for i in 1:gp.dim_ext
        gp.logdet += log(abs(gp.a[1, i]))
    end

    gp.computed = true
    return gp.logdet
end

function apply_inverse(gp::Celerite, y)
    if !gp.computed
        error("You must compute the gp first")
    end
    if size(y, 1) != gp.n
        error("Dimension mismatch")
    end

    m = gp.width
    bex = zeros(Float64, gp.dim_ext)

    # Loop over columns
    result = Array(Float64, size(y)...)
    for k in 1:size(y, 2)
        for i in 1:gp.n
            bex[(i-1)*gp.block_size+1] = y[i, k]
        end
        banbks_trans(gp.a, gp.dim_ext, m, m, gp.al, gp.ipiv, bex)
        for i in 1:gp.n
            result[i, k] = bex[(i-1)*gp.block_size+1]
        end
    end
    return result
end

function log_likelihood(gp::Celerite, y)
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
