module celerite

export compile_matrix_symm, compute_likelihood

include("terms.jl")

include("compile_matrix_symm.jl")
include("compute_likelihood.jl")

end
