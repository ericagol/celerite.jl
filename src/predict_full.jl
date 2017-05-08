function get_matrix(gp::Celerite, xs...)
    if length(xs) > 2
        error("At most 2 arguments can be provided")
    end
    local x1::Array
    local x2::Array
    if length(xs) >= 1
        x1 = xs[1]
    else
        if !gp.computed
            error("You must compute the GP first")
        end
        x1 = gp.x
    end
    if length(xs) == 2
        x2 = xs[2]
    else
        x2 = x1
    end

    if size(x1, 2) != 1 || size(x2, 2) != 1
        error("Inputs must be 1D")
    end

    tau = broadcast(-, reshape(x1, length(x1), 1), reshape(x2, 1, length(x2)))
    return get_value(gp.kernel, tau)
end

function predict_full(gp::Celerite, y, t; return_cov=true, return_var=false)
    alpha = apply_inverse(gp, y)
    Kxs = get_matrix(gp, t, gp.x)
    mu = Kxs * alpha
    if !return_cov && !return_var
        return mu
    end

    KxsT = transpose(Kxs)
    if return_var
        v = -sum(KxsT .* apply_inverse(gp, KxsT), 1)
        v = v + get_value(gp.kernel, [0.0])[1]
        return mu, v[1, :]
    end

    cov = get_matrix(gp, t)
    cov = cov - Kxs * apply_inverse(gp, KxsT)
    return mu, cov
end
