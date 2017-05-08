function predict!(gp::Celerite, t, y, x)
    a_real, c_real, a_comp, b_comp, c_comp, d_comp = get_all_coefficients(gp.kernel)
    println("a_real: ",a_real)
    println("c_real: ",c_real)
    println("a_comp: ",a_comp)
    println("b_comp: ",b_comp)
    println("c_comp: ",c_comp)
    println("d_comp: ",d_comp)
    N = length(y)
    M = length(x)
    println("M: ",M)
    println("N: ",N)
    J_real = length(a_real)
    J_comp = length(a_comp)
    J = J_real + 2*J_comp

    b = apply_inverse(gp,y)
    println("b: ",minimum(b),maximum(b))
    Q = zeros(J)
    X = zeros(J)
    pred = zeros(x)
    
    # Forward pass
    m = 1
    while m < M && x[m] <= t[1]
      m += 1
    end
    for n=1:N
        if n < N
          tref = t[n+1] 
        else 
          tref = t[N]
        end
        Q[1:J_real] = (Q[1:J_real] + b[n]).*exp(-c_real.*(tref-t[n]))
        Q[J_real+1:J_real+J_comp] += b[n].*cos(d_comp.*t[n])
        Q[J_real+1:J_real+J_comp] = Q[J_real+1:J_real+J_comp].*exp(-c_comp.*(tref-t[n]))
        Q[J_real+J_comp+1:J] += b[n].*sin(d_comp.*t[n])
        Q[J_real+J_comp+1:J] = Q[J_real+J_comp+1:J].*exp(-c_comp.*(tref-t[n]))

        while m < M+1 && (n == N || x[m] <= t[n+1])
            X[1:J_real] = a_real.*exp(-c_real.*(x[m]-tref))
            X[J_real+1:J_real+J_comp]  = a_comp.*exp(-c_comp.*(x[m]-tref)).*cos(d_comp.*x[m])
            X[J_real+1:J_real+J_comp] += b_comp.*exp(-c_comp.*(x[m]-tref)).*sin(d_comp.*x[m])
            X[J_real+J_comp+1:J]  = a_comp.*exp(-c_comp.*(x[m]-tref)).*sin(d_comp.*x[m])
            X[J_real+J_comp+1:J] -= b_comp.*exp(-c_comp.*(x[m]-tref)).*cos(d_comp.*x[m])

            pred[m] = dot(X, Q)
            m += 1
        end
    end

    # Backward pass
    m = M
    while m >= 1 && x[m] > t[N]
        m -= 1
    end
    fill!(Q,0.0)
    for n=N:-1:1
        if n > 1
          tref = t[n-1] 
        else 
          tref = t[1]
        end
        Q[1:J_real] += b[n].*a_real
        Q[1:J_real] = Q[1:J_real].*exp(-c_real.*(t[n]-tref))
        Q[J_real+1:J_real+J_comp] += b[n].*a_comp.*cos(d_comp.*t[n])
        Q[J_real+1:J_real+J_comp] += b[n].*b_comp.*sin(d_comp.*t[n])
        Q[J_real+1:J_real+J_comp] = Q[J_real+1:J_real+J_comp].*exp(-c_comp.*(t[n]-tref))
        Q[J_real+J_comp+1:J] += b[n].*a_comp.*sin(d_comp.*t[n])
        Q[J_real+J_comp+1:J] -= b[n].*b_comp.*cos(d_comp.*t[n])
        Q[J_real+J_comp+1:J] = Q[J_real+J_comp+1:J].*exp(-c_comp.*(t[n]-tref))

        while m >= 1 && (n == 1 || x[m] > t[n-1])
            X[1:J_real] = exp(-c_real.*(tref-x[m]))
            X[J_real+1:J_real+J_comp] = exp(-c_comp.*(tref-x[m])).*cos(d_comp.*x[m])
            X[J_real+J_comp+1:J] = exp(-c_comp.*(tref-x[m])).*sin(d_comp.*x[m])

            pred[m] += dot(X, Q)
            m -= 1
        end
    end 
  return pred
end
