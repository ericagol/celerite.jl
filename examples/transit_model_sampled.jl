function transit_model_sampled(time,param)
# Computes a sub-sampled transit model using the same
# parameterization as that used by Foreman-Mackey's model.

# Need to translate these to parameters ("x") called by Agol's transit_orb.jl:
x = zeros(10)
# param = {
#   0.0,            # [1] mean flux
#   np.log(8.0),    # [2] period
#   np.log(0.015),  # [3] Rp / Rs
#   np.log(0.5),    # [4] duration in days
#   0.0,            # [5] t_0
#   0.5,            # [6] impact
#   0.5,            # [7] q_1
#   0.5,            # [8] q_2
# }
period = exp(param[2]) 
x[1] = period # log(Period) -> Period
k = exp(param[3])
#println("k:   ",k)
tdur = exp(param[4])
#println("Tdur: ",tdur)
b = param[6]
#println("b: ",b)
aonr = sqrt(b^2+((1.+k)^2-b^2)/sin(pi*tdur/period)^2)
#println("a/R: ",aonr)
#println("P: ",period)
arg = sqrt((1.+k)^2-b^2)/sin(pi*tdur/period)/aonr
#println("arg: ",arg)
if abs(arg) <= 1
  inc = 180./pi*asin(arg)
else
  inc = 90.
end
#print("Inc: ",inc)
x[2] = inc

x[3] = k
x[4] = param[5]
u1 = 2.*sqrt(param[7])*param[8]
u2 = sqrt(param[7])*(1-2*param[8])
#println("u1: ",u1)
#println("u2: ",u2)
x[5] = u1
x[6] = u2
x[7] = 1.0
x[8] = aonr
x[9] = 0.0 # set eccentricity to zero

# Now we can call the transit_orb.jl routine:
lc = zeros(time)
for i=1:size(time)[1]
  lc[i] = transit_orb(time[i],x,texp,10)
end
# add the mean flux, and convert to ppt (parts-per-thousand):
return (lc-1.0)*1e3+param[1]
end
