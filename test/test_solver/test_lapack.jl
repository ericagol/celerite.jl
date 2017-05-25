using PyPlot
import Base.LAPACK.gbtrf!
import Base.LAPACK.gbtrs!

# Writing a simple test for LAPACK solver to see if I can get the syntax correct:
bandwidth = [1,2,4,8,16,32,64,128,256,512,1024]
#ntime = [64,256,1024,4096,16384,65536,262144]
ntime = [16384]
y = 0.5
dt = zeros(Float64,length(bandwidth))
for n in ntime
count = 1
for band in bandwidth
# First, solve it directly:
#  mat = zeros(n,n)
  ab = zeros(3*band+1,n)
  for i=1:n
    for j=i-band:i+band
      if j >=1 && j<=n
#        mat[i,j] = y^abs(i-j)
        ab[2*band+1+(j-i),i]=y^abs(i-j)
      end
    end
  end
#  println(mat)
  b = randn(n)
#  x = \(mat,b)

# Now use banded matrix:
  b0 = copy(b)
  tic()
  ab0,ipiv = gbtrf!(band,band,n,ab)
  x0 = gbtrs!('N',band,band,n,ab0,ipiv,b0)
  dt[count] = toq()
#  println(band," ",sum(abs(b - *(mat,x)))," ",sum(abs(x-x0))," ",dt[count])
  println(band," ",dt[count])
  count +=1
end
loglog(bandwidth,dt,label="Time elapsed")
loglog(bandwidth,0.1.*(bandwidth./16).^2)
xlabel("J")
ylabel("Time [sec]")
legend(loc="upper left")
end
