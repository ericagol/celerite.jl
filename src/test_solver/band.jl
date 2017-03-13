function band(a,m,n,p)
r = (m+1)/2

for i=1:n
  for j=m+1:m+r-1
    a[i,j]=0.0
  end
end

for k=1:n
  max = 0.0
  i = k
  j = r
  while (i <= n) && (j >= 1)
    d = abs[i,j]
    if max < d 
      max = d
      p[k] = i
    end
    i += 1
    j -= 1
  end
end

if max <= eps() 
  m=0
  return 
end

if p[k] != k
  i = r
  j = r+k-p[k]
  while (i <= m+r-1) && (i <= n-k+r)
    c=a[k,i]
    a[k,i]=a[p[k],j]
    a[p[k],j]=c
    i +=1
    j +=1
  end
end

a[k,r]=1.0/a[k,r]
h = r-1
i = k+1
while (h >= 1) && (i <= n)
end

return
end
